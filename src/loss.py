import json
import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from src import dist_utils


def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _default_targets(query_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:
    target_per_query = target_embeddings.size(0) // query_embeddings.size(0)
    return torch.arange(
        0,
        query_embeddings.size(0) * target_per_query,
        target_per_query,
        device=query_embeddings.device,
        dtype=torch.long,
    )


class TrainingDynamicsLogger:
    def __init__(self, default_log_path: str = "./logs/step_dynamics.jsonl"):
        self.default_log_path = default_log_path

    def log_step_dynamics(self, *,
        log_path: str = None,
        mean_abs_dLdk: torch.Tensor,
        mean_abs_dLdt: torch.Tensor,
        kt_gradient_ratio: torch.Tensor,
        infotn_positive_mean: torch.Tensor,
        infonce_gradient_norm: torch.Tensor,
        dinfotn_dk: torch.Tensor,
        dinfotn_dt: torch.Tensor,
        infonce_similarity_grad_norm: torch.Tensor,
        infotn_similarity_grad_norm: torch.Tensor,
        infonce_loss: torch.Tensor,
        infotn_loss: torch.Tensor,
        infotn_pair_loss: torch.Tensor,
        norm_ratio_mean,
        norm_ratio_std,
        norm_ratio_min,
        norm_ratio_max,
        norm_ratio_values,
        extra: dict = None,
    ):
        """
        记录单步训练动态；除 `self` 外，所有参数必须用关键字传入。
        Log one-step training dynamics; all arguments except `self` must be keyword-only.
        """
        log_path = log_path or self.default_log_path

        def to_scalar(x):
            return x.item() if isinstance(x, torch.Tensor) else x

        record = {
            "timestamp": datetime.now().isoformat(),
            "mean_abs_dLdk": to_scalar(mean_abs_dLdk),
            "mean_abs_dLdt": to_scalar(mean_abs_dLdt),
            "kt_gradient_ratio": to_scalar(kt_gradient_ratio),
            "infotn_positive_mean": to_scalar(infotn_positive_mean),
            "infonce_gradient_norm": to_scalar(infonce_gradient_norm),
            "dinfotn_dk": to_scalar(dinfotn_dk),
            "dinfotn_dt": to_scalar(dinfotn_dt),
            "infonce_similarity_grad_norm": to_scalar(infonce_similarity_grad_norm),
            "infotn_similarity_grad_norm": to_scalar(infotn_similarity_grad_norm),
            "infonce_loss": to_scalar(infonce_loss),
            "infotn_loss": to_scalar(infotn_loss),
            "infotn_pair_loss": to_scalar(infotn_pair_loss),
            "norm_ratio_mean": norm_ratio_mean,
            "norm_ratio_std": norm_ratio_std,
            "norm_ratio_min": norm_ratio_min,
            "norm_ratio_max": norm_ratio_max,
            "norm_ratio_values": norm_ratio_values,
        }
        if extra:
            record.update(extra)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

class SimpleContrastiveLoss:
    def __init__(self, temperature: float = 0.02):
        self.temperature = temperature

    def __call__(
        self,
        query_embeddings: Tensor,
        target_embeddings: Tensor,
        target: Tensor = None,
        reduction: str = "mean",
    ) -> Tensor:
        if target is None:
            target = _default_targets(query_embeddings, target_embeddings)

        logits = torch.matmul(query_embeddings, target_embeddings.transpose(0, 1))
        loss = F.cross_entropy(logits / self.temperature, target, reduction=reduction)
        return loss


def js_divergence(p, q):
    """
    计算两个概率分布的 JS 散度。
    Compute the Jensen-Shannon divergence between two probability distributions.
    """
    midpoint = 0.5 * (p + q)
    return 0.5 * F.kl_div(p.log(), midpoint, reduction="batchmean") + 0.5 * F.kl_div(
        q.log(), midpoint, reduction="batchmean"
    )


def compute_js_loss(similarity_matrix):
    """
    计算相似度矩阵行分布与列分布之间的 JS 散度。
    Compute JS divergence between row-wise and column-wise similarity distributions.
    """
    row_probabilities = F.softmax(similarity_matrix, dim=1)
    column_probabilities = F.softmax(similarity_matrix, dim=0)
    return js_divergence(row_probabilities, column_probabilities.T)


class InfoTNLoss:
    def __init__(self, temperature: float = 0.02, init_white=1e-4):
        self.temperature = temperature
        self.init_whitening = torch.nn.Parameter(torch.tensor(init_white))
        self.logger = TrainingDynamicsLogger("./logs/infotn_training_dynamics.jsonl")

    def safe_logdet(self, covariance: Tensor, eps: float = 1e-6) -> Tensor:
        """
        稳定计算协方差矩阵的 log(det(.))。
        Safely compute log(det(.)) for a covariance matrix.
        """
        try:
            cholesky_factor = torch.linalg.cholesky(covariance)
            logdet = 2 * torch.diagonal(cholesky_factor, dim1=-2, dim2=-1).log().sum()
        except RuntimeError:
            _, singular_values, _ = torch.svd(covariance)
            logdet = torch.log(singular_values.clamp(min=eps)).sum()
        return logdet

    def fused_similarity(
        self,
        query_embeddings: Tensor,
        target_embeddings: Tensor,
        query_embeddings_unnorm: Tensor,
        target_embeddings_unnorm: Tensor,
        eps: float = 1e-8,
    ):
        """
        计算方向相似度、InfoTN 模长相似度和诊断量。
        Compute directional similarity, InfoTN magnitude-aware similarity, and diagnostics.
        """
        cosine_similarity = torch.matmul(query_embeddings, target_embeddings.transpose(0, 1))
        unnormalized_dot_similarity = torch.matmul(
            query_embeddings_unnorm, target_embeddings_unnorm.transpose(0, 1)
        )

        pairwise_difference = query_embeddings_unnorm.unsqueeze(1) - target_embeddings_unnorm.unsqueeze(0)
        pairwise_distance = torch.norm(pairwise_difference, dim=2, p=2)
        query_norm = torch.norm(query_embeddings_unnorm, dim=1, p=2).unsqueeze(1)
        target_norm = torch.norm(target_embeddings_unnorm, dim=1, p=2).unsqueeze(0)
        norm_sum = (query_norm + target_norm).clamp(min=eps)

        target_to_query_norm_ratio = (
            torch.norm(target_embeddings_unnorm, dim=1, p=2)
            / torch.norm(query_embeddings_unnorm, dim=1, p=2).clamp(min=eps)
        ).mean()
        infotn_similarity = 1.0 - pairwise_distance / norm_sum
        norm_difference_similarity = self.norm_difference_similarity(query_embeddings, target_embeddings)

        return (
            infotn_similarity,
            cosine_similarity,
            target_to_query_norm_ratio,
            unnormalized_dot_similarity,
            norm_difference_similarity,
        )

    def norm_difference_similarity(self, query_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:
        """
        用模长差定义相似度，用于消融和诊断。
        Define similarity by norm difference for ablations and diagnostics.
        """
        query_norm = torch.norm(query_embeddings, dim=1, p=2).unsqueeze(1)
        target_norm = torch.norm(target_embeddings, dim=1, p=2).unsqueeze(0)
        return query_norm - target_norm

    def infonce_positive_gradient_norm(self, cosine_similarity: Tensor) -> Tensor:
        """
        计算 InfoNCE 对正样本相似度梯度的 L2 范数。
        Compute the L2 norm of the InfoNCE gradient on positive-pair similarities.
        """
        probabilities = F.softmax(cosine_similarity / self.temperature, dim=1)
        positive_probabilities = torch.diag(probabilities)
        return torch.norm(positive_probabilities - 1.0, p=2)

    def infotn_positive_gradients(
        self,
        infotn_similarity: Tensor,
        query_embeddings_unnorm: Tensor,
        target_embeddings_unnorm: Tensor,
        temperature: float = 0.02,
        eps: float = 1e-12,
    ):
        """
        按正样本对计算 InfoTN 关于 k 和 t 的梯度诊断量。
        Compute positive-pair InfoTN gradient diagnostics with respect to k and t.
        """
        query_norm = torch.norm(query_embeddings_unnorm, dim=-1)
        target_norm = torch.norm(target_embeddings_unnorm, dim=-1)
        norm_ratio = (target_norm / (query_norm + eps)).clamp(min=eps, max=1e6)
        cosine_positive = F.cosine_similarity(query_embeddings_unnorm, target_embeddings_unnorm, dim=-1)
        cosine_positive = cosine_positive.clamp(-1.0 + eps, 1.0 - eps)

        sqrt_term = torch.sqrt(1.0 + norm_ratio**2 - 2.0 * norm_ratio * cosine_positive + eps)
        d_infotn_dk = (norm_ratio - 1.0) * (1.0 + cosine_positive) / (
            (1.0 + norm_ratio) ** 2 * sqrt_term + eps
        )
        d_infotn_dt = -norm_ratio / ((1.0 + norm_ratio) * sqrt_term + eps)

        positive_probabilities = F.softmax(infotn_similarity / temperature, dim=1).diagonal()
        chain_scale = (1.0 - positive_probabilities) / temperature
        grad_k = chain_scale * d_infotn_dk
        grad_t = chain_scale * d_infotn_dt

        return grad_k, grad_t, torch.abs(d_infotn_dk).mean(), torch.abs(d_infotn_dt).mean()

    def infotn_gradient_norms(
        self,
        infotn_similarity: Tensor,
        query_embeddings_unnorm: Tensor,
        target_embeddings_unnorm: Tensor,
        temperature: float = 0.02,
        eps: float = 1e-12,
    ):
        """
        返回 InfoTN 关于 k 和 t 的梯度范数。
        Return InfoTN gradient norms with respect to k and t.
        """
        grad_k, grad_t, _, _ = self.infotn_positive_gradients(
            infotn_similarity,
            query_embeddings_unnorm,
            target_embeddings_unnorm,
            temperature=temperature,
            eps=eps,
        )
        return torch.norm(grad_k, p=2), torch.norm(grad_t, p=2)

    def norm_ratio_stats(self, query_embeddings_unnorm: Tensor, target_embeddings_unnorm: Tensor, eps: float = 1e-8):
        """
        统计每个样本的查询/目标模长比。
        Summarize per-sample query-to-target norm ratios.
        """
        ratio = torch.linalg.norm(query_embeddings_unnorm, dim=1) / torch.linalg.norm(
            target_embeddings_unnorm, dim=1
        ).clamp(min=eps)
        ratio_values = ratio.detach().cpu().numpy().tolist()
        return ratio_values, ratio.mean().item(), ratio.std().item(), ratio.min().item(), ratio.max().item()

    def __call__(
        self,
        query_embeddings: Tensor,
        target_embeddings: Tensor,
        query_embeddings_unnorm: Tensor,
        target_embeddings_unnorm: Tensor,
        target: Tensor = None,
        reduction: str = "mean",
    ) -> Tensor:
        if target is None:
            target = _default_targets(query_embeddings, target_embeddings)

        (
            infotn_similarity,
            cosine_similarity,
            _,
            _,
            norm_difference_similarity,
        ) = self.fused_similarity(
            query_embeddings, target_embeddings, query_embeddings_unnorm, target_embeddings_unnorm
        )
        _, _, mean_abs_dLdk, mean_abs_dLdt = self.infotn_positive_gradients(
            infotn_similarity, query_embeddings_unnorm, target_embeddings_unnorm
        )
        kt_gradient_ratio = mean_abs_dLdk / mean_abs_dLdt.clamp(min=1e-12)
        infotn_positive_mean = torch.diagonal(infotn_similarity).mean()
        infonce_gradient_norm = torch.norm(
            (1 - F.softmax(cosine_similarity, dim=1)) / self.temperature, p=2
        )
        dinfotn_dk, dinfotn_dt = self.infotn_gradient_norms(
            infotn_similarity, query_embeddings_unnorm, target_embeddings_unnorm
        )
        infonce_similarity_grad_norm = self.infonce_positive_gradient_norm(cosine_similarity)
        infotn_similarity_grad_norm = self.infonce_positive_gradient_norm(infotn_similarity)
        norm_ratio_values, ratio_mean, ratio_std, ratio_min, ratio_max = self.norm_ratio_stats(
            query_embeddings_unnorm, target_embeddings_unnorm
        )

        infonce_loss = F.cross_entropy(cosine_similarity / self.temperature, target, reduction=reduction)
        infotn_loss = F.cross_entropy(infotn_similarity / self.temperature, target, reduction=reduction)
        norm_difference_loss = F.cross_entropy(
            norm_difference_similarity / self.temperature, target, reduction=reduction
        )
        infotn_pair_loss = infotn_pair_distance_loss(
            query_embeddings, target_embeddings, query_embeddings_unnorm, target_embeddings_unnorm
        )

        if _is_main_process():
            self.logger.log_step_dynamics(
                mean_abs_dLdk=mean_abs_dLdk,
                mean_abs_dLdt=mean_abs_dLdt,
                kt_gradient_ratio=kt_gradient_ratio,
                infotn_positive_mean=infotn_positive_mean,
                infonce_gradient_norm=infonce_gradient_norm,
                dinfotn_dk=dinfotn_dk,
                dinfotn_dt=dinfotn_dt,
                infonce_similarity_grad_norm=infonce_similarity_grad_norm,
                infotn_similarity_grad_norm=infotn_similarity_grad_norm,
                infonce_loss=infonce_loss,
                infotn_loss=infotn_loss,
                infotn_pair_loss=infotn_pair_loss,
                norm_ratio_mean=ratio_mean,
                norm_ratio_std=ratio_std,
                norm_ratio_min=ratio_min,
                norm_ratio_max=ratio_max,
                norm_ratio_values=norm_ratio_values,
            )

        loss_mix_weight = 0.5
        return infotn_pair_loss * (1 - loss_mix_weight) + loss_mix_weight * norm_difference_loss


class InfoTNRegularizedLoss(InfoTNLoss):
    """
    消融损失：用显式模长正则替代 InfoTN。
    Ablation loss: replace InfoTN with explicit norm regularization.
    """

    def __init__(self, temperature: float = 0.02, init_white=1e-4, reg_weight: float = 0.5, lambda_: float = 0.5):
        super().__init__(temperature=temperature, init_white=init_white)
        self.reg_weight = reg_weight
        self.lambda_ = lambda_
        self.logger = TrainingDynamicsLogger("./logs/infotn_reg_only_dynamics.jsonl")

    def norm_regularizer(self, query_embeddings_unnorm: Tensor, target_embeddings_unnorm: Tensor) -> Tensor:
        """
        惩罚向量模长偏离 1 的程度。
        Penalize vector norms that deviate from 1.
        """
        all_norms = torch.cat(
            [
                torch.norm(query_embeddings_unnorm, dim=-1),
                torch.norm(target_embeddings_unnorm, dim=-1),
            ]
        )
        return ((all_norms - 1.0) ** 2).mean()

    def __call__(
        self,
        query_embeddings: Tensor,
        target_embeddings: Tensor,
        query_embeddings_unnorm: Tensor,
        target_embeddings_unnorm: Tensor,
        target: Tensor = None,
        reduction: str = "mean",
    ) -> Tensor:
        if target is None:
            target = _default_targets(query_embeddings, target_embeddings)

        infotn_similarity, cosine_similarity, _, _, _ = self.fused_similarity(
            query_embeddings, target_embeddings, query_embeddings_unnorm, target_embeddings_unnorm
        )
        _, _, mean_abs_dLdk, mean_abs_dLdt = self.infotn_positive_gradients(
            infotn_similarity, query_embeddings_unnorm, target_embeddings_unnorm
        )
        dinfotn_dk, dinfotn_dt = self.infotn_gradient_norms(
            infotn_similarity, query_embeddings_unnorm, target_embeddings_unnorm
        )
        norm_ratio_values, ratio_mean, ratio_std, ratio_min, ratio_max = self.norm_ratio_stats(
            query_embeddings_unnorm, target_embeddings_unnorm
        )

        infonce_loss = F.cross_entropy(cosine_similarity / self.temperature, target, reduction=reduction)
        regularization_loss = self.norm_regularizer(query_embeddings_unnorm, target_embeddings_unnorm)
        infotn_pair_loss = infotn_pair_distance_loss(
            query_embeddings, target_embeddings, query_embeddings_unnorm, target_embeddings_unnorm
        )

        if _is_main_process():
            self.logger.log_step_dynamics(
                mean_abs_dLdk=mean_abs_dLdk,
                mean_abs_dLdt=mean_abs_dLdt,
                kt_gradient_ratio=mean_abs_dLdk / mean_abs_dLdt.clamp(min=1e-12),
                infotn_positive_mean=torch.diagonal(infotn_similarity).mean(),
                infonce_gradient_norm=torch.norm(
                    (1 - F.softmax(cosine_similarity, dim=1)) / self.temperature, p=2
                ),
                dinfotn_dk=dinfotn_dk,
                dinfotn_dt=dinfotn_dt,
                infonce_similarity_grad_norm=self.infonce_positive_gradient_norm(cosine_similarity),
                infotn_similarity_grad_norm=self.infonce_positive_gradient_norm(infotn_similarity),
                infonce_loss=infonce_loss,
                infotn_loss=regularization_loss,
                infotn_pair_loss=infotn_pair_loss,
                norm_ratio_mean=ratio_mean,
                norm_ratio_std=ratio_std,
                norm_ratio_min=ratio_min,
                norm_ratio_max=ratio_max,
                norm_ratio_values=norm_ratio_values,
                extra={"regularization_loss": regularization_loss.item(), "ablation": "reg_only"},
            )

        return self.lambda_ * infonce_loss + self.reg_weight * regularization_loss

def infotn_pair_distance_loss(
    query_embeddings: Tensor,
    target_embeddings: Tensor,
    query_embeddings_unnorm: Tensor,
    target_embeddings_unnorm: Tensor,
    device=None,
    eps: float = 1e-8,
) -> Tensor:
    """
    基于成对样本的 InfoTN 距离损失。
    Pairwise InfoTN distance loss for adjacent positive pairs.
    """
    batch_size = query_embeddings.shape[0]
    device = device or query_embeddings.device
    assert batch_size % 2 == 0, "Batch size must be even."
    assert (
        query_embeddings.shape
        == target_embeddings.shape
        == query_embeddings_unnorm.shape
        == target_embeddings_unnorm.shape
    )

    pair_indices = torch.arange(batch_size, device=device)
    original_indices = pair_indices[::2]
    augmented_indices = pair_indices[1::2]

    positive_similarity = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        target_embeddings.unsqueeze(0),
        dim=-1,
    )[original_indices, augmented_indices]

    pair_distance = torch.norm(
        query_embeddings_unnorm[original_indices] - target_embeddings_unnorm[augmented_indices],
        p=2,
        dim=-1,
    )
    query_pair_norm = torch.norm(query_embeddings_unnorm[original_indices], p=2, dim=-1)
    target_pair_norm = torch.norm(target_embeddings_unnorm[augmented_indices], p=2, dim=-1)
    infotn_distance_weight = pair_distance / (query_pair_norm + target_pair_norm + eps) / 2.0

    positive_similarity = positive_similarity.clamp(min=1e-6, max=1.0)
    weighted_loss = -torch.log(positive_similarity) * infotn_distance_weight
    return weighted_loss.mean()


class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        loss = super().__call__(dist_x, dist_y, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class DistributedInfoTNLoss(InfoTNLoss):
    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, x_unnorm: Tensor, y_unnorm: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_x_unnorm = self.gather_tensor(x_unnorm)
        dist_y_unnorm = self.gather_tensor(y_unnorm)
        loss = super().__call__(dist_x, dist_y, dist_x_unnorm, dist_y_unnorm, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class DistributedInfoTNRegularizedLoss(InfoTNRegularizedLoss):
    """
    DDP 下使用显式模长正则的消融损失。
    DDP ablation loss with explicit norm regularization.
    """

    def __init__(self, n_target: int = 0, scale_loss: bool = True, temperature: float = 0.02,
                 reg_weight: float = 0.5, lambda_: float = 0.5):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__(temperature=temperature, reg_weight=reg_weight, lambda_=lambda_)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.scale_loss = scale_loss
        self.temperature = temperature

    def __call__(self, x: Tensor, y: Tensor, x_unnorm: Tensor, y_unnorm: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)
        dist_x_unnorm = self.gather_tensor(x_unnorm)
        dist_y_unnorm = self.gather_tensor(y_unnorm)
        loss = super().__call__(dist_x, dist_y, dist_x_unnorm, dist_y_unnorm, **kwargs)
        if self.scale_loss:
            loss = loss * self.world_size
        return loss

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class InExampleContrastiveLoss:
    """
    样本内分类式对比损失：从 K 个候选中选择 1 个目标。
    In-example classification contrastive loss: choose one target from K candidates.
    """

    def __init__(self, n_hard_negatives: int = 0, temperature: float = 1.0, ndim: int = None, *args, **kwargs):
        self.target_per_qry = n_hard_negatives + 1
        self.temperature = temperature
        self.ndim = ndim

    def __call__(self, x: Tensor, y: Tensor, reduction: str = 'mean'):
        if torch.distributed.is_initialized():
            x = dist_utils.dist_gather(x)
            y = dist_utils.dist_gather(y)
        bsz, ndim = x.size(0), x.size(1)
        target = torch.zeros(bsz, dtype=torch.long, device=x.device)
        if self.ndim:
            ndim = self.ndim
            x = x[:, :ndim]
            y = y[:, :ndim]
        logits = torch.einsum('bod,bsd->bs', x.view(bsz, 1, ndim), y.view(bsz, -1, ndim)) * self.temperature
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, target, reduction=reduction)
        loss_detail = {"logits": logits, "labels": target, "preds": preds}
        return loss, loss_detail
