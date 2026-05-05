# L2Dir VLM 训练与评测代码

本目录包含论文中 VLM 版本训练与评测所需的开源代码，可直接放入论文仓库。

本目录不包含模型 checkpoint、生成的 embedding、缓存数据、W&B 日志、私有 token 或任何机器相关路径。

运行前，请将所有占位路径、模型标识、缓存位置以及可选服务配置替换为你自己的值。可参考 `env.example`；不要把真实私有账号信息提交到仓库。

## 目录内容

- `scripts/train_qwen2vl_2b_imageonly.sh`：基于 Qwen2-VL-2B 的 image-only VLM 训练脚本。
- `scripts/eval_qwen2vl_2b_image_8gpu.sh`：训练后 VLM checkpoint 的 image-only MMEB 评测脚本。
- `configs/train_image.yaml`：image-only 训练数据配置模板。
- `configs/eval_image.yaml`：image-only 评测数据配置模板。
- `train.py`：训练入口。
- `eval.py`：评测入口。
- `src/`：训练和评测所需的模型、processor、trainer、dataset、collator、prompt 和 metric 源码。
- `requirements.txt`：VLM2Vec 相关 Python 依赖。
- `env.example`：用户配置示例。请将其中所有 `<...>` 占位符替换为你自己的路径或配置。

## 环境

VLM 实验使用 VLM2Vec 环境。原实验使用的 conda 环境名为 `vlm2v`：

```bash
conda activate vlm2v
```

开源复现时，请在兼容的 Python/CUDA 环境中安装 `requirements.txt` 中的依赖。

注意：GME baseline 不应使用该 VLM2Vec 环境运行。GME 模型请按照 GME 官方仓库及其官方依赖版本单独配置环境。

## 数据与 Checkpoint

脚本需要用户通过环境变量提供自己的数据路径和模型/checkpoint 路径。本开源目录不包含任何 checkpoint。

训练需要 MMEB 训练图片目录，例如：

```bash
export TRAIN_IMAGE_DIR=<replace-with-your-local-MMEB-train-image-dir>
```

评测需要 MMEB image-task 根目录，例如：

```bash
export EVAL_IMAGE_ROOT=<replace-with-your-local-MMEB-V2-image-task-root>
```

如果需要 Hugging Face cache、mirror、proxy 或 W&B 设置，请在你自己的 shell 环境中设置。不要把真实 token 或账号相关配置写入受版本控制的文件。

## 训练

在本目录下运行：

```bash
export TRAIN_IMAGE_DIR=<replace-with-your-local-MMEB-train-image-dir>
export OUTPUT_DIR=<replace-with-your-training-output-dir>

bash scripts/train_qwen2vl_2b_imageonly.sh
```

常用可覆盖配置：

```bash
export TRAIN_MODEL_NAME=<replace-with-your-base-model-id-or-local-model-path>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export MAX_STEPS=1000
export REPORT_TO=tensorboard
```

如需使用 W&B，请设置 `REPORT_TO=wandb`，并在你自己的 shell 环境中提供所需 W&B 配置。不要在脚本中硬编码账号相关值。

## 评测

设置 checkpoint 路径后，在本目录下运行：

```bash
export EVAL_MODEL_NAME=<replace-with-your-trained-checkpoint-or-model-id>
export EVAL_IMAGE_ROOT=<replace-with-your-local-MMEB-V2-image-task-root>
export OUTPUT_BASEDIR=<replace-with-your-evaluation-output-dir>

bash scripts/eval_qwen2vl_2b_image_8gpu.sh
```

常用可覆盖配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export BATCH_SIZE=16
export MODEL_BACKBONE=qwen2_vl
```

## 说明

- YAML 文件是模板。脚本会先替换其中的路径占位符，再调用 `train.py` 或 `eval.py`。
- 请将 checkpoint 和生成产物保存在本目录之外，或确保它们被 git 忽略。
