# L2Dir VLM Training and Evaluation Code

Before running, replace all placeholder paths, model identifiers, cache locations, and optional service settings with your own values. See `env.example`; do not commit real private account values.

## Contents

- `scripts/train_qwen2vl_2b_imageonly.sh`: image-only VLM training script based on Qwen2-VL-2B.
- `scripts/eval_qwen2vl_2b_image_8gpu.sh`: image-only MMEB evaluation script for a trained VLM checkpoint.
- `configs/train_image.yaml`: image-only training dataset configuration template.
- `configs/eval_image.yaml`: image-only evaluation dataset configuration template.
- `train.py`: training entry point.
- `eval.py`: evaluation entry point.
- `src/`: model, processor, trainer, dataset, collator, prompt, and metric source code used by training and evaluation.
- `requirements.txt`: Python dependencies for the VLM2Vec-based code.
- `env.example`: example user configuration. Replace every `<...>` placeholder with your own paths/settings.

## Environment

For the VLM experiments, use the VLM2Vec environment. The original experiments used a conda environment named `vlm2v`:

```bash
conda activate vlm2v
```

For a public release, users should install the dependencies in `requirements.txt` in a compatible Python/CUDA environment.

Important: GME baselines are not expected to run in this VLM2Vec environment. Please configure GME models according to the official GME repository and its official dependency versions.

Reproduction note: this code release currently cannot reproduce the original VLM2Vec V2 baseline scores exactly. The discrepancy may be related to the `num_workers` hyperparameter and the installed `datasets` package version. We do not currently have a confirmed solution for this issue.

## Data and Checkpoints

The scripts require users to provide their own data paths and model/checkpoint paths through environment variables. The release intentionally does not include any checkpoints.

Training expects the MMEB training images, for example:

```bash
export TRAIN_IMAGE_DIR=<replace-with-your-local-MMEB-train-image-dir>
```

Evaluation expects the MMEB image-task root, for example:

```bash
export EVAL_IMAGE_ROOT=<replace-with-your-local-MMEB-V2-image-task-root>
```

If you need Hugging Face cache, mirror, proxy, or W&B settings, set them in your shell with your own values. Never write real tokens or account-specific values into tracked files.

## Training

Run from this folder:

```bash
export TRAIN_IMAGE_DIR=<replace-with-your-local-MMEB-train-image-dir>
export OUTPUT_DIR=<replace-with-your-training-output-dir>

bash scripts/train_qwen2vl_2b_imageonly.sh
```

Useful overrides:

```bash
export TRAIN_MODEL_NAME=<replace-with-your-base-model-id-or-local-model-path>
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export MAX_STEPS=1000
export REPORT_TO=tensorboard
```

If using W&B, set `REPORT_TO=wandb` and provide the required W&B settings through your own shell environment. Do not hard-code account values in scripts.

## Evaluation

Run from this folder after setting the checkpoint path:

```bash
export EVAL_MODEL_NAME=<replace-with-your-trained-checkpoint-or-model-id>
export EVAL_IMAGE_ROOT=<replace-with-your-local-MMEB-V2-image-task-root>
export OUTPUT_BASEDIR=<replace-with-your-evaluation-output-dir>

bash scripts/eval_qwen2vl_2b_image_8gpu.sh
```

Useful overrides:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export BATCH_SIZE=16
export MODEL_BACKBONE=qwen2_vl
```

## Notes

- The YAML files are templates. The scripts render them into the output directory by replacing path placeholders before launching `train.py` or `eval.py`.
- Keep checkpoints and generated artifacts outside this folder, or make sure they are ignored by git.
