#!/usr/bin/env bash
set -euo pipefail

# Before running, replace all paths and optional service settings with your own
# values. See ../env.example for a complete list of user-provided variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${RELEASE_DIR}}"

echo "==> Environment"
echo "repo root: ${REPO_ROOT}"
echo "python: $(which python)"
python --version
echo ""

: "${TRAIN_IMAGE_DIR:?Replace TRAIN_IMAGE_DIR with your own local MMEB-train image directory.}"

# Replace MODEL_NAME or TRAIN_MODEL_NAME with your own base model path/id if needed.
MODEL_NAME="${TRAIN_MODEL_NAME:-${MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}}"
EXP_NAME="${EXP_NAME:-l2dir_vlm_qwen2vl_2b_imageonly}"
# Replace OUTPUT_DIR with a writable output directory on your own machine.
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${EXP_NAME}}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2208}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-64}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-8}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-8}"
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
REPORT_TO="${REPORT_TO:-tensorboard}"

mkdir -p "${OUTPUT_DIR}"
RENDERED_CONFIG="${OUTPUT_DIR}/train_image.yaml"
TEMPLATE_CONFIG="${RELEASE_DIR}/configs/train_image.yaml"

TEMPLATE_CONFIG="${TEMPLATE_CONFIG}" \
RENDERED_CONFIG="${RENDERED_CONFIG}" \
TRAIN_IMAGE_DIR="${TRAIN_IMAGE_DIR}" \
python - <<'PY'
import os
from pathlib import Path

template = Path(os.environ["TEMPLATE_CONFIG"])
output = Path(os.environ["RENDERED_CONFIG"])
text = template.read_text()
text = text.replace("{{MMEB_TRAIN_IMAGE_DIR}}", os.environ["TRAIN_IMAGE_DIR"])
output.write_text(text)
PY

export CUDA_VISIBLE_DEVICES

cmd=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  --max_restarts=0
  "${REPO_ROOT}/train.py"
  --lora
  --lora_r 16
  --model_name "${MODEL_NAME}"
  --bf16
  --pooling eos
  --normalize True
  --temperature 0.02
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataset_config "${RENDERED_CONFIG}"
  --run_name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --grad_cache True
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gc_q_chunk_size "${GC_Q_CHUNK_SIZE}"
  --gc_p_chunk_size "${GC_P_CHUNK_SIZE}"
  --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}"
  --lr_scheduler_type linear
  --learning_rate "${LEARNING_RATE}"
  --max_steps "${MAX_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --save_safetensors True
  --remove_unused_columns False
  --resume_from auto
  --report_to "${REPORT_TO}"
)

echo "==> Training command"
printf '%q ' "${cmd[@]}"
echo ""

"${cmd[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
