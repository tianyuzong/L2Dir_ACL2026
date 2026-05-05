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

: "${EVAL_IMAGE_ROOT:?Replace EVAL_IMAGE_ROOT with your own local MMEB-V2 image-task root.}"

# Replace EVAL_MODEL_NAME or MODEL_NAME with your trained checkpoint path/model id.
MODEL_NAME="${EVAL_MODEL_NAME:-${MODEL_NAME:-}}"
: "${MODEL_NAME:?Replace EVAL_MODEL_NAME or MODEL_NAME with your own trained checkpoint path/model id.}"

MODEL_BACKBONE="${MODEL_BACKBONE:-qwen2_vl}"
# Replace OUTPUT_BASEDIR with a writable output directory on your own machine.
OUTPUT_BASEDIR="${OUTPUT_BASEDIR:-${REPO_ROOT}/outputs/eval_l2dir_vlm_qwen2vl_2b}"
DATA_BASEDIR="${DATA_BASEDIR:-${EVAL_IMAGE_ROOT}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2277}"
BATCH_SIZE="${BATCH_SIZE:-16}"

mkdir -p "${OUTPUT_BASEDIR}/configs"
RENDERED_CONFIG="${OUTPUT_BASEDIR}/configs/eval_image.yaml"
TEMPLATE_CONFIG="${RELEASE_DIR}/configs/eval_image.yaml"

TEMPLATE_CONFIG="${TEMPLATE_CONFIG}" \
RENDERED_CONFIG="${RENDERED_CONFIG}" \
EVAL_IMAGE_ROOT="${EVAL_IMAGE_ROOT}" \
python - <<'PY'
import os
from pathlib import Path

template = Path(os.environ["TEMPLATE_CONFIG"])
output = Path(os.environ["RENDERED_CONFIG"])
text = template.read_text()
text = text.replace("{{MMEB_EVAL_IMAGE_ROOT}}", os.environ["EVAL_IMAGE_ROOT"])
output.write_text(text)
PY

export CUDA_VISIBLE_DEVICES

OUTPUT_PATH="${OUTPUT_BASEDIR}/image"
mkdir -p "${OUTPUT_PATH}"

cmd=(
  torchrun
  --nproc_per_node="${NUM_GPUS}"
  --master_port="${MASTER_PORT}"
  --max_restarts=0
  "${REPO_ROOT}/eval.py"
  --pooling eos
  --normalize true
  --per_device_eval_batch_size "${BATCH_SIZE}"
  --model_backbone "${MODEL_BACKBONE}"
  --model_name "${MODEL_NAME}"
  --dataset_config "${RENDERED_CONFIG}"
  --encode_output_path "${OUTPUT_PATH}"
  --data_basedir "${DATA_BASEDIR}"
)

echo "==> Evaluation command"
printf '%q ' "${cmd[@]}"
echo ""

"${cmd[@]}" 2>&1 | tee "${OUTPUT_PATH}/eval.log"

echo "All jobs completed."
