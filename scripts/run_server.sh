#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/seed${SEED}}"
SFT_CONFIG="${SFT_CONFIG:-${PROJECT_ROOT}/configs/sft_qwen25_3b.yaml}"
GRPO_CONFIG="${GRPO_CONFIG:-${PROJECT_ROOT}/configs/gv_grpo_seed42.yaml}"
GROUPS="${GROUPS:-64}"
K="${K:-4}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" scripts/preflight.py
"${PYTHON_BIN}" scripts/run_pipeline.py \
  --sft-config "${SFT_CONFIG}" \
  --grpo-config "${GRPO_CONFIG}" \
  --run-root "${RUN_ROOT}" \
  --groups "${GROUPS}" --k "${K}"
