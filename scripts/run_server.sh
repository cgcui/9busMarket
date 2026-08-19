#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/seed${SEED}}"
SFT_CONFIG="${SFT_CONFIG:-${PROJECT_ROOT}/configs/sft_qwen3_1p7b.yaml}"
GRPO_CONFIG="${GRPO_CONFIG:-${PROJECT_ROOT}/configs/gv_grpo_qwen3_1p7b_seed42.yaml}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m paper9bus_gv_grpo.prepare
"${PYTHON_BIN}" -m paper9bus_gv_grpo.audit
"${PYTHON_BIN}" scripts/train_sft.py \
  --config "${SFT_CONFIG}" \
  --output "${RUN_ROOT}/sft"
"${PYTHON_BIN}" scripts/sample_candidates.py \
  --run-dir "${RUN_ROOT}/sft" --split TRAIN --groups 64 --k 4 \
  --output "${RUN_ROOT}/sft/train_candidates_k4.jsonl"
"${PYTHON_BIN}" -m paper9bus_gv_grpo.signal_gate \
  --candidates "${RUN_ROOT}/sft/train_candidates_k4.jsonl" \
  --split TRAIN --k 4 --expected-groups 64 \
  --output "${RUN_ROOT}/signal_gate"
"${PYTHON_BIN}" scripts/train_gv_grpo.py \
  --config "${GRPO_CONFIG}" \
  --run-dir "${RUN_ROOT}/sft" \
  --candidates "${RUN_ROOT}/sft/train_candidates_k4.jsonl" \
  --gate-report "${RUN_ROOT}/signal_gate/GATE_GRPO_SIGNAL_RESULT.json" \
  --output "${RUN_ROOT}/gv_grpo"
