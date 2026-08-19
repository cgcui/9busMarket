# 9busMarket 安装与训练说明

## 1. 推荐服务器

- Ubuntu 22.04/24.04
- Python 3.10 或 3.11
- 可用的 NVIDIA 驱动和 `nvidia-smi`
- 推荐显存不低于 16 GB；NF4 + LoRA 用于 Qwen2.5-3B
- 至少 20 GB 磁盘空间，用于环境、模型缓存和 checkpoint
- `git`、`python3-venv`、`python3-pip`、`build-essential`

数据审计和 schema 测试可在 CPU 上运行；SFT、候选采样、评估和 GV-GRPO 必须使用 CUDA。

## 2. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip build-essential
python3 --version
nvidia-smi
```

## 3. 创建环境

```bash
git clone https://github.com/cgcui/9busMarket.git
cd 9busMarket

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装 Python 依赖

项目依赖已经写入 `requirements.txt`：

```text
torch, transformers, accelerate, peft, bitsandbytes,
pyarrow, pandas, numpy, PyYAML, pytest
```

常规 CUDA 环境直接执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果服务器需要指定 CUDA 版本，请先按照 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装对应的 CUDA PyTorch wheel，再执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

检查关键依赖和 GPU：

```bash
python scripts/preflight.py
```

预期能看到 `cuda_available: True`、GPU 名称以及 `preflight: PASS`。

## 5. 模型下载

正式配置默认使用 `Qwen/Qwen2.5-3B-Instruct`。首次运行会从 Hugging Face 下载模型：

```bash
python -m pip install -U huggingface_hub
huggingface-cli login  # 仅在服务器需要登录时执行
```

也可以把 `configs/sft_qwen25_3b.yaml` 和 `configs/gv_grpo_seed42.yaml` 中的 `model_name` 改成本地模型目录。

## 6. 安装后自检

```bash
python -m paper9bus_gv_grpo.prepare
python -m paper9bus_gv_grpo.audit
pytest -q tests
```

应满足：

- `prepare` 输出 `final_accessed: false`
- `audit` 输出 `status: PASS`
- 核心 schema 和 group advantage 测试通过

两年公共状态层的审计也可以单独运行：

```bash
python scripts/audit_public_state_sufficiency.py
```

仓库已经包含构造后的 `data/public/isone_2y_public_energy_state.parquet`。如果需要从原始工作区重新生成，需要提供同时包含 `ercot_llm_bidding/data_external/isone` 和 ISO-NE LMP 文件的 `--source-root`；生成器会严格按 cutoff 选择 forecast vintage，并不会把没有可验证发布时间的目标日 DA-LMP 写入 prompt。

## 7. 完整训练

推荐一键执行：

```bash
bash scripts/run_server.sh
```

它会依次执行 preflight、数据准备、审计、SFT、DEV 评估、TRAIN 候选采样、signal gate 和 GV-GRPO。可通过环境变量调整：

```bash
SEED=42 GROUPS=64 K=4 RUN_ROOT=/root/runs/seed42 bash scripts/run_server.sh
```

也可以直接调用统一 Python 入口：

```bash
python scripts/run_pipeline.py \
  --sft-config configs/sft_qwen25_3b.yaml \
  --grpo-config configs/gv_grpo_seed42.yaml \
  --groups 64 --k 4 --run-root runs/seed42
```

如果需要分阶段运行：

```bash
python scripts/train_sft.py \
  --config configs/sft_qwen25_3b.yaml \
  --output runs/esr_sft_seed42

python scripts/evaluate_sft.py \
  --run-dir runs/esr_sft_seed42 --split DEV --n 64 \
  --output runs/esr_sft_seed42/dev_eval

python scripts/sample_candidates.py \
  --run-dir runs/esr_sft_seed42 --split TRAIN --groups 64 --k 4 \
  --output runs/esr_sft_seed42/train_candidates_k4.jsonl

python -m paper9bus_gv_grpo.signal_gate \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --split TRAIN --k 4 --expected-groups 64 \
  --output runs/esr_sft_seed42/signal_gate
```

只有 `GATE_GRPO_SIGNAL_RESULT.json` 中的 `status` 为 `PASS_GRPO_SIGNAL` 时，才运行：

```bash
python scripts/train_gv_grpo.py \
  --config configs/gv_grpo_seed42.yaml \
  --run-dir runs/esr_sft_seed42 \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --gate-report runs/esr_sft_seed42/signal_gate/GATE_GRPO_SIGNAL_RESULT.json \
  --output runs/gv_grpo_seed42
```

## 8. 断点续训

SFT 支持 Trainer 原生 checkpoint：

```bash
python scripts/train_sft.py \
  --config configs/sft_qwen25_3b.yaml \
  --output runs/esr_sft_seed42 \
  --resume-from-checkpoint runs/esr_sft_seed42/checkpoints/checkpoint-100
```

GV-GRPO 默认每 25 step 保存 checkpoint，可通过 `--checkpoint-every 10` 调整；最终 adapter 在 `runs/gv_grpo_seed42/adapter`。

## 9. Public-Energy-State-v1 输出

两年状态卡及审计产物位于：

```text
data/public/isone_2y_public_energy_state.parquet
data_examples/public_energy_state_examples.jsonl
configs/public_feature_registry_v1.json
configs/public_interpretation_rules_v1.json
reports/PUBLIC_FIELD_LEGALITY_AUDIT.json
reports/PUBLIC_STATE_FEATURE_SUFFICIENCY.json
reports/PUBLIC_STATE_FEATURE_ABLATION.parquet
reports/PUBLIC_STATE_COLLISION_CLASSES.parquet
reports/PUBLIC_STATE_CARD_REPORT_CN.md
```

这批产物只用于公共观测审计和后续数据重构，尚未替换现有 SFT/GRPO 数据集。

## 10. 常见问题

### CUDA unavailable

检查 `nvidia-smi`、CUDA PyTorch wheel 和 `torch.cuda.is_available()`。正式训练不会强制在 CPU 上运行。

### No module named paper9bus_gv_grpo

执行 `python -m pip install -e .`，或临时设置：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### FAIL_GRPO_SIGNAL

这表示同一 snapshot 的 K 个模型候选没有产生足够 reward 或策略差异。保留 gate 报告并检查 SFT 训练步数、采样参数和模型输出，不要注入 oracle 候选绕过门控。

### bitsandbytes 加载失败

检查 NVIDIA 驱动、PyTorch CUDA 版本和 bitsandbytes 版本是否匹配。仅做数据审计时可以不运行训练脚本。
