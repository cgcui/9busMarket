# 9busMarket

独立的 9-bus strategic bidding 测试环境，面向 `Paper9Bus-3Gen-C3` 冻结基准和
GV-GRPO 训练。项目不依赖上层 ERCOT 仿真仓库的 Python 包、AMES Java 运行时或
绝对路径，适合直接复制到云服务器进行数据审计、SFT、候选采样和 GRPO 测试。

## 训练流程

```text
数据审计 → Core-ESR QLoRA SFT → K=4 候选采样
        → GATE_GRPO_SIGNAL → 组内 GV-GRPO → DEV 评估
```

默认模型和训练参数对齐云服务器说明：

- `Qwen/Qwen2.5-3B-Instruct`
- NF4 4-bit quantization
- LoRA `r=8, alpha=16, dropout=0.05`
- learning rate `2e-4`
- SFT batch `2`、gradient accumulation `8`
- GV-GRPO `K=4, H=4, gamma=0.95, clip=0.20, KL=0.05, 100 steps`

## 快速开始

详细安装说明见 [INSTALL.md](INSTALL.md)。最小运行顺序：

```bash
python scripts/preflight.py
python -m paper9bus_gv_grpo.prepare
python -m paper9bus_gv_grpo.audit

python scripts/train_sft.py \
  --config configs/sft_qwen25_3b.yaml \
  --output runs/esr_sft_seed42

python scripts/sample_candidates.py \
  --run-dir runs/esr_sft_seed42 \
  --split TRAIN --groups 64 --k 4 \
  --output runs/esr_sft_seed42/train_candidates_k4.jsonl

python -m paper9bus_gv_grpo.signal_gate \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --split TRAIN --k 4 --expected-groups 64 \
  --output runs/esr_sft_seed42/signal_gate

python scripts/train_gv_grpo.py \
  --config configs/gv_grpo_seed42.yaml \
  --run-dir runs/esr_sft_seed42 \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --gate-report runs/esr_sft_seed42/signal_gate/GATE_GRPO_SIGNAL_RESULT.json \
  --output runs/gv_grpo_seed42
```

`GATE_GRPO_SIGNAL` 失败时会阻断 GV-GRPO，不能用 oracle 候选绕过门控。

## 数据边界

- 工程只携带 `TRAIN/DEV` cell bank；`FINAL` 没有打包。
- prompt 只包含公开 market evidence。
- hidden opponent state、payoff、oracle action 不进入模型输入。
- simulator-grounded reward 由只读 `data/benchmark/cell_bank.parquet` 计算。

## 目录

```text
9busMarket/
├─ data/                       # 冻结 TRAIN/DEV 数据和 cell bank
├─ configs/                    # SFT / GV-GRPO 配置
├─ scripts/preflight.py        # 环境和 CUDA 检查
├─ scripts/train_sft.py        # QLoRA SFT
├─ scripts/sample_candidates.py
├─ scripts/train_gv_grpo.py
└─ src/paper9bus_gv_grpo/      # schema、reward、audit、signal gate
```

