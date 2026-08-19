# 9busMarket

独立的 9-bus strategic bidding 测试环境，包含 `Paper9Bus-3Gen-C3` 冻结数据、Core-ESR QLoRA SFT、候选采样、`GATE_GRPO_SIGNAL` 和 simulator-grounded GV-GRPO 训练流程。

项目不依赖上层 ERCOT 仿真仓库、AMES Java 运行时或绝对路径，可以直接复制到云服务器运行。

当前仓库同时保留原始 Paper9Bus v1 训练流程，并新增基于 ISO-NE 2020–2021 的 `Public-Energy-State-v1` 公共状态层。两年数据只作为公开外生时间序列；没有把 ISO-NE 未来真实值、hidden opponent、payoff 或 oracle 标签写入 9-bus prompt。

问题定义和两年数据切分见 [PROBLEM_REFORMULATION_CN.md](PROBLEM_REFORMULATION_CN.md)。

## 训练流程

```text
数据准备/审计 -> Core-ESR QLoRA SFT -> K=4 候选采样
    -> GATE_GRPO_SIGNAL -> 组内 GV-GRPO -> DEV 评估
```

默认正式配置：

- 模型：`Qwen/Qwen2.5-3B-Instruct`
- NF4 4-bit + LoRA（`r=8, alpha=16, dropout=0.05`）
- SFT：batch 2、gradient accumulation 8、learning rate `2e-4`
- GV-GRPO：`K=4, H=4, gamma=0.95, clip=0.20, KL=0.05, 100 steps`

## 快速开始

完整安装和 CUDA 检查见 [INSTALL.md](INSTALL.md)。安装完成后：

```bash
python scripts/preflight.py
python -m paper9bus_gv_grpo.prepare
python -m paper9bus_gv_grpo.audit
bash scripts/run_server.sh
```

`run_server.sh` 默认使用正式配置、64 个 TRAIN groups 和每组 `K=4`。也可以使用 Python 统一入口：

```bash
python scripts/run_pipeline.py \
  --sft-config configs/sft_qwen25_3b.yaml \
  --grpo-config configs/gv_grpo_seed42.yaml \
  --groups 64 --k 4 --run-root runs/seed42
```

## 本地 smoke 测试

如果 Qwen3-1.7B-Base 已经在 Hugging Face cache 中，可运行 2-step 端到端联调：

```bash
python scripts/run_pipeline.py \
  --sft-config configs/sft_smoke_qwen3.yaml \
  --grpo-config configs/gv_grpo_smoke.yaml \
  --groups 4 --k 4 --run-root runs/pipeline_smoke
```

smoke 的短 SFT 通常无法产生足够的策略差异，因此可能安全停止在 signal gate；这说明门控在工作，不代表正式训练失败。GV-GRPO 只有在 gate 为 `PASS_GRPO_SIGNAL` 时才会启动。

## 两年公共能量状态层

仓库已包含 `2020-01-01` 至 `2021-12-31` 的 17,496 条小时级公共状态卡：

- cutoff 前可见的 ISO-NE 日前负荷预测（8 个区域、24 小时向量、峰值、爬坡）；
- cutoff 前的系统负荷历史和最近 4 小时历史；
- cutoff 前的历史实时 LMP 结构；目标日 DA-LMP 因缺少可验证发布时间不进入 prompt；
- 已发布网络 binding 历史信号；
- TRAIN-only 阈值和确定性语义解释；
- 明确标记不可用的风/光日前预测和 focal 9-bus generator 状态，不填造。

重新从工作区外部源构造时：

```bash
python scripts/build_public_energy_state_2y.py \
  --source-root D:/code/ERCOTsimulation
python scripts/audit_public_state_sufficiency.py
```

TRAIN 消融显示，原有 371-state collision class 在加入合法当前总负荷 MW 后，decision-conflicting class 从 1 降到 0；因此当前分类为 `LEGAL_PUBLIC_REPRESENTATION_SUFFICIENT`。这一步不生成新的 SFT 数据，也不运行 SFT/GRPO。

## 断点和输出

- SFT：`runs/.../sft/adapter/`，Trainer checkpoint 在 `sft/checkpoints/`。
- SFT 断点续训：`python scripts/train_sft.py --resume-from-checkpoint runs/.../sft/checkpoints/checkpoint-100 ...`。
- GV-GRPO：`runs/.../gv_grpo/adapter/`，每 25 step 保存一个 checkpoint，可用 `--checkpoint-every` 调整。
- DEV 评估：`EVALUATION_SUMMARY.json` 和 `EVALUATION_RECORDS.jsonl`。
- 所有训练输出都在 `runs/` 下，默认不会提交到 Git。

## 数据边界

工程只携带 `TRAIN/DEV` 数据和 benchmark cell bank；`FINAL` 未打包。prompt 不包含 hidden opponent state、payoff 或 oracle action，reward 由公开的 benchmark cell bank 计算。

## 目录

```text
9busMarket/
├─ data/                         # 冻结 TRAIN/DEV 数据和 cell bank
├─ configs/                      # 正式与 smoke 配置
├─ scripts/preflight.py          # 依赖和 CUDA 检查
├─ scripts/run_pipeline.py       # 一键完整流程
├─ scripts/train_sft.py          # QLoRA SFT
├─ scripts/sample_candidates.py  # 候选采样
├─ scripts/evaluate_sft.py       # DEV/训练集严格评估
├─ scripts/train_gv_grpo.py      # GV-GRPO
├─ scripts/build_public_energy_state_2y.py  # 两年公共状态构造
├─ scripts/audit_public_state_sufficiency.py # TRAIN 特征消融
└─ src/paper9bus_gv_grpo/        # schema、reward、audit、public state、signal gate
```
