# Paper9Bus GV-GRPO

一个可单独拷贝到云服务器运行的 9-bus 小工程。它把现有的
`Paper9Bus-3Gen-C3` 冻结基准、Core-ESR 结构化输出、matched-game reward
和 GV-GRPO 训练入口放在同一个目录内，不再依赖上层 ERCOT 仓库的绝对路径。

## 训练链

```text
准备/审计冻结数据
        ↓
Power-ESR Core SFT（QLoRA）
        ↓
同一 snapshot 采样 K=4 个候选
        ↓
GATE_GRPO_SIGNAL（strict validity + 经济 reward 方差 + 策略多样性）
        ↓ 只有 PASS 才继续
GV-GRPO（组内 advantage、PPO clip、KL、100 optimizer steps）
        ↓
DEV 评估 / 独立报告
```

当前默认使用 `Qwen/Qwen3-1.7B`、NF4 4-bit、LoRA 和 seed 42 配置；如果服务器
沿用其他 checkpoint，可通过 YAML 中的 `model_name` 覆盖。训练脚本在没有 CUDA
时会主动退出，不会悄悄退化为 CPU。

## 云服务器运行

```bash
cd /root/autodl-tmp/paper9bus_gv_grpo
python -m pip install -r requirements.txt
python -m pip install -e .

python -m paper9bus_gv_grpo.prepare
python -m paper9bus_gv_grpo.audit

python scripts/train_sft.py \
  --config configs/sft_qwen3_1p7b.yaml \
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
  --config configs/gv_grpo_qwen3_1p7b_seed42.yaml \
  --run-dir runs/esr_sft_seed42 \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --gate-report runs/esr_sft_seed42/signal_gate/GATE_GRPO_SIGNAL_RESULT.json \
  --output runs/gv_grpo_seed42
```

seed 43/44 只需复制对应 YAML 并改变 `seed` 与输出目录。`DEV` 只能用于
审计和最终评估，不能用于调 reward 权重或启动正式训练。

## Windows RTX 5070 Ti

5070 Ti 是 Blackwell `sm_120`，建议使用 CUDA 12.8 版 PyTorch，再安装本项目依赖：

```powershell
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e .
python -m bitsandbytes
```

本机验证过 QLoRA 的 4-bit NF4 加载和 LoRA 反向传播；16 GB 显存先保持
`max_seq_length: 512`、`per_device_train_batch_size: 2`、
`gradient_accumulation_steps: 8`。第一次运行会从 Hugging Face 下载约 6.2 GB
的 Qwen3-1.7B 权重，建议配置 `HF_TOKEN` 或提前下载模型。

本项目保留 Transformers + PEFT + bitsandbytes，是因为后半段 GV-GRPO 是自定义
simulator-grounded 更新；Unsloth/TRL 可以加速标准 SFT，但不能直接替代当前的
候选采样、signal gate 和自定义 GRPO 更新逻辑。

## Public-Energy-State 更新分支

已合并 GitHub `cgcui/9busMarket` 的 Public-Energy-State-v1 更新，但保留当前
Qwen3-1.7B recovery、T27/T30/T31 审计脚本和历史运行产物。新增的 public-state
层只使用合法公开能源状态，不把 hidden opponent、payoff 或 oracle label 写入
prompt；原始 six-action benchmark 与 `data/core` 保持不变。

新增入口：

```powershell
python scripts/preflight.py
python scripts/audit_public_state_sufficiency.py
```

公共状态构造、字段注册表、两年数据和审计报告分别位于
`scripts/build_public_energy_state_2y.py`、`configs/public_*`、
`data/public/` 与 `reports/PUBLIC_STATE_*`。在新的 public-state 数据通过独立
审计前，不启动新的 SFT 或 GRPO。

### 风电预测补充 v2

用户提供的 ISO-NE `Seven Day Wind Power Forecast` 已整理为独立的
`Public-Energy-State-v2-WindForecast`。它保留 IEEE-9 三机物理模型不变，且不把
光伏缺失时的 `load - wind` 命名为 net load。由于原始 CSV 没有历史原始发布时间，
风电字段目前是 `DATE_ONLY_NOT_CUTOFF_PROVEN`，不能直接作为严格 D-1 10:00 ET
训练输入。

详见 `WIND_FORECAST_V2_ADDENDUM_CN.md`，重新摄取和生成命令为：

```powershell
python scripts/ingest_isone_wind_forecast.py --source-dir "D:\code\ERCOTsimulation\data\风电预测"
python scripts/build_public_energy_state_wind_v2.py
```

旧的 `data/public/isone_2y_public_energy_state.parquet` v1 保持不变。

### Renewable-Aware 9-bus 负荷版本

当前 9bus renewable-aware 版本先只接入 ISO-NE 总负荷，不使用风电/光伏预测：
`ISO-NE hourly_load_mw -> TRAIN-only median-ratio alpha -> IEEE-9 bus 5/7/9`。
生成入口是 `scripts/build_paper9bus_isone2y_load_bridge.py`，公式、产物和冻结边界见
`RENEWABLE_AWARE_9BUS_PROBLEM_CN.md`。风光输入保留为后续可插拔接口，当前不会进入
物理负荷或 net-load。

## 目录

```text
paper9bus_gv_grpo/
├─ data/                         # 小型冻结数据，约 1.4 MB
├─ configs/                      # SFT / GV-GRPO 配置
├─ src/paper9bus_gv_grpo/        # schema、reward、数据审计、signal gate
├─ scripts/train_sft.py          # QLoRA SFT
├─ scripts/sample_candidates.py  # K 候选采样
├─ scripts/train_gv_grpo.py      # 组内 GV-GRPO 更新
└─ tests/
```

`data/benchmark/cell_bank.parquet` 是 simulator-grounded reward 的只读证据。
模型 prompt 只包含 public market evidence；hidden opponent state、payoff、
oracle action 不会进入 prompt。signal gate 失败时返回非零退出码，必须先修复
候选生成/多样性问题再训练。
