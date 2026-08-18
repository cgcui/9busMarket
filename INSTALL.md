# 9busMarket 安装说明

## A. 推荐服务器环境

正式训练推荐：

- Ubuntu 22.04 或 24.04
- Python 3.10 或 3.11
- NVIDIA 驱动正常，且 `nvidia-smi` 可用
- 建议显存不低于 16 GB；3B 模型使用 NF4/LoRA
- 至少 20 GB 磁盘空间用于 Python 环境、模型缓存和 checkpoint
- Git、`python3-venv`、`build-essential`

数据审计和 schema 测试可以在 CPU 上运行；SFT、候选采样和 GV-GRPO 必须使用 CUDA。

## B. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip build-essential
```

确认 Python：

```bash
python3 --version
nvidia-smi
```

## C. 创建虚拟环境

```bash
git clone https://github.com/cgcui/9busMarket.git
cd 9busMarket

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## D. 安装 Python 依赖

项目依赖已经写入 `requirements.txt`：

```text
torch
transformers
accelerate
peft
bitsandbytes
pyarrow
pandas
numpy
PyYAML
pytest
```

普通环境可以直接安装：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果云服务器需要 CUDA 专用 PyTorch wheel，请先根据服务器 CUDA 版本在 PyTorch 官网选择并安装对应的 `torch`，再执行项目依赖安装：

```bash
# 这里替换成 PyTorch 官网给出的 CUDA 对应安装命令
python -m pip install torch
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果已安装的 CUDA 版 `torch` 满足 `requirements.txt`，pip 会复用它；不要在 CUDA 版本不匹配时强行覆盖。

检查关键依赖：

```bash
python - <<'PY'
import torch, transformers, peft, pandas, pyarrow
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## E. 模型下载与 Hugging Face

默认模型是 `Qwen/Qwen2.5-3B-Instruct`。首次运行时，Transformers 会自动下载模型到 Hugging Face cache。

如果服务器需要登录 Hugging Face：

```bash
python -m pip install -U huggingface_hub
huggingface-cli login
```

也可以通过修改 `configs/sft_qwen25_3b.yaml` 和 `configs/gv_grpo_seed42.yaml` 的 `model_name` 使用本地模型目录，例如：

```yaml
model_name: /root/autodl-tmp/models/Qwen2.5-3B-Instruct
```

## F. 安装后自检

```bash
python scripts/preflight.py
python -m paper9bus_gv_grpo.prepare
python -m paper9bus_gv_grpo.audit
pytest -q tests
```

预期结果：

- `preflight`: 依赖导入成功；CUDA 服务器上显示 GPU；
- `prepare`: 输出 `final_accessed: false`；
- `audit`: 输出 `status: PASS`；
- `pytest`: 核心 schema 和 group advantage 测试通过。

## G. 正式训练

```bash
python scripts/train_sft.py \
  --config configs/sft_qwen25_3b.yaml \
  --output runs/esr_sft_seed42

python scripts/sample_candidates.py \
  --run-dir runs/esr_sft_seed42 --split TRAIN \
  --groups 64 --k 4 \
  --output runs/esr_sft_seed42/train_candidates_k4.jsonl

python -m paper9bus_gv_grpo.signal_gate \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --split TRAIN --k 4 --expected-groups 64 \
  --output runs/esr_sft_seed42/signal_gate
```

只有 `GATE_GRPO_SIGNAL_RESULT.json` 中的 `status` 为 `PASS_GRPO_SIGNAL` 时，才执行：

```bash
python scripts/train_gv_grpo.py \
  --config configs/gv_grpo_seed42.yaml \
  --run-dir runs/esr_sft_seed42 \
  --candidates runs/esr_sft_seed42/train_candidates_k4.jsonl \
  --gate-report runs/esr_sft_seed42/signal_gate/GATE_GRPO_SIGNAL_RESULT.json \
  --output runs/gv_grpo_seed42
```

## H. 常见问题

### `CUDA unavailable`

检查 `nvidia-smi`、PyTorch CUDA wheel 和 `torch.cuda.is_available()`。不要在 CPU 上强行运行正式训练。

### `No module named paper9bus_gv_grpo`

确认已执行：

```bash
python -m pip install -e .
```

或者临时设置：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### `FAIL_GRPO_SIGNAL`

这表示同一 snapshot 的 K 个模型候选没有产生足够的 reward 或策略差异。保留失败报告，先检查模型输出、采样温度和 SFT checkpoint，不要注入 oracle 候选。

### bitsandbytes 加载失败

确认 NVIDIA 驱动、PyTorch CUDA 版本和 bitsandbytes 版本匹配；如果只是做 CPU 数据审计，可以暂时不运行训练脚本。
