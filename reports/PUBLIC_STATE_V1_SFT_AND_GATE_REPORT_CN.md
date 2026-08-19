# Paper9Bus-Public-State-v1 T41-T46 报告

## Gate 结论

- T41：`PASS_TARGET_IDENTIFIABILITY_V1`。
- T43：`PASS_FRESH_DATASET_V1`。
- T44：fresh Qwen3-1.7B，三 epoch 已完成；checkpoint 选择 `epoch1`。
- T45：`PASS_CONSTRAINED_INTERFACE_V1`；FREE strict=`0.4883`，Grammar strict=`1.0000`。
- T46：`FAIL_TRUE_ECONOMIC_SIGNAL`；动作变化组=`0/64`，真实 payoff 变化组=`0/64`，advantage ordering violations=`0`。

## 停止边界

- K=4 只使用 TRAIN 64 个 frozen physical states，K=4；没有强制动作多样性、oracle、post-hoc repair 或跨 state reward normalization。
- 本阶段在 GRPO 之前停止；未启动 GRPO，未访问 FINAL，未运行 ISO2Y 训练。
