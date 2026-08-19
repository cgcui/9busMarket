# Public-Energy-State-v1 特征充分性审计

本审计只使用冻结 9-bus `TRAIN`（1571 条），不使用 DEV/FINAL，不训练 SFT，不运行 GRPO。

## 结果

- 原始 X0 的 decision-conflicting visible class：**1**
- 原始 collision class 中的 physical state 数：**371**
- 选择表示：**X1**（X0 + current total load MW）
- 选择后 action conflict：**0**
- 选择后 belief conflict：**0**
- 选择后 plan conflict：**0**
- 结论：`LEGAL_PUBLIC_REPRESENTATION_SUFFICIENT`

X1 只增加 cell bank 中合法的当前总负荷 MW；没有把 ISO-NE 两年未来真实负荷、payoff、regret 或 oracle action 写进 9-bus prompt。ISO-NE 两年数据单独产出 Public-Energy-State-v1 时序层，只有决策 cutoff 之前发布的 forecast 和历史数据进入状态卡。
