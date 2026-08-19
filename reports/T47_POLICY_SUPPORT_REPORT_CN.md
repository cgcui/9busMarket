# T47 Action-Policy Support and Sampling Audit

- Primary classification: `MODEL_POLICY_SUPPORT_COLLAPSE`。
- checkpoint：epoch1；T45 selected Grammar；temperature=`0.9`，top-p=`0.95`。
- mean runtime top1 probability：`1.000000`。
- mean runtime action entropy：`0.000000`。
- mean runtime effective support：`1.000000`。
- mean P(vary,K=4)：`0.000000`。
- expected varying groups：`0.000000/64`。
- P(observe zero varying groups)：`1.000000e+00`。
- grammar action support：6/6 legal actions reachable at all 64 states。
- full-generation diagnostic subset：8 states × 64 exact Grammar generations；见 `T47_MONTE_CARLO_SANITY.json`。

## 决策边界

- 未训练、未启动 GRPO、未访问 FINAL。
- 只有在本 T47 分类为 `MODEL_POLICY_SUPPORT_COLLAPSE` 时，才解锁下一项 soft-policy SFT 设计；本脚本不会自动启动下一项。
