# 日前 24 小时报价接口

分类：`PASS_DAY_AHEAD_24H_BID_SMOKE`

本测试一次生成次日 24 个 G1 报价，报价单位为 USD/MWh；使用前一日同小时 TRAIN 负荷持久性作为公开预测。现有 private cell bank 仅用于离线评价，未重新求解 OPF，未读取 DEV/HOLDOUT，未训练。

- 目标日：365
- 报价小时：8760
- 隐藏 k_G3 情景：`[1.0, 1.05, 1.1, 1.2, 1.3, 1.5]`
