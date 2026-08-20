# Paper9Bus-ISONE-FixedRenewable-Physics-v1

状态：`PASS_FIXED_RENEWABLE_PHYSICS_V1_168H`

- TRAIN smoke：168 小时
- 时间：`2024-06-01T00:00:00+00:00` 至 `2024-06-07T23:00:00+00:00`
- C0：无可再生
- C1：BTM 光伏负荷扣减
- C2：BTM 光伏 + 固定风电 proxy
- BTM 光伏：负荷项，非发电机
- 风电：固定外生注入，非 OPF 决策变量，`wind_available_mw` 未使用
- 预测字段进入物理求解次数：0
- utility solar 注入：0 MW
- C0 等价性：PASS
- C2 最大系统残差：6.253e-13 MW
- 最大节点残差：1.023e-12 MW
- 物理效果差异：`True`

已停止；未处理全量 TRAIN、DEV、HOLDOUT，未修改 Action-Space-v2。
