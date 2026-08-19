# Public-Energy-State-v1 字段合法性审计

- 时间范围：`2020-01-01` 至 `2021-12-31`
- TRAIN：8760 行；DEV：8736 行
- 决策 cutoff：目标日 D-1 10:00 ET
- forecast 时间规则：`published_datetime_utc <= decision_cutoff_utc`
- 风电/光伏日前预测：**UNAVAILABLE**，未填造
- 9-bus focal generator 对 ISO-NE 两年公共数据：**UNAVAILABLE**，未填造
- future realized / payoff / regret / oracle：未进入输出
- 结论：`PASS_PUBLIC_STATE_V1`
