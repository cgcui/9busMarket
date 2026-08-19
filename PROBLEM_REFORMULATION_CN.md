# 9-bus 两年数据驱动问题重构

## 1. 重构目标

将原来的静态 Paper9Bus strategic bidding 问题改造成一个“公开能源状态条件下的日前报价决策”问题：

\[
X_t^{public}
\longrightarrow
\{\text{belief},\text{game interpretation},\text{action},\text{contingent plan},\text{confidence}\}.
\]

其中 `X_t^{public}` 只能由决策时已经公开的信息组成。ISO-NE 2020–2021 数据提供真实的外生时间序列和日前负荷预测；9-bus cell bank 继续提供固定网络下的物理/经济评价，不把两者不受验证的字段强行拼接。

## 2. 时间与数据切分

- 时间范围：2020-01-01 至 2021-12-31，覆盖 729 个目标日期、17,496 个小时状态；
- 时区：`America/New_York`；
- 目标粒度：ISO-NE operating hour；
- 决策 cutoff：目标日 `D-1 10:00 ET`；
- forecast 选择：同一目标小时/区域中，`published_datetime_utc <= decision_cutoff_utc` 的最新 vintage；
- TRAIN：2020 年本地日期；
- DEV：2021 年本地日期；
- FINAL：本任务不访问、不生成、不打包。

这意味着未来真实负荷可以用于离线质量审计或后续 simulator 评价，但不能作为当天决策时的 forecast 输入。

## 3. Public-Energy-State-v1

状态卡由两类信息构成：

### 当前/历史公开能源状态

- cutoff 前最新系统负荷 `total_load_mw`；
- cutoff 前 24 小时计算的 `load_factor`；
- 最近 4 小时系统负荷和负荷趋势；
- 最近 4 小时区域平均实时 LMP；
- cutoff 前最后公开的 binding constraint 历史信号。

### 日前公开信息

- 八个 ISO-NE 负荷区的 24 小时日前负荷预测；
- 日前预测总负荷的峰值、峰值小时、谷值、日能量、上下爬坡；
- cutoff 前四个历史小时的区域平均实时 LMP、均值、最小值、最大值和 spread。

目标日 DA-LMP 文件虽然已经归档，但没有独立可验证的发布时间字段。在 D-1 10:00 ET cutoff 下，它被标记为 `UNAVAILABLE`，不会作为未来市场结果写入状态卡。

所有语义解释由固定规则确定，例如：

- `load_level`：TRAIN 的当前负荷 1/3、2/3 分位数；
- `price_dispersion`：TRAIN 的日前 LMP spread 1/3、2/3 分位数；
- `network_stress`：binding count 为 0/1–2/>2 时分别为 LOW/ELEVATED/HIGH；
- `congestion_status`：binding count 大于 0 时为 CONGESTED。

阈值保存在 `configs/public_interpretation_rules_v1.json`，不会由 LLM 生成。

## 4. 明确不可用的信息

当前两年公开数据没有经过验证的 focal 9-bus generator 对应 dispatch、capacity headroom 或 own-node LMP，因此这些字段不填造。当前数据也没有合法的风电/光伏日前预测，因此不使用历史 renewable observation 冒充 renewable forecast。

只有同时存在合法的 load、wind 和 solar forecast 时，才计算：

\[
\text{net load}_h
=
\text{load forecast}_h
-
\text{wind forecast}_h
-
\text{solar forecast}_h.
\]

## 5. 9-bus 可辨识性结论

对冻结的 9-bus TRAIN 做了 X0–X8 累积特征消融：

- X0：原有公共 dispatch/LMP/network 表示；
- X1：X0 + 当前总负荷 MW；
- X2：X1 + 9-bus 当前负荷空间分布；
- X3–X8：依次检查机组、市场、网络、历史、日前预测、renewable/net-load 的额外价值。

原 X0 中存在一个包含 371 个 physical states 的 collision class，并且 action target 冲突。X1 仅加入合法的当前总负荷 MW 后：

- decision-conflicting classes：`1 -> 0`；
- `H(action | X)`：`0.033723 -> 0` nats；
- belief conflict：`0`；
- plan conflict：`1 -> 0`。

因此当前审计分类为：

```text
LEGAL_PUBLIC_REPRESENTATION_SUFFICIENT
```

## 6. 后续训练问题

本次重构只完成公共状态层和 TRAIN-only sufficiency audit，尚未替换 SFT 数据，也没有运行 SFT/GRPO。后续正式顺序应为：

```text
Public-Energy-State-v1
  -> 用合法状态重新生成 SFT dataset
  -> 3-epoch QLoRA SFT
  -> DEV strict Core evaluation
  -> TRAIN K=4 candidate sampling
  -> GATE_GRPO_SIGNAL
  -> gate 通过后才运行 GV-GRPO
```

训练 reward 仍然只从公开 benchmark cell bank 的 simulator-grounded evaluation 表计算；payoff、regret、oracle best action 和 hidden opponent state 不进入 prompt。

## 7. 产物

- `data/public/isone_2y_public_energy_state.parquet`：两年公共状态卡；
- `data_examples/public_energy_state_examples.jsonl`：状态卡和确定性 prompt 示例；
- `configs/public_feature_registry_v1.json`：字段来源与合法性；
- `configs/public_interpretation_rules_v1.json`：TRAIN-only 规则；
- `reports/PUBLIC_FIELD_LEGALITY_AUDIT.json`：时间戳和字段审计；
- `reports/PUBLIC_STATE_FEATURE_SUFFICIENCY.json`：消融结论；
- `reports/PUBLIC_STATE_FEATURE_ABLATION.parquet`：X0–X8 指标；
- `reports/PUBLIC_STATE_COLLISION_CLASSES.parquet`：碰撞类明细。
