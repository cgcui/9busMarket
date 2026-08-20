# ISO-NE 风电预测补充：Public-Energy-State-v2

本补充在原 `Public-Energy-State-v1` 之外新增风电预测字段。原 v1 文件不覆盖、不修改。

## 数据来源与解析

来源目录：用户提供的 `data/风电预测` 压缩包集合。

每个 CSV 是 ISO-NE `Seven Day Wind Power Forecast Report`，本项目采用：

```text
目标日 D <- 报告日期 D-1 的 Day 2 列
```

风电值保持 ISO-NE 原始 `Hour Ending`，单位为 MW。重复压缩包中的同一报告经过语义去重；53 个重复成员的预测数值一致，只是下载生成行不同。

## 覆盖情况

- 报告日期：2020-01-16 至 2021-12-31；
- 可用于目标日：2020-01-17 至 2021-12-31；
- 目标日完整覆盖：713 天、17,112 行；
- 2020-03-08 和 2021-03-14 是春季夏令时日期，原始报告没有 HE2，因此保留为 23 小时，不补值；
- 2020-01-01 至 2020-01-16 没有对应的 D-1 报告，公共状态中标记为 `UNAVAILABLE_SOURCE_COVERAGE`。

## 重要的 cutoff 限制

原始 CSV 的 `Report generated` 行是本次下载/导出时间，不是历史报告的原始发布时间。因此当前字段的状态是：

```text
PUBLIC_EX_ANTE_DATE_ONLY_NOT_CUTOFF_PROVEN
```

它可以用于数据审计、特征工程和问题重构，但不能直接宣称满足目标日 D-1 10:00 ET 的严格可用性。严格训练 gate 应在取得独立的历史发布时间后再打开。

## 与负荷、净负荷的关系

风电预测已经加入公共状态。随后审计的 `data/光伏预测` 压缩包虽然包含
`Seven Day Solar Power Forecast` 文件名，但 760 份 CSV 的逐小时数值全部为空，
因此光伏日前预测仍然没有加入。因此：

```text
wind forecast: available on covered dates
solar forecast: source archive present, but no numeric values
net-load forecast: not emitted
```

不能把 `load - wind` 直接命名为 `net_load`。IEEE-9 的三台物理机组、网络参数、成本和 Action-Space-v2 物理评价不由 ISO-NE 风电数据替换；风电只作为外生公共状态。

## 生成命令

```powershell
$py = ".venv\Scripts\python.exe"

& $py scripts\ingest_isone_wind_forecast.py `
  --source-dir "D:\code\ERCOTsimulation\data\风电预测"

& $py scripts\build_public_energy_state_wind_v2.py
```

产物：

- `data/public/isone_wind_forecast_2020_2021.parquet`：规范化风电预测表；
- `reports/WIND_FORECAST_INGESTION_AUDIT.json`：压缩包、哈希、覆盖和 cutoff 审计；
- `data/public/isone_2y_public_energy_state_wind_v2.parquet`：加入风电字段的新公共状态；
- `configs/public_feature_registry_iso2y_wind_v2.json`：字段合法性与限制；
- `reports/PUBLIC_FIELD_LEGALITY_AUDIT_WIND_V2.json`：v2 状态审计。

训练默认仍指向 v1；使用 v2 前必须显式选择新的状态文件，并确认 `strict_training_gate` 的限制。

光伏来源审计见 `reports/SOLAR_FORECAST_INGESTION_AUDIT.json`。如果后续拿到有数值的
光伏报表，必须重新执行审计，确认报告日期、Hour Ending、重复文件和发布时间后，才能
把 `solar_mw` 接入并重新生成 v2 状态。
