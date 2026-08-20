# IEEE-9 Renewable-Aware 负荷版本（当前冻结）

当前版本将 9bus 问题改成“可接入风光、但暂不使用风光预测”的 renewable-aware 版本。

## 当前问题定义

```text
ISO-NE 2020-2021 gross load forecast
        │
        │ TRAIN-only median-ratio mapping
        ▼
case9_blv 的 IEEE-9 三个原始负荷母线
        │
        ▼
IEEE-9 frozen 3-generator DC-OPF / strategic market
```

IEEE-9 网络、三台机组 G1/G2/G3、机组成本、支路参数、负荷空间比例和动作空间保持冻结。
ISO-NE 的机组、燃料、天气、互联线和可再生能源不会替换 IEEE-9 物理机组或网络。

## 负荷公式

目标小时 `t` 使用 ISO-NE 公共状态里的目标日逐小时总负荷预测：

```text
L_iso(t) = hourly_load_mw[target_he - 1]
alpha(t) = clip(L_iso(t) / median_TRAIN(L_iso), 0.5, 1.5)
```

然后保持冻结 IEEE-9 的原始负荷空间比例：

```text
P5(t) = 90  × alpha(t)
P7(t) = 100 × alpha(t)
P9(t) = 125 × alpha(t)
其他母线负荷 = 0
```

当前 TRAIN 中位数为约 `12,680 MW`，因此典型 ISO-NE 负荷对应 IEEE-9 基准总负荷
`315 MW`。映射只使用 TRAIN 统计量，DEV 不参与拟合。

## 风光接口状态

风电和光伏接口已经预留，但当前物理计算固定为：

```text
wind_forecast_used = false
solar_forecast_used = false
net_load_forecast_used = false
renewable_input_mode = DISABLED_FORECASTS_NOT_ADDED
```

将来只有同时获得完整、可审计的风电和光伏目标日预测后，才允许切换为：

```text
L_net(t) = L_iso(t) - W(t) - S(t)
```

并重新冻结一份新版本；不能在当前版本里用空光伏报表、历史实际出力或 0 MW 代替。

## 生成命令

```powershell
$py = ".venv\Scripts\python.exe"
& $py scripts\build_paper9bus_isone2y_load_bridge.py
```

产物：

- `data/public/paper9bus_isone2y_gross_load_bridge_v1.parquet`：逐小时 ISO-NE→IEEE-9 负荷桥；
- `configs/paper9bus_isone2y_renewable_aware_load_v1.json`：冻结公式和 TRAIN 统计量；
- `reports/PAPER9BUS_ISONE2Y_RENEWABLE_AWARE_LOAD_AUDIT.json`：来源、覆盖和不使用风光审计。
