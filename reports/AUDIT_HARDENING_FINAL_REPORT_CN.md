# Paper9Bus Audit Hardening T46.5

- 结论：`PASS_AUDIT_HARDENING_V1`。
- T41/T43/T45/T46 历史科学结论保持：`True`。
- X1 字段注册与 K=4 exact prompt identity：`True`。
- 重复检测器已从标准答案假阳性中修复；重验证后的 T45 candidate-bank repetition 统计：FREE `1`，Grammar `0`。
- 原始 T41-T46 报告、数据集、adapter、candidate parquet 均未覆盖；本轮只写入新的 hardening 报告。
- 当前 checkout 没有 `.git`，记录为 `GIT_PROVENANCE_UNAVAILABLE`；未声称验证 commit `6be1ca7`。
- 未访问 FINAL，未启动 GRPO，未执行 T47/T48/T49。

## 历史结论对比

| 指标 | 原始 | 重验证 | changed | scientific conclusion changed |
|---|---|---|---|---|
| T41 | PASS_TARGET_IDENTIFIABILITY_V1 | PASS_TARGET_IDENTIFIABILITY_V1 | False | False |
| T43 | PASS_FRESH_DATASET_V1 | PASS_FRESH_DATASET_V1 | False | False |
| T45 | PASS_CONSTRAINED_INTERFACE_V1 | PASS_CONSTRAINED_INTERFACE_V1 | False | False |
| T46 | FAIL_TRUE_ECONOMIC_SIGNAL | FAIL_TRUE_ECONOMIC_SIGNAL | False | False |

T47 只有在 `PASS_AUDIT_HARDENING_V1` 时才解锁；本脚本不会自动执行 T47。
