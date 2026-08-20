# Action-Space-v2 Jump Audit

分类：`CROSSING_DOMINATED_JUMP_STRUCTURE`

本审计只读取现有 8760h × 6 × 43 private cell bank，没有重新运行 OPF，没有读取 DEV/HOLDOUT，没有解决 opponent belief，也没有训练。

- 曲线：52,560；事件：30,552
- dispatch tolerance：`1e-05 MW`（solver primal tolerance `1e-07`）
- crossing-aligned fraction：1.0000
- jump fractions：`{"0": 0.4187214611872146, "1": 0.5812785388127854, "2": 0.0, ">=3": 0.0}`

跳变主判据是 conventional generator dispatch 的 tolerance-aware 变化；profit/LMP 和 active/binding set 仅作辅助证据。当前结果仅用于结构审计，不直接决定 compact action space。
