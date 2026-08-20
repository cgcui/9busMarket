# Paper9Bus-ISONE-FixedRenewable-Physics-v1

Status: `PASS_FULL_TRAIN_FIXED_RENEWABLE_PHYSICS_V1`

- TRAIN materialized hours: 8760
- Time: `2024-06-01T00:00:00+00:00` to `2025-05-31T23:00:00+00:00`
- C0: no renewable injection
- C1: estimated BTM solar as negative load
- C2: estimated BTM solar plus fixed historical wind dispatch proxy
- Wind is fixed exogenous input, not an OPF decision variable; `wind_available_mw` is not used.
- Forecast columns are carried for provenance and are not consumed by the physical solver.
- Utility solar injection: 0 MW
- C0 exact equivalence: `PASS_ZERO_RENEWABLE_EQUIVALENCE`
- Maximum C2 system residual: 9.663e-13 MW
- Maximum nodal residual: 1.364e-12 MW
- Physical effect present: `True`
- Raw negative residual-load hours: 15
- Surplus accounting gate: `PASS_SURPLUS_ACCOUNTING_GATE`; explicit `surplus_export_mw` is retained and is not wind/BTM curtailment, G1 profit, or a strategic action.

DEV and HOLDOUT were not read, and Action-Space-v2 was not changed.
