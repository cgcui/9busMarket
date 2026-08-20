# Paper9Bus-ISONE-FixedRenewable-Physics-v1

Status: `PASS_FIXED_RENEWABLE_PHYSICS_V1_8760H`

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
- Full-TRAIN surplus rule active: `True`; solver-side residual load is nonnegative and at least the frozen 30 MW aggregate generator Pmin, with raw/clipped/uplift MW retained in outputs.

DEV and HOLDOUT were not read, and Action-Space-v2 was not changed.
