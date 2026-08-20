# ISO-NE fixed-renewable physical inputs

These are the processed inputs used by `Paper9Bus-ISONE-FixedRenewable-Physics-v1`.

- `isone_btm_solar_5min_estimated...`: estimated BTM PV, 5-minute, `NativeLoadBtmPv - Load`.
- `isone_btm_solar_hourly_estimated...`: hourly mean of 12 five-minute estimates.
- `isone_wind_dispatch_expected_hourly...`: hourly mean of irregular fuel-mix wind snapshots.
- `isone_btm_solar_wind_dispatch_proxy_hourly...`: merged convenience table.

Wind is intentionally labeled `DISPATCH_EXPECTED_WIND_GENERATION` and is not
`wind_available_mw`; it is a fixed exogenous proxy and never an OPF variable.
BTM solar is estimated, not metered, and is modeled only as negative load.
