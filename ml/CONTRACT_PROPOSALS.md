# Contract proposals from the ML core

`docs/contract.md` is the source of truth. All proposals below were **ACCEPTED** and are in contract v1.1 (2026-10-24).

## 1. Add `events.csv` to section D — ACCEPTED
`data/processed/events.csv` with columns `event,start,end` (ISO 8601 with +05:30; start inclusive, end exclusive; `event` matches `hotspots.csv`). Owner: data.
Until it exists, `ml/config.py` uses `EVENTS_TODO` (same format, **unverified**: 2015-12-01..2015-12-04 and 2023-12-02..2023-12-06, Cyclone Michaung) and warns.
Note: v1.1 made `end` exclusive and the offset mandatory; the earlier "inclusive date" wording in this file is superseded.

## 2. Rainfall conventions for `rainfall.parquet` — ACCEPTED
One row per `(timestamp, lat, lon)`; `timestamp` = start of the hour in +05:30; `precip_mm` = hourly accumulation; the file covers every event window plus at least 72 h of forecast; `data/SOURCES.md` names the historical and forecast sources.
Enforced on load: timestamps need an offset and must be on the hour in +05:30 (error naming `rainfall.parquet`); missing event coverage or a short forecast only warns.

## 3. `cell_id` rows/cols are contiguous integers on one grid — ACCEPTED
Plus `cells.geojson` carries `cell_id`, `row`, `col`; row 0 southernmost, col 0 westernmost. Enforced on load with errors naming `cells.geojson`.

## 4. `as_of` — ACCEPTED (and extended to the API)
`predict_risk` keeps its signature (uses env `ML_AS_OF`, else now IST). `predict_risk_at(..., as_of=None, data_dir=None)` takes an ISO 8601 string with offset (no offset -> `ValueError`). `get_model_info()` reports `data_source`, `trained_at`, `data_dir`. API responses gain `as_of` and `data_source`.

## Open (not yet proposed in the contract)
- If the band-share report in `BACKTEST.md` shows too many "high" cells at 100 mm, it prints a PROPOSAL (a new `RAIN_R0` or `BAND_HIGH`). Changing `BAND_HIGH` would be a contract change; nothing is applied automatically.
