# Contract proposals from the ML core

These are proposals only. `docs/contract.md` is unchanged; the lead decides.

## 1. Add `events.csv` to section D (needed for the backtest and rain fitting)

`hotspots.csv` has an `event` column but no dates, so the ML core cannot pick the matching
rainfall window. Proposed new file, produced by `data/`:

- `data/processed/events.csv` — columns `event,start,end`.
  - `event` matches `hotspots.csv` (`"2015"`, `"2023"`).
  - `start` and `end` are inclusive IST dates (`2023-12-02`, `2023-12-05`); an `end` that includes a time is read as exact and exclusive.

Until it exists, `ml/config.py` uses `EVENTS_TODO` — 2015: 2015-12-01..2015-12-03, 2023 (Cyclone Michaung): 2023-12-02..2023-12-05 — **unverified, confirm with the data owner**.

## 2. Rainfall conventions for `rainfall.parquet` (section D)

- One row per (hour, rainfall point). `timestamp` is the START of the hour (ISO 8601 +05:30, or tz-aware).
- `precip_mm` is the accumulation during that hour (not a rate in another unit).
- It must cover the 2015 and 2023 event windows plus the current forecast. If a requested forecast window has no rows, `predict_risk` raises a clear error.

## 3. `cell_id` rows/cols must be contiguous integers on one grid

Local relief (cell elevation minus the ~1.25 km neighbourhood mean) and the spatial CV use `row`/`col`
parsed from `cell_id`. Orientation does not matter, but neighbouring cells must have adjacent indices.

## 4. (Informational) `as_of` is not part of the API

`predict_risk` keeps the contract signature. The forecast window is `[as_of, as_of + horizon_h)`.
`as_of` is the current IST time unless env `ML_AS_OF` is set. Backend can replay an event by
setting `ML_AS_OF`, or can import `ml.predict.predict_risk_at(..., as_of=...)`.
