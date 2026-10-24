# VellamWatch ML core

Ranks ~250 m cells by flood-hotspot risk for the next 24–72 h. **Ranking only, never flood depth.**

Two stages:
1. **Susceptibility** (static): XGBoost on terrain features, calibrated on spatial-block out-of-fold predictions. Each cell gets a percentile `u` (0-1) of its susceptibility over the model's **full study grid**.
2. **Budgeted ranking** (`budget.py`): the forecast's *rain severity* decides **how much** of the study area is flagged; `u` decides **which** cells. Severity = max(severity of the 24 h total, severity of the wettest 3 h), averaged over the whole grid. Policy targets (all in `config.py`, **policy targets, not fitted**): high cells are 0% of the grid below 25 mm / 24 h, 2.5% at 50 mm, 15% at 100 mm, 30% at 150 mm, 40% at 200 mm+ (hard cap 50%); medium-or-high is about twice that (cap 80%). The top share by `u` is "high", the next band "medium", the rest "low".

`risk_score` is a continuous ranking score: low in [0, 0.33), medium [0.33, 0.66), high [0.66, 1], rising with susceptibility inside each band and never decreasing as rainfall rises; the order of cells by risk never changes with rainfall. It is not a probability and not a depth.

**Viewport independence:** percentiles and budgets are computed over the full study grid, never the requested bbox, so the same street gets the same score and band whatever viewport the frontend asks for. Consequence: in a bbox full of susceptible streets the share of high cells can exceed the cap (the cap applies to the whole grid).

## Setup (from the repo root)
Tested on **Python 3.10.11** (Windows 11) with the exact versions pinned in `requirements.txt`.

```
python -m venv ml/.venv
ml/.venv/Scripts/python -m pip install -r ml/requirements.txt      # Windows
# source ml/.venv/bin/activate && pip install -r ml/requirements.txt  # macOS/Linux
```
All commands below run from the **repo root**.

## Synthetic data (pipeline test only — NOT results)
```
python -m ml.synthetic.make_synthetic          # writes ml/synthetic_out/ (+ SYNTHETIC.flag); --full for the whole bbox
python -m ml.train                             # trains, saves ml/artifacts/model_synthetic/
python -m ml.backtest                          # writes ml/BACKTEST.md + ml/artifacts/plots/synthetic_*.png
python -m ml.predict --bbox 13.0 80.2 13.05 80.25 --horizon 24
python -m pytest ml/tests -q
```
Synthetic runs print and write the banner `SYNTHETIC DATA - PIPELINE TEST ONLY - NOT RESULTS`.

## Real data
Put the four contract files in `data/processed/` (see "Files needed"), then the same commands, no code change:
```
python -m ml.train
python -m ml.backtest
python -m ml.predict --bbox 12.95 80.15 13.15 80.30 --horizon 24
```
Data folder choice: `--data-dir` > env `ML_DATA_DIR` > `data/processed/` (if `features.parquet` exists) > `ml/synthetic_out/`. A folder is "synthetic" iff it contains `SYNTHETIC.flag`; real and synthetic models live in separate folders under `ml/artifacts/`.

## For the backend teammate: one command to a working synthetic model
```
python -m ml.bootstrap_synthetic        # make_synthetic -> train -> warm-up; prints "data source: SYNTHETIC"
```
It writes `ml/synthetic_out/` and `ml/artifacts/model_synthetic/`, then `from ml.predict import predict_risk` works from the repo root. Everything it makes is synthetic, so `get_model_info()["data_source"]` is `"synthetic"` and the frontend must show its warning banner. The synthetic forecast lasts 72 h from generation time, so re-run it to refresh, or pin the clock with `ML_AS_OF=2023-12-03T00:00:00+05:30` (the synthetic 2023 event).

At server start call `warm_up()` once:
```python
from ml.predict import warm_up, get_model_info, predict_risk, predict_risk_at
info = warm_up()       # loads model + data, caches, runs a tiny prediction (~1-5 s); returns get_model_info() + timings
```
`warm_up()` never fails just because the forecast is stale: it falls back to the last 24 h in `rainfall.parquet` (and warns). It raises `MissingModelError` (message tells you to run `python -m ml.train` or `ml.bootstrap_synthetic`) if no model exists, and a `MissingDataError` naming the file if data is missing.

## Replaying a past event (as_of)
The forecast window is `[as_of, as_of + horizon_h)`. `as_of` defaults to now in IST; override with env `ML_AS_OF` or `--as-of`. It must be ISO 8601 **with a UTC offset** (contract v1.1); values without an offset raise `ValueError`.
```
ML_AS_OF=2023-12-03T00:00:00+05:30 python -m ml.predict --bbox 12.95 80.15 13.15 80.30 --horizon 24   # Cyclone Michaung replay
```
PowerShell: `$env:ML_AS_OF="2023-12-03T00:00:00+05:30"`. In code:
```python
from ml.predict import predict_risk, predict_risk_at, get_model_info
predict_risk(min_lat, min_lon, max_lat, max_lon, 24)                       # contract signature; uses ML_AS_OF or now
predict_risk_at(min_lat, min_lon, max_lat, max_lon, 24, as_of="2023-12-03T00:00:00+05:30")
get_model_info()   # {"data_source": "real"|"synthetic", "trained_at": ISO, "data_dir": ..., "rain_stage": ...}
```
If `rainfall.parquet` has nothing in the window you get an error saying so.

## Load-time validation (errors name the file to fix)
- `cells.geojson`: properties `cell_id`, `row`, `col`; row/col agree with `cell_id`; contiguous from 0; row 0 southernmost, col 0 westernmost.
- `rainfall.parquet`: timestamps carry an offset and are on the hour in +05:30.
- `events.csv`: columns `event,start,end`, offsets required, end after start.
- Warnings only (no crash): rainfall does not cover every event window; forecast shorter than 72 h from now (when `ML_AS_OF` is unset); `events.csv` missing (placeholder dates used); `hotspots.csv` events not in `events.csv`.

## Latency benchmark (synthetic, ~23k cells)
`python -m ml.scripts.bench_predict` generates a full-bbox grid in a temp folder and times `predict_risk`. Measured on the author's Windows laptop (synthetic, 23,140 cells): whole grid ~330 ms median (max ~375 ms), 1k-cell viewport ~10 ms, one-off warm-up ~4-5 s (model load, features, cached contributions, first rainfall window).

## Map preview (optional)
`pip install folium`, then `python -m ml.scripts.preview_map --horizon 24 --as-of 2023-12-03T00:00:00+05:30` → `ml/artifacts/preview.html`.

## Files needed from the data owner (data/processed/, contract section D)
`cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` — plus (proposed) `events.csv`; see `CONTRACT_PROPOSALS.md`.

## Layout
`config.py` all tunables (incl. the rain-to-budget policy) · `features.py` loading, derived features, labels, spatial CV · `rain.py` as_of window, IDW, study-area rainfall · `budget.py` severity, budgets, risk_score mapping · `train.py` · `backtest.py` · `predict.py` · `explain.py` driver text · `bootstrap_synthetic.py` · `synthetic/` · `scripts/` (preview_map, bench_predict) · `tests/`

## Notes
- Rainfall per cell = inverse-distance weighting (4 nearest points, power 2) of `rainfall.parquet`; severity uses the mean over the whole grid, and the "Heavy rain forecast: X mm in 24 h" driver quotes that same study-area number. Peak windows are per cell.
- Light or no rain (24 h total < 15.6 mm) gives no medium and no high cells.
- The old additive-logit rain term (`k`, `r0`) is gone: it saturated (everything "high") in heavy events and could not be identified from two events.
- Do not tune stage-2 parameters on synthetic data; they are policy targets until real flood-impact evidence exists.
- Real model files live in `ml/artifacts/model/` (`model.json` native XGBoost JSON + `meta.json`); synthetic ones in `ml/artifacts/model_synthetic/` (never committed). The root `.gitignore` un-ignores `ml/artifacts/model/` and `ml/artifacts/plots/`; `ml/.gitignore` then allows only `model.json`, `meta.json` and `plots/real_*.png`.
