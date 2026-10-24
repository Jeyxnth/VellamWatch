# VellamWatch ML core

Ranks ~250 m cells by flood-hotspot risk for the next 24–72 h. **Ranking only, never flood depth.**

Two stages: (1) XGBoost *susceptibility* from static terrain, calibrated on spatial-block out-of-fold predictions; (2) forecast *rainfall conditioning*: `logit(risk) = logit(u) + k·(rain_index − r0)`, where `u` is the cell's susceptibility percentile and `rain_index = max 24 h total + 0.5 × max 3 h total` (mm). `k` and `r0` are heuristic defaults (labelled as such) unless they can be fitted on past events.

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
`python -m ml.scripts.bench_predict` generates a full-bbox grid in a temp folder and times `predict_risk`. Measured on the author's Windows laptop: whole grid ~310 ms median (max ~400 ms), 1k-cell viewport ~7 ms, one-off warm-up ~3 s (model load, features, cached contributions).

## Map preview (optional)
`pip install folium`, then `python -m ml.scripts.preview_map --horizon 24 --as-of 2023-12-03T00:00:00+05:30` → `ml/artifacts/preview.html`.

## Files needed from the data owner (data/processed/, contract section D)
`cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` — plus (proposed) `events.csv`; see `CONTRACT_PROPOSALS.md`.

## Layout
`config.py` all tunables · `features.py` loading, derived features, labels, spatial CV · `rain.py` as_of window, IDW, rain index · `train.py` · `backtest.py` · `predict.py` · `explain.py` driver text · `synthetic/` · `scripts/preview_map.py` · `tests/`

## Notes
- Rainfall per cell = inverse-distance weighting (4 nearest points, power 2) of `rainfall.parquet`.
- No/light rain (rain index < 15.6 mm) can never give a "high" cell; risk never decreases as rain increases.
- Real model files live in `ml/artifacts/model/` (`model.json` native XGBoost JSON + `meta.json`); synthetic ones in `ml/artifacts/model_synthetic/` (never committed). `ml/.gitignore` allows only `artifacts/model/{model,meta}.json` and `artifacts/plots/real_*.png`.
- The root `.gitignore` ignores `ml/artifacts/` as a whole, which overrides `ml/.gitignore`. Until the root rule becomes `ml/artifacts/*`, commit real artifacts with `git add -f ml/artifacts/model/model.json ml/artifacts/model/meta.json ml/artifacts/plots/real_*.png`.
- The rainfall stage is described as **"heuristic, not fitted"** everywhere unless the event fit was accepted (see `meta.json` -> `rain.fitted`).
