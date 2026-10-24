# VellamWatch ML core

Ranks ~250 m cells by flood-hotspot risk for the next 24–72 h. **Ranking only, never flood depth.**

Two stages: (1) XGBoost *susceptibility* from static terrain, calibrated on spatial-block out-of-fold predictions; (2) forecast *rainfall conditioning*: `logit(risk) = logit(u) + k·(rain_index − r0)`, where `u` is the cell's susceptibility percentile and `rain_index = max 24 h total + 0.5 × max 3 h total` (mm). `k` and `r0` are heuristic defaults (labelled as such) unless they can be fitted on past events.

## Setup (from the repo root)
```
python -m venv ml/.venv
ml/.venv/Scripts/python -m pip install -r ml/requirements.txt      # Windows
# source ml/.venv/bin/activate && pip install -r ml/requirements.txt  # macOS/Linux
```
All commands below run from the **repo root**.

## Synthetic data (pipeline test only — NOT results)
```
python -m ml.synthetic.make_synthetic          # writes ml/synthetic_out/ (+ SYNTHETIC.flag); --full for the whole bbox
python -m ml.train                             # trains, saves ml/artifacts/synthetic/
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
The forecast window is `[as_of, as_of + horizon_h)`. `as_of` defaults to now in IST; override with env `ML_AS_OF` or `--as-of`:
```
ML_AS_OF=2023-12-03T00:00:00+05:30 python -m ml.predict --bbox 12.95 80.15 13.15 80.30 --horizon 24   # Cyclone Michaung replay
```
PowerShell: `$env:ML_AS_OF="2023-12-03T00:00:00+05:30"`. In code: `from ml.predict import predict_risk` (contract signature, uses `ML_AS_OF`) or `predict_risk_at(..., as_of=...)`. If `rainfall.parquet` has no data in the window you get an error saying so.

## Map preview (optional)
`pip install folium`, then `python -m ml.scripts.preview_map --horizon 24 --as-of 2023-12-03T00:00:00+05:30` → `ml/artifacts/preview.html`.

## Files needed from the data owner (data/processed/, contract section D)
`cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` — plus (proposed) `events.csv`; see `CONTRACT_PROPOSALS.md`.

## Layout
`config.py` all tunables · `features.py` loading, derived features, labels, spatial CV · `rain.py` as_of window, IDW, rain index · `train.py` · `backtest.py` · `predict.py` · `explain.py` driver text · `synthetic/` · `scripts/preview_map.py` · `tests/`

## Notes
- Rainfall per cell = inverse-distance weighting (4 nearest points, power 2) of `rainfall.parquet`.
- No/light rain (rain index < 15.6 mm) can never give a "high" cell; risk never decreases as rain increases.
- Models, synthetic data and synthetic plots are git-ignored. Only `ml/artifacts/plots/real_*.png` may be committed (see below).
- Root `.gitignore` ignores `ml/artifacts/` as a whole, so committing a real plot needs `git add -f ml/artifacts/plots/real_pr_curve.png`, or the root rule changing to `ml/artifacts/*`.
