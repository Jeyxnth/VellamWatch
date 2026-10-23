# ROLE — ML core (LEAD)

**Owner:** [LEAD NAME]

## Mission
Build the XGBoost risk engine that ranks ~250 m grid cells in Chennai by flood risk for the next 24–72 hours, and prove it with a backtest on a real past event. Also own `docs/contract.md` so the five lanes stay compatible. We claim hotspot RANKING, never exact flood depth.

## What you own
- XGBoost risk model, training and validation code in `ml/`.
- Feature engineering from `data/processed/features.parquet` and `rainfall.parquet`.
- Labels from `data/processed/hotspots.csv` (hotspot cells = positive, all others = negative; handle class imbalance).
- `ml/predict.py` exactly per contract section C.
- Model artifacts in `ml/artifacts/` (git-ignored).
- `ml/BACKTEST.md` with real numbers.
- `docs/contract.md` (all changes go through you) and the integration checkpoints.

## Deliverables
- [ ] H+2: Contract frozen (review with all four teammates, then announce "frozen")
- [ ] H+3: `ml/predict.py` stub returning mock cells in the exact contract format, so backend can import it
- [ ] H+4: Notebook/script that loads the ~50-row data samples and builds a feature matrix
- [ ] H+9: First trained XGBoost model on full data (class-imbalance handled via `scale_pos_weight` and/or sampling)
- [ ] H+12: Backtest on one past event (2015 or 2023): recall of true hotspots within top-k riskiest cells, and precision@k, for several k
- [ ] H+14: Real `predict_risk()` wired into backend (checkpoint)
- [ ] H+17: `ml/BACKTEST.md` written with real numbers, plus a top-drivers explanation (feature importance / SHAP) that fills the `drivers` field
- [ ] H+20: Feature freeze; final model saved to `ml/artifacts/`
- [ ] H+22: Slide-ready backtest figure (top-k recall chart or map)
- [ ] H+23: Demo rehearsal done

## Definition of done
- `predict_risk(min_lat, min_lon, max_lat, max_lon, horizon_h)` returns a list of dicts that validate against the contract's `cells` object (cell_id, lat, lon, risk_score 0–1, risk_band, peak_window, drivers).
- Backtest uses a time/event split so the evaluated event was NOT in training; the numbers in `BACKTEST.md` are reproducible with one command.
- Output is described everywhere as ranking, not flood depth.
- Predict call for a typical map viewport takes under 2 seconds.

## Out of scope (do NOT touch)
- Downloading or cleaning raw data (data/).
- API routes, CORS, storage (backend/).
- Any UI code (frontend/).
- Alert text, TTS, photo verification (voice_vision/).
- Do not predict flood depth or claim it.

## Git branch and folder
- Branch: `ml-core`
- Folder: `ml/` (plus ownership of `docs/`)

## Cut-first list (drop in this order if time runs short)
1. SHAP plots (use plain XGBoost feature importance for `drivers`).
2. Hyperparameter search (use sensible defaults).
3. 72 h horizon (ship 24 h only).
4. Second backtest event (keep just one).
5. Probability calibration.
Never cut: `predict.py` per contract and the one real backtest.
