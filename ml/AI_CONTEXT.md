# AI_CONTEXT — ML core (LEAD)

**Owner:** [LEAD NAME]

You are helping me build one lane of a hackathon project. Read this fully before answering.

## Project summary
VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai. Tagline: "Know your street before the water does." We combine an hourly rainfall forecast (Open-Meteo) with terrain features (elevation, slope, low-lying sinks, distance to drains/water bodies, road density) and past Chennai flood hotspots (2015, 2023) to rank flood risk per ~250 m grid cell for the next 24–72 hours. An XGBoost risk engine produces a score and a band (low/medium/high). An LLM turns the risk into a short Tamil and English action message and text-to-speech reads it aloud. Residents can upload a waterlogging photo; a pretrained vision model verifies it and verified reports refine the hotspot map. We claim hotspot RANKING, never exact flood depth. It is a 24-hour sprint with a 5-person team.

## My lane and folder
I am the LEAD and own **ML core**. My folder is `ml/` (branch `ml-core`). I also own `docs/contract.md`. Other lanes: `data/`, `backend/`, `frontend/`, `voice_vision/` — not mine.

## Contract
The contract is at `docs/contract.md`. Follow it exactly: field names, types, risk bands (low <0.33, medium 0.33–0.66, high >=0.66), cell_id format `r{row}_c{col}`, ISO 8601 timestamps with +05:30, WGS84 (lat, lon). I may update the contract myself, but only deliberately and after telling the team.

## Rules for the AI
- Only create or edit files inside `ml/`. For contract changes, tell me what to change and I will edit `docs/contract.md` myself.
- Prefer simple, readable code that works over clever code.
- Small commits; suggest a commit message after each piece of work.
- No secrets or API keys in code. Use a `.env` file and provide `.env.example`.
- Write a runnable example or test for every function.
- If blocked on another module (e.g. data not ready), create a mock that matches the contract.
- State assumptions explicitly.
- Keep answers short.

## Tech stack
Python 3.11, pandas, pyarrow, numpy, scikit-learn, XGBoost, (optional) SHAP, joblib, pytest, matplotlib.

## Exact deliverables
1. Feature engineering from `data/processed/features.parquet` and `rainfall.parquet` (rolling rainfall sums e.g. 3/6/12/24 h, max hourly intensity, interaction of rain with low elevation/sink depth).
2. Labels from `data/processed/hotspots.csv`: hotspot cells positive, others negative; handle class imbalance.
3. XGBoost training and validation; hold out one whole event.
4. BACKTEST on one past Chennai flood event: recall of true hotspots within the top-k riskiest cells and precision@k (k = 50, 100, 250, 500...). No flood depth claims.
5. `ml/predict.py` with `predict_risk(min_lat, min_lon, max_lat, max_lon, horizon_h) -> list[dict]`, each dict = a contract `cells` object (`cell_id, lat, lon, risk_score, risk_band, peak_window{start,end}, drivers[]`).
6. Model artifacts saved to `ml/artifacts/` (git-ignored).
7. `ml/BACKTEST.md` containing the real numbers.

## Acceptance checks to help me pass
- Output of `predict_risk` validates against the contract (write a small validator test).
- risk_score is within 0–1 and risk_band matches the thresholds.
- Backtest event is excluded from training; results reproducible with one command.
- Works with a stub/mock when real data is missing.
- Viewport prediction returns in under 2 seconds.

## Starter prompts
1. "Write `ml/predict.py` as a stub that returns realistic mock cells for any bounding box in the exact contract format, plus a pytest that validates the schema."
2. "Given features.parquet and rainfall.parquet columns from the contract, write `ml/features.py` that builds one row per (cell, forecast time) with rolling rainfall features."
3. "Write `ml/train.py` that trains XGBoost on hotspot labels with class-imbalance handling and holds out the 2023 event; save artifacts to ml/artifacts."
4. "Write `ml/backtest.py` computing recall@k and precision@k of true hotspots among the top-k ranked cells, and print a markdown table I can paste into BACKTEST.md."
5. "Add a function mapping XGBoost feature importance to short human-readable `drivers` strings (e.g. 'low-lying area', 'heavy rain forecast') per cell."
6. "Write a contract validator in `ml/validate_contract.py` that checks a `cells` list against section B/C of docs/contract.md."
