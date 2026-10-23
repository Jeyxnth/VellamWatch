# HANDOFF — ML core (LEAD)

**Owner:** [LEAD NAME]

## What you need from others
| From | What | Contract | By |
|------|------|----------|----|
| Data pipeline ([MEMBER 2 NAME]) | ~50-row samples of `cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` | D | H+3 |
| Data pipeline | Full processed files, with confirmed bounding box | A, D | H+8 |
| Data pipeline | `data/SOURCES.md` (needed for the backtest write-up) | D | H+8 |
| Backend ([MEMBER 3 NAME]) | Confirmation that the stub `predict_risk` imports and runs inside the API | C | H+4 |
| All four teammates | Review feedback on the contract before it is frozen | all | H+2 |

## What others need from you
| To | What | By |
|----|------|----|
| Everyone | Frozen `docs/contract.md` | H+2 |
| Backend | `ml/predict.py` stub returning mock cells in contract format | H+3 |
| Data pipeline | Answers on required columns, dtypes and the `hotspots.csv` event labels | H+2 |
| Voice/vision ([MEMBER 5 NAME]) | Sample of real `cells` objects (with `drivers` and `peak_window`) as input for alert generation | H+6 |
| Backend | Real `predict_risk()` using the trained model | H+14 |
| Presenter / everyone | Backtest numbers and one chart for slides | H+17 |

## Mocking plan if blocked
- No data yet: generate a synthetic `features.parquet` and `hotspots.csv` locally (in `ml/`, not in `data/`) with the contract columns, to develop the pipeline.
- Data late after H+8: train on whatever partial files exist and report the limitation in `BACKTEST.md`.
- Backend not ready: test `predict_risk` with a plain Python script that validates output against the contract.

## Integration checkpoints that involve you
- **H+2** contract frozen — you run this.
- **H+10** first end-to-end run — the stub (or first model) must serve through `/api/risk`.
- **H+14** real model wired in — you hand over the trained model and confirm backend calls it.
- **H+20** feature freeze — lock the model and `BACKTEST.md`.
- **H+23** demo rehearsal done — know the numbers and the "ranking, not depth" wording.
