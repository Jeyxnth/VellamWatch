# HANDOFF — Backend API

**Owner:** [MEMBER 3 NAME]

## What you need from others
| From | What | Contract | By |
|------|------|----------|----|
| Lead ([LEAD NAME]) | Frozen contract | B, C | H+2 |
| Lead (ml/) | `ml/predict.py` stub (mock cells in contract format) | C | H+3 |
| Lead (ml/) | Real `predict_risk()` with trained model | C | H+14 |
| Voice/vision ([MEMBER 5 NAME]) | `alerts.py`, `tts.py`, `vision.py` stubs returning contract-shaped data | C | H+4 |
| Voice/vision | Real implementations | C | H+14 (alerts, tts), H+16 (vision) |
| Data ([MEMBER 2 NAME]) | `cells.geojson` sample (for realistic mock cell ids and coordinates) and bounding box | A, D | H+3 |
| Frontend ([MEMBER 4 NAME]) | Their API base URL / CORS origin, and bug reports on response shapes | B | H+6 |

## What others need from you
| To | What | By |
|----|------|----|
| Frontend | Running mock API (local) + `backend/README.md` | H+1 / H+2 |
| Frontend | Public deployed URL (mock mode first) | H+8 |
| Lead | Confirmation the `predict_risk` import path works in the API | H+4 |
| Voice/vision | Confirmation how audio files are served so `synthesize()` can return a path/URL that works | H+6 |
| Frontend, lead | Real-mode deployment | H+14 |
| Everyone | Final stable deploy | H+20 |

## Mocking plan if blocked
- `USE_MOCK=1` serves everything fake (default). Mock cells use realistic Chennai coordinates and the exact response shapes.
- Per-module fallbacks: wrap each import (`predict_risk`, `generate_alert`, `synthesize`, `verify_photo`) in try/except and fall back to the mock for that endpoint only, logging the failure.
- If a teammate's module isn't delivered, write a tiny mock with the contract signature inside `backend/mocks/` — never in their folder.

## Integration checkpoints that involve you
- **H+2** contract frozen — you validate endpoints are implementable.
- **H+10** first end-to-end run — you own making it happen.
- **H+14** real model wired in — you integrate ml/ and voice_vision/.
- **H+20** feature freeze — final deployment.
- **H+23** demo rehearsal done — the live URL and mock fallback are ready.
