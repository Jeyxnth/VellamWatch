# VellamWatch — Interface Contract

**Version 1.1**

**Project:** VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai.
**Tagline:** "Know your street before the water does."

This file is the single source of truth. Everyone builds to it. Owned by the lead (ml/).

---

## A. Conventions

- **Coordinates:** WGS84, always ordered `(lat, lon)`.
- **Timestamps:** ISO 8601 with the IST offset, e.g. `2025-11-30T18:00:00+05:30`.
- **Language codes:** `"ta"` (Tamil) and `"en"` (English).
- **risk_score:** float in the range 0–1. It is a RANKING score between 0 and 1, not a calibrated probability and not a flood depth. Do not describe it as a probability in the UI or slides.
- **risk_band:** string, one of
  - `"low"` — risk_score < 0.33
  - `"medium"` — 0.33 <= risk_score < 0.66
  - `"high"` — risk_score >= 0.66
  - Thresholds may be tuned only by the ML lead, via a contract update.
- **Bounding box (proposal):** lat 12.85–13.25, lon 80.05–80.35. The data owner confirms it.
- **Grid:** the study area is divided into a fixed ~250 m grid.
- **cell_id:** `"r{row}_c{col}"` (e.g. `r12_c34`), where `row` and `col` are contiguous non-negative integers on ONE grid. Row 0 is the southernmost row, col 0 is the westernmost column.
- **Source of truth for cells:** `data/processed/cells.geojson` defines all cell ids and geometry. Every feature must carry properties `cell_id`, `row`, `col`, and the polygon geometry.
- **We claim hotspot RANKING, never exact flood depth.**

---

## B. REST API (served by `backend/`, base path `/api`)

### GET /api/health
Response:
```json
{"status": "ok"}
```

### GET /api/risk?lat=&lon=&radius_m=&horizon_h=&as_of=
Optional `as_of`: ISO 8601 timestamp WITH a UTC offset, e.g. `2023-12-03T00:00:00+05:30`. The forecast window is `[as_of, as_of + horizon_h]`. Default: current time in IST. A value without an offset returns HTTP 400 with `{"error": "..."}`. Use case: replaying Cyclone Michaung for demos and backtests.

Response:
```json
{
  "generated_at": "ISO",
  "as_of": "ISO (the value actually used)",
  "data_source": "real | synthetic",
  "horizon_h": 24,
  "cells": [
    {
      "cell_id": "r12_c34",
      "lat": 13.0827,
      "lon": 80.2707,
      "risk_score": 0.72,
      "risk_band": "high",
      "peak_window": {"start": "ISO", "end": "ISO"},
      "drivers": ["low elevation", "heavy rain forecast"]
    }
  ]
}
```

### GET /api/alert?lat=&lon=&lang=ta|en&as_of=
Optional `as_of`: same meaning and validation as in `GET /api/risk` (offset required, else HTTP 400 `{"error": "..."}`; default current IST).

The frontend MUST show a visible warning banner whenever `data_source` is `"synthetic"`.

Response:
```json
{
  "as_of": "ISO (the value actually used)",
  "data_source": "real | synthetic",
  "cell_id": "r12_c34",
  "risk_band": "high",
  "peak_window": {"start": "ISO", "end": "ISO"},
  "message_en": "string",
  "message_ta": "string",
  "actions": ["string"],
  "audio_url": "string | null"
}
```

### POST /api/report
Multipart form: `photo` (file), `lat`, `lon`, `timestamp`.
Response:
```json
{
  "report_id": "string",
  "status": "pending|verified|rejected",
  "confidence": 0.0,
  "cell_id": "string",
  "reason": "string"
}
```

### GET /api/reports?min_lat=&min_lon=&max_lat=&max_lon=
Response:
```json
{
  "reports": [
    {"report_id": "string", "lat": 13.08, "lon": 80.27, "timestamp": "ISO", "status": "verified"}
  ]
}
```

### Errors
All errors are JSON `{"error": "string"}` with a proper HTTP status (400 bad input, 404 not found, 500 server error).

---

## C. Python module interfaces (so backend can import them)

- `ml/predict.py`
  `predict_risk(min_lat, min_lon, max_lat, max_lon, horizon_h) -> list[dict]`
  Each dict matches a `cells` object in `GET /api/risk`. Uses the current IST time (or env `ML_AS_OF`) as the forecast start.
- `ml/predict.py` additionally exposes:
  - `predict_risk_at(min_lat, min_lon, max_lat, max_lon, horizon_h, as_of=None, data_dir=None) -> list[dict]` — same cell dicts. `as_of` is an ISO 8601 string with offset. The backend calls it when `as_of` is supplied.
  - `get_model_info() -> dict` with at least `{"data_source": "real"|"synthetic", "trained_at": ISO, "data_dir": str}`. The backend reads `data_source` from it.
- `voice_vision/alerts.py`
  `generate_alert(cell: dict) -> {"message_en", "message_ta", "actions"}`
- `voice_vision/tts.py`
  `synthesize(text: str, lang: str) -> str`
  Returns a URL or file path under a served audio folder.
- `voice_vision/vision.py`
  `verify_photo(image_bytes: bytes, lat: float, lon: float, timestamp: str) -> {"status", "confidence", "reason"}`

---

## D. Data files (produced by `data/`, consumed by `ml/`; all in `data/processed/`)

- `cells.geojson` — grid polygons; properties `cell_id`, `row`, `col` (see section A).
- `features.parquet` — `cell_id, lat, lon, elevation_m, slope_deg, sink_depth_m, dist_drain_m, dist_water_m, road_density, built_up_frac`.
- `hotspots.csv` — `cell_id, event, source, notes` (`event` e.g. `"2015"` or `"2023"`).
- `rainfall.parquet` — `timestamp, lat, lon, precip_mm` (hourly forecast and historical).
  - One row per `(timestamp, lat, lon)`. `timestamp` is the START of the hour, in +05:30. `precip_mm` is the accumulation during that hour, in mm.
  - The file must cover every event window in `events.csv` AND at least 72 hours of forecast from the current time.
  - `data/SOURCES.md` must state which source supplies historical rainfall and which supplies the forecast, because they may differ.
- `events.csv` (owner: data) — `event, start, end`. `start` and `end` are ISO 8601 with +05:30; `start` is inclusive, `end` is exclusive. `event` values must match the `event` column of `hotspots.csv`.

A small sample of each (about 50 rows) must be delivered within the first 3 hours so others can start.

---

## E. Change rule

Nobody edits this file directly. Request changes from the lead. Everyone builds to this contract; when blocked by another module, mock it.

---

## F. Integration checkpoints

| Hour | Checkpoint |
|------|------------|
| H+2  | Contract frozen |
| H+3  | Sample files delivered in the final formats (including events.csv) |
| H+10 | First end-to-end run |
| H+14 | Real model wired in |
| H+20 | Feature freeze |
| H+23 | Demo rehearsal done |

---

## Changelog

### Version 1.1 — 2026-10-24
- A: `risk_score` is defined as a RANKING score, not a calibrated probability or flood depth; do not call it a probability in the UI or slides.
- A: grid defined — `row`/`col` contiguous non-negative integers on one grid; row 0 southernmost, col 0 westernmost; `cells.geojson` features carry `cell_id`, `row`, `col` and polygon geometry.
- B: `GET /api/risk` and `GET /api/alert` gain optional `as_of` (ISO 8601 with UTC offset; window `[as_of, as_of + horizon_h]`; default current IST; no offset -> HTTP 400 `{"error": "..."}`).
- B: both responses gain top-level `as_of` and `data_source` (`real`|`synthetic`); the frontend must show a warning banner when `data_source` is `synthetic`.
- C: `ml/predict.py` also exposes `predict_risk_at(..., as_of=None, data_dir=None)` and `get_model_info()`; the backend uses them for `as_of` and `data_source`.
- D: new `data/processed/events.csv` (`event,start,end`, +05:30, start inclusive, end exclusive; owner: data).
- D: rainfall conventions — one row per `(timestamp, lat, lon)`, hour-start timestamps in +05:30, hourly accumulation in mm, must cover all event windows plus 72 h of forecast; `data/SOURCES.md` names the historical and forecast sources.
- F: new checkpoint H+3 — sample files delivered in final formats (including `events.csv`).

### Version 1.0
- Initial contract.
