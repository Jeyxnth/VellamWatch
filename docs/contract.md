# VellamWatch — Interface Contract

**Project:** VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai.
**Tagline:** "Know your street before the water does."

This file is the single source of truth. Everyone builds to it. Owned by the lead (ml/).

---

## A. Conventions

- **Coordinates:** WGS84, always ordered `(lat, lon)`.
- **Timestamps:** ISO 8601 with the IST offset, e.g. `2025-11-30T18:00:00+05:30`.
- **Language codes:** `"ta"` (Tamil) and `"en"` (English).
- **risk_score:** float in the range 0–1.
- **risk_band:** string, one of
  - `"low"` — risk_score < 0.33
  - `"medium"` — 0.33 <= risk_score < 0.66
  - `"high"` — risk_score >= 0.66
  - Thresholds may be tuned only by the ML lead, via a contract update.
- **Bounding box (proposal):** lat 12.85–13.25, lon 80.05–80.35. The data owner confirms it.
- **Grid:** the study area is divided into a fixed ~250 m grid.
- **cell_id:** `"r{row}_c{col}"` (e.g. `r12_c34`).
- **Source of truth for cells:** `data/processed/cells.geojson` defines all cell ids and geometry.
- **We claim hotspot RANKING, never exact flood depth.**

---

## B. REST API (served by `backend/`, base path `/api`)

### GET /api/health
Response:
```json
{"status": "ok"}
```

### GET /api/risk?lat=&lon=&radius_m=&horizon_h=
Response:
```json
{
  "generated_at": "ISO",
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

### GET /api/alert?lat=&lon=&lang=ta|en
Response:
```json
{
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
  Each dict matches a `cells` object in `GET /api/risk`.
- `voice_vision/alerts.py`
  `generate_alert(cell: dict) -> {"message_en", "message_ta", "actions"}`
- `voice_vision/tts.py`
  `synthesize(text: str, lang: str) -> str`
  Returns a URL or file path under a served audio folder.
- `voice_vision/vision.py`
  `verify_photo(image_bytes: bytes, lat: float, lon: float, timestamp: str) -> {"status", "confidence", "reason"}`

---

## D. Data files (produced by `data/`, consumed by `ml/`; all in `data/processed/`)

- `cells.geojson` — grid polygons with `cell_id`.
- `features.parquet` — `cell_id, lat, lon, elevation_m, slope_deg, sink_depth_m, dist_drain_m, dist_water_m, road_density, built_up_frac`.
- `hotspots.csv` — `cell_id, event, source, notes` (`event` e.g. `"2015"` or `"2023"`).
- `rainfall.parquet` — `timestamp, lat, lon, precip_mm` (hourly forecast and historical).

A small sample of each (about 50 rows) must be delivered within the first 3 hours so others can start.

---

## E. Change rule

Nobody edits this file directly. Request changes from the lead. Everyone builds to this contract; when blocked by another module, mock it.

---

## F. Integration checkpoints

| Hour | Checkpoint |
|------|------------|
| H+2  | Contract frozen |
| H+10 | First end-to-end run |
| H+14 | Real model wired in |
| H+20 | Feature freeze |
| H+23 | Demo rehearsal done |
