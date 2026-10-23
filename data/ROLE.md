# ROLE — Data pipeline

**Owner:** [MEMBER 2 NAME]

## Mission
Turn raw public data (rainfall, elevation, OpenStreetMap, documented past floods) into the four clean files the ML model needs, on a fixed ~250 m grid over Chennai. Everyone is blocked on your samples early, so ship small samples fast, then the full files.

## What you own
- Everything in `data/`, including `data/scripts/`, `data/raw/` and `data/processed/`.
- Open-Meteo hourly rainfall for Chennai (forecast + historical).
- SRTM or Copernicus elevation (DEM).
- OpenStreetMap roads, waterways and drains.
- The 250 m grid: `cells.geojson` (source of truth for cell ids).
- Terrain features per cell: elevation, slope, sink depth, distance to drain/water, road density, built-up fraction.
- `hotspots.csv` compiled from documented 2015 and 2023 waterlogging/flood locations, each with a source.
- `data/SOURCES.md` documenting every source and licence.

## Deliverables
- [ ] H+1: Confirm bounding box (proposal: lat 12.85–13.25, lon 80.05–80.35) with the lead
- [ ] H+2: Grid script producing `cells.geojson` with `cell_id` as `r{row}_c{col}`
- [ ] H+3: ~50-row samples of `cells.geojson`, `features.parquet`, `hotspots.csv`, `rainfall.parquet` delivered to the team (see HANDOFF for how, since processed files are git-ignored)
- [ ] H+5: Open-Meteo hourly rainfall downloader (forecast + historical incl. the 2015 and 2023 event windows)
- [ ] H+6: DEM download and terrain features: elevation, slope, sink depth
- [ ] H+7: OSM extraction: dist_drain_m, dist_water_m, road_density, built_up_frac
- [ ] H+8: Full `features.parquet`, `hotspots.csv`, `rainfall.parquet`, `cells.geojson`; `data/SOURCES.md` complete
- [ ] H+10: Data QA report: row counts, null checks, one map screenshot of hotspots vs grid
- [ ] H+14: Scheduled/one-command refresh of the rainfall forecast
- [ ] H+20: Feature freeze; support the team with fixes only

## Definition of done
- All four files exist in `data/processed/` with exactly the contract D column names and types.
- Every `cell_id` in features/hotspots exists in `cells.geojson`; no duplicates; no NaN in feature columns (impute and document).
- Every hotspot row has a `source` and the 2015 and 2023 events are both present.
- Each script is re-runnable from a clean clone (`python data/scripts/<name>.py`), and `SOURCES.md` lists source, URL, licence and date accessed.

## Out of scope (do NOT touch)
- Model training or labels logic (ml/).
- API code (backend/), UI (frontend/), alerts/TTS/vision (voice_vision/).
- `docs/contract.md` — request changes from the lead.

## Git branch and folder
- Branch: `data`
- Folder: `data/`

## Cut-first list (drop in this order if time runs short)
1. built_up_frac (fill from road_density proxy or drop the column via contract update).
2. Separate drain layer (use waterways only for dist_drain_m, documented).
3. Forecast refresh automation (fetch manually before the demo).
4. 72 h forecast window (24 h only).
5. 2015 hotspots beyond the 30–50 best-documented locations.
Never cut: samples at H+3, cells.geojson, hotspots with sources.
