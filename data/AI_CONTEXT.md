# AI_CONTEXT — Data pipeline

**Owner:** [MEMBER 2 NAME]

You are helping me build one lane of a hackathon project. Read this fully before answering.

## Project summary
VellamWatch — AI-powered street-level flood risk and "what to do" alerts for Chennai. Tagline: "Know your street before the water does." We combine an hourly rainfall forecast (Open-Meteo) with terrain features (elevation, slope, low-lying sinks, distance to drains/water bodies, road density from SRTM/Copernicus + OpenStreetMap) and past Chennai flood hotspots (2015, 2023) to rank flood risk per ~250 m grid cell for the next 24–72 hours. An XGBoost risk engine produces a score and band. An LLM writes short Tamil and English action messages; text-to-speech reads them. Residents upload waterlogging photos that a pretrained vision model verifies. We claim hotspot RANKING, never exact flood depth. 24-hour sprint, 5-person team.

## My lane and folder
I am the **Data pipeline** owner. My folder is `data/` (branch `data`), with scripts in `data/scripts/`, raw downloads in `data/raw/`, outputs in `data/processed/`. Other lanes (ml/, backend/, frontend/, voice_vision/) are not mine.

## Contract
`docs/contract.md`, section D (and A for conventions). Follow it exactly. Never change it; ask me to request a change from the lead. Key points: WGS84 (lat, lon); `cell_id` = `r{row}_c{col}`; bounding box proposal lat 12.85–13.25, lon 80.05–80.35; ~250 m grid.

## Rules for the AI
- Only create or edit files inside `data/`. Never edit `docs/contract.md`; tell me what change to request from the lead.
- Prefer simple, readable code that works over clever code.
- Small commits; suggest a message after each piece.
- No secrets or API keys in code. Use `.env` and `.env.example`.
- Write a runnable example or test for every function.
- If blocked on another module, create a mock that matches the contract.
- State assumptions explicitly (e.g. CRS used for metric calculations).
- Keep answers short.

## Tech stack
Python 3.11, requests, pandas, pyarrow, geopandas, shapely, pyproj, rasterio, numpy, osmnx (or Overpass API), richdem or scipy for slope/sinks, pytest. Open-Meteo (forecast + archive APIs, no key), SRTM or Copernicus DEM, OpenStreetMap.

## Exact deliverables (all in `data/processed/` unless noted)
1. `cells.geojson` — ~250 m grid polygons, property `cell_id` = `r{row}_c{col}`.
2. `features.parquet` — `cell_id, lat, lon, elevation_m, slope_deg, sink_depth_m, dist_drain_m, dist_water_m, road_density, built_up_frac`.
3. `hotspots.csv` — `cell_id, event, source, notes`, event "2015" or "2023", from documented waterlogging locations (geocoded and mapped to cells).
4. `rainfall.parquet` — `timestamp, lat, lon, precip_mm` hourly, forecast + historical (include the 2015 and 2023 event windows).
5. `data/SOURCES.md` — every source, URL, licence, date accessed.
6. Re-runnable scripts in `data/scripts/`.
7. ~50-row samples of each file by H+3; full files by H+8.

## Acceptance checks to help me pass
- Exact column names/types as in contract D; timestamps ISO 8601 +05:30.
- All cell_ids in features/hotspots exist in cells.geojson; no duplicates; no nulls.
- Metric computations (distance, slope, 250 m grid) done in a projected CRS (e.g. EPSG:32644), output in WGS84.
- Every hotspot row has a source; both events present.
- Scripts run from a clean clone with one command each.

## Starter prompts
1. "Write `data/scripts/make_grid.py` that builds a 250 m grid over lat 12.85–13.25, lon 80.05–80.35 in EPSG:32644 and saves cells.geojson with cell_id r{row}_c{col} in WGS84."
2. "Write `data/scripts/fetch_rainfall.py` that pulls hourly precipitation from Open-Meteo forecast and archive APIs for a coarse grid of points and writes rainfall.parquet with timestamp (+05:30), lat, lon, precip_mm."
3. "Write `data/scripts/terrain_features.py` computing elevation, slope, and sink depth (fill-depressions minus DEM) per cell from a DEM raster."
4. "Write `data/scripts/osm_features.py` using OSMnx to compute dist_drain_m, dist_water_m, road_density (km/km²) and built_up_frac per cell."
5. "Help me build hotspots.csv: given a CSV of place names with lat/lon and event year, map points to cell_ids and require a source column."
6. "Write `data/scripts/make_samples.py` that takes ~50 representative rows (including some hotspots) from each processed file and writes them to `data/samples/`-style files I can share, plus a QA script that checks everything against contract section D."
