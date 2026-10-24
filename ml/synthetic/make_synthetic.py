"""Write SYNTHETIC data in the exact contract (v1.1) section D formats, including events.csv (pipeline tests only).

*** SYNTHETIC DATA - PIPELINE TEST ONLY - NOT RESULTS ***

Run: python -m ml.synthetic.make_synthetic [--out-dir DIR] [--seed 42] [--full]
Row 0 is the southern-most row, col 0 the western-most column.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from ml import config
from ml.features import box_nanmean, event_windows

SUB_BBOX = (12.95, 80.15, 13.15, 80.30)     # min_lat, min_lon, max_lat, max_lon (inside the contract bbox)
FULL_BBOX = (12.85, 80.05, 13.25, 80.35)
CELL_M = 250.0
EVENT_CELL_FRACTION = {"2015": 0.03, "2023": 0.045}
EVENT_PEAK_MM_H = {"2015": 14.0, "2023": 22.0}


def _smooth_noise(rng, shape, n_waves=8):
    yy, xx = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing="ij")
    field = np.zeros(shape)
    for _ in range(n_waves):
        fy, fx = rng.uniform(0.5, 4.0, 2)
        field += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * (fy * yy + fx * xx) + rng.uniform(0, 6.28))
    return (field - field.mean()) / field.std()


def _z(a):
    return (a - a.mean()) / a.std()


def _terrain(rng, rows, cols):
    yy, xx = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
    elev = 2 + 25 * (1 - xx) ** 1.5 + 1.5 * _smooth_noise(rng, (rows, cols))
    for _ in range(4):  # low-lying basins
        cy, cx = rng.uniform(0.15, 0.85, 2)
        elev -= rng.uniform(2, 4) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(0.04, 0.08) ** 2))
    river_y = 0.5 + 0.1 * np.sin(2 * np.pi * xx)
    elev -= 3 * np.exp(-((yy - river_y) ** 2) / (2 * 0.015 ** 2))
    elev = np.clip(elev, 0.5, None)

    water = np.abs(yy - river_y) < 0.012
    water |= (yy - 0.25) ** 2 + (xx - 0.35) ** 2 < 0.03 ** 2
    drains = np.zeros((rows, cols), dtype=bool)
    for r in rng.integers(0, rows, 4):
        drains[r, :] = True
    for c in rng.integers(0, cols, 4):
        drains[:, c] = True
    gy, gx = np.gradient(elev, CELL_M)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    sink = np.clip(box_nanmean(elev, 4) - elev, 0, 6)
    built = np.clip(0.25 + 0.5 * np.exp(-((xx - 0.7) ** 2 + (yy - 0.5) ** 2) / (2 * 0.3 ** 2))
                    + 0.1 * _smooth_noise(rng, (rows, cols)), 0, 1)
    roads = np.clip(4 + 14 * built + 2 * _smooth_noise(rng, (rows, cols)), 0, None)
    return dict(
        elevation_m=elev, slope_deg=slope, sink_depth_m=sink,
        dist_drain_m=distance_transform_edt(~drains) * CELL_M,
        dist_water_m=distance_transform_edt(~water) * CELL_M,
        road_density=roads, built_up_frac=built)


def _rain_block(rng, start, hours, peak_mm_h, points, peak_hour):
    ts = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    rows = []
    for plat, plon, mult in points:
        burst = peak_mm_h * mult * np.exp(-(((t - peak_hour) / 5.0) ** 2))
        rate = (0.2 * rng.lognormal(0, 0.5, hours) + burst * rng.lognormal(0, 0.25, hours)).round(1)
        rows.append(pd.DataFrame({"timestamp": ts, "lat": plat, "lon": plon, "precip_mm": rate}))
    return pd.concat(rows, ignore_index=True)


def _calibrate_offset(score, target_count):
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(score + mid)))).sum() > target_count:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(out_dir, seed=config.SEED, bbox=SUB_BBOX, now=None) -> dict:
    """Write the four contract files plus SYNTHETIC.flag into out_dir. Returns summary counts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    min_lat, min_lon, max_lat, max_lon = bbox
    dlat = CELL_M / 111320.0
    dlon = CELL_M / (111320.0 * math.cos(math.radians((min_lat + max_lat) / 2)))
    rows = int(math.ceil((max_lat - min_lat) / dlat))
    cols = int(math.ceil((max_lon - min_lon) / dlon))

    t = _terrain(rng, rows, cols)
    feats, feats_cells = [], []
    for r in range(rows):
        for c in range(cols):
            lat0, lon0 = min_lat + r * dlat, min_lon + c * dlon
            cid = f"r{r}_c{c}"
            ring = [[lon0, lat0], [lon0 + dlon, lat0], [lon0 + dlon, lat0 + dlat],
                    [lon0, lat0 + dlat], [lon0, lat0]]
            feats_cells.append({"type": "Feature", "properties": {"cell_id": cid, "row": r, "col": c},
                                "geometry": {"type": "Polygon", "coordinates": [ring]}})
            rec = {"cell_id": cid, "lat": lat0 + dlat / 2, "lon": lon0 + dlon / 2}
            rec.update({k: float(v[r, c]) for k, v in t.items()})
            feats.append(rec)
    (out / "cells.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats_cells}), encoding="utf-8")
    df = pd.DataFrame(feats)
    df.to_parquet(out / "features.parquet", index=False)

    # hotspots: probability driven by low elevation, sinks, water proximity, built-up + noise
    base = (-1.0 * _z(df.elevation_m) + 0.8 * _z(df.sink_depth_m) - 0.4 * _z(df.dist_water_m)
            + 0.3 * _z(df.built_up_frac) + rng.normal(0, 0.6, len(df)))
    hot = []
    for event, frac in EVENT_CELL_FRACTION.items():
        score = (base + rng.normal(0, 0.8, len(df))).to_numpy()
        p = 1 / (1 + np.exp(-(score + _calibrate_offset(score, frac * len(df)))))
        for cid in df.cell_id[rng.random(len(df)) < p]:
            hot.append({"cell_id": cid, "event": event,
                        "source": f"synthetic generator seed={seed}", "notes": "SYNTHETIC - not a real location"})
    pd.DataFrame(hot, columns=["cell_id", "event", "source", "notes"]).to_csv(out / "hotspots.csv", index=False)

    # rainfall: 3x3 points; burst during each event window + a forecast block starting now
    pts = [(min_lat + (max_lat - min_lat) * i / 2, min_lon + (max_lon - min_lon) * j / 2, 0.85 + 0.15 * j)
           for i in range(3) for j in range(3)]
    ev_rows = [{"event": e, "start": a, "end": b} for e, (a, b) in config.EVENTS_TODO.items()]
    pd.DataFrame(ev_rows).to_csv(out / "events.csv", index=False)  # SYNTHETIC dates = the config placeholders
    windows, _ = event_windows(out)
    blocks = []
    for event, (start, end) in windows.items():
        hours = int((end - start) / np.timedelta64(1, "h"))
        blocks.append(_rain_block(rng, start, hours, EVENT_PEAK_MM_H.get(event, 15.0), pts, 30))
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=config.IST)
    now = (now.tz_localize(config.IST) if now.tzinfo is None else now.tz_convert(config.IST)).floor("h")
    blocks.append(_rain_block(rng, now, 72, 12.0, pts, 10))
    rain = pd.concat(blocks, ignore_index=True)
    rain.to_parquet(out / "rainfall.parquet", index=False)

    (out / "SYNTHETIC.flag").write_text(config.SYNTHETIC_BANNER + "\n", encoding="utf-8")
    return {"rows": rows, "cols": cols, "cells": len(df),
            "hotspot_rows": len(hot), "rain_rows": len(rain)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC contract-format data (pipeline test only)")
    ap.add_argument("--out-dir", default=str(config.SYNTHETIC_DIR))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--full", action="store_true", help="use the full contract bounding box (~23k cells)")
    a = ap.parse_args(argv)
    print(f"*** {config.SYNTHETIC_BANNER} ***")
    info = generate(a.out_dir, a.seed, FULL_BBOX if a.full else SUB_BBOX)
    print(f"wrote {a.out_dir}: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
