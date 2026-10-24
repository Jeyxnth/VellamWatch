"""Flood-hotspot RANKING per ~250 m cell (never flood depth). Contract section C.

    from ml.predict import predict_risk
    predict_risk(min_lat, min_lon, max_lat, max_lon, horizon_h) -> list[dict]

Forecast window: [as_of, as_of + horizon_h). as_of defaults to the current IST time and can be
replayed with env ML_AS_OF (e.g. 2023-12-03T00:00:00+05:30 for Cyclone Michaung), or via
predict_risk_at(..., as_of=...). Rainfall per cell is inverse-distance weighted (4 nearest
points, power 2) from rainfall.parquet.

CLI: python -m ml.predict --bbox MIN_LAT MIN_LON MAX_LAT MAX_LON --horizon 24 [--as-of ISO]
"""
import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ml import config, explain
from ml import rain as rainmod
from ml.features import (FEATURE_COLUMNS, build_static_table, data_source, event_windows, find_data_dir,
                         percentile_rank)


class MissingModelError(FileNotFoundError):
    """No trained model for the selected data source."""


@dataclass
class _State:
    booster: xgb.Booster
    meta: dict
    data_dir: Path
    cell_id: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    X: np.ndarray
    u: np.ndarray          # susceptibility percentile over ALL cells
    medians: np.ndarray
    contribs: np.ndarray   # (N, F) static XGBoost contributions; independent of rainfall, so cached
    k: float
    r0: float


_STATE = {}


def _model_paths(d, data_dir=None):
    adir = config.model_dir(data_source(d))
    meta_path, model_path = adir / "meta.json", adir / "model.json"
    if not meta_path.exists() or not model_path.exists():
        hint = f" --data-dir {d}" if data_dir else ""
        raise MissingModelError(f"No trained {data_source(d)} model in {adir}. "
                                f"Train one with: python -m ml.train{hint}")
    return adir, meta_path, model_path


def get_model_info(data_dir=None) -> dict:
    """What the loaded model was trained on (contract section C). Cheap: reads meta.json only."""
    d = find_data_dir(data_dir)
    _, meta_path, _ = _model_paths(d, data_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "data_source": meta["data_source"],
        "trained_at": meta["trained_at"],
        "data_dir": meta["data_dir"],
        "n_cells": meta["n_cells"],
        "rain_stage": "fitted" if meta["rain"].get("fitted") else "heuristic, not fitted",
        "rain_k": meta["rain"]["k"],
        "rain_r0": meta["rain"]["r0"],
    }


def get_state(data_dir=None) -> _State:
    """Load (and cache) the model and static per-cell table. Reloads if files change."""
    d = find_data_dir(data_dir)
    adir, meta_path, model_path = _model_paths(d, data_dir)
    key = (str(d.resolve()), meta_path.stat().st_mtime_ns, model_path.stat().st_mtime_ns,
           (d / "features.parquet").stat().st_mtime_ns if (d / "features.parquet").exists() else 0)
    if key in _STATE:
        return _STATE[key]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    table = build_static_table(d)
    if len(table) != meta["n_cells"]:
        raise RuntimeError(f"data changed since training ({len(table)} cells now, {meta['n_cells']} at "
                           f"training). Re-run: python -m ml.train")
    if not os.environ.get("ML_AS_OF") and (d / "rainfall.parquet").exists():
        windows, _ = event_windows(d)
        for msg in rainmod.coverage_warnings(rainmod.load_rainfall(d), windows, check_forecast=True):
            warnings.warn(msg)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    X = table[FEATURE_COLUMNS].to_numpy(float)
    m = booster.predict(xgb.DMatrix(X), output_margin=True)
    p = 1.0 / (1.0 + np.exp(-(meta["platt"]["a"] * m + meta["platt"]["b"])))
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    _STATE.clear()
    _STATE[key] = _State(booster, meta, d, table.cell_id.to_numpy(), table.lat.to_numpy(),
                         table.lon.to_numpy(), X, percentile_rank(p),
                         np.asarray(meta["feature_medians"]), contribs, meta["rain"]["k"], meta["rain"]["r0"])
    return _STATE[key]


def predict_risk_at(min_lat, min_lon, max_lat, max_lon, horizon_h, as_of=None, data_dir=None):
    """Same as predict_risk, with optional overrides.

    as_of: ISO 8601 string WITH a UTC offset (e.g. "2023-12-03T00:00:00+05:30") or a tz-aware
    datetime; strings without an offset raise ValueError. Default: env ML_AS_OF, else now (IST).
    """
    min_lat, min_lon, max_lat, max_lon = (float(v) for v in (min_lat, min_lon, max_lat, max_lon))
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("bounding box must satisfy min_lat < max_lat and min_lon < max_lon")
    horizon_h = int(horizon_h)
    if horizon_h < 1:
        raise ValueError("horizon_h must be >= 1")
    as_of_ts = rainmod.resolve_as_of(as_of)   # ISO 8601 WITH offset, else ValueError
    st = get_state(data_dir)
    idx = np.flatnonzero((st.lat >= min_lat) & (st.lat <= max_lat) &
                         (st.lon >= min_lon) & (st.lon <= max_lon))
    if idx.size == 0:
        return []
    rain_df = rainmod.load_rainfall(st.data_dir)
    cr = rainmod.cell_rain(rain_df, st.lat[idx], st.lon[idx], as_of_ts, horizon_h)

    score = np.round(rainmod.combine_risk(st.u[idx], cr.rain_index, st.k, st.r0), 3)
    drivers = explain.build_drivers(st.contribs[idx], st.X[idx], st.medians, cr.rain24, len(cr.window.hours))

    # ISO strings are built once per forecast hour, not once per cell (pandas boxing is slow)
    hours = cr.window.hours
    iso = [h.isoformat() for h in hours] + [(hours[-1] + pd.Timedelta(hours=1)).isoformat()]
    starts = cr.peak_start
    ends = starts + cr.peak_len
    scores = score.tolist()
    out = [{
        "cell_id": cid, "lat": lat, "lon": lon, "risk_score": sc, "risk_band": config.risk_band(sc),
        "peak_window": {"start": iso[a], "end": iso[b]}, "drivers": dr,
    } for cid, lat, lon, sc, a, b, dr in zip(
        st.cell_id[idx].tolist(), st.lat[idx].tolist(), st.lon[idx].tolist(), scores,
        starts.tolist(), ends.tolist(), drivers)]
    out.sort(key=lambda c: (-c["risk_score"], c["cell_id"]))
    return out


def predict_risk(min_lat, min_lon, max_lat, max_lon, horizon_h):
    """Contract signature. as_of comes from env ML_AS_OF, else now (IST)."""
    return predict_risk_at(min_lat, min_lon, max_lat, max_lon, horizon_h)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print flood-hotspot risk cells as JSON")
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--as-of", help="ISO time; default env ML_AS_OF or now (IST)")
    ap.add_argument("--data-dir")
    a = ap.parse_args(argv)
    st = get_state(a.data_dir)
    if st.meta["data_source"] == "synthetic":
        print(f"*** {config.SYNTHETIC_BANNER} ***", file=sys.stderr)
    print(f"# data source: {st.meta['data_source']}; as_of: {rainmod.resolve_as_of(a.as_of).isoformat()}",
          file=sys.stderr)
    cells = predict_risk_at(*a.bbox, a.horizon, as_of=a.as_of, data_dir=a.data_dir)
    print(json.dumps(cells, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    warnings.simplefilter("default")
    sys.exit(main())
