"""Data loading, per-cell static features, labels and spatial cross-validation.

Reads the contract section D files with pandas + plain json (no geopandas).
"""
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config

RAW_FEATURES = [
    "elevation_m", "slope_deg", "sink_depth_m", "dist_drain_m",
    "dist_water_m", "road_density", "built_up_frac",
]
# Model inputs (distances are logged; trees are invariant to this, it only helps the LR baseline/readability).
FEATURE_COLUMNS = [
    "elevation_m", "elev_pct", "local_relief_m", "sink_depth_m", "slope_deg",
    "log_dist_drain", "log_dist_water", "road_density", "built_up_frac",
]
# Physically justified monotone constraints (+1 = higher value -> higher risk).
MONOTONE = {"elevation_m": -1, "elev_pct": -1, "local_relief_m": -1, "sink_depth_m": 1}


class MissingDataError(FileNotFoundError):
    """A required contract data file is missing or malformed."""


# ---- locating data ----------------------------------------------------------
def find_data_dir(data_dir=None) -> Path:
    """--data-dir > env ML_DATA_DIR > data/processed (if features.parquet exists) > ml/synthetic_out."""
    if data_dir:
        return Path(data_dir)
    env = os.environ.get("ML_DATA_DIR")
    if env:
        return Path(env)
    if (config.REAL_DATA_DIR / "features.parquet").exists():
        return config.REAL_DATA_DIR
    return config.SYNTHETIC_DIR


def is_synthetic(data_dir) -> bool:
    return (Path(data_dir) / "SYNTHETIC.flag").exists()


def data_source(data_dir) -> str:
    return "synthetic" if is_synthetic(data_dir) else "real"


def _require(data_dir, name: str) -> Path:
    path = Path(data_dir) / name
    if not path.exists():
        if is_synthetic(data_dir) or Path(data_dir) == config.SYNTHETIC_DIR:
            hint = "Generate synthetic data with: python -m ml.synthetic.make_synthetic"
        else:
            hint = f"Ask the data owner to deliver data/processed/{name} (contract section D), or pass --data-dir."
        raise MissingDataError(f"Missing data file '{name}' (looked in {Path(data_dir).resolve()}). {hint}")
    return path


# ---- loaders -----------------------------------------------------------------
_CELL_RE = re.compile(r"^r(\d+)_c(\d+)$")


def parse_cell_id(cell_id: str):
    m = _CELL_RE.match(str(cell_id))
    if not m:
        raise ValueError(f"cell_id '{cell_id}' does not match r{{row}}_c{{col}}")
    return int(m.group(1)), int(m.group(2))


def _polygon_centroid(ring: np.ndarray):
    """Area centroid of a polygon ring (lon, lat); falls back to the vertex mean."""
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    x, y = ring[:, 0], ring[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-18:
        return x.mean(), y.mean()
    return ((x + x1) * cross).sum() / (6 * area), ((y + y1) * cross).sum() / (6 * area)


def load_cells(data_dir) -> pd.DataFrame:
    """cells.geojson -> DataFrame[cell_id, lat, lon, row, col] (centroids computed here)."""
    path = _require(data_dir, "cells.geojson")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    recs = []
    for feat in gj.get("features", []):
        cid = feat["properties"]["cell_id"]
        geom = feat["geometry"]
        poly = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        lon, lat = _polygon_centroid(np.asarray(poly, dtype=float)[:, :2])
        row, col = parse_cell_id(cid)
        recs.append((cid, lat, lon, row, col))
    df = pd.DataFrame(recs, columns=["cell_id", "lat", "lon", "row", "col"])
    if df.empty:
        raise MissingDataError(f"{path} contains no features")
    if df.cell_id.duplicated().any():
        raise ValueError(f"{path} has duplicate cell_ids")
    return df


def load_features(data_dir) -> pd.DataFrame:
    path = _require(data_dir, "features.parquet")
    df = pd.read_parquet(path)
    missing = [c for c in ["cell_id"] + RAW_FEATURES if c not in df.columns]
    if missing:
        raise MissingDataError(f"{path} is missing columns {missing} (contract section D)")
    if df.cell_id.duplicated().any():
        raise ValueError(f"{path} has duplicate cell_ids")
    return df


def load_hotspots(data_dir) -> pd.DataFrame:
    path = _require(data_dir, "hotspots.csv")
    df = pd.read_csv(path, dtype={"cell_id": str, "event": str})
    missing = [c for c in ["cell_id", "event"] if c not in df.columns]
    if missing:
        raise MissingDataError(f"{path} is missing columns {missing} (contract section D)")
    df["event"] = df["event"].str.strip()
    df["cell_id"] = df["cell_id"].str.strip()
    return df


def event_windows(data_dir):
    """event -> (start, end_exclusive) tz-aware IST, plus a provenance string.

    Uses data/processed/events.csv (event,start,end) if present, else config.EVENTS_TODO.
    Date-only `end` is inclusive (one day is added).
    """
    def to_window(start, end):
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        s = s.tz_localize(config.IST) if s.tzinfo is None else s.tz_convert(config.IST)
        e_exact = len(str(end).strip()) > 10
        e = e.tz_localize(config.IST) if e.tzinfo is None else e.tz_convert(config.IST)
        return s, (e if e_exact else e + pd.Timedelta(days=1))

    path = Path(data_dir) / "events.csv"
    if path.exists():
        ev = pd.read_csv(path, dtype=str)
        return {r.event.strip(): to_window(r.start, r.end) for r in ev.itertuples()}, "events.csv (data owner)"
    return ({k: to_window(*v) for k, v in config.EVENTS_TODO.items()},
            "config.EVENTS_TODO (UNVERIFIED dates; see CONTRACT_PROPOSALS.md)")


# ---- static feature table ----------------------------------------------------
def box_nanmean(arr: np.ndarray, radius: int) -> np.ndarray:
    """NaN-aware mean over a (2r+1)x(2r+1) window (includes the centre cell)."""
    valid = ~np.isnan(arr)
    a = np.where(valid, arr, 0.0)
    pad_a = np.pad(a, radius)
    pad_v = np.pad(valid.astype(float), radius)
    rows, cols = arr.shape
    s = np.zeros_like(a)
    n = np.zeros_like(a)
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            s += pad_a[dr:dr + rows, dc:dc + cols]
            n += pad_v[dr:dr + rows, dc:dc + cols]
    return s / np.maximum(n, 1.0)


def build_static_table(data_dir) -> pd.DataFrame:
    """One row per cell: cell_id, lat, lon, row, col, raw + derived features (sorted by cell_id)."""
    cells = load_cells(data_dir)
    feats = load_features(data_dir)
    df = cells.merge(feats[["cell_id"] + RAW_FEATURES], on="cell_id", how="inner")
    if df.empty:
        raise ValueError("cells.geojson and features.parquet share no cell_id")
    dropped = len(cells) - len(df)
    if dropped:
        warnings.warn(f"{dropped} cells in cells.geojson have no row in features.parquet and are ignored")
    n_nan = int(df[RAW_FEATURES].isna().sum().sum())
    if n_nan:
        warnings.warn(f"{n_nan} NaN feature values imputed with column medians (data owner should fix)")
        df[RAW_FEATURES] = df[RAW_FEATURES].fillna(df[RAW_FEATURES].median())
    df = df.sort_values("cell_id").reset_index(drop=True)

    grid = np.full((int(df.row.max()) + 1, int(df.col.max()) + 1), np.nan)
    grid[df.row.to_numpy(), df.col.to_numpy()] = df.elevation_m.to_numpy()
    mean_nb = box_nanmean(grid, config.LOCAL_RELIEF_RADIUS_CELLS)
    df["local_relief_m"] = df.elevation_m - mean_nb[df.row.to_numpy(), df.col.to_numpy()]
    df["elev_pct"] = df.elevation_m.rank(pct=True)
    df["log_dist_drain"] = np.log1p(df.dist_drain_m.clip(lower=0))
    df["log_dist_water"] = np.log1p(df.dist_water_m.clip(lower=0))
    return df


def make_labels(table: pd.DataFrame, hotspots: pd.DataFrame, events=None,
                neighbour_weight: float = 0.0):
    """Return (y_true, y_train, sample_weight).

    y_true: 1 for hotspot cells of the chosen events (events=None means all). Used for evaluation.
    y_train: same, plus the 8-neighbours of hotspots as soft positives with `neighbour_weight`.
    """
    sel = hotspots if events is None else hotspots[hotspots.event.isin(list(events))]
    pos = set(sel.cell_id)
    unknown = pos - set(table.cell_id)
    if unknown:
        raise ValueError(f"hotspots.csv has {len(unknown)} cell_ids not in cells/features, e.g. {sorted(unknown)[:3]}")
    y_true = table.cell_id.isin(pos).to_numpy().astype(int)
    y_train = y_true.copy()
    w = np.ones(len(table))
    if neighbour_weight > 0 and y_true.sum() > 0:
        rows, cols = table.row.to_numpy(), table.col.to_numpy()
        pos_grid = np.zeros((rows.max() + 1, cols.max() + 1), dtype=bool)
        pos_grid[rows, cols] = y_true == 1
        near = dilate(pos_grid, 1)[rows, cols] & (y_true == 0)
        y_train[near] = 1
        w[near] = neighbour_weight
    return y_true, y_train, w


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square of half-width `radius` (numpy only)."""
    pad = np.pad(mask, radius)
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = mask.shape
    for dr in range(2 * radius + 1):
        for dc in range(2 * radius + 1):
            out |= pad[dr:dr + rows, dc:dc + cols]
    return out


def percentile_rank(p: np.ndarray) -> np.ndarray:
    """Percentile of each value among all values, strictly inside (0, 1)."""
    return (pd.Series(p).rank(method="average").to_numpy() - 0.5) / len(p)


# ---- spatial block cross-validation -------------------------------------------
def spatial_block_folds(rows, cols, block=None, n_splits=None, buffer=None, seed=config.SEED):
    """List of (train_idx, test_idx). Geographic blocks are assigned to folds at random;
    training cells within `buffer` cells of any test cell are dropped (no neighbour leakage)."""
    block = block or config.CV_BLOCK_CELLS
    n_splits = n_splits or config.CV_SPLITS
    buffer = config.CV_BUFFER_CELLS if buffer is None else buffer
    rows, cols = np.asarray(rows), np.asarray(cols)
    key = (rows // block) * 100000 + (cols // block)
    uniq = np.unique(key)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of_block = {int(uniq[i]): rank % n_splits for rank, i in enumerate(order)}
    fold = np.array([fold_of_block[int(k)] for k in key])
    shape = (rows.max() + 1, cols.max() + 1)
    folds = []
    for f in range(n_splits):
        test = fold == f
        if not test.any():
            continue
        grid = np.zeros(shape, dtype=bool)
        grid[rows[test], cols[test]] = True
        near_test = dilate(grid, buffer)[rows, cols]
        train = ~test & ~near_test
        folds.append((np.flatnonzero(train), np.flatnonzero(test)))
    return folds
