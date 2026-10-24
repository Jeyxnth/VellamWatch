"""Rainfall handling: as_of window selection, IDW to cells, rain index, risk combination.

Forecast window = [floor_hour(as_of), floor_hour(as_of) + horizon_h). Hourly rows in
rainfall.parquet are read as "precip_mm accumulated in the hour starting at timestamp".
as_of = argument > env ML_AS_OF > current time in IST.
"""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config
from ml.features import _require


class NoRainfallError(ValueError):
    """rainfall.parquet has no data for the requested window."""


def resolve_as_of(as_of=None) -> pd.Timestamp:
    value = as_of if as_of is not None else os.environ.get("ML_AS_OF")
    if not value:
        return pd.Timestamp.now(tz=config.IST)
    ts = pd.Timestamp(value)
    return ts.tz_localize(config.IST) if ts.tzinfo is None else ts.tz_convert(config.IST)


_RAIN_CACHE = {}


def load_rainfall(data_dir) -> pd.DataFrame:
    """rainfall.parquet -> DataFrame[timestamp (IST), lat, lon, precip_mm]; cached by file mtime."""
    path = _require(data_dir, "rainfall.parquet")
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key in _RAIN_CACHE:
        return _RAIN_CACHE[key]
    df = pd.read_parquet(path)
    missing = [c for c in ["timestamp", "lat", "lon", "precip_mm"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing} (contract section D)")
    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_localize(config.IST) if ts.dt.tz is None else ts.dt.tz_convert(config.IST)
    out = pd.DataFrame({
        "timestamp": ts,
        "lat": df["lat"].astype(float),
        "lon": df["lon"].astype(float),
        "precip_mm": df["precip_mm"].astype(float).fillna(0.0).clip(lower=0.0),
    }).sort_values("timestamp").reset_index(drop=True)
    _RAIN_CACHE.clear()
    _WINDOW_CACHE.clear()  # windows are keyed by id(df); never reuse them across files
    _RAIN_CACHE[key] = out
    return out


@dataclass
class RainWindow:
    hours: pd.DatetimeIndex   # T hourly starts
    points: np.ndarray        # (P, 2) lat, lon
    mm: np.ndarray            # (T, P) mm per hour
    coverage: float           # share of expected (hour, point) values that were present


_WINDOW_CACHE = {}


def rain_window(rain_df: pd.DataFrame, as_of, horizon_h: int) -> RainWindow:
    start = resolve_as_of(as_of).floor("h")
    end = start + pd.Timedelta(hours=int(horizon_h))
    key = (id(rain_df), start, int(horizon_h))
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    sub = rain_df[(rain_df.timestamp >= start) & (rain_df.timestamp < end)]
    if sub.empty:
        raise NoRainfallError(
            f"rainfall.parquet has no rows in [{start.isoformat()}, {end.isoformat()}); it covers "
            f"{rain_df.timestamp.min().isoformat()} .. {rain_df.timestamp.max().isoformat()}. "
            "Set ML_AS_OF (ISO time, e.g. 2023-12-03T00:00:00+05:30) to replay a covered period, "
            "or ask the data owner for a fresh forecast.")
    sub = sub.assign(pid=sub.groupby(["lat", "lon"]).ngroup())
    pts = sub.groupby("pid")[["lat", "lon"]].first().to_numpy()
    hours = pd.date_range(start, periods=int(horizon_h), freq="h")
    piv = sub.pivot_table(index="timestamp", columns="pid", values="precip_mm", aggfunc="mean")
    piv = piv.reindex(hours)
    coverage = float(piv.notna().to_numpy().mean())
    if coverage < 0.5:
        warnings.warn(f"rainfall covers only {coverage:.0%} of the requested window; missing hours count as 0 mm")
    win = RainWindow(hours, pts, piv.fillna(0.0).to_numpy(), coverage)
    if len(_WINDOW_CACHE) > 16:
        _WINDOW_CACHE.clear()
    _WINDOW_CACHE[key] = win
    return win


def idw_weights(lat, lon, points, power=config.IDW_POWER, nearest=config.IDW_NEAREST) -> np.ndarray:
    """(N, P) inverse-distance weights (rows sum to 1) from the `nearest` rainfall points.

    Distances use an equirectangular approximation in km, floored at 50 m.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    dy = (lat[:, None] - points[None, :, 0]) * 111.32
    dx = (lon[:, None] - points[None, :, 1]) * 111.32 * np.cos(np.radians(lat.mean()))
    d = np.maximum(np.hypot(dx, dy), 0.05)
    w = 1.0 / d ** power
    p = points.shape[0]
    if p > nearest:
        keep = np.argpartition(d, nearest - 1, axis=1)[:, :nearest]
        mask = np.zeros_like(w, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        w = np.where(mask, w, 0.0)
    return w / w.sum(axis=1, keepdims=True)


def _max_rolling(series: np.ndarray, n: int):
    """Max over rolling sums of length n (clipped to T) and the start index of the max."""
    t = series.shape[1]
    n = min(n, t)
    c = np.concatenate([np.zeros((series.shape[0], 1)), np.cumsum(series, axis=1)], axis=1)
    sums = c[:, n:] - c[:, :-n]
    return sums.max(axis=1), sums.argmax(axis=1), n


@dataclass
class CellRain:
    rain24: np.ndarray        # (N,) max rolling 24 h total, mm
    max3h: np.ndarray         # (N,) max rolling 3 h total, mm
    rain_index: np.ndarray    # (N,) rain24 + INTENSITY_WEIGHT * max3h
    peak_start: np.ndarray    # (N,) index into hours of the wettest 3 h window
    peak_len: int
    window: RainWindow


def cell_rain(rain_df: pd.DataFrame, lat, lon, as_of, horizon_h: int) -> CellRain:
    win = rain_window(rain_df, as_of, horizon_h)
    w = idw_weights(lat, lon, win.points)
    series = w @ win.mm.T                                 # (N, T) mm/h per cell
    rain24, _, _ = _max_rolling(series, 24)
    max3h, peak, plen = _max_rolling(series, config.PEAK_WINDOW_H)
    return CellRain(rain24, max3h, rain24 + config.INTENSITY_WEIGHT * max3h, peak, plen, win)


def combine_risk(u, rain_index, k=config.RAIN_K, r0=config.RAIN_R0):
    """risk = sigmoid(logit(u) + k * (rain_index - r0)); non-decreasing in rain_index.

    u is the susceptibility percentile (0-1). With light rain (< LIGHT_RAIN_MM) risk is
    capped below the "high" band.
    """
    u = np.clip(np.asarray(u, float), *config.U_CLIP)
    z = np.log(u / (1.0 - u)) + k * (np.asarray(rain_index, float) - r0)
    risk = 1.0 / (1.0 + np.exp(-z))
    return np.where(np.asarray(rain_index) < config.LIGHT_RAIN_MM,
                    np.minimum(risk, config.LIGHT_RAIN_CAP), risk)
