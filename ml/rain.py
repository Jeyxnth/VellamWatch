"""Rainfall handling: as_of window selection, IDW to cells, rain index, risk combination.

Forecast window = [floor_hour(as_of), floor_hour(as_of) + horizon_h). Hourly rows in
rainfall.parquet are read as "precip_mm accumulated in the hour starting at timestamp"
(contract section D). as_of = argument > env ML_AS_OF > current time in IST; an explicit
as_of must be ISO 8601 WITH a UTC offset (contract section B).
"""
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml import budget, config
from ml.features import DataFormatError, _require


class NoRainfallError(ValueError):
    """rainfall.parquet has no data for the requested window."""


def parse_as_of(value, source="as_of") -> pd.Timestamp:
    """ISO 8601 string (or tz-aware datetime) with a UTC offset -> Timestamp in IST."""
    hint = "ISO 8601 with a UTC offset, e.g. 2023-12-03T00:00:00+05:30"
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{source} '{value}' is not a valid timestamp; use {hint}") from e
    if pd.isna(ts):
        raise ValueError(f"{source} '{value}' is not a valid timestamp; use {hint}")
    if ts.tzinfo is None:
        raise ValueError(f"{source} '{value}' has no UTC offset; use {hint}")
    return ts.tz_convert(config.IST)


def resolve_as_of(as_of=None) -> pd.Timestamp:
    """as_of argument > env ML_AS_OF > now (IST). Values without a UTC offset are rejected."""
    if as_of is not None:
        return parse_as_of(as_of, "as_of")
    env = os.environ.get("ML_AS_OF")
    if env:
        return parse_as_of(env, "env ML_AS_OF")
    return pd.Timestamp.now(tz=config.IST)


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
    if ts.dt.tz is None:
        raise DataFormatError(
            f"rainfall.parquet ({path}): timestamps have no UTC offset. Contract section D requires hour-start "
            "timestamps in +05:30. Fix rainfall.parquet.")
    ts = ts.dt.tz_convert(config.IST)  # hour starts are checked in +05:30, the contract timezone
    off_hour = ~((ts.dt.minute == 0) & (ts.dt.second == 0) & (ts.dt.microsecond == 0) & (ts.dt.nanosecond == 0))
    if off_hour.any():
        raise DataFormatError(
            f"rainfall.parquet ({path}): {int(off_hour.sum())} timestamps are not on the hour in +05:30, e.g. "
            f"{ts[off_hour].iloc[0].isoformat()}. Contract section D requires the START of each hour in +05:30. "
            "Fix rainfall.parquet.")
    if df.duplicated(["timestamp", "lat", "lon"]).any():
        warnings.warn("rainfall.parquet has duplicate (timestamp, lat, lon) rows; they are averaged. "
                      "Contract section D wants one row per (timestamp, lat, lon).")
    out = pd.DataFrame({
        "timestamp": ts,
        "lat": df["lat"].astype(float),
        "lon": df["lon"].astype(float),
        "precip_mm": df["precip_mm"].astype(float).fillna(0.0).clip(lower=0.0),
    }).sort_values("timestamp").reset_index(drop=True)
    _RAIN_CACHE.clear()
    _WINDOW_CACHE.clear()  # windows/areas are keyed by id(df); never reuse them across files
    _AREA_CACHE.clear()
    _RAIN_CACHE[key] = out
    return out


def coverage_warnings(rain_df: pd.DataFrame, windows: dict, check_forecast: bool = False) -> list:
    """Human-readable problems with rainfall coverage (warn, never crash).

    windows: event -> (start, end_exclusive). check_forecast adds the 72 h-from-now requirement.
    """
    msgs = []
    present = set(rain_df.timestamp)
    for event, (start, end) in windows.items():
        want = pd.date_range(start.floor("h"), end, freq="h", inclusive="left")
        have = sum(1 for t in want if t in present)
        if have < len(want):
            msgs.append(f"rainfall.parquet covers only {have} of {len(want)} hours of event '{event}' "
                        f"({start.isoformat()} to {end.isoformat()}). Contract section D: it must cover every "
                        "event window. Ask the data owner to extend rainfall.parquet.")
    if check_forecast:
        need = pd.Timestamp.now(tz=config.IST).floor("h") + pd.Timedelta(hours=72)
        if rain_df.timestamp.max() < need:
            msgs.append(f"rainfall.parquet ends at {rain_df.timestamp.max().isoformat()}, less than 72 h of "
                        "forecast from now. Contract section D requires at least 72 h; refresh the forecast.")
    return msgs


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
    peak_start: np.ndarray    # (N,) index into hours of the wettest 3 h window
    peak_len: int
    window: RainWindow


def cell_rain(rain_df: pd.DataFrame, lat, lon, as_of, horizon_h: int) -> CellRain:
    """Per-cell forecast statistics (IDW from the rainfall points) for [as_of, as_of + horizon_h)."""
    win = rain_window(rain_df, as_of, horizon_h)
    w = idw_weights(lat, lon, win.points)
    series = w @ win.mm.T                                 # (N, T) mm/h per cell
    rain24, _, _ = _max_rolling(series, 24)
    max3h, peak, plen = _max_rolling(series, config.PEAK_WINDOW_H)
    return CellRain(rain24, max3h, peak, plen, win)


@dataclass
class AreaRain:
    """Study-area rainfall summary: sets the high / medium budgets for the whole grid."""
    rain24: float     # mean over ALL study-area cells of the per-cell max rolling 24 h total, mm
    max3h: float      # mean over ALL cells of the per-cell max rolling 3 h total, mm
    severity: float
    f_high: float
    f_med: float
    coverage: float


_AREA_CACHE = {}


def area_rain(rain_df: pd.DataFrame, lat_all, lon_all, as_of, horizon_h: int) -> AreaRain:
    """Rain severity and budgets from the FULL study grid (never from the requested bbox), so a
    street gets the same band in any viewport. Cached per (file, hour, horizon)."""
    start = resolve_as_of(as_of).floor("h")
    key = (id(rain_df), id(lat_all), start, int(horizon_h))
    if key in _AREA_CACHE:
        return _AREA_CACHE[key]
    cr = cell_rain(rain_df, lat_all, lon_all, start, horizon_h)
    r24, r3 = float(cr.rain24.mean()), float(cr.max3h.mean())
    sev = budget.severity(r24, r3)
    f_high, f_med = budget.budgets(sev)
    out = AreaRain(r24, r3, sev, f_high, f_med, cr.window.coverage)
    if len(_AREA_CACHE) > 16:
        _AREA_CACHE.clear()
    _AREA_CACHE[key] = out
    return out
