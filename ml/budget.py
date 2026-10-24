"""Stage 2: budgeted ranking. Pure functions, no I/O.

Rain severity decides HOW MUCH of the study area is flagged (a budget of high / medium cells);
the stage-1 susceptibility percentile `u` decides WHICH cells. All numbers live in config.py and
are policy targets, NOT fitted. Output is a hotspot ranking, never flood depth.

    severity = max(sev24(rain24), sev3(max3h))               piecewise-linear, monotone
    f_high   = min(HIGH_CAP, high_budget(severity))          share of ALL study-area cells that are high
    f_med    = min(MED_CAP, max(f_high, med_budget(severity)))   share that are medium-or-high
    high   : u >= 1 - f_high            (top f_high share by susceptibility)
    medium : 1 - f_med <= u < 1 - f_high
    low    : the rest

risk_score is continuous in [0, 1] and consistent with the contract bands (BAND_MEDIUM, BAND_HIGH),
increasing with u inside each band and continuous across band edges:
    low    : BAND_MEDIUM * u / (1 - f_med)                       in [0, BAND_MEDIUM)
    medium : BAND_MEDIUM + (BAND_HIGH-BAND_MEDIUM) * (u-m) / (h-m)   in [BAND_MEDIUM, BAND_HIGH)   m=1-f_med, h=1-f_high
    high   : BAND_HIGH + (1-BAND_HIGH) * (u-h) / (1-h)           in [BAND_HIGH, 1]
For a fixed cell, raising the rainfall lowers m and h, which can only raise the score; for fixed
rainfall the score is increasing in u, so the cell order never changes with rainfall.
u must be the percentile over the model's FULL study grid so a street keeps its rank in any viewport.
"""
import numpy as np

from ml import config

BANDS = ("low", "medium", "high")
_EPS = 0.001


def _interp(x, knots):
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.interp(x, xs, ys))  # constant outside the knots


def severity(rain24_mm: float, max3h_mm: float) -> float:
    """Rain severity in [0, 1] from the 24 h total and the wettest-3 h total (mm)."""
    return max(_interp(rain24_mm, config.SEVERITY_24H_KNOTS), _interp(max3h_mm, config.SEVERITY_3H_KNOTS))


def _budget(sev: float, table) -> float:
    # tables are given by 24 h mm; convert their x-axis to severity with the 24 h mapping
    xs = [_interp(mm, config.SEVERITY_24H_KNOTS) for mm, _ in table]
    return float(np.interp(sev, xs, [f for _, f in table]))


def budgets(sev: float):
    """(f_high, f_med): shares of ALL study-area cells that are high / medium-or-high."""
    f_high = min(config.HIGH_CAP, _budget(sev, config.HIGH_BUDGET_24H))
    f_med = min(config.MED_CAP, max(f_high, _budget(sev, config.MED_BUDGET_24H)))
    return f_high, f_med


def risk_scores(u, f_high: float, f_med: float):
    """(score rounded to 3 decimals, band codes 0=low 1=medium 2=high) for susceptibility percentiles u."""
    u = np.asarray(u, float)
    lo_t, hi_t = config.BAND_MEDIUM, config.BAND_HIGH
    h, m = 1.0 - f_high, 1.0 - f_med
    high = u >= h if f_high > 0 else np.zeros(u.shape, bool)
    med = (u >= m) & ~high if f_med > 0 else np.zeros(u.shape, bool)
    low = ~high & ~med
    score = np.zeros(u.shape)
    if high.any():
        score[high] = hi_t + (1 - hi_t) * (u[high] - h) / (1 - h)
    if med.any():
        score[med] = lo_t + (hi_t - lo_t) * (u[med] - m) / (h - m)
    if low.any():
        score[low] = lo_t * u[low] / m if m > 0 else 0.0
    score = np.round(score, 3)
    score[low] = np.minimum(score[low], lo_t - _EPS)
    score[med] = np.clip(score[med], lo_t, hi_t - _EPS)
    score[high] = np.maximum(score[high], hi_t)
    band = np.where(high, 2, np.where(med, 1, 0))
    return np.clip(score, 0.0, 1.0), band


def band_names(codes) -> list:
    return [BANDS[c] for c in np.asarray(codes).tolist()]


def shares(u, f_high: float, f_med: float):
    """(low, medium, high) shares of the cells in u."""
    _, band = risk_scores(u, f_high, f_med)
    return tuple(float((band == c).mean()) for c in (0, 1, 2))


def budget_table():
    """Rows (24 h mm, severity, f_high, f_med) at the policy points, for reports."""
    mms = sorted({mm for mm, _ in config.HIGH_BUDGET_24H} | {mm for mm, _ in config.MED_BUDGET_24H}
                 | {mm for mm, _ in config.SEVERITY_24H_KNOTS})
    return [(mm, severity(mm, 0.0), *budgets(severity(mm, 0.0))) for mm in mms]
