"""Short human-readable reasons ("drivers") for each cell's risk.

Static reasons come from XGBoost pred_contribs (log-odds contribution of each feature);
the rain reason comes from the forecast rain index. Ranking language only, no depth.
"""
import numpy as np

from ml import config
from ml.features import FEATURE_COLUMNS


def rain_label(rain24: float):
    label = None
    for name, lower in config.RAIN_BANDS_24H:
        if rain24 >= lower:
            label = name
    return label


def _static_phrase(feature: str, value: float, median: float):
    high = value > median
    if feature in ("elevation_m", "elev_pct"):
        return "Low-lying area"
    if feature == "local_relief_m":
        return "Lower than the surrounding streets"
    if feature == "sink_depth_m":
        return "Natural dip where water collects"
    if feature == "log_dist_drain":
        return "Far from storm drains" if high else "Next to a drain channel"
    if feature == "log_dist_water":
        return "Far from water bodies" if high else "Close to a river or lake"
    if feature == "slope_deg":
        return "Steep slope funnels runoff" if high else "Very flat ground, slow runoff"
    if feature == "road_density":
        return "Dense road network" if high else "Sparse road network"
    if feature == "built_up_frac":
        return "Heavily built-up, paved area" if high else "Little built-up cover"
    return feature


def build_drivers(contribs, X, medians, rain24, window_h: int):
    """contribs/X: (N, F) arrays in FEATURE_COLUMNS order; rain24: (N,) mm. Returns list[list[str]].

    Rain wording first (if any rain class is reached), then the strongest positive terrain contributions.
    """
    n, nf = contribs.shape
    k = config.MAX_DRIVERS
    order = np.argsort(-contribs, axis=1)[:, :k]
    top = np.take_along_axis(contribs, order, axis=1)
    above = (X > medians).astype(np.intp)                 # (N, F): 1 if the value is above its median
    # phrase lookup per (feature, above-median); built once, not per cell
    table = [[_static_phrase(FEATURE_COLUMNS[j], 1.0 if hi else -1.0, 0.0) for hi in (False, True)] for j in range(nf)]
    span = "24 h" if window_h >= 24 else f"{window_h} h"
    rain_cache = {}
    out = []
    for i in range(n):
        reasons = []
        r = round(float(rain24[i]))
        if r not in rain_cache:
            label = rain_label(float(r))
            rain_cache[r] = f"{label} rain forecast: {r:.0f} mm in {span}" if label else None
        if rain_cache[r]:
            reasons.append(rain_cache[r])
        for slot in range(k):
            if len(reasons) >= k or top[i, slot] < config.MIN_DRIVER_CONTRIB:
                break
            j = order[i, slot]
            phrase = table[j][above[i, j]]
            if phrase not in reasons:
                reasons.append(phrase)
        out.append(reasons or ["No strong risk factors"])
    return out
