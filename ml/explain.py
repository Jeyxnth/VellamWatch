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
    """contribs/X: (N, F) arrays in FEATURE_COLUMNS order; rain24: (N,) mm. Returns list[list[str]]."""
    order = np.argsort(-contribs, axis=1)
    out = []
    for i in range(contribs.shape[0]):
        reasons = []
        label = rain_label(float(rain24[i]))
        if label:
            span = "24 h" if window_h >= 24 else f"{window_h} h"
            reasons.append(f"{label} rain forecast: {rain24[i]:.0f} mm in {span}")
        for j in order[i]:
            if len(reasons) >= config.MAX_DRIVERS or contribs[i, j] < config.MIN_DRIVER_CONTRIB:
                break
            phrase = _static_phrase(FEATURE_COLUMNS[j], X[i, j], medians[j])
            if phrase not in reasons:
                reasons.append(phrase)
        out.append(reasons or ["No strong risk factors"])
    return out
