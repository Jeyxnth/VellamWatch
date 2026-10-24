"""Central configuration for the VellamWatch ML core.

Everything tunable lives here. Values marked "heuristic" are NOT fitted to data.
We rank flood hotspots; we never estimate flood depth.
"""
import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent

SEED = 42
IST = "Asia/Kolkata"  # fixed +05:30, no daylight saving

# ---- data locations -------------------------------------------------------
REAL_DATA_DIR = REPO_ROOT / "data" / "processed"
SYNTHETIC_DIR = ML_DIR / "synthetic_out"
ARTIFACT_ROOT = ML_DIR / "artifacts"
SYNTHETIC_BANNER = "SYNTHETIC DATA - PIPELINE TEST ONLY - NOT RESULTS"


def artifact_root() -> Path:
    """Artifact folder; override with env ML_ARTIFACT_DIR."""
    return Path(os.environ.get("ML_ARTIFACT_DIR") or ARTIFACT_ROOT)


def model_dir(source: str) -> Path:
    """model.json + meta.json for a data source. Real: artifacts/model/ (committable);
    synthetic: artifacts/model_synthetic/ (always git-ignored)."""
    return artifact_root() / ("model" if source == "real" else "model_synthetic")


# ---- risk bands (contract section A; only the ML lead tunes these) ---------
BAND_MEDIUM = 0.33
BAND_HIGH = 0.66


def risk_band(score: float) -> str:
    if score >= BAND_HIGH:
        return "high"
    if score >= BAND_MEDIUM:
        return "medium"
    return "low"


# ---- stage 1: susceptibility model ----------------------------------------
# Weight given to the 8 neighbours of a hotspot cell when used as soft positives
# in TRAINING only (0 = off). Evaluation always uses the true hotspot cells.
NEIGHBOUR_WEIGHT = 0.0

XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=5.0,
    reg_alpha=0.5,
    gamma=0.5,
)

# Spatial block cross-validation: blocks of BLOCK x BLOCK cells (~2 km), and
# training cells within BUFFER cells of any test cell are dropped (no neighbour leakage).
CV_BLOCK_CELLS = 8
CV_SPLITS = 5
CV_BUFFER_CELLS = 2
LOCAL_RELIEF_RADIUS_CELLS = 2  # 5x5 window, ~1.25 km across

# ---- stage 2: rainfall conditioning ----------------------------------------
# risk logit = logit(u) + RAIN_K * (rain_index - RAIN_R0)
# u = percentile rank (0-1) of the cell's calibrated susceptibility over the whole grid.
# Why a percentile: hotspot base rate is only a few percent, so a raw probability
# could never reach the "high" band. The percentile keeps the ranking and lets
# rainfall decide how many cells turn amber/red.
RAIN_K = 0.05          # per mm. heuristic, not fitted (replaced if fitted on events)
RAIN_R0 = 100.0        # mm of rain_index at which a median-susceptible cell is at 50 %. heuristic
RAIN_K_BOUNDS = (0.005, 0.3)
RAIN_R0_BOUNDS = (30.0, 250.0)
RAIN_FIT_MIN_SPREAD_MM = 20.0   # min difference in event rain_index needed to fit
RAIN_FIT_MIN_COVERAGE = 0.9

# rain_index = (max rolling 24 h total) + INTENSITY_WEIGHT * (max rolling 3 h total), in mm
INTENSITY_WEIGHT = 0.5
LIGHT_RAIN_MM = 15.6   # below this rain_index risk is capped under the "high" band
LIGHT_RAIN_CAP = BAND_HIGH - 0.01
U_CLIP = (0.02, 0.98)
PEAK_WINDOW_H = 3
IDW_POWER = 2.0
IDW_NEAREST = 4

# IMD-style 24 h rainfall classes in mm (lower bounds). VERIFY against IMD before quoting.
RAIN_BANDS_24H = [
    ("Moderate", 15.6),
    ("Heavy", 64.5),
    ("Very heavy", 115.6),
    ("Extremely heavy", 204.5),
]

# ---- event dates -----------------------------------------------------------
# TODO(verify with data owner): used only if data/processed/events.csv is absent.
# Same format as events.csv (contract section D): ISO 8601 with +05:30, start inclusive, end exclusive.
EVENTS_TODO = {
    "2015": ("2015-12-01T00:00:00+05:30", "2015-12-04T00:00:00+05:30"),
    "2023": ("2023-12-02T00:00:00+05:30", "2023-12-06T00:00:00+05:30"),  # Cyclone Michaung
}

# ---- band-share check (BACKTEST.md section C) ---------------------------------
BAND_CHECK_TOTALS_MM = (0, 25, 50, 100, 150, 200)      # 24 h rainfall scenarios
BAND_CHECK_PROFILES = {                                  # share of the 24 h total falling in the wettest 3 h
    "uniform over 24 h": 3 / 24,
    "burst: half of it in 3 h": 0.5,
}
BAND_CHECK_AT_MM = 100
BAND_CHECK_MAX_HIGH_SHARE = 0.35   # above this at 100 mm, BACKTEST.md PROPOSES (never applies) changes
BAND_CHECK_TARGET_HIGH_SHARE = 0.20

# ---- explanations ----------------------------------------------------------
MIN_DRIVER_CONTRIB = 0.15  # min XGBoost contribution (log-odds) for a static driver
MAX_DRIVERS = 3
