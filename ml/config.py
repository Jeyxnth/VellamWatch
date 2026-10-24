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

# ---- stage 2: budgeted ranking ---------------------------------------------------
# Rain severity decides HOW MUCH of the study area is at risk (a budget of high / medium
# cells); the stage-1 susceptibility percentile decides WHICH cells. Every number below is a
# POLICY TARGET or heuristic. NOTHING in this section is fitted to data.
#
# rain_severity in [0, 1] = max( sev24(24 h total), sev3(max 3 h total) ), both piecewise-linear,
# monotone, evaluated on the study-area mean of the per-cell forecast (IDW from rainfall points).
SEVERITY_24H_KNOTS = [     # (mm in 24 h, severity). IMD-style lower bounds, VERIFY against IMD.
    (15.6, 0.0),           # below this: light rain, nothing flagged
    (64.5, 1 / 3),         # heavy
    (115.6, 2 / 3),        # very heavy
    (204.5, 1.0),          # extremely heavy
]
SEVERITY_3H_KNOTS = [      # (mm in the wettest 3 h, severity). Heuristic, NOT an IMD table. VERIFY.
    (10.0, 0.0),
    (40.0, 1 / 3),
    (70.0, 2 / 3),
    (120.0, 1.0),
]

# Policy: share of ALL study-area cells that are "high" / "medium-or-high", by 24 h rain total (mm).
# Interpolated in severity space between these points; constant beyond the last point.
HIGH_BUDGET_24H = [(25, 0.0), (50, 0.025), (100, 0.15), (150, 0.30), (200, 0.40)]
# Rule for the medium budget (cumulative, includes high): about 2x the high budget, plus a small
# early "watch" band; nothing at light rain.
MED_BUDGET_24H = [(15.6, 0.0), (25, 0.02), (50, 0.06), (100, 0.30), (150, 0.60), (200, 0.80)]
HIGH_CAP = 0.50    # the high band never exceeds this share of study-area cells
MED_CAP = 0.80     # medium-or-high never exceeds this share
LIGHT_RAIN_MM = 15.6   # 24 h total below this (and light 3 h rain) counts as "light rain" in tests/reports

# Policy check ranges (BACKTEST.md section D and tests): allowed high-cell share by 24 h total.
POLICY_HIGH_RANGE = {0: (0.0, 0.0), 25: (0.0, 0.0), 50: (0.0, 0.03), 100: (0.10, 0.20),
                     150: (0.25, 0.35), 200: (0.35, 0.50)}
CHECK_TOTALS_MM = (0, 25, 50, 100, 150, 200)
CHECK_PROFILES = {                         # share of the 24 h total that falls in the wettest 3 h
    "uniform over 24 h": 3 / 24,
    "burst: half of it in 3 h": 0.5,
}

PEAK_WINDOW_H = 3
IDW_POWER = 2.0
IDW_NEAREST = 4

# IMD-style 24 h rainfall classes in mm (lower bounds), used for driver wording. VERIFY against IMD.
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

# ---- explanations ----------------------------------------------------------
MIN_DRIVER_CONTRIB = 0.15  # min XGBoost contribution (log-odds) for a static driver
MAX_DRIVERS = 3
