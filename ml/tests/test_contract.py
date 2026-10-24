import time

import pandas as pd

from ml.predict import predict_risk, predict_risk_at
from ml.tests.conftest import SMALL_BBOX

KEYS = {"cell_id", "lat", "lon", "risk_score", "risk_band", "peak_window", "drivers"}


def _band(s):
    return "high" if s >= 0.66 else "medium" if s >= 0.33 else "low"


def test_output_matches_contract(env):
    cells = predict_risk(*SMALL_BBOX, 24)
    assert len(cells) > 1000
    for c in cells:
        assert set(c) == KEYS
        assert isinstance(c["cell_id"], str) and c["cell_id"].startswith("r")
        assert isinstance(c["lat"], float) and isinstance(c["lon"], float)
        assert isinstance(c["risk_score"], float) and 0.0 <= c["risk_score"] <= 1.0
        assert round(c["risk_score"], 3) == c["risk_score"]
        assert c["risk_band"] == _band(c["risk_score"])
        assert isinstance(c["drivers"], list) and 1 <= len(c["drivers"]) <= 3
        assert all(isinstance(d, str) and d for d in c["drivers"])
        start = pd.Timestamp(c["peak_window"]["start"])
        end = pd.Timestamp(c["peak_window"]["end"])
        assert c["peak_window"]["start"].endswith("+05:30") and c["peak_window"]["end"].endswith("+05:30")
        assert end - start == pd.Timedelta(hours=3)


def test_peak_window_inside_forecast_window(env):
    cells = predict_risk(*SMALL_BBOX, 24)
    lo = pd.Timestamp("2023-12-03T00:00:00+05:30")
    hi = lo + pd.Timedelta(hours=24)
    for c in cells[:50]:
        assert lo <= pd.Timestamp(c["peak_window"]["start"]) and pd.Timestamp(c["peak_window"]["end"]) <= hi


def test_bbox_filter_and_empty(env):
    assert predict_risk(10.0, 70.0, 10.1, 70.1, 24) == []
    sub = predict_risk(13.00, 80.20, 13.02, 80.22, 24)
    assert 0 < len(sub) < len(predict_risk(*SMALL_BBOX, 24))
    assert all(13.00 <= c["lat"] <= 13.02 and 80.20 <= c["lon"] <= 80.22 for c in sub)


def test_bad_arguments(env):
    import pytest
    with pytest.raises(ValueError):
        predict_risk(13.1, 80.2, 13.0, 80.3, 24)
    with pytest.raises(ValueError):
        predict_risk(*SMALL_BBOX, 0)


def test_speed_after_warmup(env):
    box = SMALL_BBOX                          # ~1600 cells
    predict_risk(*box, 24)                    # warm-up
    t = time.perf_counter()
    cells = predict_risk(*box, 24)
    dt = time.perf_counter() - t
    print(f"predict_risk: {len(cells)} cells in {dt * 1000:.0f} ms")
    assert len(cells) >= 1000
    assert dt < 1.0  # target ~0.3 s; generous ceiling to avoid flaky CI


def test_deterministic(env):
    a = predict_risk(*SMALL_BBOX, 24)
    b = predict_risk_at(*SMALL_BBOX, 24)
    assert a == b
