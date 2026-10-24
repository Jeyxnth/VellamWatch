import numpy as np
import pytest

from ml import config, rain
from ml.predict import predict_risk_at
from ml.tests.conftest import SMALL_BBOX, clone_data


def test_zero_rain_gives_no_high_cells(env):
    d = clone_data(env, "zero", zero_rain=True)
    cells = predict_risk_at(*SMALL_BBOX, 24, data_dir=d)
    assert cells and all(c["risk_band"] != "high" for c in cells)
    assert max(c["risk_score"] for c in cells) < config.BAND_HIGH


def test_heavy_rain_produces_high_cells(env):
    cells = predict_risk_at(*SMALL_BBOX, 24)  # synthetic 2023 burst
    assert any(c["risk_band"] == "high" for c in cells)
    assert any("rain forecast" in dr for c in cells[:20] for dr in c["drivers"])


def test_end_to_end_monotone_in_rainfall(env):
    scores = []
    for i, scale in enumerate((0.0, 0.5, 1.0, 2.0)):
        d = clone_data(env, f"scale{i}", rain_scale=scale)
        cells = predict_risk_at(*SMALL_BBOX, 24, data_dir=d)
        scores.append({c["cell_id"]: c["risk_score"] for c in cells})
    for lo, hi in zip(scores, scores[1:]):
        assert all(hi[k] >= lo[k] for k in lo)


def test_as_of_selects_forecast_window(env, monkeypatch):
    base = predict_risk_at(*SMALL_BBOX, 24, as_of="2023-12-03T00:00:00+05:30")
    quiet = predict_risk_at(*SMALL_BBOX, 24, as_of="2023-12-04T06:00:00+05:30")
    assert max(c["risk_score"] for c in base) > max(c["risk_score"] for c in quiet)
    monkeypatch.setenv("ML_AS_OF", "2023-12-04T06:00:00+05:30")
    via_env = predict_risk_at(*SMALL_BBOX, 24)
    assert via_env == quiet


def test_as_of_outside_data_gives_clear_error(env):
    with pytest.raises(rain.NoRainfallError) as e:
        predict_risk_at(*SMALL_BBOX, 24, as_of="2001-01-01T00:00:00+05:30")
    assert "ML_AS_OF" in str(e.value) and "rainfall.parquet" in str(e.value)


def test_resolve_as_of_default_is_now_ist(monkeypatch):
    monkeypatch.delenv("ML_AS_OF", raising=False)
    ts = rain.resolve_as_of()
    assert ts.utcoffset().total_seconds() == 5.5 * 3600
    assert rain.resolve_as_of("2030-01-01T00:00:00Z").utcoffset().total_seconds() == 5.5 * 3600


def test_idw_weights_sum_to_one_and_prefer_near_points():
    pts = np.array([[13.0, 80.0], [13.2, 80.2]])
    w = rain.idw_weights(np.array([13.01, 13.19]), np.array([80.01, 80.19]), pts)
    assert np.allclose(w.sum(1), 1)
    assert w[0, 0] > w[0, 1] and w[1, 1] > w[1, 0]
