"""Properties of the budgeted-ranking rain stage (ml/budget.py) and its use in predict_risk."""
import numpy as np
import pytest

from ml import budget, config
from ml.predict import get_state, predict_risk_at, warm_up
from ml.tests.conftest import SMALL_BBOX, clone_data

U = (np.arange(5000) + 0.5) / 5000   # a uniform study grid of percentiles


def _scores(total, share3=3 / 24, u=U):
    f_high, f_med = budget.budgets(budget.severity(total, total * share3))
    return budget.risk_scores(u, f_high, f_med)


# (a) light / no rain ----------------------------------------------------------------------------
@pytest.mark.parametrize("total", [0.0, 5.0, 15.0, config.LIGHT_RAIN_MM - 0.1])
def test_light_rain_flags_nothing(total):
    score, band = _scores(total, share3=3 / 24)
    assert (band == 0).all() and score.max() < config.BAND_MEDIUM


def test_light_rain_end_to_end(env):
    for i, scale in enumerate((0.0, 0.03)):   # synthetic 2023 burst scaled to ~0 / ~6 mm
        d = clone_data(env, f"light{i}", rain_scale=scale)
        cells = predict_risk_at(*SMALL_BBOX, 24, data_dir=d)
        assert {c["risk_band"] for c in cells} == {"low"}


# (b) non-decreasing in rainfall; (c) order never changes ----------------------------------------------
@pytest.mark.parametrize("share3", [3 / 24, 0.5])
def test_monotone_in_rain_and_rank_preserving(share3):
    prev = None
    order = np.argsort(U, kind="stable")
    for total in np.linspace(0, 400, 201):
        score, _ = _scores(total, share3)
        assert np.all(np.diff(score[order]) >= -1e-12)          # rank-preserving at every rain level
        if prev is not None:
            assert np.all(score >= prev - 1e-12)                # non-decreasing in rainfall
        prev = score


def test_end_to_end_monotone_and_order_stable(env):
    cells_by_scale = []
    for i, scale in enumerate((0.1, 0.4, 1.0, 2.0)):
        d = clone_data(env, f"mono{i}", rain_scale=scale)
        cells_by_scale.append({c["cell_id"]: c for c in predict_risk_at(*SMALL_BBOX, 24, data_dir=d)})
    u = dict(zip(get_state().cell_id.tolist(), get_state().u.tolist()))
    for lo, hi in zip(cells_by_scale, cells_by_scale[1:]):
        assert all(hi[k]["risk_score"] >= lo[k]["risk_score"] for k in lo)
    for cells in cells_by_scale:
        ordered = sorted(cells, key=lambda k: u[k])
        scores = [cells[k]["risk_score"] for k in ordered]
        assert all(b >= a for a, b in zip(scores, scores[1:]))


# (d) shares inside the policy ranges, bursty and uniform -------------------------------------------------
@pytest.mark.parametrize("profile", list(config.CHECK_PROFILES))
@pytest.mark.parametrize("total", config.CHECK_TOTALS_MM)
def test_high_share_inside_policy_range(total, profile):
    share3 = config.CHECK_PROFILES[profile]
    _, band = _scores(total, share3)
    lo, hi = config.POLICY_HIGH_RANGE[total]
    high = (band == 2).mean()
    assert lo - 1e-3 <= high <= hi + 1e-3, (total, profile, high)


def test_medium_budget_rule():
    for total in config.CHECK_TOTALS_MM:
        f_high, f_med = budget.budgets(budget.severity(total, total / 8))
        assert f_med >= f_high
        if total <= config.LIGHT_RAIN_MM:
            assert f_med == 0 and f_high == 0


# (e) cap ---------------------------------------------------------------------------------------------------
def test_high_share_never_exceeds_cap(monkeypatch):
    monkeypatch.setattr(config, "HIGH_BUDGET_24H", [(25, 0.0), (50, 0.5), (100, 0.9), (200, 1.0)])
    monkeypatch.setattr(config, "HIGH_CAP", 0.25)
    for total in (0, 50, 100, 200, 500):
        f_high, f_med = budget.budgets(budget.severity(total, total))
        _, band = budget.risk_scores(U, f_high, f_med)
        assert (band == 2).mean() <= 0.25 + 1e-3
        assert f_high <= 0.25


# score mapping consistency -----------------------------------------------------------------------------
@pytest.mark.parametrize("total", [30, 64.5, 100, 150, 250])
def test_score_mapping_matches_contract_bands(total):
    score, band = _scores(total, 0.5)
    for code, name in enumerate(budget.BANDS):
        sel = score[band == code]
        if sel.size:
            assert {config.risk_band(float(x)) for x in sel} == {name}
    assert score.min() >= 0 and score.max() <= 1


def test_score_is_continuous_across_band_edges():
    f_high, f_med = 0.2, 0.5
    eps = 1e-6
    h, m = 1 - f_high, 1 - f_med
    s_lo, _ = budget.risk_scores(np.array([m - eps, m, h - eps, h]), f_high, f_med)
    assert abs(s_lo[0] - s_lo[1]) < 0.01 and abs(s_lo[2] - s_lo[3]) < 0.01


def test_extremes_and_degenerate_budgets():
    s, b = budget.risk_scores(U, 0.0, 0.0)
    assert (b == 0).all()
    s, b = budget.risk_scores(U, 0.3, 0.3)   # no medium band
    assert (b == 1).sum() == 0 and (b == 2).mean() == pytest.approx(0.3, abs=1e-3)


# (f) deterministic; viewport independence ---------------------------------------------------------------------
def test_deterministic(env):
    assert predict_risk_at(*SMALL_BBOX, 24) == predict_risk_at(*SMALL_BBOX, 24)


def test_same_street_same_band_in_any_viewport(env):
    full = {c["cell_id"]: c for c in predict_risk_at(*SMALL_BBOX, 24)}
    for box in [(13.00, 80.20, 13.03, 80.23), (13.02, 80.22, 13.07, 80.28), (13.05, 80.25, 13.06, 80.26)]:
        for c in predict_risk_at(*box, 24):
            assert (c["risk_score"], c["risk_band"]) == (full[c["cell_id"]]["risk_score"], full[c["cell_id"]]["risk_band"])


def test_end_to_end_high_share_equals_budget(env):
    cells = predict_risk_at(*SMALL_BBOX, 24)
    share = sum(c["risk_band"] == "high" for c in cells) / len(cells)
    assert 0.35 <= share <= config.HIGH_CAP   # synthetic 2023 event: ~200 mm in 24 h -> policy range 35-50%


def test_severity_uses_the_larger_of_24h_and_3h():
    assert budget.severity(10, 10) == budget.severity(10, 0) == 0.0               # both below their light-rain knots
    assert budget.severity(20, 100) > budget.severity(20, 0)                     # intense 3 h raises severity
    assert budget.severity(300, 0) == 1.0 and budget.severity(0, 0) == 0.0


# warm-up ---------------------------------------------------------------------------------------------------------
def test_warm_up_returns_model_info(env):
    info = warm_up()
    assert info["data_source"] == "synthetic" and info["rain_stage"] == "policy targets, not fitted"
    assert info["warm_up_s"] >= 0
