import json
import shutil

import numpy as np
import pytest

from ml import backtest, config, explain, features, train
from ml.features import MissingDataError
from ml.predict import MissingModelError, predict_risk_at
from ml.tests.conftest import SMALL_BBOX


def test_training_is_deterministic(env, tmp_path):
    a = train.train(env["data"], tmp_path / "a")
    b = train.train(env["data"], tmp_path / "b")
    assert a["oof_pr_auc"] == b["oof_pr_auc"] and a["platt"] == b["platt"]
    assert (tmp_path / "a" / "model.json").read_bytes() == (tmp_path / "b" / "model.json").read_bytes()


def test_meta_states_synthetic_source(env):
    meta = json.loads((env["art"] / "synthetic" / "meta.json").read_text())
    assert meta["data_source"] == "synthetic" and meta["banner"] == config.SYNTHETIC_BANNER
    assert meta["rain"]["k_source"] and meta["rain"]["r0_source"]


def test_missing_feature_file_error_names_the_file(env, tmp_path):
    d = tmp_path / "partial"
    shutil.copytree(env["data"], d)
    (d / "features.parquet").unlink()
    with pytest.raises(MissingDataError) as e:
        train.train(d, tmp_path / "art")
    msg = str(e.value)
    assert "features.parquet" in msg and "partial" in msg


def test_missing_hotspots_and_rainfall_errors(env, tmp_path):
    d = tmp_path / "p2"
    shutil.copytree(env["data"], d)
    (d / "hotspots.csv").unlink()
    with pytest.raises(MissingDataError, match="hotspots.csv"):
        train.train(d, tmp_path / "art")
    d3 = tmp_path / "p3"
    shutil.copytree(env["data"], d3)
    (d3 / "rainfall.parquet").unlink()
    with pytest.raises(MissingDataError, match="rainfall.parquet"):
        predict_risk_at(*SMALL_BBOX, 24, data_dir=d3)


def test_missing_model_error_tells_how_to_train(env, monkeypatch, tmp_path):
    monkeypatch.setenv("ML_ARTIFACT_DIR", str(tmp_path / "empty"))
    with pytest.raises(MissingModelError, match="python -m ml.train"):
        predict_risk_at(*SMALL_BBOX, 24)


def test_real_data_error_asks_data_owner(tmp_path):
    with pytest.raises(MissingDataError, match="data owner"):
        features.load_cells(tmp_path)


def test_cell_id_parse_and_local_relief():
    assert features.parse_cell_id("r12_c34") == (12, 34)
    with pytest.raises(ValueError):
        features.parse_cell_id("12_34")
    grid = np.array([[1.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, np.nan]])
    m = features.box_nanmean(grid, 1)
    assert m[1, 1] == pytest.approx(17 / 8)  # 8 valid cells: seven 1s and one 10 (NaN ignored)


def test_spatial_folds_have_no_neighbour_leakage():
    rr, cc = np.meshgrid(np.arange(40), np.arange(40), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    buffer = 2
    folds = features.spatial_block_folds(rows, cols, block=8, n_splits=5, buffer=buffer, seed=1)
    assert len(folds) == 5
    covered = np.concatenate([te for _, te in folds])
    assert len(covered) == len(set(covered)) == len(rows)
    for tr, te in folds:
        assert not set(tr) & set(te)
        d = np.maximum(np.abs(rows[tr][:, None] - rows[te][None, :]),
                       np.abs(cols[tr][:, None] - cols[te][None, :])).min(axis=1)
        assert d.min() > buffer


def test_labels_and_unknown_hotspot_cells(env):
    t = features.build_static_table(env["data"])
    h = features.load_hotspots(env["data"])
    y, y_train, w = features.make_labels(t, h)
    assert y.sum() == h.cell_id.nunique()
    y2, y2_train, w2 = features.make_labels(t, h, neighbour_weight=0.3)
    assert (y2 == y).all() and y2_train.sum() > y.sum() and w2.min() == pytest.approx(0.3)
    bad = h.copy()
    bad.loc[0, "cell_id"] = "r9999_c9999"
    with pytest.raises(ValueError, match="not in cells"):
        features.make_labels(t, bad)


def test_monotone_constraints_hold(env):
    t = features.build_static_table(env["data"])
    h = features.load_hotspots(env["data"])
    y, y_train, w = features.make_labels(t, h)
    X = t[features.FEATURE_COLUMNS].to_numpy(float)
    booster = train.fit_booster(X, y_train, w)
    j = features.FEATURE_COLUMNS.index("sink_depth_m")
    lo, hi = X.copy(), X.copy()
    hi[:, j] += 1.0
    assert np.all(train.margins(booster, hi) >= train.margins(booster, lo) - 1e-9)
    j = features.FEATURE_COLUMNS.index("elevation_m")
    hi = X.copy()
    hi[:, j] += 5.0
    assert np.all(train.margins(booster, hi) <= train.margins(booster, X) + 1e-9)


def test_metrics_known_values():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.0, 0.05, 0.15, 0.25])
    assert backtest.recall_at_k(y, s, 1) == 0.5
    assert backtest.recall_at_k(y, s, 3) == 1.0
    assert backtest.precision_at_k(y, s, 2) == 0.5
    m = backtest.evaluate(y, s)
    assert m["recall@20%"] == 0.5 and m["lift@20%"] == pytest.approx(2.5)


def test_drivers_text():
    contribs = np.array([[0.0, 0.9, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    X = np.array([[3.0, 0.02, -2.0, 1.5, 1.0, 5.0, 4.0, 5.0, 0.5]])
    med = np.zeros(9)
    d = explain.build_drivers(contribs, X, med, np.array([85.0]), 24)[0]
    assert d[0] == "Heavy rain forecast: 85 mm in 24 h" and len(d) <= 3
    assert "Low-lying area" in d
    assert explain.build_drivers(contribs * 0, X, med, np.array([0.0]), 24)[0] == ["No strong risk factors"]


def test_backtest_writes_banner_and_report(env, tmp_path):
    out = tmp_path / "BACKTEST.md"
    assert backtest.main(["--data-dir", str(env["data"]), "--out", str(out),
                          "--plot-dir", str(tmp_path / "plots")]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# **SYNTHETIC DATA - PIPELINE TEST ONLY - NOT RESULTS**")
    assert "Spatial block cross-validation" in text and "Limitations" in text and "PASS" in text
    assert (tmp_path / "plots" / "synthetic_pr_curve.png").exists()
    assert (tmp_path / "plots" / "synthetic_recall_at_k.png").exists()
