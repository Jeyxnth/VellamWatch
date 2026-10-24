"""Train the two-stage risk engine.

Stage 1: XGBoost susceptibility (static terrain -> P(hotspot)), calibrated on
         spatial-block out-of-fold predictions.
Stage 2: budgeted ranking (ml/budget.py). Nothing is fitted: the rain-to-budget
         table is a policy target in config.py. Training only checks the rainfall data.

Run: python -m ml.train [--data-dir DIR]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from ml import config
from ml import rain as rainmod
from ml.features import (FEATURE_COLUMNS, MONOTONE, build_static_table, data_source,
                         event_windows, find_data_dir, load_hotspots, make_labels,
                         spatial_block_folds)


def _display_path(d) -> str:
    try:
        return Path(d).resolve().relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return str(Path(d).resolve())


def monotone_string() -> str:
    return "(" + ",".join(str(MONOTONE.get(c, 0)) for c in FEATURE_COLUMNS) + ")"


def fit_booster(X, y, w, seed=config.SEED) -> xgb.Booster:
    """Shallow, regularised, monotone-constrained XGBoost; imbalance via scale_pos_weight."""
    pos = float((w * (y == 1)).sum())
    neg = float((w * (y == 0)).sum())
    if pos <= 0:
        raise ValueError("training labels contain no positives")
    clf = xgb.XGBClassifier(
        **config.XGB_PARAMS, scale_pos_weight=neg / pos, monotone_constraints=monotone_string(),
        objective="binary:logistic", tree_method="hist", random_state=seed, n_jobs=1)
    clf.fit(X, y, sample_weight=w)
    return clf.get_booster()


def margins(booster: xgb.Booster, X) -> np.ndarray:
    return booster.predict(xgb.DMatrix(np.asarray(X, float)), output_margin=True)


def oof_margins(X, y_train, w, folds, seed=config.SEED) -> np.ndarray:
    """Out-of-fold margins from spatial block CV (every cell is in exactly one test fold)."""
    out = np.full(len(X), np.nan)
    for train_idx, test_idx in folds:
        if y_train[train_idx].sum() == 0:
            raise ValueError("a CV training fold has no hotspots; too few labelled cells for spatial CV")
        booster = fit_booster(X[train_idx], y_train[train_idx], w[train_idx], seed)
        out[test_idx] = margins(booster, X[test_idx])
    return out


def fit_platt(margin, y_true):
    """Platt scaling on out-of-fold margins -> (a, b): p = sigmoid(a * margin + b)."""
    lr = LogisticRegression(C=1.0).fit(margin.reshape(-1, 1), y_true)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def check_rain_data(data_dir, hotspots) -> None:
    """Warn (never crash) about event dates and rainfall coverage problems. Fits nothing."""
    windows, win_src = event_windows(data_dir)
    if "UNVERIFIED" in win_src:
        warnings.warn(f"events.csv not found in {data_dir}; using {win_src}")
    unknown = sorted(set(hotspots.event) - set(windows))
    if unknown:
        warnings.warn(f"hotspots.csv events {unknown} have no entry in events.csv; they are ignored for rain checks")
    try:
        rain_df = rainmod.load_rainfall(data_dir)
    except FileNotFoundError:
        return  # predict/backtest report the missing file with a clear message
    for msg in rainmod.coverage_warnings(rain_df, windows):
        warnings.warn(msg)


def rain_stage_description() -> dict:
    """What goes into meta.json: the policy, clearly labelled as not fitted."""
    return {
        "stage": "budgeted ranking: rain severity sets the share of high/medium cells, "
                 "susceptibility percentile picks which",
        "fitted": False,
        "label": "policy targets, not fitted",
        "severity_24h_knots": config.SEVERITY_24H_KNOTS,
        "severity_3h_knots": config.SEVERITY_3H_KNOTS,
        "high_budget_24h": config.HIGH_BUDGET_24H,
        "med_budget_24h": config.MED_BUDGET_24H,
        "high_cap": config.HIGH_CAP,
        "med_cap": config.MED_CAP,
    }


def train(data_dir=None, artifact_dir=None, verbose=False) -> dict:
    """Train, save artifacts under <artifact_root>/<real|synthetic>/ and return the meta dict."""
    d = find_data_dir(data_dir)
    src = data_source(d)
    out = Path(artifact_dir) if artifact_dir else config.model_dir(src)
    table = build_static_table(d)
    hotspots = load_hotspots(d)
    y_true, y_train, w = make_labels(table, hotspots, None, config.NEIGHBOUR_WEIGHT)
    n_pos = int(y_true.sum())
    if n_pos < 5:
        raise ValueError(f"only {n_pos} hotspot cells in hotspots.csv; need at least 5 to train")
    X = table[FEATURE_COLUMNS].to_numpy(float)

    folds = spatial_block_folds(table.row, table.col)
    oof = oof_margins(X, y_train, w, folds)
    a, b = fit_platt(oof, y_true)
    booster = fit_booster(X, y_train, w)
    check_rain_data(d, hotspots)

    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.json"))
    per_event = {e: int(g.cell_id.nunique()) for e, g in hotspots.groupby("event")}
    meta = {
        "data_source": src,
        "banner": config.SYNTHETIC_BANNER if src == "synthetic" else None,
        "data_dir": _display_path(d),
        "n_cells": int(len(table)),
        "n_hotspot_cells": n_pos,
        "hotspot_cells_per_event": per_event,
        "features": FEATURE_COLUMNS,
        "monotone": monotone_string(),
        "feature_medians": [float(x) for x in np.median(X, axis=0)],
        "platt": {"a": a, "b": b},
        "oof_pr_auc": float(average_precision_score(y_true, oof)),
        "base_rate": n_pos / len(table),
        "rain": rain_stage_description(),
        "seed": config.SEED,
        "xgb_params": config.XGB_PARAMS,
        "neighbour_weight": config.NEIGHBOUR_WEIGHT,
        "trained_at": pd.Timestamp.now(tz=config.IST).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the VellamWatch risk engine")
    ap.add_argument("--data-dir", help="folder with cells.geojson, features.parquet, hotspots.csv, rainfall.parquet")
    args = ap.parse_args(argv)
    warnings.simplefilter("always")
    meta = train(args.data_dir)
    if meta["data_source"] == "synthetic":
        print(f"*** {config.SYNTHETIC_BANNER} ***")
    print(f"data source : {meta['data_source'].upper()}  ({meta['data_dir']})")
    print(f"cells       : {meta['n_cells']}  hotspot cells: {meta['n_hotspot_cells']} {meta['hotspot_cells_per_event']}")
    print(f"OOF PR-AUC  : {meta['oof_pr_auc']:.3f}  (base rate {meta['base_rate']:.3f}, spatial block CV)")
    print(f"rain stage  : {meta['rain']['label']} (see ml/config.py)")
    print(f"artifacts   : {config.model_dir(meta['data_source'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
