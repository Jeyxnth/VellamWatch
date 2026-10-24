"""Train the two-stage risk engine.

Stage 1: XGBoost susceptibility (static terrain -> P(hotspot)), calibrated on
         spatial-block out-of-fold predictions.
Stage 2: rainfall conditioning parameters (k, r0); fitted on past events when
         possible, otherwise documented heuristic defaults.

Run: python -m ml.train [--data-dir DIR]
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from ml import config
from ml import rain as rainmod
from ml.features import (FEATURE_COLUMNS, MONOTONE, build_static_table, data_source,
                         event_windows, find_data_dir, load_hotspots, make_labels,
                         percentile_rank, spatial_block_folds)


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


def fit_rain_params(table, hotspots, data_dir, u_oof) -> dict:
    """Fit k and r0 from past events via a logistic model with offset logit(u_oof).

    z = logit(u) + k * rain_index + b, r0 = -b / k, samples = (cell, event).
    Needs events with dates, hotspots and rainfall coverage, and at least
    RAIN_FIT_MIN_SPREAD_MM between event rain indexes. Otherwise defaults are kept.
    """
    res = {"k": config.RAIN_K, "r0": config.RAIN_R0,
           "k_source": "heuristic, not fitted", "r0_source": "heuristic, not fitted", "note": ""}
    windows, win_src = event_windows(data_dir)
    try:
        rain_df = rainmod.load_rainfall(data_dir)
    except FileNotFoundError as e:
        res["note"] = f"rain params not fitted: {e}"
        return res
    samples = []  # (event, y, rain_index per cell)
    skipped = []
    for event, (start, end) in windows.items():
        y, _, _ = make_labels(table, hotspots, [event])
        if y.sum() == 0:
            skipped.append(f"{event}: no hotspots")
            continue
        hours = int((end - start) / np.timedelta64(1, "h"))
        try:
            cr = rainmod.cell_rain(rain_df, table.lat.to_numpy(), table.lon.to_numpy(), start, hours)
        except rainmod.NoRainfallError:
            skipped.append(f"{event}: no rainfall data in window")
            continue
        if cr.window.coverage < config.RAIN_FIT_MIN_COVERAGE:
            skipped.append(f"{event}: rainfall coverage {cr.window.coverage:.0%}")
            continue
        samples.append((event, y, cr.rain_index))
    if len(samples) < 2:
        res["note"] = f"need >= 2 events with hotspots and rainfall to fit; usable: {len(samples)}; skipped: {skipped}"
        return res
    means = [float(s[2].mean()) for s in samples]
    if max(means) - min(means) < config.RAIN_FIT_MIN_SPREAD_MM:
        res["note"] = f"event rain indexes too similar to fit ({means}); defaults kept"
        return res

    offset = np.log(np.clip(u_oof, *config.U_CLIP) / (1 - np.clip(u_oof, *config.U_CLIP)))
    y_all = np.concatenate([s[1] for s in samples]).astype(float)
    x_all = np.concatenate([s[2] for s in samples])
    off_all = np.tile(offset, len(samples))

    def nll(theta):
        z = off_all + theta[0] * x_all + theta[1]
        return float(np.sum(np.logaddexp(0.0, z) - y_all * z))

    fit = minimize(nll, x0=[config.RAIN_K, -config.RAIN_K * config.RAIN_R0],
                   bounds=[(0.0, 0.5), (-60.0, 10.0)], method="L-BFGS-B")
    names = [s[0] for s in samples]
    k_hat = float(fit.x[0])
    r0_hat = float(-fit.x[1] / k_hat) if k_hat > 1e-6 else float("nan")
    k_ok = config.RAIN_K_BOUNDS[0] < k_hat < config.RAIN_K_BOUNDS[1]
    r0_ok = config.RAIN_R0_BOUNDS[0] < r0_hat < config.RAIN_R0_BOUNDS[1]
    if k_ok and r0_ok:  # k and r0 are only meaningful together; accept both or neither
        res.update(k=k_hat, r0=r0_hat, k_source=f"fitted on events {names}", r0_source=f"fitted on events {names}")
    else:
        res["note"] += (f"fit rejected (k={k_hat:.4f}, r0={r0_hat:.1f} outside bounds {config.RAIN_K_BOUNDS} / "
                        f"{config.RAIN_R0_BOUNDS}); heuristic defaults kept for both. ")
    idx_mm = {n: round(m, 1) for n, m in zip(names, means)}
    res["note"] += f"event mean rain_index mm={idx_mm}; event dates from {win_src}."
    return res


def train(data_dir=None, artifact_dir=None, verbose=False) -> dict:
    """Train, save artifacts under <artifact_root>/<real|synthetic>/ and return the meta dict."""
    d = find_data_dir(data_dir)
    src = data_source(d)
    out = Path(artifact_dir) if artifact_dir else config.artifact_root() / src
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
    p_oof = sigmoid(a * oof + b)
    u_oof = percentile_rank(p_oof)
    booster = fit_booster(X, y_train, w)
    rain_params = fit_rain_params(table, hotspots, d, u_oof)

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
        "rain": rain_params,
        "seed": config.SEED,
        "xgb_params": config.XGB_PARAMS,
        "neighbour_weight": config.NEIGHBOUR_WEIGHT,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
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
    r = meta["rain"]
    print(f"rain k={r['k']:.4f} ({r['k_source']}), r0={r['r0']:.1f} ({r['r0_source']})")
    if r["note"]:
        print(f"note        : {r['note']}")
    print(f"artifacts   : {config.artifact_root() / meta['data_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
