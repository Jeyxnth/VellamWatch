"""Backtest the susceptibility ranking and write ml/BACKTEST.md (+ PNG plots).

Run: python -m ml.backtest [--data-dir DIR] [--out PATH]

Evaluates hotspot RANKING only (never flood depth):
  A. spatial block cross-validation (buffered, no neighbour leakage)
  B. event hold-out (train on one event's hotspots, test on the other, both directions)
plus baselines (random, lowest-elevation-first, logistic regression on elevation + distance to
water) and a band-distribution check. Synthetic data always gets the NOT RESULTS banner.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from ml import config  # noqa: E402
from ml import predict as predmod  # noqa: E402
from ml import rain as rainmod  # noqa: E402
from ml import train as trainmod  # noqa: E402
from ml.features import (FEATURE_COLUMNS, build_static_table, data_source, event_windows,  # noqa: E402
                         find_data_dir, load_hotspots, make_labels, percentile_rank, spatial_block_folds)

KS = (50, 100, 250, 500)
FRACS = (0.10, 0.20)
N_RANDOM = 20


# ---- metrics ------------------------------------------------------------------
def _order(score):
    return np.lexsort((np.arange(len(score)), -np.asarray(score)))  # stable, ties by index


def recall_at_k(y, score, k):
    k = min(int(k), len(y))
    return float(y[_order(score)[:k]].sum() / max(y.sum(), 1))


def precision_at_k(y, score, k):
    k = min(int(k), len(y))
    return float(y[_order(score)[:k]].mean())


def evaluate(y, score) -> dict:
    n = len(y)
    base = float(y.mean())
    out = {"pr_auc": float(average_precision_score(y, score)), "base_rate": base, "n": n, "n_pos": int(y.sum())}
    for f in FRACS:
        k = int(round(f * n))
        out[f"recall@{int(f * 100)}%"] = recall_at_k(y, score, k)
        out[f"lift@{int(f * 100)}%"] = precision_at_k(y, score, k) / base if base > 0 else float("nan")
    for k in KS:
        out[f"precision@{k}"] = precision_at_k(y, score, k)
    out["precision@n_pos"] = precision_at_k(y, score, int(y.sum()))
    return out


def evaluate_random(y, seed=config.SEED) -> dict:
    """Average of N_RANDOM seeded random rankings."""
    rng = np.random.default_rng(seed)
    runs = [evaluate(y, rng.random(len(y))) for _ in range(N_RANDOM)]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


# ---- baselines ------------------------------------------------------------------
def _lr_inputs(table):
    return np.column_stack([table.elevation_m.to_numpy(), table.log_dist_water.to_numpy()])


def lr_baseline(table, train_idx, test_idx, y_train):
    X = _lr_inputs(table)
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-9
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit((X[train_idx] - mu) / sd, y_train[train_idx])
    return lr.decision_function((X[test_idx] - mu) / sd)


def lowest_elevation_score(table, idx):
    return -table.elevation_m.to_numpy()[idx]


# ---- A. spatial block CV -----------------------------------------------------------
def spatial_cv(table, y_true, y_train, w):
    X = table[FEATURE_COLUMNS].to_numpy(float)
    folds = spatial_block_folds(table.row, table.col)
    n = len(table)
    scores = {"XGBoost (ours)": np.full(n, np.nan), "Lowest elevation first": np.full(n, np.nan),
              "Logistic regression (elev + dist water)": np.full(n, np.nan)}
    for tr, te in folds:
        booster = trainmod.fit_booster(X[tr], y_train[tr], w[tr])
        scores["XGBoost (ours)"][te] = trainmod.margins(booster, X[te])
        scores["Lowest elevation first"][te] = lowest_elevation_score(table, te)
        scores["Logistic regression (elev + dist water)"][te] = lr_baseline(table, tr, te, y_train)
    results = {name: evaluate(y_true, s) for name, s in scores.items()}
    results["Random (mean of 20)"] = evaluate_random(y_true)
    return results, scores, len(folds)


# ---- B. event hold-out ---------------------------------------------------------------
def event_holdout(table, hotspots, events):
    """Train on event a, test on event b. 'new only' drops cells that were hotspots in a."""
    X = table[FEATURE_COLUMNS].to_numpy(float)
    all_idx = np.arange(len(table))
    rows = []
    curves = {}
    for a in events:
        for b in events:
            if a == b:
                continue
            y_a_true, y_a_train, w = make_labels(table, hotspots, [a], config.NEIGHBOUR_WEIGHT)
            y_b, _, _ = make_labels(table, hotspots, [b])
            if y_a_true.sum() < 5 or y_b.sum() == 0:
                continue
            booster = trainmod.fit_booster(X, y_a_train, w)
            s_model = trainmod.margins(booster, X)
            s_elev = lowest_elevation_score(table, all_idx)
            s_lr = lr_baseline(table, all_idx, all_idx, y_a_train)
            for variant, keep in (("all test hotspots", np.ones(len(table), bool)),
                                  ("new hotspots only", y_a_true == 0)):
                y = y_b[keep]
                if y.sum() == 0:
                    continue
                for name, s in (("XGBoost (ours)", s_model), ("Lowest elevation first", s_elev),
                                ("Logistic regression (elev + dist water)", s_lr)):
                    rows.append({"train": a, "test": b, "variant": variant, "model": name, **evaluate(y, s[keep])})
                rows.append({"train": a, "test": b, "variant": variant, "model": "Random (mean of 20)",
                             **evaluate_random(y)})
                if variant == "all test hotspots":
                    curves[f"{a}->{b}"] = (y, s_model[keep])
    return rows, curves


# ---- C. event replay through the rainfall stage --------------------------------------------
REPLAY_HORIZON_H = 72


def event_replay(table, hotspots, events, windows, rain_df, k, r0):
    """Train stage 1 on event a's hotspots, replay event b via as_of = start of b's window.

    Reports how many of b's hotspot cells land in the high band vs. how many of ALL cells do.
    """
    X = table[FEATURE_COLUMNS].to_numpy(float)
    rows = []
    for a in events:
        for b in events:
            if a == b or b not in windows:
                continue
            y_a, y_a_train, w = make_labels(table, hotspots, [a], config.NEIGHBOUR_WEIGHT)
            y_b, _, _ = make_labels(table, hotspots, [b])
            if y_a.sum() < 5 or y_b.sum() == 0:
                continue
            try:
                cr = rainmod.cell_rain(rain_df, table.lat.to_numpy(), table.lon.to_numpy(),
                                       windows[b][0], REPLAY_HORIZON_H)
            except rainmod.NoRainfallError as e:
                rows.append({"train": a, "replay": b, "error": str(e)})
                continue
            booster = trainmod.fit_booster(X, y_a_train, w)
            u = percentile_rank(trainmod.margins(booster, X))
            risk = np.round(rainmod.combine_risk(u, cr.rain_index, k, r0), 3)
            high = risk >= config.BAND_HIGH
            medhigh = risk >= config.BAND_MEDIUM
            new = y_b.astype(bool) & (y_a == 0)
            rows.append({
                "train": a, "replay": b, "as_of": windows[b][0].isoformat(), "n_hot": int(y_b.sum()),
                "n_new": int(new.sum()), "rain24_mean": float(cr.rain24.mean()),
                "hot_high": float(high[y_b == 1].mean()), "hot_medhigh": float(medhigh[y_b == 1].mean()),
                "new_high": float(high[new].mean()) if new.any() else float("nan"),
                "cells_high": float(high.mean()), "cells_medhigh": float(medhigh.mean()),
            })
    return rows


# ---- D. band-share report ------------------------------------------------------------------
def rain_index_for(total_mm, max3h_share):
    """Rain index for a 24 h total whose wettest 3 h hold `max3h_share` of it."""
    return total_mm * (1 + config.INTENSITY_WEIGHT * max3h_share)


def band_shares(u, rain_index, k, r0):
    risk = np.round(rainmod.combine_risk(u, rain_index, k, r0), 3)
    high = risk >= config.BAND_HIGH
    med = (risk >= config.BAND_MEDIUM) & ~high
    return float((~high & ~med).mean()), float(med.mean()), float(high.mean())


def band_distribution(u, k, r0):
    """Rows (24 h mm, profile, rain index, low, medium, high) for the config.BAND_CHECK_* scenarios."""
    rows = []
    for total in config.BAND_CHECK_TOTALS_MM:
        for name, share in config.BAND_CHECK_PROFILES.items():
            ri = rain_index_for(total, share)
            rows.append((total, name, ri, *band_shares(u, ri, k, r0)))
    return rows


def band_proposal(u, k, r0):
    """If too many cells are high at 100 mm, PROPOSE (never apply) an r0 or threshold change."""
    shares = {name: band_shares(u, rain_index_for(config.BAND_CHECK_AT_MM, s), k, r0)[2]
              for name, s in config.BAND_CHECK_PROFILES.items()}
    worst = max(shares.values())
    if worst <= config.BAND_CHECK_MAX_HIGH_SHARE:
        return None
    burst = rain_index_for(config.BAND_CHECK_AT_MM, max(config.BAND_CHECK_PROFILES.values()))
    target = config.BAND_CHECK_TARGET_HIGH_SHARE
    lo, hi = 0.0, 500.0
    for _ in range(60):  # larger r0 -> fewer high cells
        mid = (lo + hi) / 2
        if band_shares(u, burst, k, mid)[2] > target:
            lo = mid
        else:
            hi = mid
    r0_new = (lo + hi) / 2
    risk = np.round(rainmod.combine_risk(u, burst, k, r0), 3)
    return {
        "worst_high_share": worst, "target": target, "r0_new": r0_new,
        "threshold_new": float(np.quantile(risk, 1 - target)),
        "shares_with_r0": {name: band_shares(u, rain_index_for(config.BAND_CHECK_AT_MM, s), k, r0_new)[2]
                           for name, s in config.BAND_CHECK_PROFILES.items()},
    }


def monotone_ok(u, k, r0) -> bool:
    grid = np.linspace(0, 400, 161)
    prev = None
    for ri in grid:
        cur = rainmod.combine_risk(u, ri, k, r0)
        if prev is not None and np.any(cur < prev - 1e-12):
            return False
        prev = cur
    return True


# ---- plots ---------------------------------------------------------------------------------
def make_plots(src, y_true, cv_scores, holdout_curves, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC - NOT RESULTS" if src == "synthetic" else "real data"
    base = y_true.mean()
    colors = {"XGBoost (ours)": "tab:red", "Lowest elevation first": "tab:blue",
              "Logistic regression (elev + dist water)": "tab:green"}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in cv_scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, color=colors[name], label=f"{name} (AP={average_precision_score(y_true, s):.2f})")
    ax.axhline(base, color="gray", ls=":", label=f"Random (base rate {base:.3f})")
    ax.set_xlabel("Recall of hotspot cells")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall, spatial block CV [{tag}]", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pr_path = plot_dir / f"{src}_pr_curve.png"
    fig.savefig(pr_path, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    frac = np.linspace(0, 1, 101)

    def curve(y, s):
        o = _order(s)
        c = np.cumsum(y[o]) / max(y.sum(), 1)
        return [c[max(int(round(f * len(y))) - 1, 0)] if f > 0 else 0 for f in frac]

    for name, s in cv_scores.items():
        ax.plot(frac * 100, curve(y_true, s), color=colors[name], label=name)
    for lbl, (y, s) in holdout_curves.items():
        ax.plot(frac * 100, curve(y, s), ls="--", lw=1, label=f"XGBoost event hold-out {lbl}")
    ax.plot([0, 100], [0, 1], color="gray", ls=":", label="Random")
    ax.set_xlabel("% of cells inspected (riskiest first)")
    ax.set_ylabel("Share of hotspot cells found")
    ax.set_title(f"Recall@top-k% of cells [{tag}]", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    rk_path = plot_dir / f"{src}_recall_at_k.png"
    fig.savefig(rk_path, dpi=130)
    plt.close(fig)
    return pr_path, rk_path


# ---- report -----------------------------------------------------------------------------------
def _fmt(v, pct=False):
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _metric_table(rows):
    head = "| Model | PR-AUC | Recall@top 10% | Recall@top 20% | Lift@10% | Lift@20% | P@50 | P@100 | P@250 | P@500 |"
    lines = [head, "|" + "---|" * 10]
    for name, m in rows:
        lines.append(f"| {name} | {_fmt(m['pr_auc'])} | {_fmt(m['recall@10%'], True)} | {_fmt(m['recall@20%'], True)} | "
                     f"{m['lift@10%']:.2f}x | {m['lift@20%']:.2f}x | {_fmt(m['precision@50'])} | "
                     f"{_fmt(m['precision@100'])} | {_fmt(m['precision@250'])} | {_fmt(m['precision@500'])} |")
    return "\n".join(lines)


def render_markdown(meta, n_folds, cv_results, holdout_rows, replay_rows, bands, proposal, mono, events, win_src, plot_rel):
    src = meta["data_source"]
    r_cfg0 = meta["rain"]["r0"]
    rain_label = "fitted on past events" if meta["rain"].get("fitted") else "heuristic, not fitted"
    L = []
    if src == "synthetic":
        L += [f"# **{config.SYNTHETIC_BANNER}**", "",
              "> Every number below comes from a synthetic generator whose hotspots are built from the same terrain "
              "features the model uses. It only proves the pipeline runs. It says nothing about Chennai.", ""]
    L += ["# VellamWatch ML backtest", "",
          f"- **Data source:** {src.upper()} (`{meta['data_dir']}`)",
          f"- **Cells:** {meta['n_cells']}; hotspot cells: {meta['n_hotspot_cells']} "
          f"(per event: {meta['hotspot_cells_per_event']}); base rate {meta['base_rate']:.3f}",
          f"- **Seed:** {meta['seed']}; model: XGBoost depth {config.XGB_PARAMS['max_depth']}, "
          f"{config.XGB_PARAMS['n_estimators']} trees, monotone constraints {meta['monotone']}",
          "- **What is measured:** how well the susceptibility ranking puts known hotspot cells near the top. "
          "This is hotspot RANKING, not flood depth.", ""]

    L += ["## A. Spatial block cross-validation", "",
          f"{n_folds} folds over ~{config.CV_BLOCK_CELLS * 250 / 1000:.0f} km geographic blocks; training cells within "
          f"{config.CV_BUFFER_CELLS} cells of any test cell are dropped. Pooled out-of-fold scores, all events' "
          "hotspots as positives.", "",
          _metric_table(list(cv_results.items())), ""]

    L += ["## B. Event hold-out", ""]
    if len(events) < 2 or not holdout_rows:
        L += [f"Only {len(events)} event(s) with hotspots, so cross-event hold-out was **not run**. "
              "Spatial CV above is the only validation.", ""]
    else:
        L += ["Train on one event's hotspots only, test on the other event. "
              "'New hotspots only' removes cells that were hotspots in the training event, so repeat flooding at "
              "known sites cannot inflate the score.", ""]
        for variant in ("all test hotspots", "new hotspots only"):
            for a, b in [(r["train"], r["test"]) for r in holdout_rows if r["variant"] == variant and r["model"] == "XGBoost (ours)"]:
                rows = [(r["model"], r) for r in holdout_rows
                        if r["train"] == a and r["test"] == b and r["variant"] == variant]
                n_pos = rows[0][1]["n_pos"]
                L += [f"**Train {a} -> test {b}, {variant}** (n_pos={n_pos}, base rate {rows[0][1]['base_rate']:.3f})", "",
                      _metric_table(rows), ""]

    L += ["## C. Event replay through the rainfall stage", ""]
    if not replay_rows:
        L += ["Not run (needs two events with hotspots and rainfall covering the event windows).", ""]
    else:
        L += [f"Stage 1 is trained on one event's hotspots only; the other event is replayed with as_of = start of its "
              f"window, horizon {REPLAY_HORIZON_H} h, through the full risk pipeline (contract band thresholds). "
              f"Rainfall stage: **{rain_label}**. A useful result has far more of the event's hotspots in the high band "
              "than cells overall. 'New' = hotspots of the replayed event that were not hotspots in the training event.", "",
              "| Train | Replay (as_of) | mean rain24 (mm) | hotspots | hotspots in HIGH | new hotspots in HIGH | "
              "ALL cells in HIGH | hotspots medium+high | ALL cells medium+high |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in replay_rows:
            if "error" in r:
                L.append(f"| {r['train']} | {r['replay']} | not run: no rainfall in that window | | | | | | |")
                continue
            new_txt = "n/a" if np.isnan(r["new_high"]) else f"{r['new_high']:.1%} (n={r['n_new']})"
            L.append(f"| {r['train']} | {r['replay']} ({r['as_of']}) | {r['rain24_mean']:.0f} | {r['n_hot']} | "
                     f"{r['hot_high']:.1%} | {new_txt} | {r['cells_high']:.1%} | {r['hot_medhigh']:.1%} | "
                     f"{r['cells_medhigh']:.1%} |")
        L.append("")

    L += ["## D. Band-share report", "",
          f"Share of ALL grid cells per band for a 24 h rainfall total. Rain index = rain24 + "
          f"{config.INTENSITY_WEIGHT} x (wettest 3 h); the two profiles bracket how peaked the storm is. "
          f"Bands: low < {config.BAND_MEDIUM}, medium < {config.BAND_HIGH}, high >= {config.BAND_HIGH}. "
          f"Rainfall stage: **{rain_label}**.", "",
          "| 24 h rain (mm) | profile | rain index (mm) | low | medium | high |", "|---|---|---|---|---|---|"]
    for total, prof, ri, lo, me, hi in bands:
        L.append(f"| {total} | {prof} | {ri:.1f} | {lo:.1%} | {me:.1%} | {hi:.1%} |")
    L += ["", f"Risk non-decreasing in rainfall for every cell (0-400 mm grid): **{'PASS' if mono else 'FAIL'}**. "
          f"With rain index below {config.LIGHT_RAIN_MM} mm risk is capped below the high band.", ""]
    if proposal:
        shares = ", ".join(f"{h:.0%} ({p})" for p, h in proposal["shares_with_r0"].items())
        L += [f"### PROPOSAL (not applied): too many high cells at {config.BAND_CHECK_AT_MM} mm", "",
              f"{proposal['worst_high_share']:.0%} of cells are high at {config.BAND_CHECK_AT_MM} mm, above the "
              f"{config.BAND_CHECK_MAX_HIGH_SHARE:.0%} guideline. Nothing in `config.py` was changed. Options for the lead:", "",
              f"1. Raise `RAIN_R0` from {r_cfg0:.0f} to about **{proposal['r0_new']:.0f}** (k unchanged). "
              f"High share at {config.BAND_CHECK_AT_MM} mm would become {shares}.",
              f"2. Or raise `BAND_HIGH` (a contract change) from {config.BAND_HIGH} to about "
              f"**{proposal['threshold_new']:.2f}** to get ~{proposal['target']:.0%} high cells in the burst scenario.", "",
              "Either choice is a judgement about how alarming the map should look; neither is fitted to real flood data.", ""]

    r = meta["rain"]
    L += ["## E. Rainfall conditioning parameters", "",
          f"- k = {r['k']:.4f} per mm: {r['k_source']}",
          f"- r0 = {r['r0']:.1f} mm: {r['r0_source']}",
          f"- {r['note']}" if r["note"] else "- (no notes)",
          "- Rain bands (IMD-style, 24 h): " + ", ".join(f"{n} >= {v} mm" for n, v in config.RAIN_BANDS_24H)
          + " - **verify** against IMD before quoting.",
          "- Event windows (start inclusive, end exclusive): " + ", ".join(f"{e}: {s.isoformat()} to {en.isoformat()}" for e, (s, en) in events.items())
          + f" (source: {win_src}).",
          "- Within a single event, forecast rain is nearly uniform over the city, so cell ranking inside an event is "
          "set by susceptibility; rainfall decides how many cells reach amber/red.", ""]

    L += ["## Plots", "", f"![Precision-recall]({plot_rel[0]})", "", f"![Recall at top-k]({plot_rel[1]})", ""]
    L += ["## Limitations", "",
          "- **Label noise:** a cell not in hotspots.csv is treated as 'not a hotspot', but it may simply be unreported.",
          "- **Only reported hotspots:** labels come from documented/news waterlogging sites, biased toward busy roads and "
          "well-reported areas.",
          "- **Coarse rainfall:** Open-Meteo points are kilometres apart; street-level rain variation is invisible.",
          "- **Few events:** at most two events, so rain sensitivity is weakly identified and defaults are heuristic.",
          "- **Reanalysis historical rainfall (~10 km) tends to understate extreme local rain; training and live "
          "forecast rainfall come from different sources.**",
          "- **Ranking, not depth:** scores order cells by relative hotspot risk. They are not water depths, "
          "return periods or probabilities of flooding.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest the VellamWatch ranking and write BACKTEST.md")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default=str(config.ML_DIR / "BACKTEST.md"))
    ap.add_argument("--plot-dir", help="default: <artifacts>/plots")
    args = ap.parse_args(argv)

    d = find_data_dir(args.data_dir)
    src = data_source(d)
    if src == "synthetic":
        print(f"*** {config.SYNTHETIC_BANNER} ***")
    meta = trainmod.train(d)
    table = build_static_table(d)
    hotspots = load_hotspots(d)
    y_true, y_train, w = make_labels(table, hotspots, None, config.NEIGHBOUR_WEIGHT)

    cv_results, cv_scores, n_folds = spatial_cv(table, y_true, y_train, w)
    events = sorted(hotspots.event.unique())
    holdout_rows, curves = event_holdout(table, hotspots, events) if len(events) >= 2 else ([], {})
    st = predmod.get_state(d)
    bands = band_distribution(st.u, st.k, st.r0)
    proposal = band_proposal(st.u, st.k, st.r0)
    mono = monotone_ok(st.u, st.k, st.r0)
    windows, win_src = event_windows(d)
    replay_rows = (event_replay(table, hotspots, events, windows, rainmod.load_rainfall(d), st.k, st.r0)
                   if len(events) >= 2 else [])
    plot_dir = Path(args.plot_dir) if args.plot_dir else config.artifact_root() / "plots"
    pr_path, rk_path = make_plots(src, y_true, cv_scores, curves, plot_dir)
    out = Path(args.out)
    try:
        rel = [Path(p).resolve().relative_to(out.resolve().parent).as_posix() for p in (pr_path, rk_path)]
    except ValueError:
        rel = [str(pr_path), str(rk_path)]
    out.write_text(render_markdown(meta, n_folds, cv_results, holdout_rows, replay_rows, bands, proposal, mono,
                                   windows, win_src, rel), encoding="utf-8")
    m = cv_results["XGBoost (ours)"]
    print(f"spatial CV: PR-AUC {m['pr_auc']:.3f}, recall@10% {m['recall@10%']:.1%}, lift@10% {m['lift@10%']:.2f}x")
    if proposal:
        print(f"PROPOSAL (not applied): {proposal['worst_high_share']:.0%} of cells are high at "
              f"{config.BAND_CHECK_AT_MM} mm; see BACKTEST.md section D")
    print(f"wrote {out} and plots in {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
