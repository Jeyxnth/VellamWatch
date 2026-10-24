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

from ml import budget, config  # noqa: E402
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


# ---- C. event replay through the budgeted rainfall stage ------------------------------------
REPLAY_HORIZON_H = 72


def _top_budget(score, n):
    """Boolean mask of the n highest-scoring cells (stable, ties by index)."""
    mask = np.zeros(len(score), bool)
    mask[_order(score)[:n]] = True
    return mask


def event_replay(table, hotspots, events, windows, rain_df):
    """Train stage 1 on event a's hotspots only; replay event b with as_of = start of b's window.

    Reports (a) recall: share of b's hotspot cells that land in the HIGH band, (b) the share of ALL
    cells that are high, (c) lift = recall / share. The baseline is lowest-elevation-first given the
    SAME number of high cells. Also reported for medium+high and for 'new' hotspots (not in event a).
    """
    X = table[FEATURE_COLUMNS].to_numpy(float)
    elev_score = lowest_elevation_score(table, np.arange(len(table)))
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
                area = rainmod.area_rain(rain_df, table.lat.to_numpy(), table.lon.to_numpy(),
                                         windows[b][0], REPLAY_HORIZON_H)
            except rainmod.NoRainfallError:
                rows.append({"train": a, "replay": b, "error": "no rainfall data in that event window"})
                continue
            booster = trainmod.fit_booster(X, y_a_train, w)
            u = percentile_rank(trainmod.margins(booster, X))
            _, band = budget.risk_scores(u, area.f_high, area.f_med)
            hot = y_b == 1
            new = hot & (y_a == 0)
            row = {"train": a, "replay": b, "as_of": windows[b][0].isoformat(), "n_hot": int(hot.sum()),
                   "n_new": int(new.sum()), "rain24": area.rain24, "severity": area.severity,
                   "f_high": area.f_high, "f_med": area.f_med}
            for name, mask in (("high", band == 2), ("medhigh", band >= 1)):
                n = int(mask.sum())
                base_mask = _top_budget(elev_score, n)
                share = n / len(mask)
                row[f"{name}_share"] = share
                for tag, sel in (("", hot), ("_new", new)):
                    if sel.any() and n > 0:
                        rec, rec_b = float(mask[sel].mean()), float(base_mask[sel].mean())
                        row[f"{name}_recall{tag}"], row[f"{name}_lift{tag}"] = rec, rec / share
                        row[f"{name}_base_recall{tag}"], row[f"{name}_base_lift{tag}"] = rec_b, rec_b / share
                    else:
                        for k in ("recall", "lift", "base_recall", "base_lift"):
                            row[f"{name}_{k}{tag}"] = float("nan")
            rows.append(row)
    return rows


# ---- D. budget check -------------------------------------------------------------------------
def budget_check(u):
    """Rows for bursty/uniform storms at each rain level, with the policy range and PASS/FAIL."""
    tol = 1.0 / len(u) + 1e-6   # discretisation: shares move in steps of one cell
    rows = []
    for total in config.CHECK_TOTALS_MM:
        lo, hi = config.POLICY_HIGH_RANGE[total]
        for name, share3 in config.CHECK_PROFILES.items():
            sev = budget.severity(total, total * share3)
            f_high, f_med = budget.budgets(sev)
            low_s, med_s, high_s = budget.shares(u, f_high, f_med)
            ok = (lo - tol <= high_s <= hi + tol) and high_s <= config.HIGH_CAP + tol
            rows.append({"total": total, "profile": name, "max3h": total * share3, "severity": sev,
                         "low": low_s, "medium": med_s, "high": high_s, "range": (lo, hi), "ok": ok})
    return rows


def property_checks(u) -> dict:
    """(a) light rain flags nothing, (b) risk never decreases with rain, (c) cell order never changes."""
    totals = np.linspace(0, 400, 161)
    prev, mono, rank_ok = None, True, True
    order = np.argsort(u, kind="stable")
    for t in totals:
        f_high, f_med = budget.budgets(budget.severity(t, t / 8))
        score, _ = budget.risk_scores(u, f_high, f_med)
        if prev is not None and np.any(score < prev - 1e-12):
            mono = False
        if np.any(np.diff(score[order]) < -1e-12):
            rank_ok = False
        prev = score
    f_high, f_med = budget.budgets(budget.severity(config.LIGHT_RAIN_MM - 0.1, 5.0))
    _, band = budget.risk_scores(u, f_high, f_med)
    return {"light_rain_clean": bool((band == 0).all()), "monotone": mono, "rank_preserving": rank_ok}


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


def render_markdown(meta, n_folds, cv_results, holdout_rows, replay_rows, bands, props, events, win_src, plot_rel):
    src = meta["data_source"]
    rain_label = meta["rain"]["label"]
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

    L += ["## C. Event replay through the budgeted rainfall stage", ""]
    if not replay_rows:
        L += ["Not run (needs two events with hotspots and rainfall covering the event windows).", ""]
    else:
        L += [f"Stage 1 is trained on one event's hotspots only; the other event is replayed with as_of = start of its "
              f"window, horizon {REPLAY_HORIZON_H} h. Rain severity sets how many cells are high; the stage-1 ranking picks "
              "which. **Recall** = share of the replayed event's hotspot cells in the HIGH band; **lift** = recall / share "
              "of all cells that are high (random = 1.0). The baseline flags the lowest-elevation cells using the SAME "
              f"number of high cells. Rain stage: **{rain_label}**.", "",
              "| Train | Replay (as_of) | area rain24 (mm) | severity | high budget | hotspots | recall (ours) | "
              "recall (low-elev) | lift (ours) | lift (low-elev) | new-hotspot recall (ours / low-elev) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in replay_rows:
            if "error" in r:
                L.append(f"| {r['train']} | {r['replay']} | not run: {r['error']} | | | | | | | | |")
                continue
            if r["f_high"] == 0 or np.isnan(r["high_recall"]):
                L.append(f"| {r['train']} | {r['replay']} ({r['as_of']}) | {r['rain24']:.0f} | {r['severity']:.2f} | "
                         "0% (no high cells at this rainfall) | %d | n/a | n/a | n/a | n/a | n/a |" % r["n_hot"])
                continue
            new_txt = ("n/a" if np.isnan(r["high_recall_new"]) else
                       f"{r['high_recall_new']:.1%} / {r['high_base_recall_new']:.1%} (n={r['n_new']})")
            L.append(f"| {r['train']} | {r['replay']} ({r['as_of']}) | {r['rain24']:.0f} | {r['severity']:.2f} | "
                     f"{r['f_high']:.1%} | {r['n_hot']} | {r['high_recall']:.1%} | {r['high_base_recall']:.1%} | "
                     f"{r['high_lift']:.2f}x | {r['high_base_lift']:.2f}x | {new_txt} |")
        L += ["", "Medium+high (same replay):", "",
              "| Train | Replay | medium+high budget | recall (ours) | recall (low-elev) | lift (ours) | lift (low-elev) |",
              "|---|---|---|---|---|---|---|"]
        for r in replay_rows:
            if "error" in r or np.isnan(r.get("medhigh_recall", float("nan"))):
                continue
            L.append(f"| {r['train']} | {r['replay']} | {r['medhigh_share']:.1%} | {r['medhigh_recall']:.1%} | "
                     f"{r['medhigh_base_recall']:.1%} | {r['medhigh_lift']:.2f}x | {r['medhigh_base_lift']:.2f}x |")
        L.append("")

    L += ["## D. Budget check", "",
          "Share of ALL study-area cells per band for a 24 h rainfall total, for a uniform storm and a bursty storm "
          f"(half of the total in the wettest 3 h). Bands: low < {config.BAND_MEDIUM}, medium < {config.BAND_HIGH}, "
          f"high >= {config.BAND_HIGH}. The policy range is a TARGET set in `config.py` (not fitted); the high band is capped at "
          f"{config.HIGH_CAP:.0%}. Rain stage: **{rain_label}**.", "",
          "| 24 h rain (mm) | storm | wettest 3 h (mm) | severity | low | medium | high | policy range for high | in range |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in bands:
        lo, hi = r["range"]
        L.append(f"| {r['total']} | {r['profile']} | {r['max3h']:.1f} | {r['severity']:.2f} | {r['low']:.1%} | "
                 f"{r['medium']:.1%} | {r['high']:.1%} | {lo:.0%}-{hi:.0%} | {'PASS' if r['ok'] else 'FAIL'} |")
    L += ["", f"- Light rain (< {config.LIGHT_RAIN_MM} mm) flags no medium or high cells: **{'PASS' if props['light_rain_clean'] else 'FAIL'}**",
          f"- Every cell's risk is non-decreasing as rainfall rises (0-400 mm): **{'PASS' if props['monotone'] else 'FAIL'}**",
          f"- Cell order by risk never changes with rainfall (rank-preserving): **{'PASS' if props['rank_preserving'] else 'FAIL'}**", ""]

    L += ["## E. Stage 2: budgeted ranking (policy targets, not fitted)", "",
          "Rain severity (0-1) = max(severity of the 24 h total, severity of the wettest 3 h), averaged over the whole study "
          "grid. It sets the share of cells that are high / medium; the stage-1 susceptibility percentile (over the full grid, "
          "so a street keeps its rank in any viewport) decides which. risk_score is continuous: low [0, 0.33), medium "
          "[0.33, 0.66), high [0.66, 1], rising with susceptibility inside each band.", "",
          "| 24 h rain (mm) | severity | high budget | medium+high budget |", "|---|---|---|---|"]
    for mm, sev, fh, fm in budget.budget_table():
        L.append(f"| {mm:g} | {sev:.2f} | {fh:.1%} | {fm:.1%} |")
    L += ["",
          "- 24 h severity knots (IMD-style lower bounds, **verify**): " + ", ".join(f"{m:g} mm -> {v:.2f}" for m, v in config.SEVERITY_24H_KNOTS),
          "- Wettest-3 h severity knots (heuristic, not IMD, **verify**): " + ", ".join(f"{m:g} mm -> {v:.2f}" for m, v in config.SEVERITY_3H_KNOTS),
          f"- Caps: high <= {config.HIGH_CAP:.0%}, medium+high <= {config.MED_CAP:.0%} of study-area cells.",
          "- Event windows (start inclusive, end exclusive): " + ", ".join(f"{e}: {s.isoformat()} to {en.isoformat()}" for e, (s, en) in events.items())
          + f" (source: {win_src}).",
          "- Nothing in this stage was tuned on synthetic data, and none of it is fitted to real data either.", ""]

    L += ["## Plots", "", f"![Precision-recall]({plot_rel[0]})", "", f"![Recall at top-k]({plot_rel[1]})", ""]
    L += ["## Limitations", "",
          "- **Label noise:** a cell not in hotspots.csv is treated as 'not a hotspot', but it may simply be unreported.",
          "- **Only reported hotspots:** labels come from documented/news waterlogging sites, biased toward busy roads and "
          "well-reported areas.",
          "- **Coarse rainfall:** Open-Meteo points are kilometres apart; street-level rain variation is invisible.",
          "- **Few events:** at most two events, so rain sensitivity cannot be fitted; stage 2 uses policy targets.",
          "- **Budgets are policy, not measurement:** the share of cells flagged at each rainfall level is a design choice "
          "(see `config.py`), so a high band does not mean a measured probability of flooding.",
          "- **Study-area rainfall:** severity uses the mean forecast over the whole grid; a storm that hits only part of the "
          "city is under-weighted.",
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
    bands = budget_check(st.u)
    props = property_checks(st.u)
    windows, win_src = event_windows(d)
    replay_rows = (event_replay(table, hotspots, events, windows, rainmod.load_rainfall(d))
                   if len(events) >= 2 else [])
    plot_dir = Path(args.plot_dir) if args.plot_dir else config.artifact_root() / "plots"
    pr_path, rk_path = make_plots(src, y_true, cv_scores, curves, plot_dir)
    out = Path(args.out)
    try:
        rel = [Path(p).resolve().relative_to(out.resolve().parent).as_posix() for p in (pr_path, rk_path)]
    except ValueError:
        rel = [str(pr_path), str(rk_path)]
    out.write_text(render_markdown(meta, n_folds, cv_results, holdout_rows, replay_rows, bands, props,
                                   windows, win_src, rel), encoding="utf-8")
    m = cv_results["XGBoost (ours)"]
    print(f"spatial CV: PR-AUC {m['pr_auc']:.3f}, recall@10% {m['recall@10%']:.1%}, lift@10% {m['lift@10%']:.2f}x")
    fails = [r for r in bands if not r["ok"]]
    print(f"budget check: {len(bands) - len(fails)}/{len(bands)} rows inside the policy ranges; properties {props}")
    print(f"wrote {out} and plots in {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
