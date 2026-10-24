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
                         find_data_dir, load_hotspots, make_labels, spatial_block_folds)

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


# ---- C. band distribution ---------------------------------------------------------------
BAND_SCENARIOS = [("no rain", 0.0), ("light (15 mm)", 15.0), ("moderate (40 mm)", 40.0),
                  ("heavy (64.5 mm)", 64.5), ("very heavy (115.6 mm)", 115.6),
                  ("extremely heavy (204.5 mm)", 204.5)]


def band_distribution(u, k, r0):
    rows = []
    for label, ri in BAND_SCENARIOS:
        risk = np.round(rainmod.combine_risk(u, ri, k, r0), 3)
        bands = np.array([config.risk_band(float(x)) for x in risk])
        rows.append((label, ri, *(float((bands == b).mean()) for b in ("low", "medium", "high"))))
    return rows


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


def render_markdown(meta, n_folds, cv_results, holdout_rows, bands, mono, plots, events, plot_rel):
    src = meta["data_source"]
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

    L += ["## C. Band distribution", "",
          f"Share of ALL grid cells per band for a given rain index (rain24 + {config.INTENSITY_WEIGHT} x max3h, mm). "
          f"Bands: low < {config.BAND_MEDIUM}, medium < {config.BAND_HIGH}, high >= {config.BAND_HIGH}.", "",
          "| Scenario | rain index (mm) | low | medium | high |", "|---|---|---|---|---|"]
    for label, ri, lo, me, hi in bands:
        L.append(f"| {label} | {ri:g} | {lo:.1%} | {me:.1%} | {hi:.1%} |")
    L += ["", f"Risk non-decreasing in rainfall for every cell (0-400 mm grid): **{'PASS' if mono else 'FAIL'}**. "
          f"With rain index below {config.LIGHT_RAIN_MM} mm risk is capped below the high band.", ""]

    r = meta["rain"]
    L += ["## D. Rainfall conditioning parameters", "",
          f"- k = {r['k']:.4f} per mm: {r['k_source']}",
          f"- r0 = {r['r0']:.1f} mm: {r['r0_source']}",
          f"- {r['note']}" if r["note"] else "- (no notes)",
          "- Rain bands (IMD-style, 24 h): " + ", ".join(f"{n} >= {v} mm" for n, v in config.RAIN_BANDS_24H)
          + " - **verify** against IMD before quoting.",
          "- Event windows used for fitting: " + ", ".join(f"{e}: {s.date()} to {(en - pd.Timedelta(days=1)).date()}" for e, (s, en) in events.items())
          + " (unless data/processed/events.csv exists these are unverified placeholders).",
          "- Within a single event, forecast rain is nearly uniform over the city, so cell ranking inside an event is "
          "set by susceptibility; rainfall decides how many cells reach amber/red.", ""]

    L += ["## Plots", "", f"![Precision-recall]({plot_rel[0]})", "", f"![Recall at top-k]({plot_rel[1]})", ""]
    L += ["## Limitations", "",
          "- **Label noise:** a cell not in hotspots.csv is treated as 'not a hotspot', but it may simply be unreported.",
          "- **Only reported hotspots:** labels come from documented/news waterlogging sites, biased toward busy roads and "
          "well-reported areas.",
          "- **Coarse rainfall:** Open-Meteo points are kilometres apart; street-level rain variation is invisible.",
          "- **Few events:** at most two events, so rain sensitivity is weakly identified and defaults are heuristic.",
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
    mono = monotone_ok(st.u, st.k, st.r0)

    windows, _ = event_windows(d)
    plot_dir = Path(args.plot_dir) if args.plot_dir else config.artifact_root() / "plots"
    pr_path, rk_path = make_plots(src, y_true, cv_scores, curves, plot_dir)
    out = Path(args.out)
    try:
        rel = [Path(p).resolve().relative_to(out.resolve().parent).as_posix() for p in (pr_path, rk_path)]
    except ValueError:
        rel = [str(pr_path), str(rk_path)]
    out.write_text(render_markdown(meta, n_folds, cv_results, holdout_rows, bands, mono, (pr_path, rk_path),
                                   windows, rel), encoding="utf-8")
    m = cv_results["XGBoost (ours)"]
    print(f"spatial CV: PR-AUC {m['pr_auc']:.3f}, recall@10% {m['recall@10%']:.1%}, lift@10% {m['lift@10%']:.2f}x")
    print(f"wrote {out} and plots in {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
