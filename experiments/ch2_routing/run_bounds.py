#!/usr/bin/env python3
"""Chapter 2: routing frontier and the calibration-error bounds (statements D1, D2).

1. Frontier: routed fraction vs recall bound for the symmetric region
   (tau, 1 - tau) and the asymmetric region (tau, 1/2), from
   results/ch2/frontier_<label>.csv (written by run_routing.py).
2. Statement D2: on pooled calibrated test probabilities, compares the
   calibration-based prediction sum_{p <= tau} p of the unrecoverable misses
   with the observed count, and evaluates the bound eps(tau) * #{p <= tau},
   where eps(tau) is the binned local calibration error on [0, tau].

Writes figures/ch2_frontier.png and results/ch2/calibration_bound.json.

Usage:
    python experiments/ch2_routing/run_bounds.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thesis.routing.calibration import calibrate, fit_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TAUS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.146, 0.2]


def pooled_calibrated(scores: pd.DataFrame, model: str, seed: int) -> pd.DataFrame:
    parts = []
    for _, fg in scores[(scores["model"] == model) & (scores["seed"] == seed)].groupby("fold"):
        val = fg[fg["split"] == "val"]
        t = fit_temperature(val["score"].to_numpy(), val["label"].to_numpy())
        test = fg[fg["split"] == "test"].copy()
        test["p"] = calibrate(test["score"].to_numpy(), t)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def local_calibration_error(p: np.ndarray, y: np.ndarray, tau: float, n_bins: int = 10) -> float:
    """max over quantile bins of [0, tau] of |mean(y) - mean(p)|."""
    mask = p <= tau
    if mask.sum() < n_bins:
        return float("nan")
    pp, yy = p[mask], y[mask]
    order = np.argsort(pp)
    errs = [abs(yy[b].mean() - pp[b].mean()) for b in np.array_split(order, n_bins)]
    return float(max(errs))


def plot_frontier(out_png: Path, labels):
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for ax, label in zip(axes[0], labels):
        f = pd.read_csv(ROOT / "results" / "ch2" / f"frontier_{label}.csv")
        for region, style in (("symmetric", "--"), ("asymmetric", "-")):
            g = f[f["region"] == region].groupby("tau_n")[["routed_frac", "recall_bound"]].mean()
            g = g.sort_values("routed_frac")
            ax.plot(100 * g["routed_frac"], 100 * g["recall_bound"], style, label=region)
        ax.set_xlabel("Routed frames, %")
        ax.set_ylabel("Recall bound, %")
        ax.set_title(f"labels: {label}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=["fall", "lying"])
    parser.add_argument("--model", default="ltc24")
    args = parser.parse_args()

    (ROOT / "figures").mkdir(exist_ok=True)
    plot_frontier(ROOT / "figures" / "ch2_frontier.png", args.labels)

    rows = []
    for label in args.labels:
        scores = pd.read_parquet(ROOT / "results" / "ch3" / f"scores_{label}.parquet")
        for seed in sorted(scores["seed"].unique()):
            d = pooled_calibrated(scores, args.model, seed)
            p, y = d["p"].to_numpy(), d["label"].to_numpy()
            n_pos = int(y.sum())
            for tau in TAUS:
                below = p <= tau
                eps = local_calibration_error(p, y, tau)
                rows.append(dict(
                    label=label, seed=int(seed), tau=tau, n_below=int(below.sum()),
                    predicted_f_edge=float(p[below].sum()), observed_f_edge=int(y[below].sum()),
                    eps_local=eps, bound=eps * int(below.sum()) if not np.isnan(eps) else np.nan,
                    recall_bound_predicted=1 - float(p[below].sum()) / n_pos,
                    recall_bound_observed=1 - int(y[below].sum()) / n_pos,
                ))
    r = pd.DataFrame(rows)
    out = ROOT / "results" / "ch2" / "calibration_bound.json"
    r.to_json(out, orient="records", indent=2)
    r["gap"] = (r["observed_f_edge"] - r["predicted_f_edge"]).abs()
    r["bound_holds"] = r["gap"] <= r["bound"]
    print(r.groupby(["label", "tau"]).agg(
        n_below=("n_below", "mean"), predicted=("predicted_f_edge", "mean"), observed=("observed_f_edge", "mean"),
        eps_local=("eps_local", "mean"), bound=("bound", "mean"), bound_holds=("bound_holds", "mean"),
        rb_pred=("recall_bound_predicted", "mean"), rb_obs=("recall_bound_observed", "mean"),
    ).round(3).to_string())
    print(f"\nSaved {out} and figures/ch2_frontier.png")


if __name__ == "__main__":
    main()
