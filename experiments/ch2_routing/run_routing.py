#!/usr/bin/env python3
"""Chapter 2: calibration, routing regions and conformal choice of tau_N.

Inputs are the per-window LTC logits from chapter 3
(results/ch3/scores_<label>.parquet). For every seed and outer fold:
  1. temperature T is fitted on the validation logits and applied to
     validation and test logits;
  2. tau_N is chosen by one of
       empirical -- plug-in: largest tau with validation recall bound >= 1 - alpha;
       crc_val   -- conformal risk control on the validation sequences of the fold;
       crc_cross -- conformal risk control on the test sequences of the other
                    folds (scores from models that never saw those sequences);
     tau_A = 1/2, i.e. edge fall decisions are never revised;
  3. on the test sequences of the fold: per-sequence miss fraction L_i(tau_N)
     (the quantity controlled by CRC), pooled recall bound, routed fraction.

Also writes the routing frontier (routed fraction vs recall bound) for the
symmetric and the asymmetric region on pooled calibrated test probabilities.

Writes results/ch2/routing.json and results/ch2/frontier_<label>.csv.

Usage:
    python experiments/ch2_routing/run_routing.py --alpha 0.05 0.1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.routing.calibration import brier, calibrate, ece, fit_temperature
from thesis.routing.conformal import (
    crc_threshold,
    empirical_threshold,
    min_calibration_sequences,
    sequence_miss_fraction,
)
from thesis.routing.region import region_stats

ROOT = Path(__file__).resolve().parents[2]
TAU_GRID = np.concatenate([np.geomspace(1e-5, 0.01, 40), np.linspace(0.01, 0.5, 99)])


def positive_probs(part: pd.DataFrame) -> list[np.ndarray]:
    pos = part[part["label"] == 1]
    return [g["p"].to_numpy() for _, g in pos.groupby("seq")]


def calibrate_fold(fg: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    val = fg["split"] == "val"
    t = fit_temperature(fg.loc[val, "score"].to_numpy(), fg.loc[val, "label"].to_numpy())
    fg = fg.copy()
    fg["p"] = calibrate(fg["score"].to_numpy(), t)
    return fg, t


def run_label(scores: pd.DataFrame, alphas, model: str) -> tuple[list[dict], list[dict], pd.DataFrame]:
    scores = scores[scores["model"] == model]
    calib_rows, rows, frontier = [], [], []
    for seed, sg in scores.groupby("seed"):
        folds = {}
        for fold, fg in sg.groupby("fold"):
            folds[fold], t = calibrate_fold(fg)
            test = folds[fold][folds[fold]["split"] == "test"]
            calib_rows.append(dict(seed=int(seed), fold=int(fold), temperature=t))
        test_all = pd.concat([f[f["split"] == "test"] for f in folds.values()])
        raw_p = 1 / (1 + np.exp(-test_all["score"].to_numpy()))
        y_all = test_all["label"].to_numpy()
        calib_rows.append(dict(
            seed=int(seed), fold="pooled",
            ece_raw=ece(raw_p, y_all), ece_cal=ece(test_all["p"].to_numpy(), y_all),
            brier_raw=brier(raw_p, y_all), brier_cal=brier(test_all["p"].to_numpy(), y_all),
        ))
        if seed == sg["seed"].min():
            p_all = test_all["p"].to_numpy()
            for tau in TAU_GRID:
                for kind, tau_a in (("symmetric", tau), ("asymmetric", 0.5)):
                    frontier.append(dict(region=kind, **region_stats(p_all, y_all, tau, tau_a)))

        for alpha in alphas:
            for method in ("empirical", "crc_val", "crc_cross"):
                risks, n_cal, feasible, taus = [], [], [], []
                pooled = []
                for fold, fg in folds.items():
                    val, test = fg[fg["split"] == "val"], fg[fg["split"] == "test"]
                    if method == "empirical":
                        tau, n = empirical_threshold(val["p"].to_numpy(), val["label"].to_numpy(), 1 - alpha, TAU_GRID), None
                    else:
                        if method == "crc_val":
                            cal = positive_probs(val)
                        else:
                            others = pd.concat([f[f["split"] == "test"] for k, f in folds.items() if k != fold])
                            cal = positive_probs(others)
                        n = len(cal)
                        tau = crc_threshold(cal, alpha, TAU_GRID)
                    feasible.append(tau is not None)
                    tau = 0.0 if tau is None else tau
                    taus.append(tau)
                    n_cal.append(n)
                    risks.extend(sequence_miss_fraction(positive_probs(test), tau))
                    pooled.append(test.assign(tau=tau))
                pooled = pd.concat(pooled)
                routed = (pooled["p"] > pooled["tau"]) & (pooled["p"] < 0.5)
                pos = pooled["label"] == 1
                f_edge = int((pos & ~routed & (pooled["p"] < 0.5)).sum())
                rows.append(dict(
                    seed=int(seed), alpha=alpha, method=method,
                    tau_per_fold=taus, n_cal_per_fold=n_cal, feasible_folds=int(sum(feasible)),
                    test_risk=float(np.mean(risks)), risk_le_alpha=bool(np.mean(risks) <= alpha),
                    recall_bound=1 - f_edge / int(pos.sum()),
                    routed_frac=float(routed.mean()), n_routed=int(routed.sum()),
                ))
    return calib_rows, rows, pd.DataFrame(frontier)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", default=["fall", "lying"])
    parser.add_argument("--model", default="ltc24")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--scores-dir", default=ROOT / "results" / "ch3")
    parser.add_argument("--out-dir", default=ROOT / "results" / "ch2")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(model=args.model, min_calibration_sequences={a: min_calibration_sequences(a) for a in args.alpha})
    for label in args.labels:
        scores = pd.read_parquet(Path(args.scores_dir) / f"scores_{label}.parquet")
        calib, rows, frontier = run_label(scores, args.alpha, args.model)
        frontier.to_csv(out / f"frontier_{label}.csv", index=False)
        report[label] = dict(calibration=calib, conformal=rows)

        c = pd.DataFrame([r for r in calib if r["fold"] == "pooled"])
        t = pd.DataFrame([r for r in calib if r["fold"] != "pooled"])
        print(f"\n=== labels: {label} ===")
        print(f"temperature: {t['temperature'].min():.2f}..{t['temperature'].max():.2f}; "
              f"ECE raw {c['ece_raw'].mean():.3f} -> calibrated {c['ece_cal'].mean():.3f}; "
              f"Brier {c['brier_raw'].mean():.4f} -> {c['brier_cal'].mean():.4f}")
        r = pd.DataFrame(rows)
        r["n_cal"] = r["n_cal_per_fold"].apply(lambda v: np.nan if v[0] is None else np.mean(v))
        print(r.groupby(["alpha", "method"]).agg(
            n_cal=("n_cal", "mean"), feasible_folds=("feasible_folds", "mean"),
            test_risk=("test_risk", "mean"), risk_max=("test_risk", "max"),
            seeds_ok=("risk_le_alpha", "mean"), recall_bound=("recall_bound", "mean"),
            routed_pct=("routed_frac", lambda v: 100 * v.mean()),
        ).round(4).to_string())

    with open(out / "routing.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {out / 'routing.json'}")


if __name__ == "__main__":
    main()
