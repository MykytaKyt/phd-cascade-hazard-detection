#!/usr/bin/env python3
"""Chapter 3: empirical check of the LTC stability lemmas (thesis.kinematic.stability).

For every seed, one LTC is trained on the training sequences of outer fold 0
(same protocol as run_cv.py) and compared with the certified bounds:
  V1 -- observed max |h_i| over all windows vs tau_i |a_i|;
  V2 -- observed first-order logit sensitivity to max-norm input perturbations
        vs the certified bound ||w||_1 L_x sum_k L_h^k.

Writes results/ch3/stability.json.

Usage:
    python experiments/ch3_kinematic/run_stability.py --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from thesis.evaluation import make_windows, prepare_features, split_validation, train_nn
from thesis.kinematic.labels import add_labels
from thesis.kinematic.models import LTCClassifier
from thesis.kinematic.stability import cell_constants, empirical_sensitivity, logit_sensitivity_bound, max_state

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--label", default="fall")
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    df = prepare_features(add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data"))
    x, y, meta = make_windows(df, window=args.window, label_col=args.label)
    groups = meta["seq"].to_numpy()
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(x, y, groups))

    rows = []
    for seed in args.seeds:
        tr_seqs, _ = split_validation(groups[train_idx], 0.2, np.random.default_rng(seed * 1000))
        tr = np.flatnonzero(np.isin(groups, tr_seqs))
        flat = x[tr].reshape(-1, x.shape[-1])
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        xn = ((x - mu) / sd).astype(np.float32)
        torch.manual_seed(seed * 1000)
        model = train_nn(LTCClassifier(x.shape[-1], hidden=24), xn[tr], y[tr])

        c = cell_constants(model, x.shape[-1])
        peak = max_state(model, xn)
        sens = empirical_sensitivity(model, xn[test_idx])
        bound = logit_sensitivity_bound(c, args.window)
        rows.append(dict(
            seed=seed,
            v1_holds=bool(np.all(peak <= c["state_bound"] + 1e-6)),
            v1_max_ratio=float(np.max(peak / c["state_bound"])),
            tau_range=[float(c["tau"].min()), float(c["tau"].max())],
            l_x=c["l_x"], l_h=c["l_h"],
            v2_bound=bound,
            v2_empirical_max=float(sens.max()),
            v2_empirical_median=float(np.median(sens)),
            v2_holds=bool(sens.max() <= bound),
        ))
        print(rows[-1])

    out = ROOT / "results" / "ch3" / "stability.json"
    with open(out, "w") as f:
        json.dump(dict(label=args.label, window=args.window, runs=rows), f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
