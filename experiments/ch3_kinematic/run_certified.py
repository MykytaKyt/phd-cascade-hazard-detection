#!/usr/bin/env python3
"""Chapter 3: certified-stable LTC (Lemma V2 with L_h < 1) vs the plain LTC.

The certified model is trained with the penalty weight * sum_i relu(L_h,i - 0.95)
(thesis.kinematic.stability.contraction_penalty). Both models are evaluated with
the nested protocol on fall labels: frame-level metrics at the validation-selected
threshold and event-level outcomes. For one model per seed (outer fold 0) the
certified logit-sensitivity bound is compared with the empirical sensitivity.

Writes results/ch3/certified.json and results/ch3/scores_fall_certified.parquet.

Usage:
    python experiments/ch3_kinematic/run_certified.py --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from thesis.evaluation import (
    FEATURE_COLUMNS,
    ModelSpec,
    make_windows,
    prepare_features,
    run_nested_cv,
    split_validation,
    summarize,
    train_nn,
)
from thesis.kinematic.labels import add_labels
from thesis.kinematic.models import LTCClassifier
from thesis.kinematic.stability import (
    cell_constants,
    contraction_penalty,
    empirical_sensitivity,
    logit_sensitivity_bound,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_event import evaluate_scores  # noqa: E402

IN_DIM = len(FEATURE_COLUMNS)


def specs(weight: float):
    return [
        ModelSpec("ltc24", lambda d: LTCClassifier(d, hidden=24)),
        ModelSpec("ltc24_cert", lambda d: LTCClassifier(d, hidden=24), contraction_penalty(IN_DIM, weight=weight)),
    ]


def stability_check(df: pd.DataFrame, seed: int, spec: ModelSpec, window: int = 8) -> dict:
    x, y, meta = make_windows(prepare_features(df), window=window, label_col="fall")
    groups = meta["seq"].to_numpy()
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(x, y, groups))
    tr_seqs, _ = split_validation(groups[train_idx], 0.2, np.random.default_rng(seed * 1000))
    tr = np.flatnonzero(np.isin(groups, tr_seqs))
    flat = x[tr].reshape(-1, x.shape[-1])
    xn = ((x - flat.mean(0)) / (flat.std(0) + 1e-6)).astype(np.float32)
    torch.manual_seed(seed * 1000)
    model = train_nn(spec.build(x.shape[-1]), xn[tr], y[tr], penalty=spec.penalty)
    c = cell_constants(model, x.shape[-1])
    sens = empirical_sensitivity(model, xn[test_idx])
    return dict(model=spec.name, seed=seed, l_x=c["l_x"], l_h=c["l_h"],
                bound=logit_sensitivity_bound(c, window),
                bound_infinite_window=(c["w_out_l1"] * c["l_x"] / (1 - c["l_h"])) if c["l_h"] < 1 else None,
                empirical_max=float(sens.max()), empirical_median=float(np.median(sens)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weight", type=float, default=10.0)
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    sp = specs(args.weight)
    records, scores = run_nested_cv(df, sp, seeds=args.seeds, label_col="fall")
    scores.to_parquet(ROOT / "results" / "ch3" / "scores_fall_certified.parquet", index=False)
    frame = summarize(records)
    events = pd.DataFrame(evaluate_scores(scores, onsets, kappa=0.0))
    stab = [stability_check(df, seed, s) for seed in args.seeds for s in sp]

    with open(ROOT / "results" / "ch3" / "certified.json", "w") as f:
        json.dump(dict(weight=args.weight, seeds=args.seeds,
                       frame=frame.reset_index().to_dict(orient="records"),
                       events=events.to_dict(orient="records"), stability=stab), f, indent=2, default=float)

    print((frame[["precision_mean", "recall_mean", "f1_mean", "f1_std"]] * 100).round(1).to_string())
    cols = ["detection_rate", "adl_false_alarm_seqs", "delay_median"]
    print(events[events["rule"] == "threshold"].groupby("model")[cols].mean().round(2).to_string())
    print(pd.DataFrame(stab).groupby("model")[["l_h", "bound", "bound_infinite_window",
                                               "empirical_max"]].mean().to_string())


if __name__ == "__main__":
    main()
