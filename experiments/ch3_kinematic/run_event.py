#!/usr/bin/env python3
"""Event-level fall detection on URFD: fall sequences vs ADL sequences.

For each frame-label variant (lying: the binarisation used earlier;
fall: positives only in fall sequences from the onset) the kinematic
classifier is trained with the nested protocol of thesis.evaluation.
On the resulting logits two alarm rules are compared:
  threshold -- alarm when the logit crosses h (frame-wise rule);
  cusum     -- Page's CUSUM on the logit with drift kappa.
The rule parameter h is selected on the validation sequences of every
fold (all validation falls detected, fewest false alarms) and applied to
the test sequences. Test outcomes are pooled over folds (70 sequences per seed).

Writes results/ch3/event_level.json and results/ch3/scores_<label>.parquet.

Usage:
    python experiments/ch3_kinematic/run_event.py --seeds 0 1 2 3 4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.changepoint.cusum import (
    cusum_alarms,
    evaluate_rule,
    select_h,
    summarize_outcomes,
    threshold_alarms,
)
from thesis.evaluation import ModelSpec, run_nested_cv
from thesis.kinematic.labels import add_labels
from thesis.kinematic.models import LSTMClassifier, LTCClassifier

ROOT = Path(__file__).resolve().parents[2]
SPECS = {
    "ltc24": ModelSpec("ltc24", lambda d: LTCClassifier(d, hidden=24)),
    "lstm16": ModelSpec("lstm16", lambda d: LSTMClassifier(d, hidden=16)),
}
H_GRID = np.linspace(-6, 40, 185)


def sequences_of(part: pd.DataFrame, onsets: dict):
    for seq, g in part.groupby("seq", sort=True):
        g = g.sort_values("frame")
        yield seq, g["frame"].to_numpy(), g["score"].to_numpy(), onsets.get(seq)


def normal_frames(part: pd.DataFrame, onsets: dict) -> int:
    onset = part["seq"].map(onsets)
    return int((onset.isna() | (part["frame"] < onset)).sum())


def evaluate_scores(scores: pd.DataFrame, onsets: dict, kappa: float) -> list[dict]:
    rows = []
    rules = {
        "threshold": (threshold_alarms, {}),
        "cusum": (cusum_alarms, {"kappa": kappa}),
    }
    for (model, seed), sg in scores.groupby(["model", "seed"]):
        for rule_name, (rule, kw) in rules.items():
            outcomes, n_normal, hs = [], 0, []
            for fold, fg in sg.groupby("fold"):
                val, test = fg[fg["split"] == "val"], fg[fg["split"] == "test"]
                h = select_h(list(sequences_of(val, onsets)), rule, H_GRID, **kw)
                hs.append(h)
                outcomes += evaluate_rule(list(sequences_of(test, onsets)), rule, h, **kw)
                n_normal += normal_frames(test, onsets)
            rows.append(dict(model=model, seed=int(seed), rule=rule_name, h_per_fold=hs,
                             **summarize_outcomes(outcomes, n_normal)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=ROOT / "data" / "features.parquet")
    parser.add_argument("--data-dir", default=ROOT / "data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=["ltc24"])
    parser.add_argument("--labels", nargs="+", default=["lying", "fall"])
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--out-dir", default=ROOT / "results" / "ch3")
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(args.features), args.data_dir)
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for label in args.labels:
        path = out / f"scores_{label}.parquet"
        if path.exists():
            scores = pd.read_parquet(path)
        else:
            _, scores = run_nested_cv(df, [SPECS[m] for m in args.models], seeds=args.seeds, label_col=label)
            scores.to_parquet(path, index=False)
        for row in evaluate_scores(scores, onsets, args.kappa):
            results.append(dict(label=label, **row))

    with open(out / "event_level.json", "w") as f:
        json.dump(dict(kappa=args.kappa, seeds=args.seeds, results=results), f, indent=2)

    r = pd.DataFrame(results)
    cols = ["detection_rate", "adl_false_alarm_seqs", "fall_early_alarm_seqs", "false_alarms_per_hour",
            "delay_median", "delay_mean", "delay_max"]
    print(r.groupby(["label", "model", "rule"])[cols].agg(["mean", "std"]).round(2).to_string())
    print(f"\nSaved {out / 'event_level.json'}")


if __name__ == "__main__":
    main()
