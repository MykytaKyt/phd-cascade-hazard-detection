#!/usr/bin/env python3
"""Chapter 3: kinematic classifiers on URFD with the decision threshold
selected on held-out validation sequences (see thesis.evaluation).

Writes:
  results/ch3/kinematic_cv.json     per-run records and summary
  results/ch3/scores.parquet        per-window validation/test scores (used by chapters 2 and 4)

Usage:
    python experiments/ch3_kinematic/run_cv.py --seeds 0 1 2 3 4
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from thesis.evaluation import ModelSpec, run_nested_cv, summarize
from thesis.kinematic.models import LSTMClassifier, LTCClassifier

ROOT = Path(__file__).resolve().parents[2]

SPECS = [
    ModelSpec("threshold", None),
    ModelSpec("lstm32", lambda d: LSTMClassifier(d, hidden=32)),
    ModelSpec("lstm16", lambda d: LSTMClassifier(d, hidden=16)),
    ModelSpec("ltc24", lambda d: LTCClassifier(d, hidden=24)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=ROOT / "data" / "features.parquet")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=[s.name for s in SPECS])
    parser.add_argument("--out-dir", default=ROOT / "results" / "ch3")
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    df = pd.read_parquet(args.features)
    specs = [s for s in SPECS if s.name in args.models]
    records, scores = run_nested_cv(df, specs, seeds=args.seeds)
    summary = summarize(records)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "scores.parquet", index=False)
    with open(out / "kinematic_cv.json", "w") as f:
        json.dump(dict(
            protocol="5-fold GroupKFold over sequences; 20% of training sequences held out "
                     "(stratified fall/ADL) for threshold selection at Recall>=0.95",
            seeds=args.seeds,
            summary=summary.reset_index().to_dict(orient="records"),
            runs=records,
        ), f, indent=2)

    pd.set_option("display.width", 200)
    view = summary[["params", "precision_mean", "precision_std", "recall_mean", "recall_std",
                    "f1_mean", "f1_std", "f1_seed_std", "oracle_f1_mean", "oracle_f1_std"]] * 1
    for c in view.columns:
        if c != "params":
            view[c] = (view[c] * 100).round(1)
    print(view.to_string())
    print(f"\nSaved {out / 'kinematic_cv.json'} and {out / 'scores.parquet'}")


if __name__ == "__main__":
    main()
