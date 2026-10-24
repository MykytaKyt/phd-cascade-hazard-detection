#!/usr/bin/env python3
"""Chapter 3: ablations of the LTC filter (feature groups, window length, hidden size)
with the nested protocol, for both label variants.

Feature groups follow the earlier publication (PCI) plus a biomechanics-only set.
For every configuration: frame-level metrics at the validation-selected threshold
and event-level outcomes of the frame-wise rule.

Writes results/ch3/ablation.json.

Usage:
    python experiments/ch3_kinematic/run_ablation.py --seeds 0 1 2 --labels fall lying
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from thesis.evaluation import FEATURE_COLUMNS, ModelSpec, run_nested_cv, summarize
from thesis.kinematic.labels import add_labels
from thesis.kinematic.models import LTCClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_event import evaluate_scores  # noqa: E402

BBOX = ["aspect", "bbox_h"]
FEATURE_SETS = {
    "all": FEATURE_COLUMNS,
    "no_angle_mos_bos": [c for c in FEATURE_COLUMNS if c not in ("angle", "mos", "bos_area")],
    "no_mos": [c for c in FEATURE_COLUMNS if c != "mos"],
    "no_dynamics": [c for c in FEATURE_COLUMNS if c not in ("vy", "vx", "ay")],
    "bbox_only": BBOX,
    "biomech_only": ["com_y", "angle", "mos", "bos_area", "vy", "vx", "ay"],
}
CONFIGS = (
    [dict(features=k, window=8, hidden=24) for k in FEATURE_SETS]
    + [dict(features="all", window=w, hidden=24) for w in (4, 16, 24)]
    + [dict(features="all", window=8, hidden=h) for h in (8, 16, 32)]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--labels", nargs="+", default=["fall", "lying"])
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    rows = []
    for label in args.labels:
        for cfg in CONFIGS:
            name = f"ltc{cfg['hidden']}_w{cfg['window']}_{cfg['features']}"
            spec = ModelSpec(name, lambda d, h=cfg["hidden"]: LTCClassifier(d, hidden=h))
            records, scores = run_nested_cv(df, [spec], seeds=args.seeds, window=cfg["window"],
                                            label_col=label, feature_columns=FEATURE_SETS[cfg["features"]])
            frame = summarize(records).iloc[0]
            ev = pd.DataFrame(evaluate_scores(scores, onsets, kappa=0.0))
            ev = ev[ev["rule"] == "threshold"].mean(numeric_only=True)
            row = dict(label=label, **cfg, params=int(frame["params"]),
                       f1=frame["f1_mean"], f1_std=frame["f1_std"], recall=frame["recall_mean"],
                       precision=frame["precision_mean"],
                       detection_rate=ev["detection_rate"], adl_false_alarm_seqs=ev["adl_false_alarm_seqs"],
                       delay_median=ev["delay_median"])
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    with open(ROOT / "results" / "ch3" / "ablation.json", "w") as f:
        json.dump(dict(seeds=args.seeds, rows=rows), f, indent=2, default=float)
    print("Saved results/ch3/ablation.json")


if __name__ == "__main__":
    main()
