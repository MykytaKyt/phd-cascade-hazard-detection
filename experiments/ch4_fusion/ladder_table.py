#!/usr/bin/env python3
"""Chapter 4: summary of the VLM size ladder on URFD (event level, clip variant).

For every model directory results/ch4/<tag>/ written by analyze_vlm.py --out-dir:
ROC AUC of p_V for true vs false alarms, the outcome of the fused decision
(Dempster, alpha = 0, beta = 0.3; mean over seeds) and the cost condition (2.15)
in the form of section 4.7.3: with TPR_V = detected / detected_kinematic and
FPR_V = false alarms / false alarms_kinematic, the cascade pays off for a free VLM
call iff C_FN / C_FP < r_max = FA_kin (1 - FPR_V) / (D_kin (1 - TPR_V)), where D_kin
is the number of falls detected by kinematics per seed (r_max = inf if no fall is lost).

Writes results/ch4/ladder.json.

Usage:
    python experiments/ch4_fusion/ladder_table.py gemma-4-E4B qwen3-vl-8b gemma-4-31b qwen3-vl-32b
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_FALLS = 30
METRICS = ("detection_rate", "false_alarm_episodes", "adl_with_alarm")


def mean_outcomes(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["method"], r.get("alpha"), r.get("beta"))].append(r)
    return {k: {m: sum(r[m] for r in v) / len(v) for m in METRICS} for k, v in groups.items()}


def main():
    out = []
    for tag in sys.argv[1:]:
        d = ROOT / "results" / "ch4" / tag
        disc = json.load(open(d / "vlm_discrimination.json"))
        ev = mean_outcomes(json.load(open(d / "fusion_events.json")))
        kin = ev[("clip6", "kinematics", None, None)]
        row = dict(model=tag, auc_clip=disc["clip6"]["auc"], auc_frame=disc["frame"]["auc"])
        for variant in ("clip6", "frame"):
            fused = ev[(variant, "dempster", 0.0, 0.3)]
            tpr = fused["detection_rate"] / kin["detection_rate"]
            fpr = fused["false_alarm_episodes"] / kin["false_alarm_episodes"]
            d_kin = kin["detection_rate"] * N_FALLS
            r_max = (kin["false_alarm_episodes"] * (1 - fpr) / (d_kin * (1 - tpr))) if tpr < 1 else float("inf")
            row[variant] = dict(fused, tpr_v=tpr, fpr_v=fpr, r_max=r_max,
                                vlm_only=ev[(variant, "vlm_only", None, None)])
        out.append(row)
        c = row["clip6"]
        print(f"{tag:14s} AUC clip {row['auc_clip']:.3f} frame {row['auc_frame']:.3f} | fused clip: "
              f"det {c['detection_rate']:.3f} FA {c['false_alarm_episodes']:.1f} ADL {c['adl_with_alarm']:.1f} "
              f"TPR_V {c['tpr_v']:.3f} FPR_V {c['fpr_v']:.3f} r_max {c['r_max']:.1f}")
    with open(ROOT / "results" / "ch4" / "ladder.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
