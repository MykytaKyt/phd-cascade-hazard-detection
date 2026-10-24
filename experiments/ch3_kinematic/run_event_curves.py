#!/usr/bin/env python3
"""Event-level operating characteristics and the CUSUM false-alarm bound.

The classifier logits are turned into per-frame log-likelihood ratios
z_t = s_t / T - logit(pi), with temperature T fitted on the validation
logits of the fold and pi the validation share of positive frames. For
each alarm rule the threshold h is swept on the pooled test sequences
(no selection), giving detection rate, false-alarm episodes per hour and
detection delay as functions of h, averaged over seeds.

The false-alarm bound of statement B2: if z_t are conditional LLRs and
kappa >= 0, P(CUSUM alarm within T normal frames) <= T exp(-h). The
observed share of ADL sequences with an alarm is compared with the mean
bound min(1, T_i exp(-h)).

Writes figures/ch3_event_tradeoff.png and results/ch3/event_curves.json.

Usage:
    python experiments/ch3_kinematic/run_event_curves.py --label fall
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logit  # noqa: E402

from thesis.changepoint.cusum import cusum_alarms, evaluate_rule, summarize_outcomes, threshold_alarms  # noqa: E402
from thesis.kinematic.labels import add_labels  # noqa: E402
from thesis.routing.calibration import fit_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RULES = {"threshold": threshold_alarms, "cusum": cusum_alarms}
H_GRID = {"threshold": np.linspace(-4, 8, 61), "cusum": np.geomspace(0.5, 400, 60)}


def llr_test(scores: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, fg in scores.groupby("fold"):
        val = fg[fg["split"] == "val"]
        t = fit_temperature(val["score"].to_numpy(), val["label"].to_numpy())
        prior = float(np.clip(val["label"].mean(), 1e-4, 1 - 1e-4))
        test = fg[fg["split"] == "test"].copy()
        test["z"] = test["score"] / t - logit(prior)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def sequences(test: pd.DataFrame, onsets: dict):
    for seq, g in test.groupby("seq", sort=True):
        g = g.sort_values("frame")
        yield seq, g["frame"].to_numpy(), g["z"].to_numpy(), onsets.get(seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="fall")
    parser.add_argument("--model", default="ltc24")
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    scores = pd.read_parquet(ROOT / "results" / "ch3" / f"scores_{args.label}.parquet")
    scores = scores[scores["model"] == args.model]

    rows = []
    for seed, sg in scores.groupby("seed"):
        test = llr_test(sg)
        seqs = list(sequences(test, onsets))
        onset_s = test["seq"].map(onsets)
        n_normal = int((onset_s.isna() | (test["frame"] < onset_s)).sum())
        adl_len = {s: len(f) for s, f, _, o in seqs if o is None}
        for rule_name, rule in RULES.items():
            for h in H_GRID[rule_name]:
                out = evaluate_rule(seqs, rule, h)
                row = dict(seed=int(seed), rule=rule_name, h=float(h), **summarize_outcomes(out, n_normal))
                if rule_name == "cusum":
                    row["adl_alarm_share"] = float(np.mean([o.false_alarm for o in out if not o.is_fall]))
                    row["adl_alarm_bound"] = float(np.mean([min(1.0, n * np.exp(-h)) for n in adl_len.values()]))
                rows.append(row)

    r = pd.DataFrame(rows)
    (ROOT / "results" / "ch3").mkdir(parents=True, exist_ok=True)
    r.to_json(ROOT / "results" / "ch3" / f"event_curves_{args.label}.json", orient="records", indent=2)
    m = r.groupby(["rule", "h"]).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for rule_name, style in (("threshold", "--"), ("cusum", "-")):
        g = m[m["rule"] == rule_name].sort_values("false_alarms_per_hour")
        axes[0].plot(g["false_alarms_per_hour"], 100 * g["detection_rate"], style, label=rule_name)
        axes[1].plot(g["false_alarms_per_hour"], g["delay_median"] / 30.0, style, label=rule_name)
    axes[0].set_ylabel("Detected falls, %")
    axes[1].set_ylabel("Median detection delay, s")
    for ax in axes:
        ax.set_xlabel("False-alarm episodes per hour")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Event-level operating characteristics (labels: {args.label})")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"ch3_event_tradeoff_{args.label}.png", dpi=200)

    c = m[m["rule"] == "cusum"]
    sel = c.iloc[np.searchsorted(c["h"].to_numpy(), [2, 5, 10, 20, 50, 100, 200, 400]).clip(max=len(c) - 1)]
    print(sel[["h", "detection_rate", "false_alarms_per_hour", "delay_median", "adl_alarm_share", "adl_alarm_bound"]]
          .round(3).to_string(index=False))
    for target in (0.9, 0.8):
        for rule_name in ("threshold", "cusum"):
            g = m[(m["rule"] == rule_name) & (m["detection_rate"] >= target)]
            best = g.sort_values("false_alarms_per_hour").iloc[0] if len(g) else None
            if best is not None:
                print(f"detection >= {target}: {rule_name:9s} FA/h {best['false_alarms_per_hour']:.1f}, "
                      f"median delay {best['delay_median']:.1f} frames (h={best['h']:.2f})")


if __name__ == "__main__":
    main()
