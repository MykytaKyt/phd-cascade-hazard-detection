#!/usr/bin/env python3
"""Chapter 4: distribution statistics of the VLM answers on the alarm episodes.

For one responses file (all distinct alarm episodes of the five seeds, variants
"clip6" and "frame"): distribution of p_V and u_V by episode type (table 4.4),
discreteness, agreement between the variants, informativeness of u_V for the
model's own errors, stability on near-identical episodes, dependence of the clip
answer on the alarm delay (section 4.6.7), the best threshold operating point
(section 4.7.3), the answers on the routed frames of seed 0 (table 4.5), keyword
categories of the explanations (table 4.6) and the format and latency of the answers.

Writes results/ch4/gemma-4-E4B/clip_stats.json (or --out).

Usage:
    python experiments/ch4_fusion/clip_stats.py --responses results/vlm/urfd_responses_<model>.jsonl \
        --out results/ch4/<model>/clip_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from thesis.kinematic.labels import add_labels
from thesis.vlm.requests import calibrate_folds, clip_frames, request_id

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("clip6", "frame")
# keyword categories of the explanations, checked in this order (first match wins)
CATEGORIES_FALL = {
    "transition": ("transition", "from standing", "from an upright", "last frames", "later frames", "final frames",
                   "lower frames", "sequence shows"),
    "posture": ("posture", "position", "on the floor", "on the ground", "lying", "fallen"),
    "balance": ("balance", "falling", "fall", "collaps", "tumble", "uncontrolled"),
}
CATEGORIES_CONTROLLED = {
    "bending": ("bend", "squat", "crouch", "pick", "reach"),
    "sit_lie": ("sit", "lying down", "lie down", "lay down", "resting", "bed", "couch", "sofa"),
    "exercise": ("exercis", "sport", "athlet", "danc", "stretch", "acrobat", "yoga", "run", "jump", "gymnast"),
    "controlled": ("control", "walk", "everyday", "intentional", "normal", "deliberate", "motion", "moving"),
}
AMBIGUITY = ("ambiguous", "unclear", "difficult", "uncertain", "hard to", "not clear")
PROMPT_EXAMPLES = ("sitting", "sit down", "sits", "lying down", "lie down", "exercis", "bending", "bend")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", default=ROOT / "results" / "vlm" / "urfd_responses_gemma-4-E4B.jsonl")
    parser.add_argument("--out", default=ROOT / "results" / "ch4" / "gemma-4-E4B" / "clip_stats.json")
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    lying_adl = {s for s, g in df.groupby("seq") if s.startswith("adl") and g["lying"].any()}
    frames_of = {s: g["frame"].to_numpy() for s, g in df.groupby("seq")}
    resp = {}
    with open(args.responses) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                resp[r["id"]] = r

    ev = pd.read_parquet(ROOT / "results" / "vlm" / "urfd_events.parquet")
    ev = ev.drop_duplicates(["seq", "alarm_frame"]).reset_index(drop=True)
    true = np.array([s in onsets and a >= onsets[s] for s, a in zip(ev["seq"], ev["alarm_frame"])])
    ev["type"] = np.where(true, "true_alarm", np.where(ev["seq"].isin(lying_adl), "false_lying_adl",
                          np.where(ev["seq"].str.startswith("adl"), "false_other_adl", "false_before_onset")))
    ev["true"] = true
    for v in VARIANTS:
        fr = [clip_frames(a, frames_of[s]) if v == "clip6" else [a] for s, a in zip(ev["seq"], ev["alarm_frame"])]
        ids = [request_id(s, f, v) for s, f in zip(ev["seq"], fr)]
        ev[f"p_{v}"] = [resp[i]["p_anomaly"] for i in ids]
        ev[f"u_{v}"] = [resp[i]["uncertainty"] for i in ids]
        if v == "clip6":
            ev["n_clip_frames"] = [len(f) for f in fr]
    out = dict(n_episodes=len(ev), clip_frames_count=ev["n_clip_frames"].value_counts().sort_index().to_dict())

    for v in VARIANTS:
        p, u = ev[f"p_{v}"].to_numpy(), ev[f"u_{v}"].to_numpy()
        d = dict(by_type={})
        for t, g in ev.groupby("type"):
            q = g[f"p_{v}"].quantile([0.25, 0.5, 0.75]).to_list()
            d["by_type"][t] = dict(n=len(g), median=q[1], q1=q[0], q3=q[2], values=sorted(g[f"p_{v}"].tolist())
                                   if len(g) <= 3 else None, frac_ge_05=float((g[f"p_{v}"] >= 0.5).mean()),
                                   mean_u=float(g[f"u_{v}"].mean()), mean_p=float(g[f"p_{v}"].mean()))
        vals, counts = np.unique(p, return_counts=True)
        d["distinct_values"] = int(len(vals))
        d["value_counts"] = {str(k): int(c) for k, c in zip(vals, counts)}
        err = (p >= 0.5) != ev["true"].to_numpy()
        d["errors"] = int(err.sum())
        d["auc_u_for_errors"] = float(roc_auc_score(err, u))
        d["mean_u_error"], d["mean_u_correct"] = float(u[err].mean()), float(u[~err].mean())
        d["auc_u_true_vs_false"] = float(roc_auc_score(ev["true"], u))
        d["spearman_u_vs_dist"] = float(spearmanr(u, np.abs(p - 0.5)).statistic)
        d["true_alarms_rejected"] = float((p[ev["true"]] < 0.5).mean())
        fpr, tpr, thr = roc_curve(ev["true"], p)
        k = int(np.argmax(tpr - fpr))
        d["max_tpr_minus_fpr"] = dict(value=float(tpr[k] - fpr[k]), threshold=float(thr[k]),
                                      tpr=float(tpr[k]), fpr=float(fpr[k]))
        hi = tpr >= 0.95
        d["min_fpr_at_tpr_095"] = float(fpr[hi].min())
        out[v] = d

    out["spearman_clip_vs_frame"] = float(spearmanr(ev["p_clip6"], ev["p_frame"]).statistic)
    out["agree_at_05"] = int(((ev["p_clip6"] >= 0.5) == (ev["p_frame"] >= 0.5)).sum())

    # near-identical episodes: same sequence, alarm frames at most 3 frames apart
    pairs = []
    for s, g in ev.groupby("seq"):
        g = g.sort_values("alarm_frame")
        a = g.to_dict("records")
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if a[j]["alarm_frame"] - a[i]["alarm_frame"] <= 3:
                    pairs.append((s, a[i], a[j]))
    near = dict(n_pairs=len(pairs), n_seqs=len({s for s, _, _ in pairs}))
    for v in VARIANTS:
        dis = sum((x[f"p_{v}"] >= 0.5) != (y[f"p_{v}"] >= 0.5) for _, x, y in pairs)
        near[v] = dict(disagree=int(dis), mean_abs_diff=float(np.mean([abs(x[f"p_{v}"] - y[f"p_{v}"])
                                                                          for _, x, y in pairs])))
    out["near_pairs"] = near
    f24 = ev[ev["seq"] == "fall-24"].sort_values("alarm_frame")
    out["fall24_clip"] = {int(a): float(p) for a, p in zip(f24["alarm_frame"], f24["p_clip6"])}

    # section 4.6.7: clip answer vs delay of the alarm after the fall onset (true alarms)
    t = ev[ev["true"]].copy()
    t["delay"] = t["alarm_frame"] - t["seq"].map(onsets)
    late = t["delay"] >= 20
    out["delay"] = dict(median=float(t["delay"].median()), n_late=int(late.sum()), n_early=int((~late).sum()),
                        frac_ge_05_late=float((t.loc[late, "p_clip6"] >= 0.5).mean()),
                        frac_ge_05_early=float((t.loc[~late, "p_clip6"] >= 0.5).mean()))

    # table 4.6: keyword categories of the explanations
    reasons = {}
    for v in VARIANTS:
        fr = [clip_frames(a, frames_of[s]) if v == "clip6" else [a] for s, a in zip(ev["seq"], ev["alarm_frame"])]
        ev[f"reason_{v}"] = [resp[request_id(s, f, v)]["reason"].lower() for s, f in zip(ev["seq"], fr)]
        cat = []
        for r, p in zip(ev[f"reason_{v}"], ev[f"p_{v}"]):
            cats = CATEGORIES_FALL if p >= 0.5 else CATEGORIES_CONTROLLED
            cat.append(next((("fall:" if p >= 0.5 else "controlled:") + k for k, words in cats.items()
                             if any(w in r for w in words)), "fall:other" if p >= 0.5 else "controlled:other"))
        ev[f"cat_{v}"] = cat
        table = {}
        for name, g in (("true", ev[ev["true"]]), ("false", ev[~ev["true"]])):
            table[name] = g[f"cat_{v}"].value_counts().to_dict()
            table[f"ambiguous_{name}"] = int(g[f"reason_{v}"].apply(lambda r: any(w in r for w in AMBIGUITY)).sum())
        ctrl = ev[ev[f"p_{v}"] < 0.5]
        table["controlled_n"] = len(ctrl)
        table["controlled_with_prompt_examples"] = int(
            ctrl[f"reason_{v}"].apply(lambda r: any(w in r for w in PROMPT_EXAMPLES)).sum())
        fall = ev[ev[f"p_{v}"] >= 0.5]
        table["fall_n"] = len(fall)
        table["fall_posture"] = int((fall[f"cat_{v}"] == "fall:posture").sum())
        lad = ev[ev["type"] == "false_lying_adl"]
        table["lying_adl_floor_or_ground"] = int(lad[f"reason_{v}"].str.contains("floor|ground").sum())
        reasons[v] = table
    out["reasons"] = reasons

    # format and latency of all answers in the file
    allr = list(resp.values())
    out["answers"] = dict(
        n=len(allr), strict_json=sum(r["raw"].strip().startswith("{") for r in allr),
        reason_words_median=float(np.median([len(r["reason"].split()) for r in allr])),
        latency_median_by_variant={f"{r_kind}/{var}": float(np.median(
            [r["latency_s"] for r in allr if r["kind"] == r_kind and r["variant"] == var]))
            for r_kind, var in (("event", "clip6"), ("event", "frame"), ("frame", "frame"))})

    # table 4.5: answers on the routed frames of seed 0 (frame variant)
    scores = pd.read_parquet(ROOT / "results" / "ch3" / "scores_fall.parquet")
    scores = scores[(scores["model"] == "ltc24") & (scores["seed"] == 0)]
    test = pd.concat([t for _, t in calibrate_folds(scores)], ignore_index=True)
    test = test[(test["p"] > 0.146) & (test["p"] < 1 - 0.3)].copy()
    ids = [request_id(s, [int(f)], "frame") for s, f in zip(test["seq"], test["frame"])]
    test["p_v"] = [resp[i]["p_anomaly"] for i in ids]
    test["u_v"] = [resp[i]["uncertainty"] for i in ids]
    is_fall_seq = test["seq"].str.startswith("fall")
    group = np.where(test["label"] == 1, "fall_frames", np.where(is_fall_seq, "fall_seq_negative",
                     np.where(test["seq"].isin(lying_adl), "lying_adl", "other_adl")))
    test["group"] = group
    routed = dict(n=len(test), groups={g: dict(n=len(x), mean_p=float(x["p_v"].mean()),
                                               frac_ge_05=float((x["p_v"] >= 0.5).mean()),
                                               mean_u=float(x["u_v"].mean()))
                                       for g, x in test.groupby("group")})
    routed["auc_p_v"] = float(roc_auc_score(test["label"], test["p_v"]))
    routed["auc_p_kinematic"] = float(roc_auc_score(test["label"], test["p"]))
    fl = test[test["group"].isin(["fall_frames", "lying_adl"])]
    routed["auc_fall_vs_lying_adl"] = float(roc_auc_score(fl["group"] == "fall_frames", fl["p_v"]))
    err = (test["p_v"] >= 0.5) != (test["label"] == 1)
    routed["auc_u_for_errors"] = float(roc_auc_score(err, test["u_v"]))
    out["routed_frames"] = routed

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
