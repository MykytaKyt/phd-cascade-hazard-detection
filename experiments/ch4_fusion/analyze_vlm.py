#!/usr/bin/env python3
"""Chapter 4: kinematic-semantic fusion on URFD from cached VLM responses.

Event level (all seeds): every alarm episode of the kinematic rule is kept or
suppressed after fusing
  m_Y: p (1 - u_Y), (1 - p)(1 - u_Y), u_Y = alpha H(p) + (1 - alpha)(1 - c),
       c = mean keypoint confidence at the alarm frame (0 if no pose);
  m_V: p_V (1 - u_V), (1 - p_V)(1 - u_V), u_V from the VLM;
with Dempster's or Yager's rule; the alarm is kept iff BetP(fall) >= beta.
Baselines: kinematics only (all episodes kept) and VLM only (p_V >= 0.5).

Frame level (seed 0): frames in the routing region are decided by the fused
argmax, the rest by p >= 1/2; precision, recall, F1 and misses are reported.

Writes results/ch4/fusion_events.json and results/ch4/fusion_frames.json.

Usage:
    python experiments/ch4_fusion/analyze_vlm.py --responses results/vlm/urfd_responses.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.fusion.dst import dempster, pignistic, simple_bba, yager
from thesis.kinematic.labels import add_labels
from thesis.routing.region import binary_entropy
from thesis.vlm.requests import calibrate_folds, request_id

ROOT = Path(__file__).resolve().parents[2]
FRAME = ["fall", "normal"]
BETAS = (0.5, 0.3, 0.2, 0.1)
ALPHAS = (0.0, 0.5, 1.0)


def fused_fall_prob(p: float, conf: float, p_v: float, u_v: float, alpha: float, rule: str) -> float:
    u_y = float(np.clip(alpha * binary_entropy(np.array([p]))[0] + (1 - alpha) * (1 - conf), 0, 1))
    m_y = simple_bba({"fall": p, "normal": 1 - p}, u_y, FRAME)
    m_v = simple_bba({"fall": p_v, "normal": 1 - p_v}, u_v, FRAME)
    if rule == "dempster":
        m, _ = dempster(m_y, m_v)
    else:
        m, _ = yager(m_y, m_v, frame=FRAME)
    return pignistic(m, FRAME)["fall"]


def load_responses(path: Path) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                out[r["id"]] = r
    return out


def event_outcomes(ev: pd.DataFrame, keep: np.ndarray, onsets: dict, all_seqs: list[str]) -> dict:
    ev = ev.assign(keep=keep)
    falls = [s for s in all_seqs if s in onsets]
    adls = [s for s in all_seqs if s not in onsets]
    kept = ev[ev["keep"]]
    onset = kept["seq"].map(onsets)
    late = kept[onset.notna() & (kept["alarm_frame"] >= onset)]
    false = kept[onset.isna() | (kept["alarm_frame"] < onset)]
    return dict(
        detection_rate=len(set(late["seq"])) / len(falls),
        false_alarm_episodes=len(false),
        adl_with_alarm=len(set(false["seq"]) & set(adls)),
        n_adl=len(adls),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", default=ROOT / "results" / "vlm" / "urfd_responses.jsonl")
    parser.add_argument("--label", default="fall")
    parser.add_argument("--model", default="ltc24")
    parser.add_argument("--tau-n", type=float, default=0.146)
    parser.add_argument("--tau-a", type=float, default=0.3)
    parser.add_argument("--out-dir", default=ROOT / "results" / "ch4")
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    conf = df.set_index(["seq", "frame"])["mean_conf"].fillna(0.0).to_dict()
    lying_adl = {s for s, g in df.groupby("seq") if s.startswith("adl") and g["lying"].any()}
    all_seqs = sorted(df["seq"].unique())
    resp = load_responses(Path(args.responses))
    events = pd.read_parquet(ROOT / "results" / "vlm" / "urfd_events.parquet")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- event level
    from thesis.vlm.requests import clip_frames
    frames_of = {s: g["frame"].to_numpy() for s, g in df.groupby("seq")}
    rows, missing = [], 0
    for variant in ("clip6", "frame"):
        def rid(e):
            fr = clip_frames(e.alarm_frame, frames_of[e.seq]) if variant == "clip6" else [e.alarm_frame]
            return request_id(e.seq, fr, variant)
        ids = [rid(e) for e in events.itertuples()]
        have = np.array([i in resp for i in ids])
        missing += int((~have).sum())
        pv = np.array([resp[i]["p_anomaly"] if i in resp else np.nan for i in ids])
        uv = np.array([resp[i]["uncertainty"] if i in resp else np.nan for i in ids])
        c = np.array([conf.get((e.seq, e.alarm_frame), 0.0) for e in events.itertuples()])
        for seed, idx in events.groupby("seed").indices.items():
            ev = events.iloc[idx]
            base = dict(seed=int(seed), variant=variant)
            rows.append(dict(base, method="kinematics", **event_outcomes(ev, np.ones(len(ev), bool), onsets, all_seqs)))
            ok = have[idx]
            rows.append(dict(base, method="vlm_only",
                             **event_outcomes(ev, ~ok | (pv[idx] >= 0.5), onsets, all_seqs)))
            for rule in ("dempster", "yager"):
                for alpha in ALPHAS:
                    bet = np.array([fused_fall_prob(e.p, c[i], pv[i], uv[i], alpha, rule) if have[i] else 1.0
                                    for i, e in zip(idx, ev.itertuples())])
                    for beta in BETAS:
                        rows.append(dict(base, method=f"{rule}", alpha=alpha, beta=beta,
                                         **event_outcomes(ev, bet >= beta, onsets, all_seqs)))
    ev_res = pd.DataFrame(rows)
    ev_res.to_json(out / "fusion_events.json", orient="records", indent=2)
    summary = ev_res.groupby(["variant", "method", "alpha", "beta"], dropna=False)[
        ["detection_rate", "false_alarm_episodes", "adl_with_alarm"]].mean().round(3)
    print("=== event level (mean over seeds) ===")
    print(summary.to_string())

    # ---- frame level, seed 0
    scores = pd.read_parquet(ROOT / "results" / "ch3" / f"scores_{args.label}.parquet")
    scores = scores[(scores["model"] == args.model) & (scores["seed"] == 0)]
    test = pd.concat([t for _, t in calibrate_folds(scores)], ignore_index=True)
    test["routed"] = (test["p"] > args.tau_n) & (test["p"] < 1 - args.tau_a)
    ids = [request_id(s, [int(f)], "frame") for s, f in zip(test["seq"], test["frame"])]
    test["has_vlm"] = [i in resp for i in ids]
    test["p_v"] = [resp[i]["p_anomaly"] if i in resp else np.nan for i in ids]
    test["u_v"] = [resp[i]["uncertainty"] if i in resp else np.nan for i in ids]
    test["conf"] = [conf.get((s, int(f)), 0.0) for s, f in zip(test["seq"], test["frame"])]
    y = test["label"].to_numpy()
    frame_rows = []
    for rule in ("dempster", "yager"):
        for alpha in ALPHAS:
            pred = (test["p"] >= 0.5).to_numpy()
            sel = (test["routed"] & test["has_vlm"]).to_numpy()
            bet = np.array([fused_fall_prob(r.p, r.conf, r.p_v, r.u_v, alpha, rule)
                            for r in test[sel].itertuples()])
            pred_f = pred.copy()
            pred_f[sel] = bet >= 0.5
            for name, pr in (("ltc", pred), ("fused", pred_f)):
                tp, fp, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
                p_, r_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                frame_rows.append(dict(rule=rule, alpha=alpha, system=name, precision=p_, recall=r_,
                                       f1=2 * p_ * r_ / max(p_ + r_, 1e-12), fn=fn, fp=fp,
                                       n_vlm=int(sel.sum()), routed=int(test["routed"].sum()),
                                       lying_adl_fp=int((pr & (y == 0) & test["seq"].isin(lying_adl)).sum())))
    fr = pd.DataFrame(frame_rows).drop_duplicates(subset=["system", "rule", "alpha"])
    fr.to_json(out / "fusion_frames.json", orient="records", indent=2)
    print("\n=== frame level, seed 0 ===")
    print(fr.round(3).to_string(index=False))
    if missing:
        print(f"\nNote: {missing} event requests have no response yet; their alarms are kept.")


if __name__ == "__main__":
    main()
