#!/usr/bin/env python3
"""Chapter 5: zero-shot level of the cascade on UCF-Crime (CPU, from clip_scores.parquet).

For several random calibration/evaluation splits of the test videos (20 % / 80 %,
stratified anomalous/normal):
  - frame-level ROC AUC of the zero-shot anomaly score 1 - P(Normal);
  - temperature calibration of the anomaly logit on the calibration videos, ECE;
  - normal-side routing threshold tau_N by conformal risk control on the calibration
    videos (per-video miss fraction), evaluated on the evaluation videos, compared
    with the plug-in choice; tau_A = 1/2;
  - text-query retrieval (AP, precision@K per class) on all videos.
Also builds the VLM requests for the routed segments of split 0.

Writes results/ch5/zeroshot.json and results/vlm/ucf_requests.jsonl.

Usage:
    python experiments/ch5_ucfcrime/run_zeroshot.py --alpha 0.05 0.1 --splits 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.routing.calibration import calibrate, ece, fit_temperature
from thesis.routing.conformal import crc_threshold, empirical_threshold, sequence_miss_fraction
from thesis.vad.analysis import (
    anomaly_logit,
    frame_auc,
    positive_probs_by_video,
    retrieval_metrics,
    routed_segments,
    segment_clip_frames,
    split_videos,
)

ROOT = Path(__file__).resolve().parents[2]
TAU_GRID = np.concatenate([np.geomspace(1e-5, 0.01, 40), np.linspace(0.01, 0.5, 99)])


def run(df: pd.DataFrame, alphas, n_splits: int, cal_fraction: float):
    z = anomaly_logit(df)
    y = df["label"].to_numpy()
    rows, split0 = [], None
    for split in range(n_splits):
        cal_v, eval_v = split_videos(df, cal_fraction, np.random.default_rng(split))
        cal, ev = df["video"].isin(cal_v).to_numpy(), df["video"].isin(eval_v).to_numpy()
        t = fit_temperature(z[cal], y[cal])
        p = calibrate(z, t)
        base = dict(split=split, temperature=t, auc=frame_auc(df[ev], z[ev]),
                    ece_raw=ece(1 / (1 + np.exp(-z[ev])), y[ev]), ece_cal=ece(p[ev], y[ev]),
                    n_cal_anomalous=int(df[cal].loc[df["label"] == 1, "video"].nunique()))
        cal_pos = positive_probs_by_video(df[cal], p[cal])
        ev_pos = positive_probs_by_video(df[ev], p[ev])
        for alpha in alphas:
            for method in ("empirical", "crc"):
                if method == "crc":
                    tau = crc_threshold(cal_pos, alpha, TAU_GRID)
                    feasible = tau is not None
                    tau = 0.0 if tau is None else tau
                else:
                    tau, feasible = empirical_threshold(p[cal], y[cal], 1 - alpha, TAU_GRID), True
                routed = (p[ev] > tau) & (p[ev] < 0.5)
                risk = float(sequence_miss_fraction(ev_pos, tau).mean())
                pos = y[ev] == 1
                rows.append(dict(base, alpha=alpha, method=method, tau=tau, feasible=feasible,
                                 test_risk=risk, risk_le_alpha=risk <= alpha,
                                 recall_bound=float(1 - (pos & (p[ev] <= tau)).sum() / pos.sum()),
                                 routed_frac=float(routed.mean())))
                if split == 0 and method == "crc" and alpha == max(alphas):
                    split0 = (df[ev], p[ev], tau)
    return pd.DataFrame(rows), split0


def build_requests(ev: pd.DataFrame, p: np.ndarray, tau_n: float, tau_a: float, max_requests: int):
    routed = (p > tau_n) & (p < 1 - tau_a)
    segs = routed_segments(ev, routed)
    segs = segs.sort_values("n_sampled", ascending=False).head(max_requests)
    reqs = []
    for s in segs.itertuples():
        frames = segment_clip_frames(s.start, s.end)
        rid = f"{s.video}:{s.start}-{s.end}:clip6"
        reqs.append(dict(id=rid, video=s.video, frames=frames, start=s.start, end=s.end,
                         variant="clip6", kind="ucf_segment"))
    return reqs, len(segs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default=ROOT / "results" / "ch5" / "clip_scores.parquet")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cal-fraction", type=float, default=0.2)
    parser.add_argument("--tau-a", type=float, default=0.3)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--out", default=ROOT / "results" / "ch5" / "zeroshot.json")
    parser.add_argument("--requests-out", default=ROOT / "results" / "vlm" / "ucf_requests.jsonl")
    args = parser.parse_args()

    df = pd.read_parquet(args.scores)
    res, split0 = run(df, args.alpha, args.splits, args.cal_fraction)
    retr = retrieval_metrics(df)

    print(f"videos: {df['video'].nunique()}, sampled frames: {len(df)}")
    print(res.groupby("split")[["auc", "ece_raw", "ece_cal", "n_cal_anomalous"]].first().mean().round(4).to_string())
    print(res.groupby(["alpha", "method"]).agg(
        feasible=("feasible", "mean"), test_risk=("test_risk", "mean"), risk_max=("test_risk", "max"),
        splits_ok=("risk_le_alpha", "mean"), recall_bound=("recall_bound", "mean"),
        routed_pct=("routed_frac", lambda v: 100 * v.mean())).round(4).to_string())
    print(retr.round(3).to_string(index=False))
    print(f"mean AP over classes: {retr['ap'].mean():.3f}")

    reqs, n_segments = build_requests(*split0, args.tau_a, args.max_requests)
    Path(args.requests_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.requests_out, "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(alphas=args.alpha, splits=args.splits, cal_fraction=args.cal_fraction,
                       runs=res.to_dict(orient="records"), retrieval=retr.to_dict(orient="records"),
                       vlm_requests=dict(split=0, tau_n=split0[2], tau_a=args.tau_a,
                                         segments=n_segments, requests=len(reqs))),
                  f, indent=2, default=float)
    print(f"\n{len(reqs)} VLM requests ({n_segments} routed segments) -> {args.requests_out}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
