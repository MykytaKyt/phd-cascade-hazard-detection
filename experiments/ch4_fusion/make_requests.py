#!/usr/bin/env python3
"""Chapter 4: build the VLM request list for URFD from the kinematic scores.

  event requests  -- every alarm episode of the frame-wise rule (all seeds) is verified
                     on a clip of 6 frames (alarm - 40 ... alarm + 10, 1/3 s apart) and on
                     the single alarm frame;
  frame requests  -- seed 0: every frame in the routing region (tau_N, 1 - tau_A) as a
                     single frame, for comparison with the earlier publications.

Writes results/vlm/urfd_events.parquet (alarm episodes per seed and fold) and
results/vlm/urfd_requests.jsonl (deduplicated requests for query_vlm.py).

Usage:
    python experiments/ch4_fusion/make_requests.py --tau-n 0.146 --tau-a 0.3
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from thesis.kinematic.labels import add_labels
from thesis.vlm.requests import alarm_events, calibrate_folds, clip_frames, request_id

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="fall")
    parser.add_argument("--model", default="ltc24")
    parser.add_argument("--tau-n", type=float, default=0.146)
    parser.add_argument("--tau-a", type=float, default=0.3)
    parser.add_argument("--frame-seed", type=int, default=0)
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    frames_of = {s: g["frame"].to_numpy() for s, g in df.groupby("seq")}
    scores = pd.read_parquet(ROOT / "results" / "ch3" / f"scores_{args.label}.parquet")
    scores = scores[scores["model"] == args.model]

    out = ROOT / "results" / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    events = alarm_events(scores, onsets)
    events.to_parquet(out / "urfd_events.parquet", index=False)

    requests = {}
    for e in events.itertuples():
        for variant, frames in (("clip6", clip_frames(e.alarm_frame, frames_of[e.seq])), ("frame", [e.alarm_frame])):
            rid = request_id(e.seq, frames, variant)
            requests[rid] = dict(id=rid, seq=e.seq, frames=frames, variant=variant, kind="event")
    n_event = len(requests)

    for _, test in calibrate_folds(scores[scores["seed"] == args.frame_seed]):
        routed = test[(test["p"] > args.tau_n) & (test["p"] < 1 - args.tau_a)]
        for r in routed.itertuples():
            rid = request_id(r.seq, [int(r.frame)], "frame")
            requests.setdefault(rid, dict(id=rid, seq=r.seq, frames=[int(r.frame)], variant="frame", kind="frame"))

    with open(out / "urfd_requests.jsonl", "w") as f:
        for req in requests.values():
            f.write(json.dumps(req) + "\n")

    per_seed = events.groupby("seed").size()
    print(f"alarm episodes per seed: {per_seed.to_dict()}")
    print(f"requests: {len(requests)} total, {n_event} for events "
          f"({sum(r['variant'] == 'clip6' for r in requests.values())} clips), "
          f"{len(requests) - n_event} additional single frames (seed {args.frame_seed}, "
          f"tau_N={args.tau_n}, tau_A={args.tau_a})")
    print(f"Saved {out / 'urfd_events.parquet'} and {out / 'urfd_requests.jsonl'}")


if __name__ == "__main__":
    main()
