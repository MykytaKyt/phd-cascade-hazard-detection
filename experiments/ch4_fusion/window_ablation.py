#!/usr/bin/env python3
"""Chapter 4: sensitivity of the clip verification to the clip window.

The base clip covers alarm - 40 ... alarm + 10 frames (6 frames, 1/3 s apart). The
late window covers alarm - 10 ... alarm + 50 with the same number of frames
(6 frames, 12 frames = 0.4 s apart), so it reaches into the lying phase. Both
windows go through the same prompt, grid and fusion; only the frames change.

  --make       writes results/vlm/urfd_requests_window.jsonl (event clips, late window);
  --responses  a responses file with answers to both the base and the late clips of
               one model; writes discrimination and event-level fusion per window.

Usage:
    python experiments/ch4_fusion/window_ablation.py --make
    python experiments/ch4_fusion/window_ablation.py --responses results/vlm/urfd_responses_<model>.jsonl \
        --out results/ch4/<model>/window_ablation.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from thesis.kinematic.labels import add_labels
from thesis.vlm.requests import CLIP_OFFSETS, clip_frames, request_id

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_vlm import discrimination, event_outcomes, fused_fall_prob, load_responses  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LATE_OFFSETS = (-10, 2, 14, 26, 38, 50)
WINDOWS = {"clip6": CLIP_OFFSETS, "clip6late": LATE_OFFSETS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make", action="store_true")
    parser.add_argument("--responses", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = add_labels(pd.read_parquet(ROOT / "data" / "features.parquet"), ROOT / "data")
    onsets = df.dropna(subset=["onset"]).groupby("seq")["onset"].first().to_dict()
    conf = df.set_index(["seq", "frame"])["mean_conf"].fillna(0.0).to_dict()
    lying_adl = {s for s, g in df.groupby("seq") if s.startswith("adl") and g["lying"].any()}
    all_seqs = sorted(df["seq"].unique())
    frames_of = {s: g["frame"].to_numpy() for s, g in df.groupby("seq")}
    events = pd.read_parquet(ROOT / "results" / "vlm" / "urfd_events.parquet")

    def rid(e, variant):
        return request_id(e.seq, clip_frames(e.alarm_frame, frames_of[e.seq], WINDOWS[variant]), variant)

    if args.make:
        requests = {}
        for e in events.itertuples():
            fr = clip_frames(e.alarm_frame, frames_of[e.seq], LATE_OFFSETS)
            i = request_id(e.seq, fr, "clip6late")
            requests[i] = dict(id=i, seq=e.seq, frames=fr, variant="clip6late", kind="event")
        path = ROOT / "results" / "vlm" / "urfd_requests_window.jsonl"
        with open(path, "w") as f:
            for r in requests.values():
                f.write(json.dumps(r) + "\n")
        print(f"{len(requests)} late-window clip requests in {path}")
        return

    resp = load_responses(Path(args.responses))
    pv_of, rows = {}, []
    for variant in WINDOWS:
        ids = [rid(e, variant) for e in events.itertuples()]
        have = np.array([i in resp for i in ids])
        if not have.all():
            print(f"{variant}: {int((~have).sum())} of {len(ids)} event clips have no response")
        pv = np.array([resp[i]["p_anomaly"] if i in resp else np.nan for i in ids])
        uv = np.array([resp[i]["uncertainty"] if i in resp else np.nan for i in ids])
        c = np.array([conf.get((e.seq, e.alarm_frame), 0.0) for e in events.itertuples()])
        pv_of[variant] = {(e.seq, e.alarm_frame): v for e, v in zip(events.itertuples(), pv)}
        for seed, idx in events.groupby("seed").indices.items():
            ev = events.iloc[idx]
            base = dict(seed=int(seed), variant=variant)
            rows.append(dict(base, method="vlm_only",
                             **event_outcomes(ev, ~have[idx] | (pv[idx] >= 0.5), onsets, all_seqs)))
            bet = np.array([fused_fall_prob(e.p, c[i], pv[i], uv[i], 0.0, "dempster") if have[i] else 1.0
                            for i, e in zip(idx, ev.itertuples())])
            for beta in (0.5, 0.3, 0.2):
                rows.append(dict(base, method="dempster", alpha=0.0, beta=beta,
                                 **event_outcomes(ev, bet >= beta, onsets, all_seqs)))
    res = pd.DataFrame(rows)
    summary = res.groupby(["variant", "method", "beta"], dropna=False)[
        ["detection_rate", "false_alarm_episodes", "adl_with_alarm"]].mean().round(3)
    print(summary.to_string())
    disc = discrimination(events, pv_of, onsets, lying_adl)
    print(json.dumps(disc, indent=2))
    out = Path(args.out or ROOT / "results" / "ch4" / "window_ablation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(offsets={k: list(v) for k, v in WINDOWS.items()}, discrimination=disc,
                       events=summary.reset_index().to_dict(orient="records")), f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
