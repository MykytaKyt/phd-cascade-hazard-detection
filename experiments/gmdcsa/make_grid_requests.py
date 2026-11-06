#!/usr/bin/env python3
"""Dense grid of 6-frame clips (anchor every STRIDE frames, same offsets as alarm clips) for a frames dir.
Usage: PYTHONPATH=src python3 experiments/gmdcsa/make_grid_requests.py <frames_dir> <out.jsonl> 10"""
import json, sys
from pathlib import Path
import numpy as np
from thesis.kinematic.dataset import frame_number, list_sequence_frames, list_sequences
from thesis.vlm.requests import clip_frames, request_id
frames_dir, out, stride = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
seen, n = set(), 0
with open(out, "w") as f:
    for seq in list_sequences(frames_dir):
        avail = np.array(sorted(frame_number(Path(p).name) for p in list_sequence_frames(frames_dir, seq)))
        for g in avail[::stride]:
            fr = [int(x) for x in clip_frames(int(g), avail)]; rid = request_id(seq, fr, "clip6")
            if rid in seen: continue
            seen.add(rid); n += 1
            f.write(json.dumps(dict(id=rid, seq=seq, frames=fr, variant="clip6", kind="grid", anchor=int(g))) + "\n")
print(n, "requests ->", out)
