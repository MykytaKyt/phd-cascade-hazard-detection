#!/usr/bin/env python3
"""GMDCSA-24: dump frames (640x360 PNG, URFD-like layout), run YOLOv8n-pose, compute DDL features.
Writes <out>/{frames/, gm_features.parquet, gm_meta.json}.
Usage (GPU): PYTHONPATH=src python3 experiments/gmdcsa/gm_extract.py <gmdcsa_dir> <out_dir>"""
import json, re, sys
from multiprocessing import Pool
from pathlib import Path
import cv2, numpy as np, pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/w/gmdcsa")   # unpacked GMDCSA-24 (Subject 1..4)
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/root/w/gm"); FR = OUT / "frames"


def videos():
    for sd in sorted(SRC.glob("Subject *")):
        s = int(sd.name.split()[-1])
        for kind in ("Fall", "ADL"):
            rows = {}
            for line in open(sd / f"{kind}.csv", encoding="utf-8", errors="replace").read().splitlines()[1:]:
                if line.strip():
                    parts = line.split(",")
                    rows[parts[0].strip()] = dict(length=parts[1].strip(), classes=parts[-1].strip(), description=",".join(parts[4:-1]).strip())
            for v in sorted((sd / kind).glob("*.mp4")):
                yield dict(subject=s, kind=kind.lower(), path=str(v), seq=f"gm-s{s}-{kind.lower()}-{int(v.stem):02d}", **rows.get(v.name, {}))


def dump(m):
    cap = cv2.VideoCapture(m["path"]); fps = cap.get(cv2.CAP_PROP_FPS); d = FR / m["seq"] / f"{m['seq']}-cam0-rgb"; d.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        ok, im = cap.read()
        if not ok: break
        n += 1
        cv2.imwrite(str(d / f"{m['seq']}-cam0-rgb-{n:03d}.png"), cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return dict(m, fps=fps, n_frames=n)


if __name__ == "__main__":
    with Pool(16) as p: meta = p.map(dump, list(videos()))
    json.dump(meta, open(OUT / "gm_meta.json", "w"), indent=1)
    print(len(meta), "videos", sum(m["n_frames"] for m in meta), "frames")
    from ultralytics import YOLO
    from thesis.kinematic.ddl import compute_ddl_features
    from thesis.kinematic.extract import _add_temporal_derivatives, keypoint_columns
    model = YOLO("yolov8n-pose.pt"); rows = []
    for m in meta:
        imgs = sorted((FR / m["seq"] / f"{m['seq']}-cam0-rgb").glob("*.png"))
        for k, res in enumerate(model.predict([str(i) for i in imgs], verbose=False, stream=True, device=0), 1):
            h, w = res.orig_shape; rec = dict(seq=m["seq"], frame=k, H=h, W=w, valid=0)
            if res.keypoints is not None and len(res.keypoints) > 0 and res.boxes is not None and len(res.boxes) > 0:
                idx = int(np.argmax((res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()))
                kp = res.keypoints.data[idx].cpu().numpy(); box = res.boxes.xywh[idx].cpu().numpy()
                rec.update(valid=1, **compute_ddl_features(kp, box, frame_w=w, frame_h=h), **keypoint_columns(kp, box))
            rows.append(rec)
    df = _add_temporal_derivatives(pd.DataFrame(rows).sort_values(["seq", "frame"]).reset_index(drop=True))
    df.to_parquet(OUT / "gm_features.parquet", index=False)
    print("features", df.shape, "valid", df["valid"].mean())
