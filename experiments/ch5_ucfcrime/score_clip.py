#!/usr/bin/env python3
"""Local run (GPU): zero-shot CLIP scoring of the UCF-Crime test videos.

Every `stride`-th frame of every test video is encoded with CLIP and compared
with the text prompts of the 13 anomaly classes and "Normal" (several templates
per class, averaged). Saves per sampled frame the 14 cosine similarities and,
optionally, the image embeddings (to change prompts later without re-encoding).

Requires `pip install -e ".[vad]"`, the UCF-Crime test videos and the test
annotation file Temporal_Anomaly_Annotation_for_Testing_Videos.txt
(UCF-Crime: Sultani et al., CVPR 2018).

Writes results/ch5/clip_scores.parquet (+ results/ch5/clip_embeddings.npz).

Usage:
    python experiments/ch5_ucfcrime/score_clip.py \
        --videos-dir /data/UCF-Crime \
        --annotations /data/UCF-Crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
        --model ViT-L-14 --pretrained openai --stride 16 --device cuda
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from thesis.vad.ucf import CLASSES, parse_annotations, prompts

ROOT = Path(__file__).resolve().parents[2]


def find_videos(videos_dir: Path) -> dict[str, Path]:
    return {p.name: p for p in videos_dir.rglob("*.mp4")}


def sample_frames(path: Path, stride: int):
    """Yields (frame_index, PIL image) for every stride-th frame; returns n_frames, fps at the end."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % stride == 0:
            yield k, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    return k, fps


@torch.no_grad()
def text_embeddings(model, tokenizer, device) -> torch.Tensor:
    out = []
    for cls, texts in prompts().items():
        e = model.encode_text(tokenizer(texts).to(device), normalize=True).mean(0)
        out.append(e / e.norm())
    return torch.stack(out)  # (14, d)


@torch.no_grad()
def encode_video(model, preprocess, path: Path, stride: int, device, batch: int):
    idx, embs, buf = [], [], []
    gen = sample_frames(path, stride)
    n_frames, fps = 0, 30.0
    while True:
        try:
            k, im = next(gen)
        except StopIteration as stop:
            n_frames, fps = stop.value
            break
        idx.append(k)
        buf.append(preprocess(im))
        if len(buf) == batch:
            embs.append(model.encode_image(torch.stack(buf).to(device), normalize=True).float().cpu())
            buf = []
    if buf:
        embs.append(model.encode_image(torch.stack(buf).to(device), normalize=True).float().cpu())
    return np.array(idx), torch.cat(embs) if embs else torch.empty(0), n_frames, fps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--out-dir", default=ROOT / "results" / "ch5")
    args = parser.parse_args()

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained,
                                                                 device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    text = text_embeddings(model, tokenizer, args.device).float().cpu()

    ann = parse_annotations(args.annotations)
    videos = find_videos(Path(args.videos_dir))
    missing = [a.video for a in ann if a.video not in videos]
    if missing:
        print(f"WARNING: {len(missing)} annotated videos not found, e.g. {missing[:3]}")

    rows, emb_store = [], {}
    t0 = time.time()
    for n, a in enumerate([a for a in ann if a.video in videos], 1):
        idx, emb, n_frames, fps = encode_video(model, preprocess, videos[a.video], args.stride, args.device, args.batch)
        if len(idx) == 0:
            continue
        sims = (emb @ text.T).numpy()
        labels = a.frame_labels(n_frames)[idx]
        part = pd.DataFrame(sims, columns=[f"sim_{c}" for c in CLASSES])
        part.insert(0, "label", labels)
        part.insert(0, "frame", idx)
        part.insert(0, "fps", fps)
        part.insert(0, "n_frames", n_frames)
        part.insert(0, "cls", a.cls)
        part.insert(0, "video", a.video)
        rows.append(part)
        if not args.no_embeddings:
            emb_store[a.video] = emb.half().numpy()
        if n % 10 == 0:
            print(f"  {n} videos, {time.time() - t0:.0f} s")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(rows, ignore_index=True)
    scores.attrs.update(model=args.model, pretrained=args.pretrained, stride=args.stride)
    scores.to_parquet(out / "clip_scores.parquet", index=False)
    if emb_store:
        np.savez_compressed(out / "clip_embeddings.npz", text=text.numpy(), **emb_store)
    print(f"Saved {out / 'clip_scores.parquet'}: {scores['video'].nunique()} videos, {len(scores)} sampled frames")


if __name__ == "__main__":
    main()
