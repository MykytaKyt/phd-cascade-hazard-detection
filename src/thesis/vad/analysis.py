"""Zero-shot hazardous-event detection on UCF-Crime from CLIP similarities.

Input: one row per sampled frame (video, cls, frame, label, sim_<class> for the
13 anomaly classes and Normal), produced by experiments/ch5_ucfcrime/score_clip.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit, softmax
from sklearn.metrics import average_precision_score, roc_auc_score

from .ucf import ANOMALY_CLASSES, CLASSES

LOGIT_SCALE = 100.0  # CLIP logit scale for cosine similarities


def class_probs(df: pd.DataFrame, scale: float = LOGIT_SCALE) -> np.ndarray:
    """Softmax over the 14 class prompts, shape (n, 14), columns ordered as CLASSES."""
    sims = df[[f"sim_{c}" for c in CLASSES]].to_numpy(dtype=np.float64)
    return softmax(scale * sims, axis=1)


def anomaly_logit(df: pd.DataFrame, scale: float = LOGIT_SCALE) -> np.ndarray:
    """logit of P(any anomaly class) = 1 - P(Normal) under the zero-shot softmax."""
    p_normal = class_probs(df, scale)[:, CLASSES.index("Normal")]
    return logit(np.clip(1.0 - p_normal, 1e-9, 1 - 1e-9))


def frame_auc(df: pd.DataFrame, score: np.ndarray) -> float:
    """Frame-level ROC AUC over all sampled frames of all videos (the standard UCF-Crime
    protocol pools frames over the whole test set)."""
    return float(roc_auc_score(df["label"].to_numpy(), score))


def split_videos(df: pd.DataFrame, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Calibration / evaluation split of videos, stratified anomalous vs normal."""
    vids = df.groupby("video")["cls"].first()
    cal = []
    for is_anom in (True, False):
        group = vids[(vids != "Normal") == is_anom].index.to_numpy()
        n = max(1, int(round(fraction * len(group))))
        cal.extend(rng.choice(group, size=n, replace=False))
    cal = np.array(sorted(cal))
    return cal, np.setdiff1d(vids.index.to_numpy(), cal)


def positive_probs_by_video(df: pd.DataFrame, p: np.ndarray) -> list[np.ndarray]:
    d = df.assign(p=p)
    return [g["p"].to_numpy() for _, g in d[d["label"] == 1].groupby("video")]


def retrieval_metrics(df: pd.DataFrame, ks=(10, 50, 100)) -> pd.DataFrame:
    """Text-query retrieval: for every anomaly class c, frames are ranked by the
    similarity to the prompt of c; relevant = anomalous frames of videos of class c.
    Returns AP and precision@K per class."""
    rows = []
    for c in ANOMALY_CLASSES:
        rel = ((df["cls"] == c) & (df["label"] == 1)).to_numpy()
        if not rel.any():
            continue
        s = df[f"sim_{c}"].to_numpy()
        order = np.argsort(-s)
        row = dict(cls=c, n_relevant=int(rel.sum()), ap=float(average_precision_score(rel, s)))
        for k in ks:
            row[f"p@{k}"] = float(rel[order[:k]].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def routed_segments(df: pd.DataFrame, routed: np.ndarray, max_gap: int = 1) -> pd.DataFrame:
    """Group consecutive routed sampled frames of a video into segments
    (gaps of at most `max_gap` sampled frames are bridged)."""
    d = df.assign(routed=routed).sort_values(["video", "frame"])
    rows = []
    for video, g in d.groupby("video", sort=True):
        frames = g["frame"].to_numpy()
        r = g["routed"].to_numpy()
        idx = np.flatnonzero(r)
        if not len(idx):
            continue
        start = prev = idx[0]
        for i in list(idx[1:]) + [None]:
            if i is not None and i - prev <= max_gap + 1:
                prev = i
                continue
            rows.append(dict(video=video, start=int(frames[start]), end=int(frames[prev]),
                             n_sampled=int(prev - start + 1)))
            if i is not None:
                start = prev = i
    return pd.DataFrame(rows, columns=["video", "start", "end", "n_sampled"])


def segment_clip_frames(start: int, end: int, n: int = 6) -> list[int]:
    return sorted({int(round(v)) for v in np.linspace(start, end, n)})
