"""Frame- and event-level labels for URFD.

URFD annotates every frame of the cam0 sequences with -1 (not lying),
0 (transitional pose) or 1 (lying on the ground). Lying also occurs in
16 ADL sequences, where the person lies down on purpose, so the usual
binarisation (1 -> positive) labels lying posture rather than falls.

Two frame-level label variants are provided:
  lying -- 1 iff the frame is annotated 1 (the binarisation used in the
           earlier publications);
  fall  -- 1 iff the frame belongs to a fall sequence and is at or after
           the fall onset; ADL frames are always 0.

The fall onset of a fall sequence is its first transitional frame (0).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import ADLS_CSV, FALLS_CSV


def load_raw_labels(data_dir: str | Path) -> pd.DataFrame:
    """Per-frame raw annotation: columns seq, frame, raw (-1/0/1)."""
    parts = []
    for name in (FALLS_CSV, ADLS_CSV):
        df = pd.read_csv(Path(data_dir) / name, header=None, usecols=[0, 1, 2])
        df.columns = ["seq", "frame", "raw"]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def fall_onsets(raw: pd.DataFrame) -> dict[str, int]:
    """Onset frame (first transitional frame) of every fall sequence."""
    falls = raw[raw["seq"].str.startswith("fall") & (raw["raw"] == 0)]
    return falls.groupby("seq")["frame"].min().to_dict()


def add_labels(features: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Attach raw annotation and both binary label variants to a feature
    table (keyed by seq, frame). Adds columns raw, lying, fall, onset
    (onset is NaN for ADL sequences)."""
    raw = load_raw_labels(data_dir)
    onsets = fall_onsets(raw)
    df = features.drop(columns=[c for c in ("raw", "lying", "fall", "onset") if c in features]).merge(
        raw, on=["seq", "frame"], how="left", validate="one_to_one"
    )
    df["lying"] = (df["raw"] == 1).astype(int)
    df["onset"] = df["seq"].map(onsets)
    df["fall"] = (df["onset"].notna() & (df["frame"] >= df["onset"])).astype(int)
    return df
