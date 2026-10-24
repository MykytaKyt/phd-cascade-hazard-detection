"""Building VLM requests from kinematic scores.

Two request kinds for URFD:
  event -- every alarm episode of the kinematic frame-wise rule on calibrated
           LLRs (threshold selected on validation sequences to detect every
           validation fall) is verified on a clip of frames around the alarm;
  frame -- every frame inside the routing region (tau_N, 1 - tau_A) for
           frame-level comparison with the earlier publications.
Requests are deduplicated by (seq, frames, variant) across seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit

from thesis.changepoint.cusum import alarm_episodes, select_h, threshold_alarms
from thesis.routing.calibration import calibrate, fit_temperature

CLIP_OFFSETS = (-40, -30, -20, -10, 0, 10)  # frames relative to the alarm (30 fps)
H_GRID = np.linspace(-4, 8, 121)


def calibrate_folds(scores: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Per fold: temperature and prior from the validation rows; adds p (calibrated
    probability) and z (log-likelihood ratio) to validation and test rows."""
    parts = []
    for _, fg in scores.groupby("fold"):
        val = fg[fg["split"] == "val"].copy()
        test = fg[fg["split"] == "test"].copy()
        t = fit_temperature(val["score"].to_numpy(), val["label"].to_numpy())
        prior_logit = logit(float(np.clip(val["label"].mean(), 1e-4, 1 - 1e-4)))
        for part in (val, test):
            part["p"] = calibrate(part["score"].to_numpy(), t)
            part["z"] = part["score"] / t - prior_logit
        parts.append((val, test))
    return parts


def _sequences(part: pd.DataFrame, onsets: dict):
    for seq, g in part.groupby("seq", sort=True):
        g = g.sort_values("frame")
        yield seq, g["frame"].to_numpy(), g["z"].to_numpy(), onsets.get(seq)


def alarm_events(scores: pd.DataFrame, onsets: dict, refractory: int = 90) -> pd.DataFrame:
    """Alarm episodes on the test sequences of every (seed, fold): seq, alarm frame, p at the alarm."""
    rows = []
    for seed, sg in scores.groupby("seed"):
        for val, test in calibrate_folds(sg):
            h = select_h(list(_sequences(val, onsets)), threshold_alarms, H_GRID)
            for seq, g in test.groupby("seq", sort=True):
                g = g.sort_values("frame")
                frames = g["frame"].to_numpy()
                starts = alarm_episodes(frames[threshold_alarms(g["z"].to_numpy(), h)], refractory)
                p_at = dict(zip(frames, g["p"]))
                for f in starts:
                    rows.append(dict(seed=int(seed), fold=int(g["fold"].iloc[0]), h=float(h),
                                     seq=seq, alarm_frame=int(f), p=float(p_at[f])))
    return pd.DataFrame(rows)


def clip_frames(alarm_frame: int, available: np.ndarray, offsets=CLIP_OFFSETS) -> list[int]:
    """Frames at the given offsets, snapped to the nearest available frame of the sequence."""
    available = np.sort(available)
    out = []
    for off in offsets:
        target = alarm_frame + off
        out.append(int(available[np.argmin(np.abs(available - target))]))
    return sorted(set(out))


def request_id(seq: str, frames: list[int], variant: str) -> str:
    return f"{seq}:{'-'.join(str(f) for f in frames)}:{variant}"
