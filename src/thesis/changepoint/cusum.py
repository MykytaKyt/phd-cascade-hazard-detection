"""Event-level fall detection on per-frame classifier scores.

A fall is treated as a change point in the sequence of classifier logits
z_t. For a calibrated classifier with output p_t and prior pi, the
per-frame log-likelihood ratio of "fall" against "normal" is
logit(p_t) - logit(pi), so the logit itself is an LLR up to a constant,
absorbed here into the drift kappa. Page's CUSUM statistic

    S_0 = 0,   S_t = max(0, S_{t-1} + z_t - kappa),

raises an alarm at the first t with S_t >= h and restarts from 0.
The frame-wise rule (alarm iff z_t >= h) is the special case without
accumulation and is evaluated alongside as the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cusum_alarms(z: np.ndarray, h: float, kappa: float = 0.0) -> np.ndarray:
    """Indices of CUSUM alarms (statistic restarts after each alarm)."""
    s, alarms = 0.0, []
    for t, v in enumerate(z):
        s = max(0.0, s + v - kappa)
        if s >= h:
            alarms.append(t)
            s = 0.0
    return np.asarray(alarms, dtype=int)


def threshold_alarms(z: np.ndarray, h: float) -> np.ndarray:
    """Indices where the frame-wise rule switches on (rising edges of z >= h)."""
    on = z >= h
    return np.flatnonzero(on & ~np.concatenate([[False], on[:-1]]))


@dataclass
class SequenceOutcome:
    seq: str
    is_fall: bool
    detected: bool          # fall sequence: an alarm at or after the onset
    false_alarm: bool       # ADL sequence: any alarm; fall sequence: an alarm before the onset
    delay: float            # frames from onset to the first alarm at or after it (NaN if none / ADL)
    n_false_alarms: int     # number of alarms outside [onset, end]


def evaluate_sequence(seq: str, frames: np.ndarray, alarms: np.ndarray, onset: float | None) -> SequenceOutcome:
    alarm_frames = frames[alarms]
    if onset is None or np.isnan(onset):
        return SequenceOutcome(seq, False, False, len(alarm_frames) > 0, np.nan, len(alarm_frames))
    early = alarm_frames[alarm_frames < onset]
    late = alarm_frames[alarm_frames >= onset]
    delay = float(late[0] - onset) if len(late) else np.nan
    return SequenceOutcome(seq, True, len(late) > 0, len(early) > 0, delay, len(early))


def evaluate_rule(sequences, rule, h: float, **kwargs) -> list[SequenceOutcome]:
    """`sequences`: iterable of (seq, frames, z, onset) with frames sorted.
    `rule`: cusum_alarms or threshold_alarms."""
    return [evaluate_sequence(seq, frames, rule(z, h, **kwargs), onset) for seq, frames, z, onset in sequences]


def summarize_outcomes(outcomes: list[SequenceOutcome], n_normal_frames: int, fps: float = 30.0) -> dict:
    falls = [o for o in outcomes if o.is_fall]
    adls = [o for o in outcomes if not o.is_fall]
    delays = np.array([o.delay for o in falls if o.detected])
    n_fa = sum(o.n_false_alarms for o in outcomes)
    hours = n_normal_frames / fps / 3600.0
    return dict(
        n_falls=len(falls), n_adl=len(adls),
        detected=sum(o.detected for o in falls),
        detection_rate=float(np.mean([o.detected for o in falls])) if falls else np.nan,
        adl_false_alarm_seqs=sum(o.false_alarm for o in adls),
        fall_early_alarm_seqs=sum(o.false_alarm for o in falls),
        false_alarms=int(n_fa),
        false_alarms_per_hour=n_fa / hours if hours else np.nan,
        delay_mean=float(delays.mean()) if len(delays) else np.nan,
        delay_median=float(np.median(delays)) if len(delays) else np.nan,
        delay_max=float(delays.max()) if len(delays) else np.nan,
    )


def select_h(sequences, rule, grid: np.ndarray, min_detection: float = 1.0, **kwargs) -> float:
    """Smallest number of false alarms subject to detection rate >= min_detection
    on the given (validation) sequences; ties broken by the smaller mean delay.
    Falls back to the h with the highest detection rate if the constraint is infeasible."""
    best, best_key, fallback, fallback_key = None, None, None, None
    for h in grid:
        out = evaluate_rule(sequences, rule, h, **kwargs)
        falls = [o for o in out if o.is_fall]
        det = np.mean([o.detected for o in falls]) if falls else 1.0
        n_fa = sum(o.n_false_alarms for o in out)
        delays = [o.delay for o in falls if o.detected]
        mean_delay = float(np.mean(delays)) if delays else np.inf
        if det >= min_detection:
            key = (n_fa, mean_delay)
            if best_key is None or key < best_key:
                best, best_key = h, key
        fkey = (-det, n_fa)
        if fallback_key is None or fkey < fallback_key:
            fallback, fallback_key = h, fkey
    return float(best if best is not None else fallback)
