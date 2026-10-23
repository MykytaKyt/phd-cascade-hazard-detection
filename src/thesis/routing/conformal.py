"""Conformal choice of the normal-side routing threshold tau_N.

Unit of exchangeability: the sequence. For a sequence i with positive
frames, let L_i(tau) be the fraction of its positive frames with
calibrated probability p <= tau, i.e. the fraction that the edge level
would miss without consulting the semantic level. L_i is non-decreasing
in tau and bounded in [0, 1]. With n exchangeable calibration sequences,
conformal risk control (Angelopoulos et al.) chooses

    tau_hat = max { tau : (n * mean_i L_i(tau) + 1) / (n + 1) <= alpha }

and guarantees E[L_test(tau_hat)] <= alpha for a new exchangeable sequence.
Because L_i(0) >= 0, the set is non-empty only if 1 / (n + 1) <= alpha,
i.e. n >= 1 / alpha - 1 calibration sequences are needed.
"""
from __future__ import annotations

import numpy as np


def min_calibration_sequences(alpha: float) -> int:
    return int(np.ceil(1.0 / alpha - 1.0))


def sequence_miss_fraction(pos_probs: list[np.ndarray], tau: float) -> np.ndarray:
    """L_i(tau) for every calibration sequence (arrays of positive-frame probabilities)."""
    return np.array([(p <= tau).mean() for p in pos_probs])


def crc_threshold(pos_probs: list[np.ndarray], alpha: float, grid: np.ndarray) -> float | None:
    """Largest tau in `grid` satisfying the conformal risk control condition;
    None if no tau is admissible (too few calibration sequences)."""
    n = len(pos_probs)
    best = None
    for tau in np.sort(grid):
        risk = (n * sequence_miss_fraction(pos_probs, tau).mean() + 1.0) / (n + 1.0)
        if risk <= alpha:
            best = float(tau)
        else:
            break
    return best


def empirical_threshold(p: np.ndarray, y: np.ndarray, recall: float, grid: np.ndarray) -> float:
    """Largest tau in `grid` whose empirical recall bound on (p, y) is >= recall
    (plug-in choice without a finite-sample guarantee)."""
    pos = p[y == 1]
    best = 0.0
    for tau in np.sort(grid):
        if 1.0 - (pos <= tau).mean() >= recall:
            best = float(tau)
        else:
            break
    return best
