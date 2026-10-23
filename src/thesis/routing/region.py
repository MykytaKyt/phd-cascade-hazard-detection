"""Routing regions of the cascade.

A frame with calibrated fall probability p is sent to the semantic level
iff tau_N < p < 1 - tau_A. Outside the region the edge decision p >= 1/2
is final. The symmetric rule of the entropy threshold h corresponds to
tau_N = tau_A = H^{-1}(h).

Positives with p <= tau_N are never seen by the semantic level, so for any
semantic model and combination rule

    Recall <= 1 - F_edge(tau_N) / P,

where F_edge(tau_N) is the number of positives with p <= tau_N.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def inverse_entropy(h: float) -> float:
    """eps in [0, 1/2] with H(eps) = h."""
    if h <= 0:
        return 0.0
    if h >= 1:
        return 0.5
    return float(brentq(lambda e: binary_entropy(np.array([e]))[0] - h, 1e-12, 0.5))


def region_stats(p: np.ndarray, y: np.ndarray, tau_n: float, tau_a: float) -> dict:
    routed = (p > tau_n) & (p < 1 - tau_a)
    edge_pred = p >= 0.5
    n_pos = int(y.sum())
    f_edge = int(((~routed) & (~edge_pred) & (y == 1)).sum())
    return dict(
        tau_n=float(tau_n), tau_a=float(tau_a),
        n_routed=int(routed.sum()), routed_frac=float(routed.mean()),
        routed_normal_side=int((routed & ~edge_pred).sum()),
        routed_fall_side=int((routed & edge_pred).sum()),
        f_edge=f_edge,
        recall_bound=1.0 - f_edge / n_pos if n_pos else np.nan,
        edge_fp=int(((~routed) & edge_pred & (y == 0)).sum()),
    )
