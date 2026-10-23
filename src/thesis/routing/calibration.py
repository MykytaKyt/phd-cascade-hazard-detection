"""Probability calibration of the kinematic classifier."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    """Mean binary negative log-likelihood of sigmoid(logits / T)."""
    z = logits / temperature
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """Temperature T minimising the NLL of sigmoid(logits / T); optimised over log T."""
    res = minimize_scalar(lambda lt: nll(logits, y, np.exp(lt)), bounds=np.log(bounds), method="bounded")
    return float(np.exp(res.x))


def calibrate(logits: np.ndarray, temperature: float) -> np.ndarray:
    return expit(logits / temperature)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins over P(y = 1)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))
