"""Certified bounds for the discretised LTC cell (models.LTCCell).

Cell update, element-wise, with f = tanh(W_f [x; h] + b_f), g = sigmoid(W_g [x; h] + b_g),
tau = softplus(tau0) + 1e-3 > 0:

    h_i <- (h_i + a_i f_i) / D_i,   D_i = 1 + 1/tau_i + g_i >= 1 + 1/tau_i = 1/rho_i.

Lemma V1 (boundedness): from h_0 = 0, |h_{t,i}| <= tau_i |a_i| (1 - rho_i^t) < tau_i |a_i|.
Lemma V2 (input sensitivity): if every input frame is perturbed by at most eps in the
max-norm, the state perturbation obeys e_t <= L_h e_{t-1} + L_x eps, so after W steps
e_W <= L_x eps sum_{k<W} L_h^k and the output logit changes by at most ||w_out||_1 e_W.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import LTCClassifier


def cell_constants(model: LTCClassifier, in_dim: int) -> dict:
    cell = model.cell
    with torch.no_grad():
        tau = (F.softplus(cell.tau0) + 1e-3).numpy().astype(np.float64)
        a = cell.a.abs().numpy().astype(np.float64)
        wf, wg = cell.w_f.weight.numpy().astype(np.float64), cell.w_tau.weight.numpy().astype(np.float64)
        w_out = model.fc.weight.numpy().astype(np.float64).ravel()
    rho = tau / (1.0 + tau)
    wf_x, wf_h = np.abs(wf[:, :in_dim]).sum(1), np.abs(wf[:, in_dim:]).sum(1)
    wg_x, wg_h = np.abs(wg[:, :in_dim]).sum(1), np.abs(wg[:, in_dim:]).sum(1)
    state_bound = tau * a                    # Lemma V1: |h_i| <= tau_i |a_i|
    num_bound = state_bound + a              # |h_i + a_i f_i| <= (tau_i + 1) |a_i|
    l_x = np.max(rho * a * wf_x + 0.25 * rho**2 * num_bound * wg_x)
    l_h = np.max(rho * (1.0 + a * wf_h) + 0.25 * rho**2 * num_bound * wg_h)
    return dict(tau=tau, a=a, rho=rho, state_bound=state_bound, l_x=float(l_x), l_h=float(l_h),
                w_out_l1=float(np.abs(w_out).sum()))


def logit_sensitivity_bound(consts: dict, window: int) -> float:
    """Certified bound on |delta logit| per unit eps (max-norm input perturbation)."""
    geom = sum(consts["l_h"] ** k for k in range(window))
    return consts["w_out_l1"] * consts["l_x"] * geom


def empirical_sensitivity(model: LTCClassifier, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Per-window first-order sensitivity sum_t ||d logit / d x_t||_1, i.e. the slope of
    |delta logit| for the worst max-norm perturbation of size eps -> 0."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.tensor(x[i:i + batch], requires_grad=True)
        model(xb).sum().backward()
        out.append(xb.grad.abs().sum(dim=(1, 2)).numpy())
    return np.concatenate(out)


@torch.no_grad()
def max_state(model: LTCClassifier, x: np.ndarray) -> np.ndarray:
    """max over windows and steps of |h_{t,i}| for every unit."""
    cell = model.cell
    xt = torch.tensor(x)
    h = torch.zeros(len(xt), cell.hidden)
    peak = torch.zeros(cell.hidden)
    for t in range(xt.shape[1]):
        inp = torch.cat([xt[:, t], h], dim=-1)
        f = torch.tanh(cell.w_f(inp))
        gate = torch.sigmoid(cell.w_tau(inp))
        tau = F.softplus(cell.tau0) + 1e-3
        h = (h + f * cell.a) / (1.0 + (1.0 / tau + gate))
        peak = torch.maximum(peak, h.abs().max(0).values)
    return peak.numpy()
