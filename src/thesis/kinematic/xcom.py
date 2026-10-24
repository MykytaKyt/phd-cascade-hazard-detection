"""Extrapolated centre of mass (XCoM) and the margin of stability in image coordinates.

Inverted-pendulum model (Hof et al., 2005): for a pendulum of length l the
extrapolated centre of mass is

    xi = x + v / omega0,   omega0 = sqrt(g / l),

and the dynamic margin of stability is the distance from xi to the boundary of
the base of support (positive inside). In a single 2D view only the image
horizontal axis is available; the base of support is the horizontal extent of
the visible ankles [x_min, x_max]. Pixel scale cancels in xi: x and v are both in
pixels, and omega0 is in 1/s, so v must be in pixels per second (v_frame * fps).

All margins are normalised by the person's bounding-box height.

Columns used: com_x (normalised by frame width), W (frame width), box_h,
kp15_x/kp15_c, kp16_x/kp16_c (left/right ankle), valid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRAVITY = 9.81
L_ANKLE, R_ANKLE = 15, 16


def omega0(leg_length_m: float = 0.9) -> float:
    return float(np.sqrt(GRAVITY / leg_length_m))


def horizontal_margin(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Signed distance from x to the interval [lo, hi]: positive inside, negative outside."""
    return np.minimum(x - lo, hi - x)


def add_xcom_features(
    df: pd.DataFrame,
    fps: float = 30.0,
    leg_length_m: float = 0.9,
    visibility_threshold: float = 0.3,
) -> pd.DataFrame:
    """Add xcom_x (pixels), mos_static_x and mos_xcom_x (normalised by box height).

    mos_static_x uses the CoM itself (static stability), mos_xcom_x the
    extrapolated CoM (dynamic stability). Frames without a visible ankle get NaN.
    """
    df = df.sort_values(["seq", "frame"]).copy()
    w0 = omega0(leg_length_m)
    com_px = df["com_x"] * df["W"]

    xs = np.stack([df[f"kp{j}_x"].to_numpy() for j in (L_ANKLE, R_ANKLE)], axis=1)
    cs = np.stack([df[f"kp{j}_c"].to_numpy() for j in (L_ANKLE, R_ANKLE)], axis=1)
    visible = cs > visibility_threshold
    lo = np.where(visible, xs, np.inf).min(axis=1)
    hi = np.where(visible, xs, -np.inf).max(axis=1)
    has_bos = visible.any(axis=1) & (df["valid"].to_numpy() == 1)

    v_px = np.zeros(len(df))
    for _, idx in df.groupby("seq").indices.items():
        c = com_px.iloc[idx].ffill().bfill().to_numpy()
        v_px[idx] = np.gradient(c) * fps if len(c) > 1 else 0.0
    xcom = com_px.to_numpy() + v_px / w0

    box_h = df["box_h"].to_numpy()
    with np.errstate(invalid="ignore"):
        df["xcom_x"] = np.where(has_bos, xcom, np.nan)
        df["mos_static_x"] = np.where(has_bos, horizontal_margin(com_px.to_numpy(), lo, hi) / box_h, np.nan)
        df["mos_xcom_x"] = np.where(has_bos, horizontal_margin(xcom, lo, hi) / box_h, np.nan)
    return df
