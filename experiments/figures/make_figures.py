#!/usr/bin/env python3
"""Figures of the dissertation (Ukrainian labels), built from saved results only.

  fig_cascade.png        -- structure of the cascade (chapter 2)
  fig_reliability.png    -- reliability diagrams of the kinematic level (chapter 2)
  fig_frontier.png       -- routing frontier, symmetric vs asymmetric region (chapter 2)
  fig_xcom.png           -- inverted pendulum, CoM, XCoM and base of support (chapter 3)
  fig_ablation.png       -- feature-group ablation for both label variants (chapter 3)
  fig_event_tradeoff.png -- event-level operating characteristics (chapter 3)

Usage:
    python experiments/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from thesis.routing.calibration import calibrate, fit_temperature  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "savefig.dpi": 300})


def box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec="black", lw=0.8))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, a, b, text=None, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12, lw=0.9))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03, text, ha="center", fontsize=8)


def fig_cascade():
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.set_xlim(0, 11.7)
    ax.set_ylim(0.0, 3.6)
    ax.axis("off")
    edge, cloud, dec = "#dceefb", "#fde9d9", "#e2f0d9"
    ax.text(0.1, 3.4, "Периферійний пристрій", fontsize=10, weight="bold")
    ax.text(7.5, 3.4, "Семантичний рівень", fontsize=10, weight="bold")
    box(ax, (0.1, 2.0), 1.3, 1.0, "Відеопотік\n$X_t$", edge)
    box(ax, (1.7, 2.0), 1.5, 1.0, "YOLOv8-Pose:\nключові точки", edge)
    box(ax, (3.5, 2.0), 1.7, 1.0, "DDL: ЦМ, XCoM,\nзапас стійкості,\nкут, швидкості", edge)
    box(ax, (5.5, 2.0), 1.4, 1.0, "LTC\n(вікно W)\n$p_t$", edge)
    box(ax, (5.2, 0.3), 2.0, 1.1, "Маршрутизатор\n$\\tau_N < p_t < 1-\\tau_A$\n(конформний $\\tau_N$)", edge)
    box(ax, (2.6, 0.3), 1.9, 1.1, "Рішення на\nпериферії\n$p_t \\geq 1/2$", dec)
    box(ax, (7.7, 0.3), 2.0, 1.1, "VLM:\n«контрольована\nдія чи нещасний\nвипадок?»", cloud)
    box(ax, (7.7, 2.0), 2.0, 1.0, "Комбінування\nсвідчень\nДемпстера–Шафера", cloud)
    box(ax, (10.1, 2.0), 1.5, 1.0, "Рішення\n$BetP(B) \\geq \\beta$", dec)
    arrow(ax, (1.4, 2.5), (1.7, 2.5))
    arrow(ax, (3.2, 2.5), (3.5, 2.5))
    arrow(ax, (5.2, 2.5), (5.5, 2.5))
    arrow(ax, (6.2, 2.0), (6.2, 1.4))
    arrow(ax, (5.2, 0.85), (4.5, 0.85))
    ax.text(4.85, 0.95, "поза\nобластю", ha="center", va="bottom", fontsize=8)
    arrow(ax, (7.2, 0.85), (7.7, 0.85))
    ax.text(7.45, 0.95, "кліп", ha="center", va="bottom", fontsize=8)
    arrow(ax, (6.9, 2.5), (7.7, 2.5))
    ax.text(7.3, 2.58, "$m_Y$", ha="center", fontsize=9)
    arrow(ax, (8.7, 1.4), (8.7, 2.0))
    ax.text(8.8, 1.65, "$m_V$", fontsize=9)
    arrow(ax, (9.7, 2.5), (10.1, 2.5))
    ax.plot([7.45, 7.45], [1.6, 3.5], ls="--", c="grey", lw=0.8)
    fig.savefig(FIG / "fig_cascade.png", bbox_inches="tight")
    plt.close(fig)


def reliability(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(ys)


def fig_reliability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, label, title in zip(axes, ["lying", "fall"], ["Мітки «лежання»", "Мітки «падіння»"]):
        sc = pd.read_parquet(ROOT / "results" / "ch3" / f"scores_{label}.parquet")
        sc = sc[sc["model"] == "ltc24"]
        raw, cal, ys = [], [], []
        for _, fg in sc.groupby(["seed", "fold"]):
            val, test = fg[fg["split"] == "val"], fg[fg["split"] == "test"]
            t = fit_temperature(val["score"].to_numpy(), val["label"].to_numpy())
            raw.append(1 / (1 + np.exp(-test["score"].to_numpy())))
            cal.append(calibrate(test["score"].to_numpy(), t))
            ys.append(test["label"].to_numpy())
        y = np.concatenate(ys)
        for p, style, name in ((np.concatenate(raw), "o--", "до масштабування"),
                               (np.concatenate(cal), "s-", "після масштабування")):
            x, f = reliability(p, y)
            ax.plot(x, f, style, ms=4, label=name)
        ax.plot([0, 1], [0, 1], ":", c="black", lw=0.8, label="ідеальне калібрування")
        ax.set_xlabel("Прогнозована ймовірність")
        ax.set_ylabel("Емпірична частота")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_reliability.png")
    plt.close(fig)


def fig_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, label, title in zip(axes, ["fall", "lying"], ["Мітки «падіння»", "Мітки «лежання»"]):
        f = pd.read_csv(ROOT / "results" / "ch2" / f"frontier_{label}.csv")
        for region, style, name in (("symmetric", "--", "симетрична область"), ("asymmetric", "-", "асиметрична область")):
            g = f[f["region"] == region].groupby("tau_n")[["routed_frac", "recall_bound"]].mean().sort_values("routed_frac")
            ax.plot(100 * g["routed_frac"], 100 * g["recall_bound"], style, label=name)
        ax.set_xlabel("Кадрів, переданих VLM, %")
        ax.set_ylabel("Межа повноти, %")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_frontier.png")
    plt.close(fig)


def fig_xcom():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_xlim(-1.2, 1.6)
    ax.set_ylim(-0.25, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    theta = np.deg2rad(18)
    foot = np.array([0.0, 0.0])
    com = foot + np.array([np.sin(theta), np.cos(theta)])
    ax.plot([-1.1, 1.5], [0, 0], c="black", lw=1)
    ax.add_patch(plt.Rectangle((-0.25, -0.06), 0.5, 0.06, fc="#9dc3e6", ec="black", lw=0.6))
    ax.text(0, -0.17, "площа опори $[u_{min}, u_{max}]$", ha="center", fontsize=8)
    ax.plot(*np.c_[foot, com], c="black", lw=2)
    ax.plot(*com, "o", ms=12, c="#2e75b6")
    ax.text(com[0] - 0.08, com[1] + 0.06, "ЦМ (u)", fontsize=9, ha="right")
    ax.annotate("", xy=(com[0] + 0.45, com[1]), xytext=com, arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(com[0] + 0.22, com[1] + 0.06, "$\\dot u$", fontsize=10)
    xi = com[0] + 0.75
    ax.plot([com[0], com[0]], [com[1], 0], ":", c="#2e75b6")
    ax.plot([xi, xi], [com[1] - 0.15, 0], ":", c="#c00000")
    ax.plot(xi, 0, "x", ms=10, c="#c00000", mew=2)
    ax.text(xi + 0.04, 0.06, "XCoM $\\xi_u = u + \\dot u/\\omega_0$", fontsize=9, color="#c00000")
    ax.annotate("", xy=(0.25, 0.25), xytext=(xi, 0.25), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#c00000"))
    ax.text((0.25 + xi) / 2, 0.29, "$\\mathrm{MoS}_{XCoM} < 0$", ha="center", fontsize=9, color="#c00000")
    ax.text(0.02, 0.45, "$\\theta$", fontsize=10)
    fig.savefig(FIG / "fig_xcom.png", bbox_inches="tight")
    plt.close(fig)


def fig_ablation():
    rows = pd.DataFrame(json.load(open(ROOT / "results" / "ch3" / "ablation.json"))["rows"])
    rows = rows[(rows["window"] == 8) & (rows["hidden"] == 24)]
    names = {"all": "усі 10", "no_angle_mos_bos": "без кута, MoS,\nплощі опори", "no_mos": "без MoS",
             "no_dynamics": "без швидкостей\nі прискорення", "bbox_only": "лише рамка",
             "biomech_only": "лише\nбіомеханічні"}
    order = list(names)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.38
    x = np.arange(len(order))
    for k, (label, name) in enumerate((("lying", "мітки «лежання»"), ("fall", "мітки «падіння»"))):
        r = rows[rows["label"] == label].set_index("features").loc[order]
        ax.bar(x + (k - 0.5) * width, 100 * r["f1"], width, yerr=100 * r["f1_std"], capsize=3, label=name)
    ax.set_xticks(x, [names[o] for o in order], fontsize=8)
    ax.set_ylabel("F1, %")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def fig_event_tradeoff():
    r = pd.DataFrame(json.load(open(ROOT / "results" / "ch3" / "event_curves_fall.json")))
    m = r.groupby(["rule", "h"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for rule, style, name in (("threshold", "--", "покадрове правило"), ("cusum", "-", "CUSUM")):
        g = m[m["rule"] == rule].sort_values("false_alarms_per_hour")
        axes[0].plot(g["false_alarms_per_hour"], 100 * g["detection_rate"], style, label=name)
        axes[1].plot(g["false_alarms_per_hour"], g["delay_median"] / 30.0, style, label=name)
    axes[0].set_ylabel("Виявлено падінь, %")
    axes[1].set_ylabel("Медіанна затримка, с")
    for ax in axes:
        ax.set_xlabel("Епізодів хибних тривог на годину")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_event_tradeoff.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    for f in (fig_cascade, fig_reliability, fig_frontier, fig_xcom, fig_ablation, fig_event_tradeoff):
        f()
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
