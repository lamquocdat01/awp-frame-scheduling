# -*- coding: utf-8 -*-
"""v3_figures.py -- redraw the three baseline figures that were 200 dpi (sch2, sch5, fig3) at 300 dpi and
at their printed size in paper_v3 (elsarticle, single column), so that every glyph is >= 8 pt in print.
Content is unchanged: sch2 = Theorem 1' schematic, sch5 = Theorem 2 converse schematic (same seed as
theory/paper_build/gen_proof_figs.py), fig3 = frontier r*(c) at beta = 0.4 / 0.8 with the SAT-/HEDGE-GATE
points of results/frontier_general.csv (validate_frontier.py output).
Usage: python code/v3_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from v3_io import save_fig  # noqa: E402  (atomic writes, C7)

TOPIC = Path(__file__).resolve().parents[1]
FIG = TOPIC / "Revision_v3" / "figures"
RES = TOPIC / "results"
BLUE, GREEN, RED, GRAY = "#2c6fbb", "#1e7a33", "#c0392b", "#888888"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7.5, "font.family": "serif",
                     "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})


def r_star(c, beta):
    if c <= beta:
        return c
    if beta <= 0.5:
        return beta * (1 - c) / (1 - beta)
    return 1 - (1 - beta) * c / beta


def timeline(ax, T, onsets, W, fires, title, fire_color):
    ax.set_xlim(-0.5, T + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(title, loc="left")
    for o in onsets:
        ax.add_patch(Rectangle((o, 0.30), W, 0.40, color=GREEN, alpha=0.22))
        ax.plot([o], [0.5], marker="v", color=GREEN, ms=6)
    for f in fires:
        ax.axvline(f, 0.12, 0.88, color=fire_color, lw=1.4)
    for o in onsets:
        caught = any(o <= f <= o + W for f in fires)
        ax.text(o + W / 2.0, 0.80, "caught" if caught else "missed", ha="center",
                color=(GREEN if caught else RED), fontsize=7.5)
    ax.set_xlabel("frame $t$")


def sch2() -> None:
    T, W = 30, 2
    onsets = [4, 12, 23]
    clutter = [1, 2, 7, 8, 9, 16, 17, 18, 27, 28]
    fig, axes = plt.subplots(2, 1, figsize=(5.35, 2.9), constrained_layout=True)
    timeline(axes[0], T, onsets, W, clutter, "(a) pure-advice gate fires the top-cue (clutter) frames: $R=0$", RED)
    for f in clutter:
        axes[1].axvline(f, 0.12, 0.88, color=RED, lw=0.9, alpha=0.25)
    timeline(axes[1], T, onsets, W, list(range(0, T + 1, 7)),
             r"(b) HEDGE-GATE: the oblivious backbone (blue) guarantees $\lambda\beta$ whatever the cue", BLUE)
    save_fig(fig, FIG / "sch2.png", dpi=300)


def sch5() -> None:
    rng = np.random.default_rng(1)
    N, advised = 16, [2, 5, 9, 12]
    p = np.clip(rng.uniform(0.05, 0.5, N), 0, 1)
    for b in advised:
        p[b] = rng.uniform(0.7, 0.95)
    fig, ax = plt.subplots(figsize=(5.0, 2.4), constrained_layout=True)
    ax.bar(range(N), p, color=[GREEN if b in advised else GRAY for b in range(N)], edgecolor="k", lw=0.5)
    nonadv = [b for b in range(N) if b not in advised]
    for b in sorted(nonadv, key=lambda b: p[b])[:4]:
        ax.bar(b, p[b], color=RED, edgecolor="k", lw=0.5)
    ax.set_xticks(range(N))
    ax.set_xlabel("block $b$")
    ax.set_ylabel(r"$p_b=\Pr[\mathrm{fire\ in\ } b]$")
    ax.legend(handles=[Patch(color=GREEN, label="advised set $A$"), Patch(color=GRAY, label="non-advised"),
                       Patch(color=RED, label=r"events at $\eta{=}1$ (smallest $p_b$)")], loc="upper right", frameon=False)
    ax.set_ylim(0, 1.15)
    save_fig(fig, FIG / "sch5.png", dpi=300)


def fig3() -> None:
    rows = list(csv.DictReader(open(RES / "frontier_general.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.5), constrained_layout=True)
    for ax, beta, title in ((axes[0], 0.4, r"$\beta=0.4$ (scarce budget)"), (axes[1], 0.8, r"$\beta=0.8$ (high budget)")):
        cc = np.linspace(0, 1, 401)
        ax.plot(cc, [r_star(c, beta) for c in cc], "k-", lw=1.4, label=r"frontier $r^\star(c)$")
        sat = [(float(r["c"]), float(r["r"])) for r in rows if r["gate"] == "SAT" and abs(float(r["beta"]) - beta) < 1e-9]
        hed = [(float(r["c"]), float(r["r"])) for r in rows if r["gate"] == "HEDGE" and abs(float(r["beta"]) - beta) < 1e-9]
        if sat:
            ax.plot(*zip(*sat), "o-", color=BLUE, ms=3, lw=0.9, label="SAT-GATE")
        lams = np.linspace(0, 1, 11)
        ax.plot([(1 - l) + l * beta for l in lams], [r_star((1 - l) + l * beta, beta) for l in lams], "o",
                mfc="none", color="tab:green", ms=6, lw=0, label="HEDGE-GATE (Alg. 1, block-aware)")
        if hed:  # validate_frontier.py model: backbone and advice arm may hit the same block
            ax.plot(*zip(*hed), "s--", color=RED, ms=3, lw=0.9, label="naive overlapping split")
        ax.plot([beta], [beta], "*", color="k", ms=8)
        ax.set_xlabel("consistency $c$")
        ax.set_ylabel("robustness $r$")
        ax.set_title(title)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 0.9)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    save_fig(fig, FIG / "fig3.png", dpi=300)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    sch2()
    sch5()
    fig3()
    print("sch2, sch5, fig3 written at 300 dpi to", FIG)
