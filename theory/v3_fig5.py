# -*- coding: utf-8 -*-
"""v3_fig5.py -- fig5 of paper_v3 (real-data trust-dial sweep), redrawn for P3C.

Geometry and style follow Revision_TPDS/figs_1col/replot_1col.py (3 panels stacked, 3.5 in wide, 300 dpi).
Solid lines + markers: HEDGE-GATE as in Algorithm 1 of v3 (block-aware split, uniform choice;
results/v3/corruption_sweep_REAL_blockaware.csv, code/v3_sweep_blockaware.py), with the 95% video-level
bootstrap band. Dashed lines: the frame-level split of the TPDS version (results/corruption_sweep_REAL.csv),
kept as the control, as in fig3. The two coincide at lambda in {0, 1} (checked by v3_sweep_blockaware.py).
"""
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from v3_io import save_fig  # noqa: E402  (atomic writes, C7)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "Revision_v3", "figures", "fig5.png")
COL_IN = 3.5
W = 7
A_GRID = [0.1, 0.2, 0.333]
LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
PG = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "lines.linewidth": 1.3, "lines.markersize": 2.6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300,
})


def cert(a):
    K = math.ceil(1 / a)
    return min(1.0, (W + 1) / K), K


def obl(a):
    return 1 - (1 - a) ** (W + 1)


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def get(rows, a, lam, p):
    for r in rows:
        if abs(float(r["a"]) - a) < 1e-6 and abs(float(r["lam"]) - lam) < 1e-6 and abs(float(r["corruption_p"]) - p) < 1e-6:
            return r
    return None


ba = rd(os.path.join(RES, "v3", "corruption_sweep_REAL_blockaware.csv"))
nv = rd(os.path.join(RES, "corruption_sweep_REAL.csv"))
colors = plt.cm.viridis(np.linspace(0, 0.88, len(LAM)))
fig, axes = plt.subplots(3, 1, figsize=(COL_IN, 2.9), sharex=True)
for ax, a in zip(axes, A_GRID):
    for ci, lam in enumerate(LAM):
        rb = [get(ba, a, lam, p) for p in PG]
        etas = [float(x["eta"]) for x in rb]
        ax.plot(etas, [float(x["recall_mean"]) for x in rb], "-o", color=colors[ci], label=r"$\lambda$=%.2f" % lam)
        ax.fill_between(etas, [float(x["ci_lo"]) for x in rb], [float(x["ci_hi"]) for x in rb], color=colors[ci], alpha=0.12)
        if 0 < lam < 1:
            rn = [get(nv, a, lam, p) for p in PG]
            ax.plot([float(x["eta"]) for x in rn], [float(x["recall_mean"]) for x in rn], "--", color=colors[ci], lw=0.9)
    c, K = cert(a)
    ax.axhline(c, ls="--", color="gray", lw=0.9)
    ax.axhline(obl(a), ls=":", color="brown", lw=0.9)
    ax.set_title("real, $a$=%.3f ($K$=%d)" % (a, K), pad=2)
for ax in axes:
    ax.grid(alpha=0.3, lw=0.4)
    ax.tick_params(length=2, pad=1.5)
    ax.set_ylabel("recall $R$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
axes[-1].set_xlabel(r"prediction error $\eta$ (real cue)")
h, l = axes[0].get_legend_handles_labels()
h += [Line2D([], [], color="k", lw=1.3, marker="o", ms=2.6), Line2D([], [], color="k", lw=0.9, ls="--")]
l += ["block-aware (Alg. 1)", "frame-level (TPDS)"]
axes[-1].legend(h, l, ncol=4, loc="lower left", frameon=True, framealpha=0.85, borderpad=0.25, handlelength=1.4,
               columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2, fontsize=5.6)
fig.tight_layout(pad=0.25, h_pad=0.35)
save_fig(fig, OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("wrote", OUT)
