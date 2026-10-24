# -*- coding: utf-8 -*-
"""v3 (AwP): copy of Revision_TPDS/figs_1col/merge_fig24.py; the red S/RF benchmark-anchor band
(numbers belonging to topic #47, see SPLIT_03_vs_47.md) and the benchmark AUC band are removed; the
measured C121 deployment error eta = 1 - AUC (numbers.json) is marked instead.

B2: merge fig2 (noise) and fig4 (adversarial) into ONE two-column figure*.

Both panels share the same structure (3 activation levels x 5 trust levels); only
the x axis differs (prediction error eta vs adversarial corruption p). Drawing
them as a 3x2 grid inside a single \\textwidth float saves one caption and, more
importantly, packs better than two separate single-column floats -- the page
audit showed those were being pushed and leaving pages 9-11 half empty.

Data is read from the same CSVs, so every plotted value is unchanged.
"""
import os, csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "..", "results")
OUT  = os.path.join(HERE, "..", "Revision_v3", "figures")
import json
from v3_io import save_fig  # noqa: E402  (atomic writes, C7)
ETA_DEP = json.load(open(os.path.join(HERE, "..", "Revision_v3", "numbers.json"), encoding="utf-8"))["eta_deployed"]

TEXT_IN = 7.16
W = 7
A_GRID = [0.1, 0.2, 0.333]
LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
PG  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
AUC = (0.4934, 0.7913)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "lines.linewidth": 1.2, "lines.markersize": 2.4,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300,
})

def cert(a):
    K = math.ceil(1 / a); return min(1.0, (W + 1) / K), K
def obl(a):
    return 1 - (1 - a) ** (W + 1)
def rd(fn):
    return list(csv.DictReader(open(os.path.join(RES, fn), encoding="utf-8")))
def get(rows, **kw):
    for r in rows:
        if all((r[k] == v) if k == "regime" else (abs(float(r[k]) - v) < 1e-6)
               for k, v in kw.items()):
            return r
    return None

colors = plt.cm.viridis(np.linspace(0, 0.88, len(LAM)))
sw = rd("corruption_sweep.csv")                       # TPDS frame-level split (dashed control)
ba = rd(os.path.join("v3", "corruption_sweep_blockaware.csv"))  # Algorithm 1 block-aware split (P4, solid)
eta_noise = {p: float(get(sw, regime="noise", a=0.1, lam=0.0, corruption_p=p)["eta"]) for p in PG}

fig, axes = plt.subplots(3, 2, figsize=(TEXT_IN, 2.55), sharey=True)
for row, a in enumerate(A_GRID):
    c, K = cert(a)
    # ---- left column: noise regime ----
    ax = axes[row][0]
    for ci, lam in enumerate(LAM):
        recs = [float(get(ba, regime="noise", a=a, lam=lam, corruption_p=p)["recall_mean"]) for p in PG]
        ax.plot([eta_noise[p] for p in PG], recs, "-o", color=colors[ci], label=r"$\lambda$=%.2f" % lam)
        if 0 < lam < 1:
            rn = [float(get(sw, regime="noise", a=a, lam=lam, corruption_p=p)["recall_mean"]) for p in PG]
            ax.plot([eta_noise[p] for p in PG], rn, "--", color=colors[ci], lw=0.8)
    ax.axvline(ETA_DEP, color="orange", lw=1.0)
    ax.set_ylabel("$R$ ($a$=%.3f)" % a)
    # ---- right column: adversarial regime ----
    ax2 = axes[row][1]
    for ci, lam in enumerate(LAM):
        recs = [float(get(ba, regime="adversarial", a=a, lam=lam, corruption_p=p)["recall_mean"]) for p in PG]
        ax2.plot(PG, recs, "-o", color=colors[ci], label=r"$\lambda$=%.2f" % lam)
        if 0 < lam < 1:
            rn = [float(get(sw, regime="adversarial", a=a, lam=lam, corruption_p=p)["recall_mean"]) for p in PG]
            ax2.plot(PG, rn, "--", color=colors[ci], lw=0.8)
    for ax_ in (ax, ax2):
        ax_.axhline(c, ls="--", color="gray", lw=0.9)
        ax_.axhline(obl(a), ls=":", color="brown", lw=0.9)
        ax_.grid(alpha=0.3, lw=0.4)
        ax_.tick_params(length=2, pad=1.5)
        ax_.set_ylim(-0.03, 1.08)
        ax_.set_yticks([0, 0.5, 1.0])
        if row < 2:
            ax_.tick_params(labelbottom=False)
axes[0][0].set_title("noise regime ($K$=10, 5, 4 top to bottom)", pad=2)
axes[0][1].set_title("adversarial regime", pad=2)
axes[2][0].set_xlabel(r"prediction error $\eta$")
axes[2][1].set_xlabel("adversarial corruption $p$")
axes[0][0].legend(ncol=5, loc="lower left", frameon=True, framealpha=0.85,
                  borderpad=0.2, handlelength=1.1, columnspacing=0.6,
                  handletextpad=0.3, borderaxespad=0.15)
fig.tight_layout(pad=0.25, h_pad=0.3, w_pad=0.8)
p = os.path.join(OUT, "fig24.png")
save_fig(fig, p, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", p)

