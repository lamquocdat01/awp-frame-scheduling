# -*- coding: utf-8 -*-
"""Redraw fig6 (per-category deployment gap) from results/percategory.csv.

Same data and ordering as awp_percategory.py's figure block; geometry/typography
changed for the TPDS supplementary: drawn at 2.94 in (= 0.84 * IEEE column width,
the size it is included at, so 8 pt in matplotlib = 8 pt on paper), and the
embedded title is dropped -- the LaTeX caption carries it (same rule as the
fig2/fig4/fig5 single-column redraw). Output: figures/fig6.png.
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "percategory.csv")
OUT = os.path.join(HERE, "..", "figures", "fig6.png")

rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["group"] != "ALL"]
rows.sort(key=lambda r: float(r["gap_a020"]))  # most negative at the bottom, as before

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
labels = [r["group"].replace("CDnet:", "") for r in rows]
gaps = [float(r["gap_a020"]) for r in rows]

fig, ax = plt.subplots(figsize=(2.94, 2.75))
ax.barh(labels, gaps, color="#c0392b")
ax.axvline(0, color="k", lw=0.8)
ax.tick_params(length=2, pad=1.5)
fig.tight_layout(pad=0.25, rect=(0, 0.14, 1, 1))
# x-label centered on the FIGURE (not the axes) so it cannot run off the edge
fig.supxlabel("gate $-$ periodic recall gap @ $a$=0.20\n(negative = periodic wins)",
              fontsize=8, y=0.015)
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT, "bars:", len(labels))
