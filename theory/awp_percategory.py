# -*- coding: utf-8 -*-
"""
awp_percategory.py — Bảng per-category / per-dataset từ corpus 121 video đã cache (cap 300, ROI).
Xuất results/percategory.csv + results/fig6_percategory.png. Không trích lại ảnh (dùng cache).
"""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awp_real_data as R
from awp_experiments import (hedge_gate_fire, recall_windowed, periodic_certificate,
                             onset_auc, eta_from_auc, W, RESULTS_DIR)

def group_of(ds, vid):
    if ds == "CDnet":            # vid = CDN_<cat>_<video>
        return "CDnet:" + vid.split("_", 2)[1]
    return ds

# nạp corpus từ cache
corpus = []
for ds, vid, f, g in R.enumerate_videos():
    cp = R._cache_path(vid)
    if not os.path.exists(cp):
        continue
    z = np.load(cp); cue = z["cue"]; gt = z["gt"].astype(bool)
    ev = R.extract_events(gt)
    if len(ev) >= 2:
        win = np.zeros(len(gt), bool)
        for e in ev:
            win[e["onset"]:min(len(gt), e["onset"] + W + 1)] = True
        corpus.append(dict(grp=group_of(ds, vid), ds=ds, cue=cue, events=ev, win=win, T=len(gt)))

print("corpus videos(>=2 ev):", len(corpus), "groups:", len(set(c["grp"] for c in corpus)))

# per-group: gate(λ=0) vs periodic(λ=1) at a=0.1,0.2; onset-AUC; n; events
def grp_stats(rows, a):
    gate = np.mean([recall_windowed(hedge_gate_fire(c["cue"], c["T"], a, 0.0), c["events"], W) for c in rows])
    per = np.mean([recall_windowed(hedge_gate_fire(c["cue"], c["T"], a, 1.0), c["events"], W) for c in rows])
    return gate, per

groups = sorted(set(c["grp"] for c in corpus))
out = []
for grp in groups:
    rows = [c for c in corpus if c["grp"] == grp]
    auc = float(np.mean([onset_auc(c["cue"], c["win"]) for c in rows]))
    nev = sum(len(c["events"]) for c in rows)
    g1, p1 = grp_stats(rows, 0.10); g2, p2 = grp_stats(rows, 0.20)
    out.append(dict(group=grp, n_videos=len(rows), n_events=nev, onset_auc=round(auc, 3),
                    gate_a010=round(g1, 3), periodic_a010=round(p1, 3), gap_a010=round(g1 - p1, 3),
                    gate_a020=round(g2, 3), periodic_a020=round(p2, 3), gap_a020=round(g2 - p2, 3)))

# overall row
allrows = corpus
auc = float(np.mean([onset_auc(c["cue"], c["win"]) for c in allrows]))
g1, p1 = grp_stats(allrows, 0.10); g2, p2 = grp_stats(allrows, 0.20)
out.append(dict(group="ALL", n_videos=len(allrows), n_events=sum(len(c["events"]) for c in allrows),
                onset_auc=round(auc, 3), gate_a010=round(g1, 3), periodic_a010=round(p1, 3),
                gap_a010=round(g1 - p1, 3), gate_a020=round(g2, 3), periodic_a020=round(p2, 3),
                gap_a020=round(g2 - p2, 3)))

fields = ["group", "n_videos", "n_events", "onset_auc", "gate_a010", "periodic_a010", "gap_a010",
          "gate_a020", "periodic_a020", "gap_a020"]
with open(os.path.join(RESULTS_DIR, "percategory.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); [w.writerow(r) for r in out]

# figure: gap@a=0.2 per group (all negative => periodic wins everywhere)
g = [r for r in out if r["group"] != "ALL"]
g = sorted(g, key=lambda r: r["gap_a020"])
fig, ax = plt.subplots(figsize=(8, 4.5))
labels = [r["group"].replace("CDnet:", "") for r in g]
ax.barh(labels, [r["gap_a020"] for r in g], color="#c0392b")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("gate − periodic recall gap @ a=0.20 (negative = periodic wins)")
ax.set_title("Per-category deployment gap (121-video corpus): the motion gate loses everywhere")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "fig6_percategory.png"), dpi=200); plt.close(fig)

print("== per-group (gap@0.2) ==")
for r in out:
    print("  %-26s n=%2d ev=%4d AUC=%.3f gate@.2=%.3f per@.2=%.3f gap=%+.3f"
          % (r["group"], r["n_videos"], r["n_events"], r["onset_auc"], r["gate_a020"],
             r["periodic_a020"], r["gap_a020"]))
print("Saved percategory.csv + fig6_percategory.png")
