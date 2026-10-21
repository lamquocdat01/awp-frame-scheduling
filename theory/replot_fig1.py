# -*- coding: utf-8 -*-
"""Fig 1 — measured recall-energy anchors (English). Deterministic S/RF are swept curves;
DQN (RL) appears only as isolated per-penalty points at high activation (cannot reach low a)."""
import os, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")
rows=list(csv.DictReader(open(os.path.join(RES,"anchors_pa.csv"),encoding="utf-8")))
a=[float(r["a"]) for r in rows]; per=[float(r["periodic"]) for r in rows]
S=[float(r["S"]) for r in rows]; RF=[float(r["RF"]) for r in rows]
# DQN (RL) per-lambda points from the detection-grounded evaluation (a3_learned_result.json)
dqn=[(0.623,0.749),(0.845,0.862),(0.503,0.608),(0.539,0.737),(0.514,0.590)]
DQN_MIN=0.476
fig,ax=plt.subplots(figsize=(7.6,6))
# deployment regime shading (a <= 0.333) and DQN-infeasible region (a < 0.476)
ax.axvspan(0.0,0.333,color="green",alpha=0.05)
ax.axvline(DQN_MIN,ls=":",color="gray",lw=1.2)
ax.text(DQN_MIN+0.006,0.06,"DQN min\nactivation 0.48",fontsize=8.5,color="dimgray",va="bottom")
ax.plot(a,per,"-o",color="green",lw=2.2,ms=7,label="periodic (oblivious)")
ax.plot(a,S,"-s",color="navy",lw=2.2,ms=7,label="S (score-gate, swept)")
ax.plot(a,RF,"-^",color="crimson",lw=2.2,ms=7,label="RF (random-forest, swept)")
ax.scatter([p[0] for p in dqn],[p[1] for p in dqn],marker="X",s=90,color="darkorange",
           edgecolor="k",zorder=5,label="DQN (RL), per-penalty points")
ax.axhline(1.0,ls="--",color="gray",lw=1.2,label="oracle always-on")
ax.set_xlim(0.05,0.9); ax.set_ylim(0,1.03)
ax.set_xlabel("activation a (energy)"); ax.set_ylabel("event recall (W=7)"); ax.grid(alpha=0.3)
ax.set_title("Measured recall-energy anchors (benchmark corpus)\n"
             "a=0.20: periodic 0.984 / S 0.284 / RF 0.200; DQN cannot reach a<=0.33")
ax.legend(loc="lower right",fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig1_anchor_frontier_PA.png"),dpi=300); plt.close(fig)
print("fig1 regenerated with DQN scatter")
