# -*- coding: utf-8 -*-
import os, csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES=os.path.join(os.path.dirname(__file__),"..","results")
W=7; A_GRID=[0.1,0.2,0.333]; LAM=[0.0,0.25,0.5,0.75,1.0]; PG=[0.0,0.2,0.4,0.6,0.8,1.0]
def cert(a):
    K=math.ceil(1/a); return min(1.0,(W+1)/K),K
def obl(a): return 1-(1-a)**(W+1)
rw=list(csv.DictReader(open(os.path.join(RES,"corruption_sweep_REAL.csv"),encoding="utf-8")))
def get(a,lam,p):
    for r in rw:
        try:
            if abs(float(r["a"])-a)<1e-6 and abs(float(r["lam"])-lam)<1e-6 and abs(float(r["corruption_p"])-p)<1e-6:
                return r
        except: pass
    return None
colors=plt.cm.viridis(np.linspace(0,0.88,len(LAM)))
fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
for ax,a in zip(axes,A_GRID):
    for ci,lam in enumerate(LAM):
        pts=[]
        for p in PG:
            x=get(a,lam,p)
            if not x: continue
            try: pts.append((float(x["eta"]),float(x["recall_mean"]),float(x["ci_lo"]),float(x["ci_hi"])))
            except (TypeError,ValueError): continue
        if not pts: continue
        etas=[q[0] for q in pts]; recs=[q[1] for q in pts]; los=[q[2] for q in pts]; his=[q[3] for q in pts]
        ax.plot(etas,recs,"-o",color=colors[ci],lw=2,ms=4,label="λ=%.2f"%lam)
        ax.fill_between(etas,los,his,color=colors[ci],alpha=0.12)
    c,K=cert(a); ax.axhline(c,ls="--",color="gray",lw=1.1,label="periodic cert")
    ax.axhline(obl(a),ls=":",color="brown",lw=1.1,label="oblivious random")
    ax.set_title("REAL a=%.3f (K=%d)"%(a,K)); ax.set_xlabel("prediction error η (real cue)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("windowed recall R (W=7)"); axes[-1].legend(fontsize=7,loc="lower left")
fig.suptitle("Real-data UNCAPPED (77-video corpus BMC+LASIESTA+CDnet, B=1000): "
             "HEDGE-GATE windowed recall vs η (bands = 95% CI).",fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig5_real_corruption.png"),dpi=300); plt.close(fig)
print("fig5 regenerated (English)")
