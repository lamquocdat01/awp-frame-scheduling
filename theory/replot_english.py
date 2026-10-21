# -*- coding: utf-8 -*-
"""Regenerate fig2 (noise), fig4 (adversarial), fig5 (real) with ENGLISH-ONLY text,
reading the existing result CSVs (no heavy pipeline)."""
import os, csv, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES=os.path.join(os.path.dirname(__file__),"..","results")
W=7; A_GRID=[0.1,0.2,0.333]; LAM=[0.0,0.25,0.5,0.75,1.0]; PG=[0.0,0.2,0.4,0.6,0.8,1.0]
AUC=(0.4934,0.7913)
def cert(a):
    K=math.ceil(1/a); return min(1.0,(W+1)/K),K
def obl(a): return 1-(1-a)**(W+1)
def rd(fn):
    return list(csv.DictReader(open(os.path.join(RES,fn),encoding="utf-8")))
colors=plt.cm.viridis(np.linspace(0,0.88,len(LAM)))

def get(rows,**kw):
    for r in rows:
        if all(abs(float(r[k])-v)<1e-6 if k!="regime" else r[k]==v for k,v in kw.items()):
            return r
    return None

# ---- fig2 noise + fig4 adversarial from corruption_sweep.csv ----
sw=rd("corruption_sweep.csv")
eta_noise={p:float(get(sw,regime="noise",a=0.1,lam=0.0,corruption_p=p)["eta"]) for p in PG}
# FIG 2
fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
for ax,a in zip(axes,A_GRID):
    for ci,lam in enumerate(LAM):
        recs=[float(get(sw,regime="noise",a=a,lam=lam,corruption_p=p)["recall_mean"]) for p in PG]
        ax.plot([eta_noise[p] for p in PG],recs,"-o",color=colors[ci],lw=2,ms=4,label="λ=%.2f"%lam)
    c,K=cert(a); ax.axhline(c,ls="--",color="gray",lw=1.1,label="periodic cert")
    ax.axhline(obl(a),ls=":",color="brown",lw=1.1,label="oblivious random")
    ax.axvspan(1-AUC[1],1-AUC[0],color="orange",alpha=0.12)
    ax.set_title("a=%.3f (K=%d)"%(a,K)); ax.set_xlabel("prediction error η"); ax.grid(alpha=0.3)
axes[0].set_ylabel("windowed recall R (W=7)"); axes[-1].legend(fontsize=7,loc="lower left")
fig.suptitle("NOISE regime — HEDGE-GATE recall degrades smoothly in η (Prop. 4). "
             "Uninformative noise slides recall toward the oblivious-random level without breaking the floor.",fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig2_noise_smoothness.png"),dpi=300); plt.close(fig)

# FIG 4
anch={float(r["a"]):(float(r["S"]),float(r["RF"])) for r in rd("anchors_pa.csv")}
fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
for ax,a in zip(axes,A_GRID):
    for ci,lam in enumerate(LAM):
        recs=[float(get(sw,regime="adversarial",a=a,lam=lam,corruption_p=p)["recall_mean"]) for p in PG]
        ax.plot(PG,recs,"-o",color=colors[ci],lw=2,ms=4,label="λ=%.2f"%lam)
    c,K=cert(a); ax.axhline(c,ls="--",color="gray",lw=1.1,label="periodic cert")
    ax.axhline(obl(a),ls=":",color="brown",lw=1.1,label="oblivious random")
    ak=min(anch,key=lambda x:abs(x-a)); s,rf=anch[ak]
    ax.axhspan(min(s,rf),max(s,rf),color="red",alpha=0.10)
    ax.axhline(s,color="red",lw=0.8,alpha=0.5); ax.axhline(rf,color="red",lw=0.8,alpha=0.5)
    ax.set_title("a=%.3f (K=%d)"%(a,K)); ax.set_xlabel("adversarial corruption p"); ax.grid(alpha=0.3)
axes[0].set_ylabel("windowed recall R (W=7)"); axes[-1].legend(fontsize=7,loc="lower left")
fig.suptitle("ADVERSARIAL regime — the pure gate (λ=0) collapses BELOW the oblivious-random level "
             "(worst case, Theorem 1′); HEDGE-GATE (λ>0) holds its backbone floor. Red band = measured P-A anchors (S/RF gates).",fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig4_adversarial_floor.png"),dpi=300); plt.close(fig)

# ---- fig5 real from corruption_sweep_REAL.csv ----
rw=rd("corruption_sweep_REAL.csv")
fig,axes=plt.subplots(1,3,figsize=(15,4.2),sharey=True)
for ax,a in zip(axes,A_GRID):
    for ci,lam in enumerate(LAM):
        rows=[get(rw,a=a,lam=lam,corruption_p=p) for p in PG]
        rows=[x for x in rows if x is not None]
        etas=[float(x["eta"]) for x in rows]; recs=[float(x["recall_mean"]) for x in rows]
        los=[float(x["ci_lo"]) for x in rows]; his=[float(x["ci_hi"]) for x in rows]
        ax.plot(etas,recs,"-o",color=colors[ci],lw=2,ms=4,label="λ=%.2f"%lam)
        ax.fill_between(etas,los,his,color=colors[ci],alpha=0.12)
    c,K=cert(a); ax.axhline(c,ls="--",color="gray",lw=1.1,label="periodic cert")
    ax.axhline(obl(a),ls=":",color="brown",lw=1.1,label="oblivious random")
    ax.set_title("REAL a=%.3f (K=%d)"%(a,K)); ax.set_xlabel("prediction error η (real cue)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("windowed recall R (W=7)"); axes[-1].legend(fontsize=7,loc="lower left")
fig.suptitle("Real-data UNCAPPED (77-video corpus BMC+LASIESTA+CDnet, B=1000): "
             "HEDGE-GATE windowed recall vs η (bands = 95% CI).",fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(RES,"fig5_real_corruption.png"),dpi=300); plt.close(fig)
print("regenerated fig2, fig4, fig5 (English)")
