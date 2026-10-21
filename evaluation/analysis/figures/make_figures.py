"""
Generate Fig.1-4 for the C4 methodology paper from results/*.json.
Vector PDF (IEEEtran column width). Colorblind-safe (Wong). Error bars = bootstrap
95% CI (B=1000, seed=42) where available in the JSON. Honest: shows debunk direction + CI.
Run:  python analysis/figures/make_figures.py   (writes paper/figures/*.pdf)
"""
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES, OUT = ROOT / "results", ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
def load(n): return json.load(open(RES / f"{n}.json"))
W = {"blue":"#0072B2","orange":"#E69F00","green":"#009E73","pink":"#CC79A7","gray":"#999999","red":"#D55E00"}
plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.bbox":"tight","pdf.fonttype":42})
def err(point, ci):  # -> yerr [[lo],[hi]] or None
    if not ci or ci[0] is None: return None
    return [[max(0,point-ci[0])],[max(0,ci[1]-point)]]

# ---------- Fig.1: protocol flip + cross-dataset (LODO) ----------
def fig1():
    n1=load("n1_crossdata")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,2.7))
    # (a) within-CDnet SS vs SI skip-rate (the flip)
    pol=["periodic","S-gate","pi-gate"]; cols=[W["blue"],W["orange"],W["green"]]
    ss=[n1["within_cdnet"]["SS"][p] for p in pol]; si=[n1["within_cdnet"]["SI"][p] for p in pol]
    x=np.arange(len(pol)); w=0.36
    ax1.bar(x-w/2,ss,w,label="Same-scene (SS)",color=cols,alpha=0.55,edgecolor="k",linewidth=.4)
    ax1.bar(x+w/2,si,w,label="Scene-indep (SI/LOCO)",color=cols,edgecolor="k",linewidth=.4)
    ax1.set_xticks(x); ax1.set_xticklabels(pol); ax1.set_ylabel("skip rate @ iso-90% recall")
    ax1.set_title(f"(a) Within-CDnet protocol flip (Spearman {n1['within_cdnet']['spearman']:+.0f})")
    ax1.legend(fontsize=6,loc="upper right")
    # (b) LODO per held-out dataset: skip @iso-90% periodic vs pi-gate (higher=better)
    ds=n1["meta"]["datasets"]; xb=np.arange(len(ds)); wb=0.36
    per=[n1["folds"][d]["SI"]["periodic"]["skip"] for d in ds]
    pig=[n1["folds"][d]["SI"]["pi-gate"]["skip"] for d in ds]
    perc=[err(per[i],n1["folds"][d]["SI_boot"]["periodic"].get("skip_ci")) for i,d in enumerate(ds)]
    pigc=[err(pig[i],n1["folds"][d]["SI_boot"]["pi-gate"].get("skip_ci")) for i,d in enumerate(ds)]
    def ye(es):
        lo=[e[0][0] if e else 0 for e in es]; hi=[e[1][0] if e else 0 for e in es]; return [lo,hi]
    ax2.bar(xb-wb/2,per,wb,yerr=ye(perc),capsize=2,label="periodic (oblivious)",color=W["blue"],edgecolor="k",linewidth=.4)
    ax2.bar(xb+wb/2,pig,wb,yerr=ye(pigc),capsize=2,label="content-aware gate ($\\hat\\pi$)",color=W["green"],edgecolor="k",linewidth=.4)
    ax2.set_xticks(xb); ax2.set_xticklabels(ds,rotation=12); ax2.set_ylabel("skip rate @ iso-90% recall")
    ax2.set_title("(b) Cross-dataset (LODO): periodic dominates"); ax2.legend(fontsize=6,loc="upper right")
    fig.tight_layout(); fig.savefig(OUT/"fig1_protocol_flip.pdf"); plt.close(fig); print("fig1_protocol_flip.pdf")

# ---------- Fig.2: per-scene-K oracle vs gate vs scene-agnostic vs estimator ----------
def fig2():
    ps=load("perscene_k")
    order=[("scene_agnostic","scene-agnostic\nglobal-K",W["gray"]),
           ("gate_S","gate (S)",W["orange"]),("gate_pi","gate ($\\hat\\pi$)",W["green"]),
           ("est_W5","zero-shot\nestimator",W["pink"]),
           ("oracle_cat","oracle\nper-category",W["blue"]),("oracle_vid","oracle\nper-video",W["red"])]
    pts=[ps["brackets_pooled"][k]["point"] for k,_,_ in order]
    cis=[ps["brackets_pooled"][k].get("ci") for k,_,_ in order]
    ye=[[max(0,p-c[0]) for p,c in zip(pts,cis)],[max(0,c[1]-p) for p,c in zip(pts,cis)]]
    fig,ax=plt.subplots(figsize=(3.5,2.8)); x=np.arange(len(order))
    ax.bar(x,pts,0.7,yerr=ye,capsize=2.5,color=[c for _,_,c in order],edgecolor="k",linewidth=.4)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l,_ in order],fontsize=6)
    ax.set_ylabel(r"skip rate $\rho_{\max}$ @ recall$\geq$0.90"); ax.set_ylim(0,0.95)
    fr=ps["fraction_of_oracle_gain"]["vs_scene_agnostic"]
    ax.set_title("Per-scene-K headroom is oracle-only",fontsize=8)
    ax.annotate(f"estimator recovers {fr['point']:.2f}\n[{fr['ci'][0]:+.2f},{fr['ci'][1]:+.2f}] of gain\n(CI incl. 0)",
                xy=(3,ps['brackets_pooled']['est_W5']['point']),xytext=(1.2,0.82),fontsize=6,
                ha="left",arrowprops=dict(arrowstyle="->",lw=.6))
    fig.tight_layout(); fig.savefig(OUT/"fig2_perscene_k.pdf"); plt.close(fig); print("fig2_perscene_k.pdf")

# ---------- Fig.3: regime trend (gate - oblivious advantage vs activation) ----------
def fig3():
    n3=load("n3_rareevent")
    hh=n3["head_to_head_sparse"]; a=sorted(float(k) for k in hh)
    def series(key,cikey):
        v=[hh[f"{x:g}"][key] for x in a]; ci=[hh[f"{x:g}"][cikey] for x in a]
        lo=np.array([c[0] for c in ci]); hi=np.array([c[1] for c in ci]); return np.array(v),lo,hi
    sa,sl,sh=series("S_adv","S_adv_ci"); pa,pl,ph=series("pi_adv","pi_adv_ci")
    fig,ax=plt.subplots(figsize=(3.5,2.7))
    ax.axhline(0,color="k",lw=.6); ax.axvspan(0,0.20,color=W["gray"],alpha=0.15)
    ax.plot(a,sa,"-o",ms=3,color=W["orange"],label="gate (S) $-$ oblivious"); ax.fill_between(a,sl,sh,color=W["orange"],alpha=.2)
    ax.plot(a,pa,"-s",ms=3,color=W["green"],label="gate ($\\hat\\pi$) $-$ oblivious"); ax.fill_between(a,pl,ph,color=W["green"],alpha=.2)
    ax.set_xlabel("activation $a$ (energy budget)"); ax.set_ylabel("equal-energy recall advantage")
    ax.set_title(f"Gate advantage only at high $a$\n(Spearman {n3['trend']['spearman_inevent_vs_advantage']:+.3f} vs in-event)",fontsize=7.5)
    ax.text(0.10,ax.get_ylim()[1]*0.92,"deployable\nlow-$a$",fontsize=6,ha="center",color=W["gray"])
    ax.legend(fontsize=6,loc="lower right")
    fig.tight_layout(); fig.savefig(OUT/"fig3_regime_trend.pdf"); plt.close(fig); print("fig3_regime_trend.pdf")

# ---------- Fig.4: in-event fraction (rare-event misnomer) + duration law ----------
def fig4():
    n3=load("n3_rareevent"); pdi=n3["inventory"]["per_dataset_inevent"]
    dl=n3["mechanism"]["sparse_duration_law"]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(7.0,2.6))
    ds=["CDnet2014","LASIESTA","BMC","VIRAT"]; means=[pdi[d]["mean"] for d in ds]
    cols=[W["blue"],W["orange"],W["green"],W["red"]]
    b=ax1.bar(ds,means,0.7,color=cols,edgecolor="k",linewidth=.4)
    ax1.set_ylabel("mean in-event fraction"); ax1.set_ylim(0,1.08); ax1.axhline(1.0,ls="--",color="k",lw=.5)
    ax1.set_xticklabels(ds,rotation=12)
    ax1.set_title('(a) "Rare-event" is a misnomer\n(detection-grounded)',fontsize=7.5)
    ax1.annotate("VIRAT = 1.000\n(detector fires on\nstatic objects)",xy=(3,1.0),xytext=(1.3,0.55),fontsize=6,
                 arrowprops=dict(arrowstyle="->",lw=.6))
    # (b) sparse-regime duration law
    labels=["single-frame\n(D=1)","D≥5","other"]; fs=dl["frac_single"]; fg=dl["frac_ge5"]; other=max(0,1-fs-fg)
    ax2.bar(labels,[fs,fg,other],0.6,color=[W["red"],W["green"],W["gray"]],edgecolor="k",linewidth=.4)
    ax2.set_ylabel("fraction of sparse-regime events"); ax2.set_ylim(0,0.75)
    ax2.set_title(f"(b) Sparse events are mostly single-frame\n(median dur {dl['median']:.0f}, n={dl['n_events']})",fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT/"fig4_duration_law.pdf"); plt.close(fig); print("fig4_duration_law.pdf")

if __name__=="__main__":
    fig1(); fig2(); fig3(); fig4()
    print("done ->",OUT)
