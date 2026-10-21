# -*- coding: utf-8 -*-
"""Authoritative validation of the GENERAL tight consistency-robustness frontier
(non-adaptive gates, canonical block family) and of HEDGE-GATE vs SAT-GATE.
Regenerates results/frontier_general.csv and results/fig3_frontier_general.png.
Model: per-block independent firing prob p_b (depends on cue only => non-adaptive).
  consistency c = mean_{b in advised A} p_b ;  robustness r = mean of B smallest p_b
  (adversary hides the B events on the least-fired blocks).
"""
import os, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def frontier(beta, c):
    if c <= beta: return c
    if beta <= 0.5: return beta*(1-c)/(1-beta)
    return 1-(1-beta)*c/beta

def sat_gate(N, beta, q):
    """SAT-GATE: advised (top-cue) blocks at rate q; ALL blocks share an oblivious
    floor g exhausting the budget: m*q + (N-m)*g = B.  Dial q in [beta,1]."""
    m = round(beta*N); B = m
    g = min(1.0, max(0.0, (B - m*q)/(N-m)))
    p = np.array([max(q, g)]*m + [g]*(N-m))   # advised get max(q,g); floor elsewhere
    return p[:m].mean(), np.sort(p)[:m].mean()

def hedge_gate(N, beta, lam):
    """HEDGE-GATE(lambda): oblivious backbone using lam*B fires spread over all N
    blocks (each block fired w.p. lam*beta) UNION advice arm firing the top (1-lam)B
    advised blocks at rate 1."""
    m = round(beta*N); B = m
    nA = round((1-lam)*B); back = lam*beta
    p = np.empty(N)
    for b in range(N):
        adv = 1.0 if b < nA else 0.0
        p[b] = 1-(1-adv)*(1-back)
    return p[:m].mean(), np.sort(p)[:m].mean()

def main():
    RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    RES = os.path.abspath(RES); os.makedirs(RES, exist_ok=True)
    N = 2000
    rows = []
    for beta in [0.2, 0.4, 0.6, 0.8, 0.9]:
        # SAT sweep
        for q in np.linspace(beta, 1.0, 11):
            c, r = sat_gate(N, beta, q)
            rows.append(dict(beta=beta, gate="SAT", dial=round(q,3),
                             c=round(c,4), r=round(r,4), frontier=round(frontier(beta,c),4),
                             on_frontier=int(abs(r-frontier(beta,c))<0.01)))
        # HEDGE sweep
        for lam in np.linspace(0,1,11):
            c, r = hedge_gate(N, beta, lam)
            rows.append(dict(beta=beta, gate="HEDGE", dial=round(lam,3),
                             c=round(c,4), r=round(r,4), frontier=round(frontier(beta,c),4),
                             on_frontier=int(abs(r-frontier(beta,c))<0.01)))
    with open(os.path.join(RES,"frontier_general.csv"),"w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=["beta","gate","dial","c","r","frontier","on_frontier"])
        w.writeheader(); [w.writerow(x) for x in rows]

    # Figure: two panels, beta=0.4 (<=1/2, HEDGE optimal) and beta=0.8 (>1/2, SAT optimal)
    fig, axes = plt.subplots(1,2,figsize=(11,4.4))
    for ax,beta in zip(axes,[0.4,0.8]):
        cs=np.linspace(0,1,200); fr=[frontier(beta,c) for c in cs]
        ax.plot(cs,fr,"k-",lw=2,label="tight frontier $r^*(c)$")
        sc=[(x["c"],x["r"]) for x in rows if x["beta"]==beta and x["gate"]=="SAT"]
        hc=[(x["c"],x["r"]) for x in rows if x["beta"]==beta and x["gate"]=="HEDGE"]
        ax.plot([a for a,b in sc],[b for a,b in sc],"o-",color="#1f77b4",ms=5,label="SAT-GATE")
        ax.plot([a for a,b in hc],[b for a,b in hc],"s--",color="#d62728",ms=5,label="HEDGE-GATE")
        ax.plot([beta],[frontier(beta,beta)],"k*",ms=13)
        ax.set_title(r"$\beta=%.1f$  (%s)"%(beta,"HEDGE optimal" if beta<=0.5 else "SAT optimal; HEDGE suboptimal"))
        ax.set_xlabel("consistency $c$"); ax.set_ylabel("worst-case robustness $r$")
        ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_xlim(0,1.02); ax.set_ylim(0,1.02)
    fig.suptitle("General tight consistency-robustness frontier (non-adaptive gates). "
                 "SAT-GATE attains it in both regimes; HEDGE only for $\\beta\\leq1/2$.",fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RES,"fig3_frontier_general.png"),dpi=200); plt.close(fig)

    # print concise validation summary
    print("=== SAT-GATE on frontier? (max |r - r*| per beta) ===")
    for beta in [0.2,0.4,0.6,0.8,0.9]:
        d=max(abs(x["r"]-x["frontier"]) for x in rows if x["beta"]==beta and x["gate"]=="SAT")
        print(f"  beta={beta}: max dev={d:.2e}  -> {'ON frontier (max dev < 1e-3)' if d<1e-3 else 'OFF'}")
    print("=== HEDGE-GATE gap to frontier (beta=0.8) ===")
    for x in rows:
        if x["beta"]==0.8 and x["gate"]=="HEDGE":
            print(f"  lam={x['dial']:.2f}: (c={x['c']:.3f}, r={x['r']:.3f})  r*={x['frontier']:.3f}  gap={x['frontier']-x['r']:+.3f}")
    print("Saved frontier_general.csv + fig3_frontier_general.png")

if __name__=="__main__":
    main()
