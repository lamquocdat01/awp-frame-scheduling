# -*- coding: utf-8 -*-
"""v3_a8c_alloc.py -- optimal tier budget allocation (Theorem 7, A8(c)), AwP v3.

W = 7. Nodes n in {8, 32, 128}; weights: equal, Zipf (w_i ~ 1/i), Dirichlet(1) (seeds 42..46);
per-node consistency c in {0.8, 0.9, 1.0}; mean activation a_bar in {0.03, 0.05, 0.0625, 0.08, 0.1}.
Compared (Model T1 = budget-scaled worst case, Revision_v3/theory/A8c_allocation.md):
  uniform    -- every node at a_bar, integer stride K = ceil(1/a_bar)  (Proposition 6 value)
  dp_opt     -- exact multiple-choice-knapsack DP over integer strides (Theorem 7(C))
  greedy     -- continuous bang-bang allocation of Theorem 7(A2)
  U          -- continuous upper bound of Theorem 7(A1) (envelope min(beta, c))
Model T2 (fixed loads) check: identical loads -> uniform optimal; heterogeneous loads -> water-filling gain.
Outputs: results/v3/a8c_alloc.csv, a8c_T2.csv, a8c_summary.json, Revision_v3/figures/F_A8c.{png,pdf}.

Usage: python code/v3_a8c_alloc.py [--seeds 42 43 44 45 46] [--grid 20000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

TOPIC = Path(__file__).resolve().parents[1]
OUT = TOPIC / "results" / "v3"
FIG = TOPIC / "Revision_v3" / "figures"
W = 7
KMAX = 400
LEVELS = [None] + list(range(W + 1, KMAX + 1))  # None = node switched off (beta = 0)


def r_star(c: float, beta: float) -> float:
    if beta <= 0:
        return 0.0
    if c <= beta:
        return c
    if beta <= 0.5:
        return beta * (1 - c) / (1 - beta)
    return 1 - (1 - beta) * c / beta


def beta_of_K(K):
    return 0.0 if K is None else min(1.0, (W + 1) / K)


def cost_of_K(K):
    return 0.0 if K is None else 1.0 / K


def dp_opt(w: np.ndarray, c: float, A: float, L: int) -> tuple[float, list]:
    """Exact DP for integer strides; costs rounded UP to the grid (allocation always feasible)."""
    n = len(w)
    delta = A / L
    vals = np.array([r_star(c, beta_of_K(K)) for K in LEVELS])
    costs = np.array([math.ceil(cost_of_K(K) / delta - 1e-9) for K in LEVELS], int)
    neg = -1e18
    D = np.full(L + 1, neg)
    D[0] = 0.0
    choice = np.zeros((n, L + 1), np.int16)
    for i in range(n):
        best = np.full(L + 1, neg)
        arg = np.zeros(L + 1, np.int16)
        for j, (cst, v) in enumerate(zip(costs, vals)):
            if cst > L:
                continue
            cand = np.full(L + 1, neg)
            cand[cst:] = D[: L + 1 - cst] + w[i] * v
            upd = cand > best + 1e-15
            best[upd] = cand[upd]
            arg[upd] = j
        D = best
        choice[i] = arg
    ell = int(np.argmax(D))
    opt = float(D[ell])
    alloc = []
    for i in range(n - 1, -1, -1):
        j = int(choice[i, ell])
        alloc.append(LEVELS[j])
        ell -= costs[j]
    alloc.reverse()
    return opt, alloc


def greedy_continuous(w: np.ndarray, c: float, A: float) -> tuple[float, float, np.ndarray]:
    b = (W + 1) * A
    order = np.argsort(-w, kind="stable")
    beta = np.zeros(len(w))
    rem = b
    U = 0.0
    for i in order:
        give = min(c, rem)
        beta[i] = give
        U += w[i] * give
        rem -= give
        if rem <= 1e-12:
            break
    val = float(sum(w[i] * r_star(c, beta[i]) for i in range(len(w))))
    return float(U), val, beta


def weights(kind: str, n: int, seed: int) -> np.ndarray:
    if kind == "equal":
        w = np.ones(n)
    elif kind == "zipf":
        w = 1.0 / np.arange(1, n + 1)
    else:
        w = np.random.default_rng(seed).dirichlet(np.ones(n))
    return w / w.sum()


def t2_frontier(beta: float, mu: float, c: float) -> float:
    if beta < mu * c - 1e-12:
        return float("nan")
    lin = (beta - mu * c) / (1 - mu) if mu <= 0.5 else (beta - (1 - mu) * c) / mu
    return min(c, lin)


def t2_waterfill(w, mu, c, b):
    """Theorem 7(B): floors mu_i c, then greedy by w_i * slope_i up to c."""
    floor = mu * c
    if floor.sum() > b + 1e-12:
        return float("nan"), None
    beta = floor.copy()
    rem = b - floor.sum()
    slope = 1.0 / np.maximum(1 - mu, mu)
    for i in np.argsort(-(w * slope), kind="stable"):
        give = min(c - beta[i], rem)
        beta[i] += give
        rem -= give
    return float(sum(w[i] * t2_frontier(beta[i], mu[i], c) for i in range(len(w)))), beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--grid", type=int, default=20000)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if a.plot_only:
        make_figure(a.grid)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (8, 32, 128):
        for wk in ("equal", "zipf", "dirichlet"):
            seeds = a.seeds if wk == "dirichlet" else [a.seeds[0]]
            for c in (0.8, 0.9, 1.0):
                for abar in (0.03, 0.05, 0.0625, 0.08, 0.1):
                    rec = {"uniform": [], "dp_opt": [], "greedy": [], "U": []}
                    for sd in seeds:
                        w = weights(wk, n, sd)
                        A = n * abar
                        Ku = math.ceil(1 / abar - 1e-12)
                        uni = float(sum(w) * r_star(c, beta_of_K(Ku)))
                        opt, _ = dp_opt(w, c, A, a.grid)
                        U, gval, _ = greedy_continuous(w, c, A)
                        rec["uniform"].append(uni)
                        rec["dp_opt"].append(opt)
                        rec["greedy"].append(gval)
                        rec["U"].append(U)
                    m = {k: float(np.mean(v)) for k, v in rec.items()}
                    gain = m["dp_opt"] - m["uniform"]
                    rows.append({"n": n, "weights": wk, "n_seeds": len(seeds), "c": c, "a_bar": abar,
                                 "beta_bar": round(min(1, (W + 1) * abar), 4),
                                 "uniform": round(m["uniform"], 4), "dp_opt": round(m["dp_opt"], 4),
                                 "greedy_cont": round(m["greedy"], 4), "U_cont": round(m["U"], 4),
                                 "gain_abs": round(gain, 4),
                                 "gain_pct": (round(100 * gain / m["uniform"], 1) if m["uniform"] > 1e-9 else "uniform=0"),
                                 "dp_le_U": bool(m["dp_opt"] <= m["U"] + 1e-6),
                                 "min_dp_over_seeds": round(min(rec["dp_opt"]), 4),
                                 "max_dp_over_seeds": round(max(rec["dp_opt"]), 4)})
                    print(rows[-1])
    with open(OUT / "a8c_alloc.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # Remark rem:skew check (n = 2, equal weights, a_bar = 0.0625, c = 1)
    w2 = np.array([0.5, 0.5])
    opt2, alloc2 = dp_opt(w2, 1.0, 2 * 0.0625, 4000)
    U2, g2, b2 = greedy_continuous(w2, 1.0, 2 * 0.0625)
    remark = {"uniform": 0.0, "remark_split_0.025_0.1": 0.5 * (r_star(1, 0.2) + r_star(1, 0.8)),
              "theorem7_greedy": g2, "U": U2, "dp_opt_integer_strides": opt2, "dp_alloc_K": alloc2}

    # Model T2 (fixed loads)
    t2 = []
    for n in (8, 32, 128):
        for sd in a.seeds:
            rng = np.random.default_rng(sd)
            for lk in ("identical", "heterogeneous"):
                mu = np.full(n, 0.2) if lk == "identical" else rng.uniform(0.05, 0.45, n)
                w = mu / mu.sum()
                c = 0.9
                for abar in (0.05, 0.08):
                    b = n * min(1, (W + 1) * abar)
                    uni_beta = b / n
                    uni = float(sum(w[i] * t2_frontier(uni_beta, mu[i], c) for i in range(n)))
                    wf, _ = t2_waterfill(w, mu, c, b)
                    t2.append({"n": n, "seed": sd, "loads": lk, "c": c, "a_bar": abar,
                               "uniform": round(uni, 4) if not math.isnan(uni) else "infeasible",
                               "waterfill_opt": round(wf, 4) if not math.isnan(wf) else "infeasible"})
    with open(OUT / "a8c_T2.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(t2[0]))
        wr.writeheader()
        wr.writerows(t2)
    ident_gain = [r["waterfill_opt"] - r["uniform"] for r in t2 if r["loads"] == "identical"
                  and isinstance(r["uniform"], float) and isinstance(r["waterfill_opt"], float)]
    summary = {"config": {"W": W, "KMAX": KMAX, "grid": a.grid, "seeds": a.seeds},
               "remark_skew_check": remark,
               "all_dp_le_U": all(r["dp_le_U"] for r in rows),
               "T2_identical_max_abs_gain": max(abs(g) for g in ident_gain) if ident_gain else None,
               "headline": [r for r in rows if r["weights"] == "equal" and r["c"] == 1.0]}
    (OUT / "a8c_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    make_figure(a.grid)


def make_figure(grid: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.7), constrained_layout=True)
    n = 32
    w = np.ones(n) / n
    abars = np.linspace(0.01, 0.125, 24)
    for c, ls in ((1.0, "-"), (0.9, "--")):
        uni = [r_star(c, beta_of_K(math.ceil(1 / x - 1e-12))) for x in abars]
        opt = [dp_opt(w, c, n * x, grid // 4)[0] for x in abars]
        U = [greedy_continuous(w, c, n * x)[0] for x in abars]
        axes[0].plot(abars, uni, color="tab:gray", ls=ls, lw=1.1, label=f"uniform, c={c}")
        axes[0].plot(abars, opt, color="tab:red", ls=ls, lw=1.1, label=f"optimal (Thm 7C), c={c}")
        axes[0].plot(abars, U, color="k", ls=":", lw=0.9, label="bound $U$, c=1" if c == 1.0 else None)
    axes[0].set_xlabel(r"mean activation $\bar a$ (W = 7)")
    axes[0].set_ylabel("certified tier robustness")
    axes[0].set_title(f"n = {n} equal-weight nodes")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[0].set_ylim(0, 1.0)
    # allocation profile
    abar, c = 0.05, 1.0
    _, alloc = dp_opt(w, c, n * abar, grid // 4)
    betas = sorted([beta_of_K(K) for K in alloc], reverse=True)
    axes[1].bar(np.arange(n), betas, color="tab:red", width=0.8, label="optimal (Thm 7)")
    axes[1].axhline(beta_of_K(math.ceil(1 / abar)), color="tab:gray", lw=1.2, label="uniform")
    axes[1].set_xlabel("node (sorted)")
    axes[1].set_ylabel(r"budget share $\beta_i$")
    axes[1].set_title(r"$\bar a=0.05$, $c=1$ ($\bar\beta=0.4$)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_ylim(0, 1.05)
    fig.savefig(FIG / "F_A8c.png", dpi=300)
    fig.savefig(FIG / "F_A8c.pdf")


if __name__ == "__main__":
    main()
