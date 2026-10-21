# -*- coding: utf-8 -*-
"""v3_a8a_mc.py -- numerical checks for A8(a) (adaptive gates), AwP v3.

Canonical block family of the paper: W = 7, T = 1500 frames -> N = 187 blocks (same horizon as the
old synthetic sweep, code/awp_experiments.py), m = B events, beta = m / N.

Checks (Revision_v3/theory/A8a_adaptive.md):
  (1) Theorem 4  -- every adaptive gate averages <= beta on uniformly random instances (ceiling).
  (2) Theorem 5  -- a c = 1 adaptive gate (SV(0)) is held to r*(1) when the advised set comes last.
  (3) Theorem 6  -- SV(x) worst case over a structured adversary family lies above the proved lower
                    bound and below the ceiling beta (exploratory: the family is not exhaustive,
                    so the empirical worst case is an UPPER estimate of the true robustness).
Outputs: results/v3/a8a_*.csv, results/v3/a8a_summary.json, Revision_v3/figures/F_A8a.{png,pdf}.

Usage: python code/v3_a8a_mc.py [--seeds 42 43 44 45 46] [--reps 400]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
from pathlib import Path

import numpy as np
from v3_io import atomic_open, atomic_write_text, save_fig  # noqa: E402  (atomic writes, C7)

TOPIC = Path(__file__).resolve().parents[1]
OUT = TOPIC / "results" / "v3"
FIG = TOPIC / "Revision_v3" / "figures"
W, T = 7, 1500
N0 = T // (W + 1)  # 187 blocks


def r_star(c: float, beta: float) -> float:
    """Non-adaptive frontier of Theorem 2."""
    if c <= beta:
        return c
    if beta <= 0.5:
        return beta * (1 - c) / (1 - beta)
    return 1 - (1 - beta) * c / beta


def theta0(beta: float) -> float:
    return max(0.0, (2 * beta - 1) / beta)


def delta(beta: float) -> float:
    return beta - r_star(1.0, beta)


# ----------------------------------------------------------------------------- instances
def advised_set(N: int, m: int, place: str, rng) -> np.ndarray:
    isA = np.zeros(N, bool)
    if place == "last":
        isA[N - m:] = True
    elif place == "first":
        isA[:m] = True
    elif place == "interleaved":
        isA[np.linspace(0, N - 1, m).round().astype(int)] = True
    else:  # random
        isA[rng.choice(N, m, replace=False)] = True
    assert isA.sum() == m
    return isA


def event_set(isA: np.ndarray, m: int, theta: float, adv_order: str, non_place: str, rng) -> np.ndarray:
    """E with |E cap A| = round(theta m); which advised / non-advised blocks carry events is set by order rules."""
    N = len(isA)
    A_idx, U_idx = np.flatnonzero(isA), np.flatnonzero(~isA)
    ka = int(round(theta * m))
    ka = max(ka, m - len(U_idx))  # theta below theta_0 = r*(1) is infeasible: clamp to |E \ A| <= N - m
    ku = m - ka
    if adv_order == "hits_first":
        a_ev = A_idx[:ka]
    elif adv_order == "hits_last":
        a_ev = A_idx[len(A_idx) - ka:]
    else:
        a_ev = rng.choice(A_idx, ka, replace=False)
    if non_place == "front":
        u_ev = U_idx[:ku]
    elif non_place == "back":
        u_ev = U_idx[len(U_idx) - ku:]
    elif non_place == "spread":
        u_ev = U_idx[np.linspace(0, len(U_idx) - 1, ku).round().astype(int)] if ku else U_idx[:0]
    else:
        u_ev = rng.choice(U_idx, ku, replace=False)
    isE = np.zeros(N, bool)
    isE[a_ev] = True
    isE[u_ev] = True
    assert isE.sum() == m
    return isE


# ----------------------------------------------------------------------------- gates (vectorised over reps)
def run_sv(isA: np.ndarray, isE: np.ndarray, m: int, x: float, reps: int, rng) -> np.ndarray:
    """SV(x): trust mode (fire advised, sprinkle non-advised w.p. g) until the first contradiction,
    then a uniformly random subset of the remaining blocks of size min(budget, remaining) (selection
    sampling). isE may be (N,) or (reps, N). Returns recall per rep."""
    N = len(isA)
    g = x / (N - m) if N > m else 0.0
    E = np.broadcast_to(isE, (reps, N))
    budget = np.full(reps, m, int)
    hits = np.zeros(reps, int)
    fb = np.zeros(reps, bool)  # fallback mode active from this block on
    for t in range(N):
        u = rng.random(reps)
        if isA[t]:
            fire_trust = np.ones(reps, bool)
        else:
            fire_trust = u < g
        p_fb = np.minimum(1.0, budget / (N - t))
        fire = np.where(fb, rng.random(reps) < p_fb, fire_trust) & (budget > 0)
        ev = E[:, t]
        hits += fire & ev
        budget -= fire
        contra = (~fb) & fire & (ev != isA[t])
        fb = fb | contra
    return hits / m


def run_uniform(isE: np.ndarray, m: int, reps: int, rng) -> np.ndarray:
    """Oblivious uniform random m-subset (the canonical-family analogue of random-phase periodic)."""
    N = isE.shape[-1]
    E = np.broadcast_to(isE, (reps, N))
    keys = rng.random((reps, N))
    thr = np.partition(keys, m - 1, axis=1)[:, m - 1:m]
    return (E & (keys <= thr)).sum(1) / m


def run_greedy_adaptive(isA, isE, m, reps, rng):
    """'Exploit feedback' heuristic (adaptive, not optimal): fire advised blocks; after the first
    advised miss, fire every block while budget lasts. Used only to show the Theorem-4 ceiling."""
    N = len(isA)
    E = np.broadcast_to(isE, (reps, N))
    budget = np.full(reps, m, int)
    hits = np.zeros(reps, int)
    explore = np.zeros(reps, bool)
    for t in range(N):
        fire = (explore | isA[t]) & (budget > 0)
        ev = E[:, t]
        hits += fire & ev
        budget -= fire
        explore = explore | (fire & isA[t] & ~ev)
    return hits / m


# ----------------------------------------------------------------------------- experiments
def ci95(v: np.ndarray) -> tuple[float, float, float]:
    mu = float(v.mean())
    h = 1.96 * float(v.std(ddof=1)) / math.sqrt(len(v))
    return mu, mu - h, mu + h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--plot-only", action="store_true", help="redraw F_A8a from results/v3/a8a_sv_worstcase.csv")
    a = ap.parse_args()
    if a.plot_only:
        rows = list(csv.DictReader(open(OUT / "a8a_sv_worstcase.csv", encoding="utf-8")))
        for r in rows:
            for k in ("beta", "c_emp", "r_worst_emp"):
                r[k] = float(r[k])
        make_figure(rows)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    betas_target = [0.4, 0.5, 0.8]
    summary = {"config": {"W": W, "T": T, "N": N0, "seeds": a.seeds, "reps_per_seed": a.reps,
                          "note": "canonical block family, m = B events, beta = m/N"}}

    # (1) Theorem 4 ceiling on uniformly random instances -------------------------------------
    rows1 = []
    for bt in betas_target:
        m = round(bt * N0)
        beta = m / N0
        for gate in ["SV(0)", "SV(5)", "SV(20)", "greedy-adaptive", "uniform-oblivious"]:
            vals = []
            for sd in a.seeds:
                rng = np.random.default_rng(sd)
                isA = advised_set(N0, m, "random", rng)
                Er = np.zeros((a.reps, N0), bool)
                for i in range(a.reps):
                    Er[i, rng.choice(N0, m, replace=False)] = True
                if gate.startswith("SV"):
                    v = run_sv(isA, Er, m, float(gate[3:-1]), a.reps, rng)
                elif gate == "greedy-adaptive":
                    v = run_greedy_adaptive(isA, Er, m, a.reps, rng)
                else:
                    v = run_uniform(Er, m, a.reps, rng)
                vals.append(v)
            v = np.concatenate(vals)
            mu, lo, hi = ci95(v)
            rows1.append({"beta": round(beta, 4), "m": m, "gate": gate, "mean_recall": round(mu, 4),
                          "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n_runs": len(v),
                          "ceiling_beta": round(beta, 4), "violates": bool(lo > beta + 1e-9)})
    with atomic_open(OUT / "a8a_ceiling.csv") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows1[0]))
        w.writeheader()
        w.writerows(rows1)
    summary["thm4_any_violation"] = any(r["violates"] for r in rows1)

    # (2)+(3) SV(x) worst case over the adversary family ---------------------------------------
    placements = ["last", "first", "interleaved", "random"]
    adv_orders = ["hits_first", "hits_last", "random"]
    non_places = ["front", "back", "spread", "random"]
    thetas_base = [i / 20 for i in range(21)]
    rows2, rows3 = [], []
    for bt in betas_target:
        m = round(bt * N0)
        beta = m / N0
        th0 = theta0(beta)
        xs = [0, 1, 2, 5, 10, 20, round(math.sqrt((1 - beta) * m / beta), 2)]
        thetas = sorted(set([round(t, 4) for t in thetas_base if t >= th0 - 1e-9] + [round(th0, 4), round(beta, 4)]))
        for x in xs:
            # consistency on E = A (A placement irrelevant for c; use 'last')
            cs = []
            for sd in a.seeds:
                rng = np.random.default_rng(sd)
                isA = advised_set(N0, m, "last", rng)
                cs.append(run_sv(isA, isA.copy(), m, x, a.reps, rng))
            c_emp = float(np.concatenate(cs).mean())
            worst = None
            for place in placements:
                for th in thetas:
                    for ao in adv_orders:
                        for npl in non_places:
                            vals = []
                            for sd in a.seeds:
                                rng = np.random.default_rng(10_000 * sd + zlib.crc32(f"{place}|{th}|{ao}|{npl}".encode()) % 9973)
                                isA = advised_set(N0, m, place, rng)
                                isE = event_set(isA, m, th, ao, npl, rng)
                                if isE is None:
                                    vals = None
                                    break
                                vals.append(run_sv(isA, isE, m, x, a.reps, rng))
                            if vals is None:
                                continue
                            v = np.concatenate(vals)
                            mu = float(v.mean())
                            rows3.append({"beta": round(beta, 4), "x": x, "place": place, "theta": th,
                                          "adv_order": ao, "non_place": npl, "mean_recall": round(mu, 4)})
                            if worst is None or mu < worst[0]:
                                worst = (mu, place, th, ao, npl, v)
            mu, lo, hi = ci95(worst[5])
            lb6 = beta - (1 - beta) / (beta * x) - (x + 2) / m if x > 0 else float("-inf")
            x_eff = (1 - c_emp) * m
            ub5 = th0 + (1 - th0) * min(1.0, (1 - th0) * beta / (1 - beta) * x_eff)
            rows2.append({"beta": round(beta, 4), "m": m, "x": x, "c_emp": round(c_emp, 4),
                          "c_bound": round(1 - x / m, 4), "r_worst_emp": round(mu, 4), "ci_lo": round(lo, 4),
                          "ci_hi": round(hi, 4), "worst_instance": f"{worst[1]}|theta={worst[2]}|{worst[3]}|{worst[4]}",
                          "thm6_lower": round(lb6, 4) if x > 0 else "", "thm5_upper(A last)": round(min(beta, ub5), 4),
                          "ceiling_beta": round(beta, 4), "r_star_c_emp": round(r_star(c_emp, beta), 4),
                          "nonadaptive_r_star_1": round(r_star(1.0, beta), 4)})
            print(rows2[-1])
    with atomic_open(OUT / "a8a_sv_worstcase.csv") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows2[0]))
        w.writeheader()
        w.writerows(rows2)
    with atomic_open(OUT / "a8a_sv_family_all.csv") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows3[0]))
        w.writeheader()
        w.writerows(rows3)
    summary["sv_worstcase"] = rows2
    summary["thm6_lower_respected"] = all((r["thm6_lower"] == "" or r["ci_hi"] >= r["thm6_lower"]) for r in rows2)
    summary["ceiling_respected"] = all(r["ci_lo"] <= r["ceiling_beta"] + 1e-9 for r in rows2)
    summary["delta_beta"] = {str(b): round(delta(b), 4) for b in [0.2, 0.4, 0.5, 0.8, 1.0]}
    atomic_write_text(OUT / "a8a_summary.json", json.dumps(summary, indent=1), encoding="utf-8")
    make_figure(rows2)
    print("thm4 violation:", summary["thm4_any_violation"], "| thm6 respected:", summary["thm6_lower_respected"],
          "| ceiling respected:", summary["ceiling_respected"])


def make_figure(rows2: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "font.family": "serif"})
    fig, ax4 = plt.subplots(2, 2, figsize=(5.35, 4.4), constrained_layout=True)
    axes = [ax4[0, 0], ax4[0, 1], ax4[1, 0], ax4[1, 1]]
    for ax, bt in zip(axes[:2], [0.4, 0.8]):
        m = round(bt * N0)
        beta = m / N0
        cc = np.linspace(0, 1, 401)
        ax.plot(cc, [r_star(c, beta) for c in cc], "k-", lw=1.2, label=r"non-adaptive $r^\star(c)$ (Thm 2)")
        ax.axhline(beta, color="tab:blue", ls="--", lw=1, label=r"adaptive ceiling $\beta$ (Thm 4)")
        xs = np.logspace(-0.3, 2.3, 200)
        lbv = np.array([beta - (1 - beta) / (beta * x) - (x + 2) / m for x in xs])
        keep = lbv > 0
        ax.plot((1 - xs / m)[keep], lbv[keep], color="tab:green",
                lw=1, label=r"SV($x$) proved lower bound (Thm 6)")
        pts = [r for r in rows2 if abs(float(r["beta"]) - beta) < 1e-3]
        ax.plot([r["c_emp"] for r in pts], [r["r_worst_emp"] for r in pts], "o", ms=3.5, color="tab:red",
                label="SV($x$) empirical worst case")
        ax.plot([1.0], [r_star(1.0, beta)], "s", ms=4, mfc="none", color="k")
        ax.annotate(r"$c{=}1$: Thm 5", (1.0, r_star(1.0, beta)), xytext=(0.30, r_star(1.0, beta) + 0.05 if beta > 0.5 else 0.06),
                    fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("consistency $c$")
        ax.set_ylabel("robustness $r$")
        ax.set_title(rf"$\beta={beta:.2f}$ ($m={m}$, $N={N0}$)")
    h, l = axes[0].get_legend_handles_labels()
    axes[3].axis("off")
    axes[3].legend(h, l, loc="center", frameon=False, fontsize=7.5)
    bb = np.linspace(0.001, 0.999, 500)
    axes[2].plot(bb, [delta(b) for b in bb], "k-", lw=1.2)
    axes[2].set_xlabel(r"oblivious rate $\beta$")
    axes[2].set_ylabel(r"$\Delta(\beta)=\beta-r^\star(1)$")
    axes[2].set_title("price of adaptivity (Cor. 7)")
    for b in (0.4, 0.8):
        axes[2].plot([b], [delta(b)], "o", ms=3.5, color="tab:red")
        axes[2].annotate(f"{delta(b):.2f}", (b, delta(b)), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[2].set_ylim(0, 0.55)
    save_fig(fig, FIG / "F_A8a.png", dpi=300)
    save_fig(fig, FIG / "F_A8a.pdf")


if __name__ == "__main__":
    main()
