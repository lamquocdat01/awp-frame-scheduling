# -*- coding: utf-8 -*-
"""p5_sweep_alg1.py -- P5B PHA 2 (24-11-2026): real-data trust-dial sweep (fig5) and Table 3 re-run with the
backbone of Algorithm 1 as stated after P5B PHA 1: random phase U ~ Unif{0..K-1}, K = ceil(T/B), every epoch
t = U (mod K) kept independently with probability lambda; advice arm block-aware with uniform draws, then the
iso-energy fill of v3_sweep_blockaware (total activations = budget = round(aT), as in every v3 experiment).
5 seeds; per video the recall is averaged over seeds; CIs are video-level bootstraps (B = 1000, same seeds as
v3_sweep_blockaware). The deterministic-backbone results (v3_sweep_blockaware, v3_numbers) are kept unchanged.
Corpus, corruption and eta-proxy rngs exactly as v3_sweep_blockaware.
Usage: python code/p5_sweep_alg1.py [--pilot]
Outputs: results/v3/p5_alg1_sweep.csv, results/v3/p5_alg1_summary.json
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from awp_experiments import (A_GRID, B, CORRUPT_GRID, LAMBDA_GRID, SEED, W, bootstrap_ci,  # noqa: E402
                             eta_from_auc, onset_auc, recall_windowed)
from v3_io import atomic_open, atomic_write_text  # noqa: E402
from v3_numbers import load_corpus  # noqa: E402
from v3_sweep_blockaware import corrupt  # noqa: E402

RES = HERE.parent / "results" / "v3"
SEEDS = [0, 1, 2, 3, 4]


def alg1_fire(score, T, a, lam, rng, W=W):
    budget = int(round(a * T))
    fired = np.zeros(T, dtype=bool)
    if budget <= 0:
        return fired
    K = -(-T // budget)                                   # stride ceil(T/B)
    U = rng.integers(K)
    lattice = np.arange(U, T, K)
    keep = lattice[rng.random(len(lattice)) < lam] if lam > 0 else lattice[:0]
    fired[keep] = True
    remaining = budget - int(fired.sum())
    if remaining <= 0:
        return fired
    cs = np.concatenate([[0], np.cumsum(fired)])
    t = np.arange(T)
    covered = (cs[t + 1] - cs[np.maximum(0, t - W)]) > 0
    advised = np.argsort(-score, kind="mergesort")[:budget]
    free = advised[~covered[advised]]
    pick = rng.choice(free, size=remaining, replace=False) if len(free) > remaining else free
    fired[pick] = True
    short = budget - int(fired.sum())
    if short > 0:
        order = np.argsort(-score, kind="mergesort")
        for mask in (~fired & ~covered, ~fired):
            cand = order[mask[order]][:short]
            fired[cand] = True
            short = budget - int(fired.sum())
            if short <= 0:
                break
    return fired


def cell(corpus, a, lam, p):
    per = np.zeros((len(SEEDS), len(corpus)))
    for si, s in enumerate(SEEDS):
        rng = np.random.default_rng(SEED + 7 + int(p * 100))           # corruption, as in v3 sweep
        brng = np.random.default_rng([SEED, s, int(a * 1000), int(lam * 100), int(round(p * 100))])
        for vi, c in enumerate(corpus):
            f = alg1_fire(corrupt(c["cue"], p, rng), c["T"], a, lam, brng)
            per[si, vi] = recall_windowed(f, c["events"], W)
    return per


def main():
    pilot = "--pilot" in sys.argv
    t0 = time.time()
    _, corpus, _ = load_corpus()
    print(f"corpus {len(corpus)} videos; load {time.time() - t0:.1f}s", flush=True)
    rows, table3, per_by = [], {}, {}
    agrid = A_GRID[:1] if pilot else A_GRID
    for a in agrid:
        for lam in LAMBDA_GRID:
            for p in CORRUPT_GRID:
                per = cell(corpus, a, lam, p)
                v = per.mean(axis=0)
                per_by[(a, lam, p)] = v
                m, lo, hi = bootstrap_ci(v, B, SEED + int(a * 1000) + int(lam * 10))
                rng2 = np.random.default_rng(SEED + int(p * 100))
                eta = eta_from_auc(float(np.mean([onset_auc(corrupt(c["cue"], p, rng2), c["win"]) for c in corpus])))
                rows.append(dict(a=a, lam=lam, corruption_p=p, proxy_1_minus_auc=round(eta, 4), recall_mean=round(m, 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                                 seed_sd=round(float(per.mean(axis=1).std(ddof=1)), 4)))
        print(f"a={a}: {time.time() - t0:.1f}s", flush=True)
    if pilot:
        print(f"PILOT one budget: {time.time() - t0:.1f}s -> full estimate {3 * (time.time() - t0):.0f}s")
        return
    for a in A_GRID:
        g, pp = per_by[(a, 0.0, 0.0)], per_by[(a, 1.0, 0.0)]
        gm, glo, ghi = bootstrap_ci(g, B, SEED + int(a * 1000))
        pm, plo, phi = bootstrap_ci(pp, B, SEED + int(a * 1000) + 10)
        dm, dlo, dhi = bootstrap_ci(g - pp, B, SEED)
        table3[f"a{a:.3f}"] = dict(gate=round(gm, 3), gate_ci=[round(glo, 3), round(ghi, 3)], periodic=round(pm, 3),
                                   periodic_ci=[round(plo, 3), round(phi, 3)], gap=round(gm - pm, 3),
                                   gap_paired_ci=[round(dlo, 3), round(dhi, 3)], gap_ci_excludes_0=bool(dhi < 0))
    inter = [r for r in rows if 0 < r["lam"] < 1]
    per_ref = {(r["a"], r["corruption_p"]): r["recall_mean"] for r in rows if r["lam"] == 1.0}
    above = [r for r in inter if r["recall_mean"] > per_ref[(r["a"], r["corruption_p"])]]
    summ = dict(backbone="Algorithm 1: random phase, stride ceil(T/B), keep w.p. lambda; block-aware uniform advice arm; iso-energy fill",
                seeds=SEEDS, table3_alg1=table3, n_intermediate_cells=len(inter),
                n_intermediate_above_periodic=len(above), above_cells=above
                )
    with atomic_open(RES / "p5_alg1_sweep.csv") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); [w.writerow(r) for r in rows]
    atomic_write_text(RES / "p5_alg1_summary.json", json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(table3, indent=1)); print("above periodic:", len(above), "/", len(inter), "runtime", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
