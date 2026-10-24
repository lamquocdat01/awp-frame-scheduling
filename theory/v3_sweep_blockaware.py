# -*- coding: utf-8 -*-
"""v3_sweep_blockaware.py -- real-data trust-dial sweep (fig5, S2.5) with HEDGE-GATE exactly as Algorithm 1 of v3.

The TPDS sweep (awp_real_data.main -> results/corruption_sweep_REAL.csv) uses awp_experiments.hedge_gate_fire,
a *frame-level* split: its advice arm skips frames the backbone has fired, but not windows the backbone has
already covered, and it takes the highest-score frames in order. Algorithm 1 of v3 (and the proof of Theorem 3)
needs the *block-aware* split: the advice arm spends its (1-lam)B activations on advised epochs whose window
[t-W, t] contains no backbone activation, chosen uniformly at random among them.

  hedge_gate_fire_blockaware(score, T, a, lam, rng)
    backbone   : identical to hedge_gate_fire (round(lam*budget) activations on np.linspace(0, T-1)),
    advised set: the `budget` epochs of highest cue score (budget-tuned threshold, |A| = B; stable order),
    advice arm : `remaining` epochs drawn uniformly without replacement from the advised epochs whose window
                 contains no backbone activation; if there are fewer such epochs, all of them, and the rest of
                 the budget goes to the highest-score epochs that are still free (then to any unfired epoch),
                 so the total number of activations is round(a*T) exactly as in the TPDS split (iso-energy).
  At lam = 0 there is no backbone and the advice arm fires the whole advised set; at lam = 1 there is no advice
  arm. Both endpoints are therefore the TPDS policy *as sets of frames*, which this script checks.

Same corpus, order and seeds as awp_real_data.main: corruption rng SEED+7+int(100p) consumed video by video,
eta rng SEED+int(100p), video-level bootstrap seed SEED+int(1000a)+int(10lam) (B = 1000). The tie rng of the
uniform choice is np.random.default_rng([SEED+int(1000a)+int(10lam), int(round(100p))]), consumed video by video.

Checks (exit 1 if any fails):
  (I1) the naive replay of this loop reproduces results/corruption_sweep_REAL.csv cell for cell;
  (I2) block-aware rows at lam in {0, 1} equal the TPDS rows cell for cell (and the fired sets are equal);
Outputs (atomic, UTF-8):
  results/v3/corruption_sweep_REAL_blockaware.csv   columns of the TPDS file
  results/v3/sweep_blockaware_summary.json          both variants side by side + invariant report
  python code/v3_sweep_blockaware.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from awp_experiments import (A_GRID, B, CORRUPT_GRID, LAMBDA_GRID, SEED, W, bootstrap_ci,  # noqa: E402
                             eta_from_auc, hedge_gate_fire, onset_auc, recall_windowed)
from v3_io import atomic_open, atomic_write_text  # noqa: E402  (atomic writes, C7)
from v3_numbers import load_corpus  # noqa: E402  (same corpus and order as awp_real_data.main)

TOPIC = HERE.parent
RES = TOPIC / "results"
OUT_CSV = RES / "v3" / "corruption_sweep_REAL_blockaware.csv"
OUT_JSON = RES / "v3" / "sweep_blockaware_summary.json"
FIELDS = ["a", "lam", "corruption_p", "eta", "recall_mean", "ci_lo", "ci_hi"]


def hedge_gate_fire_blockaware(score, T, a, lam, rng, W=W):
    budget = int(round(a * T))
    fired = np.zeros(T, dtype=bool)
    n_bb = int(round(lam * budget))
    if n_bb >= 1:
        fired[np.unique(np.linspace(0, T - 1, n_bb, dtype=int))] = True
    remaining = budget - int(fired.sum())
    if remaining <= 0:
        return fired
    cs = np.concatenate([[0], np.cumsum(fired)])
    t = np.arange(T)
    covered = (cs[t + 1] - cs[np.maximum(0, t - W)]) > 0      # window [t-W, t] holds a backbone activation
    advised = np.argsort(-score, kind="mergesort")[:budget]
    free = advised[~covered[advised]]
    if len(free) > remaining:
        pick = rng.choice(free, size=remaining, replace=False)
    else:
        pick = free
    fired[pick] = True
    short = budget - int(fired.sum())
    if short > 0:                                              # iso-energy fallback (never used at lam = 0)
        order = np.argsort(-score, kind="mergesort")
        for mask in (~fired & ~covered, ~fired):
            cand = order[mask[order]][:short]
            fired[cand] = True
            short = budget - int(fired.sum())
            if short <= 0:
                break
    return fired


def corrupt(cue, p, rng):  # verbatim from awp_real_data.main
    out = cue.copy(); k = int(round(p * len(cue)))
    if k > 1:
        idx = rng.choice(len(cue), size=k, replace=False)
        out[idx] = cue[idx][rng.permutation(k)]
    return out


def sweep(corpus, variant):
    rows, fired_sets = [], {}
    for a in A_GRID:
        for lam in LAMBDA_GRID:
            for p in CORRUPT_GRID:
                rng = np.random.default_rng(SEED + 7 + int(p * 100))
                tie = np.random.default_rng([SEED + int(a * 1000) + int(lam * 10), int(round(p * 100))])
                per, fs = [], []
                for c in corpus:
                    s = corrupt(c["cue"], p, rng)
                    f = (hedge_gate_fire(s, c["T"], a, lam) if variant == "naive"
                         else hedge_gate_fire_blockaware(s, c["T"], a, lam, tie))
                    per.append(recall_windowed(f, c["events"], W))
                    if lam in (0.0, 1.0):
                        fs.append(np.flatnonzero(f))
                m, lo, hi = bootstrap_ci(per, B, SEED + int(a * 1000) + int(lam * 10))
                rng2 = np.random.default_rng(SEED + int(p * 100))
                etas = [onset_auc(corrupt(c["cue"], p, rng2), c["win"]) for c in corpus]
                rows.append(dict(a=a, lam=lam, corruption_p=p, eta=round(eta_from_auc(float(np.mean(etas))), 4),
                                 recall_mean=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
                if fs:
                    fired_sets[(a, lam, p)] = fs
    return rows, fired_sets


def as_cells(rows):
    return [[str(r[k]) for k in FIELDS] for r in rows]


def main():
    _, corpus, _ = load_corpus()
    print(f"corpus: {len(corpus)} videos with >= 2 events, {sum(len(c['events']) for c in corpus)} events", flush=True)
    naive, fs_n = sweep(corpus, "naive")
    ba, fs_b = sweep(corpus, "blockaware")
    report, fails = [], 0

    tpds = list(csv.reader(open(RES / "corruption_sweep_REAL.csv", encoding="utf-8", newline="")))[1:]
    d1 = sum(x != y for r1, r2 in zip(as_cells(naive), tpds) for x, y in zip(r1, r2)) + abs(len(tpds) - len(naive))
    report.append(f"I1 naive replay vs results/corruption_sweep_REAL.csv: {len(naive)} rows x {len(FIELDS)} cells, {d1} differences")
    fails += d1 > 0

    end_n = [r for r in as_cells(naive) if float(r[1]) in (0.0, 1.0)]
    end_b = [r for r in as_cells(ba) if float(r[1]) in (0.0, 1.0)]
    d2 = sum(x != y for r1, r2 in zip(end_n, end_b) for x, y in zip(r1, r2))
    d2s = sum(not np.array_equal(u, v) for k in fs_n for u, v in zip(fs_n[k], fs_b[k]))
    report.append(f"I2 block-aware vs TPDS at lam in {{0,1}}: {len(end_b)} rows, {d2} cell differences; "
                  f"{sum(len(v) for v in fs_b.values())} per-video fired sets, {d2s} different")
    fails += (d2 > 0) + (d2s > 0)

    def key(r):
        return f"a{r['a']:.3f}_lam{r['lam']:.2f}_p{r['corruption_p']:.1f}"
    both = {key(r): {"eta": r["eta"], "naive": r["recall_mean"], "naive_ci": [r["ci_lo"], r["ci_hi"]],
                     "blockaware": q["recall_mean"], "blockaware_ci": [q["ci_lo"], q["ci_hi"]],
                     "diff_ba_minus_naive": round(q["recall_mean"] - r["recall_mean"], 4)}
            for r, q in zip(naive, ba)}
    inter = [v["diff_ba_minus_naive"] for k, v in both.items() if not ("lam0.00" in k or "lam1.00" in k)]
    dep = {k: v for k, v in both.items() if k.endswith("_p0.0")}
    summ = {"rows": both, "deployed_cue_p0": dep,
            "intermediate_lambda": {"n_cells": len(inter), "mean_diff": round(float(np.mean(inter)), 4),
                                    "min_diff": round(float(np.min(inter)), 4), "max_diff": round(float(np.max(inter)), 4),
                                    "n_ba_better": int(sum(x > 0 for x in inter)), "n_ba_worse": int(sum(x < 0 for x in inter)),
                                    "n_equal": int(sum(x == 0 for x in inter))},
            "invariants": report, "invariant_failures": int(fails),
            "seeds": "corrupt SEED+7+int(100p); eta SEED+int(100p); bootstrap SEED+int(1000a)+int(10lam), B=1000; "
                     "tie default_rng([SEED+int(1000a)+int(10lam), round(100p)])"}
    with atomic_open(OUT_CSV) as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); [w.writerow(r) for r in ba]
    atomic_write_text(OUT_JSON, json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    for line in report:
        print(line)
    print("intermediate lambda (block-aware - naive):", summ["intermediate_lambda"])
    print(f"invariant failures: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
