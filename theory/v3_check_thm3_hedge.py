# -*- coding: utf-8 -*-
"""v3_check_thm3_hedge.py -- audit of the HEDGE-GATE statement in Theorem 3 (baseline) at beta > 1/2.

Two block-level implementations of HEDGE-GATE(lambda) on the canonical family (N blocks, m = B events,
advised set = first m blocks):
  union   : backbone fires each block w.p. lambda*beta; the advice arm fires the first (1-lambda)B advised
            blocks REGARDLESS of the backbone (overlap wasted). This is validate_frontier.py's model.
  skip    : Algorithm 1 as written ("fire the not-yet-fired frames of highest cue score"): the advice arm
            spends its (1-lambda)B activations on advised blocks the backbone has NOT already activated.
Exact per-block activation probabilities are estimated by Monte Carlo (backbone = uniformly random subset of
lambda*B blocks); consistency c = mean p over advised blocks; robustness r = mean of the m smallest p.
Reports max_lambda (r*(c) - r). Seed 42, 2000 draws per lambda.

P3C (22-10-2026) adds an EXACT section (rational arithmetic, full enumeration of the gate's coins) on a small
instance that isolates the two symmetry conditions the proof of Theorem 3 uses: the advice arm chooses
uniformly among the free advised blocks, and the backbone's activated set is exchangeable (a uniformly random
lambda*B-subset of the blocks). Breaking either -- a fixed tie-breaking order, or a periodic backbone whose
residue classes the advised set meets unequally -- makes the advised marginals unequal, and at beta > 1/2 the
robustness falls strictly below r*(c). The Monte Carlo part above it is unchanged (same rng stream).
"""
from fractions import Fraction
from itertools import combinations

import numpy as np


def r_star(c, beta):
    if c <= beta:
        return c
    if beta <= 0.5:
        return beta * (1 - c) / (1 - beta)
    return 1 - (1 - beta) * c / beta


def probs(N, m, lam, mode, draws, rng):
    B = m
    nb = int(round(lam * B))
    na = B - nb
    cnt = np.zeros(N)
    for _ in range(draws):
        fired = np.zeros(N, bool)
        fired[rng.choice(N, nb, replace=False)] = True
        adv = np.arange(m)
        if mode == "union":
            fired[adv[:na]] = True
        else:
            free = adv[~fired[adv]]
            fired[rng.permutation(free)[:na]] = True
        cnt += fired
    return cnt / draws


def main():
    rng = np.random.default_rng(42)
    N = 200
    for beta in (0.4, 0.8):
        m = int(beta * N)
        for mode in ("union", "skip"):
            gaps = []
            for lam in np.linspace(0, 1, 11):
                p = probs(N, m, lam, mode, 2000, rng)
                c = p[:m].mean()
                r = np.sort(p)[:m].mean()
                gaps.append((round(lam, 1), round(c, 3), round(r, 3), round(r_star(c, beta) - r, 3)))
            print(f"beta={beta} mode={mode}: max gap {max(g[3] for g in gaps):.3f}")
            for g in gaps:
                print("   lam=%.1f c=%.3f r=%.3f r*(c)-r=%.3f" % g)


def exact_marginals(N, A, bb_sets, k, rule):
    """Per-block activation probabilities, averaging over the listed (equally likely) backbone sets."""
    p, w = [Fraction(0)] * N, Fraction(1, len(bb_sets))
    for S in bb_sets:
        for b in S:
            p[b] += w
        free = [b for b in A if b not in S]
        if rule == "uniform":
            kk = min(k, len(free))
            for b in free:
                p[b] += w * Fraction(kk, len(free))
        else:  # fixed order: the first k free advised blocks
            for b in free[:k]:
                p[b] += w
    return p


def exact_section():
    print("EXACT (rational arithmetic, all backbone draws enumerated)")
    for N, m, nonadv, lam in ((10, 8, (0, 1), Fraction(1, 4)), (20, 16, (0, 1, 5, 6), Fraction(1, 4)),
                              (10, 4, (0, 1, 2, 3, 4, 5), Fraction(1, 2))):
        beta = Fraction(m, N)
        nb = int(lam * m)
        k = m - nb
        A = [b for b in range(N) if b not in nonadv]
        stride = N // nb
        assert stride * nb == N, "periodic backbone needs nb | N"
        backbones = {"uniform-subset": [set(c) for c in combinations(range(N), nb)],
                     "periodic": [set(range(ph, N, stride)) for ph in range(stride)]}
        for bb in ("uniform-subset", "periodic"):
            for rule in ("uniform", "fixed-order"):
                p = exact_marginals(N, A, backbones[bb], k, rule)
                c = sum(p[b] for b in A) / m
                r = sum(sorted(p)[:m]) / m
                rs = r_star(c, beta)  # exact: r_star uses only field operations
                adv = sorted(set(p[b] for b in A))
                print("  N=%d m=%d beta=%.1f lam=%s %s backbone, %s choice: c=%.4f r=%.4f r*(c)=%.4f gap=%.4f"
                      % (N, m, float(beta), lam, bb, rule, c, r, rs, rs - r))
                print("      advised marginals " + "/".join("%.4f" % float(x) for x in adv))


if __name__ == "__main__":
    main()
    exact_section()
