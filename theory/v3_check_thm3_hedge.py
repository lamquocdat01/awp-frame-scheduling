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
Reports max_lambda (r*(c) - r). Seed 42, 20000 draws.
"""
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


if __name__ == "__main__":
    main()
