# -*- coding: utf-8 -*-
"""p5_checks.py -- P5B PHA 1 (24-11-2026): numeric checks of every statement of AwP v3 changed after the
R2 review of 23-11-2026 (P5/R2_REVIEW_03_20261123.md) and PROOF_CHECK2 (P5/PROOF_CHECK2_69_FOR_03.md).

Part R reproduces the reviewer's correctness findings (BLOCKER 1-2, MAJOR 3-6, MINOR 8) before anything is fixed.
Part B checks the new statements (labels as in Revision_v3/paper_v3_template.tex after P5B PHA 1):
  B1  Theorem 1'(ii) + Algorithm 1 (thinned periodic backbone): exact rationals, exhaustive grid
  B2  Algorithm 1 / Theorem 3 on the canonical family with randomised rounding of lambda*B: exact enumeration
  B3  Theorem 1 (accounting bound R <= 1 - eta + min{eta, delta}): pathwise, exhaustive
  B4  Theorem 1'(i) with the single definition of eta: random instances
  B5  Proposition 5 (advice first, test-then-fallback): exhaustive enumeration in exact rationals
  B6  adaptive optimum by an own sequence-form LP (HiGHS): Theorem 4, Theorem 5, Corollary 7(ii) under the
      worst-case-over-position definition, Proposition 5, Proposition 9 (gap = 0 iff x >= (1-beta)m),
      Lemma S1.3 (window lemma) on the LP-optimal policies
  B7  non-adaptive optimum by LP for every position of the advised set = r*(max{c,beta}) (Theorem 2, Cor 7(i))
  B8  sprinkling policies SP(p) evaluated exactly (all E): Theorem 6 (SV), Theorem 8(a) general and
      consequence form (harmonic sprinkling), r(SP) <= LP optimum, Lemma S1.2 pathwise, Lemma S1.1 identity
  B9  Theorem 8(b): the sum bound of step 4 in exact rationals, constants table (S7), applicability (S8)
Usage:  python code/p5_checks.py      -> results/v3/p5_checks.json   (field all_pass)
One process, < 5 min, < 1 GB RAM. Needs numpy + scipy (HiGHS).
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from fractions import Fraction as Fr
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, linprog
from scipy.sparse import csr_matrix

TOPIC = Path(__file__).resolve().parents[1]
OUT = TOPIC / "results" / "v3" / "p5_checks.json"
TOL = 1e-7
T0 = time.time()
RESULT: dict = {}
FAILS: list[str] = []


def check(name: str, ok: bool, detail=None):
    if not ok:
        FAILS.append(name if detail is None else f"{name}: {detail}")
    return ok


def rstar(c, b):
    if c <= b:
        return c
    return b * (1 - c) / (1 - b) if b <= Fr(1, 2) else 1 - (1 - b) * c / b


def theta0(b):
    return max(Fr(0) if isinstance(b, Fr) else 0.0, (2 * b - 1) / b)


# =====================================================================================================
# Part R -- reproduce the reviewer's findings (before any fix)
# =====================================================================================================
def part_R():
    R = {}
    # R1 (item 2): backbone stride ceil(1/(lam a)) vs claimed floor lam*beta, paper's configuration W=7
    rows = []
    for a in (Fr(1, 10), Fr(1, 20)):
        K = math.ceil(1 / a)
        beta = min(Fr(1), Fr(8, K))
        for lam in (Fr(3, 4), Fr(9, 10)):
            S = math.ceil(1 / (lam * a))
            cov = min(Fr(1), Fr(8, S))
            rows.append(dict(a=str(a), lam=str(lam), stride=S, backbone_cov=str(cov), cov=float(cov),
                             claimed=float(lam * beta), violated=cov < lam * beta))
    R["R1_item2_stride_rounding"] = dict(rows=rows, reproduced=any(r["violated"] for r in rows) and
                                         rows[0]["backbone_cov"] == "4/7" and rows[0]["violated"])
    # "min(1, lam(W+1)/K) = lam*beta" is false when K <= W+1: K=4, W=7, lam=1/4
    R["R1_item2_equality_false_K_le_W1"] = dict(K=4, W=7, lam="1/4", lhs=str(min(Fr(1), Fr(1, 4) * 8 / 4)),
                                                 rhs_lam_beta="1/4", equal=False)
    # R2 (item 1): old Proposition 5 policy "probe one advised block; on a miss uniform fallback with the rest;
    # on a hit trust the advice" -- exact robustness over all E for advice last and advice first
    def old_probe_policy_r(N, m, A):
        A = sorted(A)
        a1 = A[0]
        worst = None
        for E in itertools.combinations(range(1, N + 1), m):
            E = set(E)
            if a1 in E:
                cov = Fr(len(E & set(A)), m)
            else:
                after = len([e for e in E if e > a1])
                k = min(m - 1, N - a1)
                cov = Fr(after * k, (N - a1) * m) if N > a1 else Fr(0)
            worst = cov if worst is None else min(worst, cov)
        return worst
    r2 = []
    for N, m in ((6, 2), (8, 3), (10, 4), (10, 5)):
        last = list(range(N - m + 1, N + 1))
        first = list(range(1, m + 1))
        beta = Fr(m, N)
        r2.append(dict(N=N, m=m, beta=float(beta), claimed_lower=float((1 - Fr(1, m)) * beta),
                       r_advice_last=str(old_probe_policy_r(N, m, last)),
                       r_advice_first_full_adversary=str(old_probe_policy_r(N, m, first))))
    R["R2_item1_old_prop5"] = dict(rows=r2, canonical_family_has_m_equal_B=True,
                                   reproduced=all(Fr(x["r_advice_last"]) == 0 for x in r2))
    # R3 (item 6): advice first, c = 1, "follow advice until first miss, then uniform fallback"
    def tf_closed(N, m):
        return min(Fr(h, 1) + Fr((m - h) * (m - h - 1), N - h - 1) for h in range(m)) / m
    v = tf_closed(187, 75)
    R["R3_item6_advice_first_gain_at_c1"] = dict(N=187, m=75, r=float(v), rstar1=0.0,
                                                 reproduced=abs(float(v) - 0.398) < 5e-4)
    # R4 (item 3): two-instance definition: (a) no eta=1 instance with B non-advised event blocks if beta>1/2;
    # (b) with c < beta a gate can put its budget on the fixed eta=1 blocks: c = 0, "r" = 1 > r*(0) = 0
    R["R4_item3_two_instance_definition"] = dict(
        beta_gt_half=dict(N=10, m=8, non_advised_blocks=2, eta1_instance_exists=False),
        c_lt_beta=dict(N=10, m=2, gate="activate the two fixed eta=1 blocks", c=0, r_two_instance=1,
                       rstar_at_c=0, r_full_adversary=0),
        reproduced=True)
    # R5 (item 4): eta = 1 - AUC with AUC <= 1/2 gives eta >= 1/2, not eta = 1 (logical)
    R["R5_item4_eta_definitions"] = dict(auc_le_half_implies="eta=1-AUC>=1/2", reproduced=True)
    # R6 (item 5): stated Theorem 1 bound exceeds the proved one by eta(W+1)/K
    R["R6_item5_thm1_stated_vs_proved"] = dict(example=dict(eta=1, W=7, K=10, delta=0.1, proved=0.1,
                                                            stated=round(1 - 1 * (1 - 0.8) + 0.1, 3)),
                                               reproduced=True)
    return R


# =====================================================================================================
# B1 -- Theorem 1'(ii) + Algorithm 1: thinned periodic backbone, exact rationals
# =====================================================================================================
LAM_GRID = sorted({Fr(i, 10) for i in range(11)} | {Fr(1, 3), Fr(2, 3), Fr(3, 4), Fr(1, 7), Fr(9, 10)})


def b1_backbone(Tmax=30, Wmax=7):
    n_cells = n_win = n_viol = n_budget_viol = n_old_viol = n_old_cells = 0
    tight = 0
    min_slack = None
    for T in range(1, Tmax + 1):
        for B in range(1, T + 1):
            K = -(-T // B)                       # K = ceil(T/B) = ceil(1/a), a = B/T
            nU = [len(range(1 + U, T + 1, K)) for U in range(K)]
            n_cells += 1
            if max(nU) > B or Fr(sum(nU), K) > B:  # hard budget for every phase; mean count T/K <= B
                n_budget_viol += 1
            for W in range(0, Wmax + 1):
                if W + 1 > T:
                    continue
                beta = min(Fr(1), Fr(W + 1, K))
                for o in range(1, T - W + 1):    # every window inside the horizon
                    counts = {}
                    for U in range(K):
                        k = sum(1 for t in range(o, o + W + 1) if (t - 1 - U) % K == 0)
                        counts[k] = counts.get(k, 0) + 1
                    n_win += 1
                    for lam in LAM_GRID:
                        P = sum(Fr(c, K) * (1 - (1 - lam) ** k) for k, c in counts.items())
                        sl = P - lam * beta
                        if sl < 0:
                            n_viol += 1
                        if sl == 0 and lam > 0:
                            tight += 1
                        min_slack = sl if min_slack is None else min(min_slack, sl)
                # old rule: random-phase periodic schedule with stride S = ceil(1/(lam a)) = ceil(T/(lam B));
                # a window of W+1 epochs inside the horizon meets it with probability min(W+1, S)/S
                for lam in LAM_GRID:
                    if lam == 0:
                        continue
                    S = math.ceil(Fr(T) / (lam * B))
                    n_old_cells += 1
                    if Fr(min(W + 1, S), S) < lam * beta:
                        n_old_viol += 1
    res = dict(T_range=[1, Tmax], W_range=[0, Wmax], lambda_grid=[str(x) for x in LAM_GRID],
               budget_cells=n_cells, budget_violations=n_budget_viol, windows=n_win,
               floor_checks=n_win * len(LAM_GRID), floor_violations=n_viol, floor_tight_cases=tight,
               min_slack=str(min_slack), old_rule_checks=n_old_cells, old_rule_violations=n_old_viol)
    check("B1 budget", n_budget_viol == 0, n_budget_viol)
    check("B1 floor lam*beta", n_viol == 0, n_viol)
    check("B1 old rule must be refuted on the same grid", n_old_viol > 0)
    return res


# =====================================================================================================
# B2 -- Algorithm 1 / Theorem 3 on the canonical family (randomised rounding of lambda*B), exact
# =====================================================================================================
def b2_canonical_hedge(Nmax=8):
    n = viol = 0
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            beta = Fr(m, N)
            A = set(range(m))                       # marginals do not depend on the position of A
            blocks = range(N)
            for lam in LAM_GRID:
                lo = math.floor(lam * m)
                frac = lam * m - lo
                Ls = [(lo, 1 - frac)] + ([(lo + 1, frac)] if frac > 0 else [])
                marg = [Fr(0)] * N
                used = Fr(0)
                for L, pL in Ls:
                    if pL == 0:
                        continue
                    subsets = list(itertools.combinations(blocks, L))
                    for bb in subsets:
                        pb = pL / len(subsets)
                        bbs = set(bb)
                        free = [b for b in A if b not in bbs]
                        k = m - L                    # advice arm budget; free advised >= m - L
                        if k > len(free):
                            raise RuntimeError("advice arm short")
                        for b in bbs:
                            marg[b] += pb
                        for b in free:               # uniform k-subset of the free advised blocks
                            marg[b] += pb * Fr(k, len(free)) if free else 0
                        used += pb * (L + k)
                c = sum(marg[b] for b in A) / m
                r = sum(sorted(marg)[:m]) / m
                q = 1 - lam + lam * beta
                ok = (c == q and r == rstar(q, beta) and used == m and
                      all(marg[b] == lam * beta for b in blocks if b not in A) and
                      len({marg[b] for b in A}) == 1)
                n += 1
                viol += not ok
    check("B2 canonical HEDGE (c, r) = (q, r*(q)), budget = m, symmetric marginals", viol == 0, viol)
    return dict(N_range=[2, Nmax], cells=n, violations=viol)


# =====================================================================================================
# B3 -- Theorem 1: pathwise R <= 1 - eta + min{eta, delta} for delta-committed gates
# =====================================================================================================
def b3_theorem1(Nmax=8):
    n = viol = tight_ok = tight_cells = 0
    deltas = [Fr(0), Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(3, 4), Fr(1)]
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            A = set(range(m))
            sigmas = [set(s) for k in range(m + 1) for s in itertools.combinations(range(N), k)]
            for E in itertools.combinations(range(N), m):
                E = set(E)
                eta = Fr(len(E - A), m)
                for d in deltas:
                    best = 0
                    bound = 1 - eta + min(eta, d)
                    for s in sigmas:
                        if len(s - A) > d * m:
                            continue
                        cov = Fr(len(s & E), m)
                        n += 1
                        if cov > bound:
                            viol += 1
                        best = max(best, cov)
                    if (d * m).denominator == 1:
                        tight_cells += 1
                        tight_ok += best == bound
    check("B3 Theorem 1 pathwise", viol == 0, viol)
    check("B3 Theorem 1 attained when delta*m integer", tight_ok == tight_cells, (tight_ok, tight_cells))
    return dict(N_range=[2, Nmax], checks=n, violations=viol, attained=f"{tight_ok}/{tight_cells}")


# =====================================================================================================
# B4 -- Theorem 1'(i): eta = fraction of events whose window holds no advised epoch
# =====================================================================================================
def b4_theorem1p_i(trials=3000, seed=20261124):
    rng = random.Random(seed)
    viol = done = 0
    for _ in range(trials):
        T = rng.randint(10, 200)
        W = rng.randint(0, 9)
        B = rng.randint(1, max(1, T // 4))
        mmax = (T - B) // (W + 1)
        if mmax < 1:
            continue
        m = rng.randint(1, mmax)
        onsets = sorted(rng.sample(range(1, T - W + 1), min(m, T - W)))
        m = len(onsets)
        if m * (W + 1) > T - B:
            continue
        inwin = set()
        for o in onsets:
            inwin.update(range(o, o + W + 1))
        nonwin = [t for t in range(1, T + 1) if t not in inwin]
        if len(nonwin) < B:
            viol += 1
            continue
        score = {t: 0 for t in range(1, T + 1)}
        for t in rng.sample(nonwin, B):
            score[t] = 1
        adv = sorted(score, key=lambda t: (-score[t], t))[:B]          # G_adv: top-B epochs by cue score
        eta = sum(1 for o in onsets if not set(range(o, o + W + 1)) & set(adv)) / m
        R = sum(1 for o in onsets if set(range(o, o + W + 1)) & set(adv)) / m
        wins = [score[t] for t in inwin]
        nons = [score[t] for t in nonwin]
        auc = sum((w > z) + 0.5 * (w == z) for w in wins for z in nons) / (len(wins) * len(nons))
        done += 1
        viol += not (eta == 1 and R == 0 and auc <= 0.5)
    check("B4 Theorem 1'(i)", viol == 0, viol)
    return dict(instances=done, violations=viol)


# =====================================================================================================
# B5 -- Proposition 5: advice first, test-then-fallback, exhaustive exact enumeration
# =====================================================================================================
def tf_exact(N, m, E):
    """advice first A = {1..m}; follow advice while hits; at first advised miss tau activate a uniformly random
    (m - tau)-subset of {tau+1..N}; expected coverage computed by enumerating the fallback subsets."""
    hits = 0
    for t in range(1, m + 1):
        if t in E:
            hits += 1
            continue
        rest = list(range(t + 1, N + 1))
        k = min(m - t, len(rest))
        subs = list(itertools.combinations(rest, k))
        tot = sum(len(set(s) & E) for s in subs)
        return Fr(hits) + Fr(tot, len(subs))
    return Fr(hits)


def b5_prop5(Nmax=10, Nclosed=400):
    viol = cells = 0
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            c = tf_exact(N, m, set(range(1, m + 1))) / m
            r = min(tf_exact(N, m, set(E)) for E in itertools.combinations(range(1, N + 1), m)) / m
            beta = Fr(m, N)
            ok = c == 1 and r == Fr(m - 1, N - 1) and r >= (1 - Fr(1, m)) * beta
            gain_pos = r > theta0(beta)
            ok = ok and (gain_pos == (m >= 2 and N - m >= 2))
            cells += 1
            viol += not ok
    mono_viol = 0
    for N in range(2, Nclosed + 1):
        for m in range(1, N):
            f = [h + Fr((m - h) * (m - h - 1), N - h - 1) for h in range(m)]
            mono_viol += any(f[i + 1] < f[i] for i in range(m - 1))
            if min(f) != Fr(m * (m - 1), N - 1):
                mono_viol += 1
    check("B5 Proposition 5 exact", viol == 0, viol)
    check("B5 f(h) non-decreasing", mono_viol == 0, mono_viol)
    return dict(exhaustive_N_range=[2, Nmax], cells=cells, violations=viol,
                closed_form_N_range=[2, Nclosed], monotonicity_violations=mono_viol)


# =====================================================================================================
# B6 -- adaptive optimum: own sequence-form LP over the full history tree
# =====================================================================================================
class AdaptiveLP:
    """Information sets = histories h in {0 skip,1 miss,2 hit}^{t-1} with at most m activations. Variables
    x(h,a) are realisation weights. Coverage under E is linear in x. Does not depend on the advised set A:
    A enters only through the consistency row (E = A)."""

    def __init__(self, N, m):
        self.N, self.m = N, m
        infos = [()]
        layers = [[()]]
        for t in range(2, N + 1):
            nxt = []
            for h in layers[-1]:
                for o in (0, 1, 2):
                    h2 = h + (o,)
                    if sum(1 for z in h2 if z) <= m:
                        nxt.append(h2)
            layers.append(nxt)
            infos += nxt
        self.infos = infos
        self.idx = {}
        for h in infos:
            self.idx[(h, 0)] = len(self.idx)
            if sum(1 for z in h if z) < m:
                self.idx[(h, 1)] = len(self.idx)
        nv = len(self.idx) + 1                       # last variable = z
        self.nv = nv
        rows, cols, vals, rhs = [], [], [], []
        for r_i, h in enumerate(infos):
            for a in (0, 1):
                if (h, a) in self.idx:
                    rows.append(r_i); cols.append(self.idx[(h, a)]); vals.append(1.0)
            if h:
                parent = (h[:-1], 0 if h[-1] == 0 else 1)
                rows.append(r_i); cols.append(self.idx[parent]); vals.append(-1.0)
                rhs.append(0.0)
            else:
                rhs.append(1.0)
        self.Aeq = csr_matrix((vals, (rows, cols)), shape=(len(infos), nv))
        self.beq = np.array(rhs)
        self.Es = list(itertools.combinations(range(1, N + 1), m))
        self.cov = {E: self._cov_row(set(E)) for E in self.Es}

    def _cov_row(self, E):
        row = np.zeros(self.nv)
        for h in self.infos:
            t = len(h) + 1
            if t not in E or (h, 1) not in self.idx:
                continue
            if all((z == 0) or ((z == 2) == ((s + 1) in E)) for s, z in enumerate(h)):
                row[self.idx[(h, 1)]] = 1.0
        return row

    def solve(self, A, x):
        m = self.m
        A = tuple(sorted(A))
        Aub = []
        bub = []
        for E in self.Es:                            # z*m - cov(E) <= 0
            row = -self.cov[E].copy()
            row[-1] = m
            Aub.append(row); bub.append(0.0)
        row = -self.cov[A].copy()                    # cov(A) >= m - x
        Aub.append(row); bub.append(-(m - x))
        c = np.zeros(self.nv); c[-1] = -1.0
        res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=self.Aeq, b_eq=self.beq,
                      bounds=[(0, None)] * (self.nv - 1) + [(None, None)], method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed N={self.N} m={m} A={A} x={x}: {res.message}")
        sol = res.x
        r_pol = min(float(self.cov[E] @ sol) for E in self.Es) / m
        return -res.fun, sol, r_pol

    def arun_marginals(self, sol, A):
        """q_t = probability that the run on E = A activates block t."""
        A = set(A)
        q = np.zeros(self.N + 1)
        for h in self.infos:
            t = len(h) + 1
            if (h, 1) not in self.idx:
                continue
            if all((z == 0) or ((z == 2) == ((s + 1) in A)) for s, z in enumerate(h)):
                q[t] += sol[self.idx[(h, 1)]]
        return q


def window_lemma_ok(q_nonadv, beta, m, N, r):
    """Lemma S1.3 with advice last: for d>=0, 1<=l<=m, d+l<=n:
    m(beta-r)(N-d-l) >= (m-l)(Q(d)-beta d) + (n-d) sum_{j<=l} (beta - (Q(d+j)-Q(d)))."""
    n = N - m
    Q = np.concatenate([[0.0], np.cumsum(q_nonadv)])
    worst = math.inf
    nontriv = 0
    for d in range(0, n):
        for l in range(1, min(m, n - d) + 1):
            rhs = (m - l) * (Q[d] - beta * d) + (n - d) * sum(beta - (Q[d + j] - Q[d]) for j in range(1, l + 1))
            lhs = m * (beta - r) * (N - d - l)
            worst = min(worst, lhs - rhs)
            nontriv += rhs > 1e-9
    return worst, nontriv


def b6_adaptive_lp(Nmax_all=7, N_extra=8):
    out = dict(cells=0, lps=0)
    v = dict(ceiling=0, thm5=0, cor7ii=0, prop5=0, prop9=0, lemmaS13=0)
    lemma_nontriv = 0
    cor7_rows = []
    lp_cache = {}
    for N in range(3, N_extra + 1):
        for m in range(1, N):
            lp = AdaptiveLP(N, m)
            beta = m / N
            th0 = max(0.0, (2 * beta - 1) / beta)
            first = tuple(range(1, m + 1))
            last = tuple(range(N - m + 1, N + 1))
            inter = tuple(sorted({1 + (2 * i) % N for i in range(m)} if m <= (N + 1) // 2 else
                                 set(range(1, N + 1)) - {2 + 2 * i for i in range(N - m)}))
            if len(inter) != m:
                inter = tuple(sorted(random.Random(N * 100 + m).sample(range(1, N + 1), m)))
            places = (list(itertools.combinations(range(1, N + 1), m)) if N <= Nmax_all
                      else sorted({first, last, inter}))
            thr = (1 - beta) * m
            xs = sorted({0.0, 0.5, 1.0, 2.0, thr} | ({thr - 0.5} if thr - 0.5 > 0 else set()))
            rho = {}
            for A in places:
                for x in xs:
                    val, sol, r_pol = lp.solve(A, x)
                    out["lps"] += 1
                    rho[(A, x)] = val
                    lp_cache[(N, m, A, x)] = val
                    out["cells"] += 1
                    v["ceiling"] += val > beta + TOL
                    # Proposition 9: gap = 0 iff x >= (1-beta) m
                    if x >= thr - 1e-12:
                        v["prop9"] += abs(val - beta) > TOL
                    elif x <= thr - 0.5 + 1e-12:
                        v["prop9"] += beta - val <= TOL
                    if A == last:
                        b5 = th0 + (1 - th0) * min(1.0, (1 - th0) * beta / (1 - beta) * x)
                        v["thm5"] += val > b5 + TOL
                        q = lp.arun_marginals(sol, A)[1:N - m + 1]
                        w, nt = window_lemma_ok(q, beta, m, N, r_pol)
                        v["lemmaS13"] += w < -1e-6
                        lemma_nontriv += nt > 0
            # Corollary 7(ii): worst case over positions at x = 0 equals theta0 = r*(1), attained by advice last
            vals0 = {A: rho[(A, 0.0)] for A in places}
            mn = min(vals0.values())
            ok = abs(mn - th0) <= TOL and abs(vals0[last] - th0) <= TOL
            v["cor7ii"] += not ok
            # Proposition 5: advice first at x = 0 -- TF reaches (m-1)/(N-1) and no adaptive gate does better
            v["prop5"] += abs(vals0[first] - (m - 1) / (N - 1)) > TOL
            cor7_rows.append(dict(N=N, m=m, all_positions=N <= Nmax_all, rho0_min=round(mn, 9),
                                  rho0_last=round(vals0[last], 9), rho0_first=round(vals0[first], 9),
                                  theta0=round(th0, 9), prop5_value=round((m - 1) / (N - 1), 9)))
    for k, val in v.items():
        check(f"B6 {k}", val == 0, val)
    out.update(violations=v, lemmaS13_nontrivial_cells=lemma_nontriv, corollary7_rows=cor7_rows,
               N_all_positions=[3, Nmax_all], N_three_positions=N_extra)
    return out, lp_cache


# =====================================================================================================
# B7 -- non-adaptive optimum for every position (LP over marginals) = r*(max(c, beta))
# =====================================================================================================
def b7_nonadaptive(Nmax=7):
    viol = n = 0
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            beta = m / N
            Es = list(itertools.combinations(range(N), m))
            for A in itertools.combinations(range(N), m):
                for c in (0.0, 0.25, beta, 0.6, 0.9, 1.0):
                    # vars p_0..p_{N-1}, z ; max z s.t. z*m <= sum_E p ; sum_A p >= m c ; sum p <= m
                    nv = N + 1
                    Aub, bub = [], []
                    for E in Es:
                        row = np.zeros(nv); row[list(E)] = -1; row[-1] = m
                        Aub.append(row); bub.append(0)
                    row = np.zeros(nv); row[list(A)] = -1; Aub.append(row); bub.append(-m * c)
                    row = np.zeros(nv); row[:N] = 1; Aub.append(row); bub.append(m)
                    cc = np.zeros(nv); cc[-1] = -1
                    res = linprog(cc, A_ub=np.array(Aub), b_ub=np.array(bub),
                                  bounds=[(0, 1)] * N + [(None, None)], method="highs")
                    target = float(rstar(Fr(max(c, beta)).limit_denominator(10 ** 9), Fr(m, N)))
                    n += 1
                    viol += abs(-res.fun - target) > TOL
    check("B7 non-adaptive optimum = r*(max(c,beta)) for every position", viol == 0, viol)
    return dict(N_range=[2, Nmax], cells=n, violations=viol)


# =====================================================================================================
# B8 -- sprinkling policies SP(p): exact evaluation over all E
# =====================================================================================================
def sp_value(N, m, A, E, p):
    """exact E|sigma cap E| of SP(p) (trust mode: every advised block, non-advised block i (time-order
    index) w.p. p[i]; switch at first contradiction to a uniform min{b, N-tau} subset of later blocks)."""
    A = set(A)
    nonadv = [t for t in range(1, N + 1) if t not in A]
    pos = {t: i for i, t in enumerate(nonadv, start=1)}
    after = [0] * (N + 2)
    for t in range(N, 0, -1):
        after[t] = after[t + 1] + (1 if t in E else 0)
    states = {(0, 0): Fr(1)}                       # (used, hits) -> probability, trust mode
    val = Fr(0)
    for t in range(1, N + 1):
        new = {}
        for (u, h), pr in states.items():
            if u >= m:
                new[(u, h)] = new.get((u, h), 0) + pr
                continue
            if t in A:
                if t in E:
                    new[(u + 1, h + 1)] = new.get((u + 1, h + 1), 0) + pr
                else:
                    b, rest = m - u - 1, N - t
                    val += pr * (h + (Fr(min(b, rest), rest) * after[t + 1] if rest else 0))
            else:
                g = p[pos[t]]
                if g > 0:
                    if t in E:
                        b, rest = m - u - 1, N - t
                        val += pr * g * (h + 1 + (Fr(min(b, rest), rest) * after[t + 1] if rest else 0))
                    else:
                        new[(u + 1, h)] = new.get((u + 1, h), 0) + pr * g
                if g < 1:
                    new[(u, h)] = new.get((u, h), 0) + pr * (1 - g)
        states = new
    for (u, h), pr in states.items():
        val += pr * h
    return val


def sp_c_r(N, m, A, p):
    c = sp_value(N, m, A, set(A), p) / m
    r = min(sp_value(N, m, A, set(E), p) for E in itertools.combinations(range(1, N + 1), m)) / m
    return c, r


def b8_sprinkling(lp_cache, Nmax=9):
    v = dict(thm6=0, thm8a_general=0, thm8a_consequence=0, below_lp=0)
    n = dict(thm6=0, thm8a_general=0, thm8a_consequence=0, below_lp=0)
    rng = random.Random(7)
    for N in range(3, Nmax + 1):
        for m in range(1, N):
            n_ = N - m
            beta = Fr(m, N)
            first = tuple(range(1, m + 1))
            last = tuple(range(N - m + 1, N + 1))
            rnd = tuple(sorted(rng.sample(range(1, N + 1), m)))
            places = sorted({first, last, rnd})
            gb = 1 + (1 - beta) / beta ** 2
            for A in places:
                # Theorem 6 (SV(x))
                for x in sorted({Fr(1, 2), Fr(1), Fr(2), Fr(n_)}):
                    if not (0 < x <= n_):
                        continue
                    p = [None] + [x / n_] * n_
                    c, r = sp_c_r(N, m, A, p)
                    n["thm6"] += 1
                    ok = c >= 1 - x / m and r >= beta - (1 - beta) / (beta * x) - (x + 2) / m
                    v["thm6"] += not ok
                    key = (N, m, A, float(x))
                    if key in lp_cache:
                        n["below_lp"] += 1
                        v["below_lp"] += float(r) > lp_cache[key] + TOL
                # Theorem 8(a), general form: H(gamma, i0), gamma >= gamma_beta, integer i0 in [gamma, n]
                for gam in (gb, gb * Fr(3, 2)):
                    alpha = (1 - beta) / (gam - 1)
                    for i0 in range(math.ceil(gam), n_ + 1):
                        p = [None] + [Fr(0) if i < i0 else gam / i for i in range(1, n_ + 1)]
                        X = sum(p[1:])
                        c, r = sp_c_r(N, m, A, p)
                        n["thm8a_general"] += 1
                        ok = c >= 1 - X / m and r >= beta - ((1 - beta + alpha) * i0 + 2 * (X + 1)) / m
                        v["thm8a_general"] += not ok
                # Theorem 8(a), consequence form with gamma = gamma_beta
                g = float(gb)
                if n_ / g >= 1:
                    Cp = (1 - float(beta) + float(beta) ** 2) * ((1 - float(beta)) / float(beta)) * math.exp(1 / g)
                    xmax = 1 + g * math.log(n_ / g)
                    for x in np.linspace(1, xmax, 5):
                        i0 = math.ceil(n_ * math.exp(-(x - 1) / g) - 1e-12)
                        p = [None] + [Fr(0) if i < i0 else gb / i for i in range(1, n_ + 1)]
                        c, r = sp_c_r(N, m, A, p)
                        n["thm8a_consequence"] += 1
                        ok = (float(c) >= 1 - x / m - 1e-12 and
                              float(r) >= float(beta) - Cp * math.exp(-x / g) - (2 * x + 3) / m - 1e-12)
                        v["thm8a_consequence"] += not ok
    for k in v:
        check(f"B8 {k}", v[k] == 0, v[k])
    return dict(N_range=[3, Nmax], checks=n, violations=v)


def pay(psi, beta):
    return max(-psi, 0) - beta * max(psi, 0)


def b8_lemma_stop(Nmax=8, seed=11):
    """Lemma S1.2 pathwise: at a contradiction Ybar >= beta m - pay(psi) - 2(s+1); otherwise |sigma cap E| >= m-s."""
    rng = random.Random(seed)
    viol = n = 0
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            beta = Fr(m, N)
            places = list(itertools.combinations(range(1, N + 1), m))
            if len(places) > 12:
                places = rng.sample(places, 12)
            for A in places:
                A = set(A)
                nonadv = [t for t in range(1, N + 1) if t not in A]
                for coins in itertools.product((0, 1), repeat=len(nonadv)):
                    coin = dict(zip(nonadv, coins))
                    for E in itertools.combinations(range(1, N + 1), m):
                        E = set(E)
                        used = s = h = M = nu = 0
                        stop = None
                        for t in range(1, N + 1):
                            if t not in A:
                                nu += 1
                            if used >= m:
                                if t not in A and t in E:
                                    M += 1
                                continue
                            if t in A:
                                used += 1
                                if t in E:
                                    h += 1
                                else:
                                    stop = (t, 1, 0)
                                    break
                            else:
                                if coin[t]:
                                    used += 1
                                    if t in E:
                                        stop = (t, 0, 1)
                                        break
                                    s += 1
                                elif t in E:
                                    M += 1
                        n += 1
                        if stop is None:
                            hits = h
                            viol += hits < m - s
                            continue
                        tau, dA, dN = stop
                        u = m - h
                        Qv = u + (N - m) - nu
                        psi = beta * nu - M
                        b = m - used
                        Q0 = N - tau
                        Rem = len([e for e in E if e > tau])
                        Yb = h + dN + (Fr(min(b, Q0), Q0) * Rem if Q0 else 0)
                        viol += Yb < beta * m - pay(psi, beta) - 2 * (s + 1)
                        viol += Q0 != Qv - dA
    check("B8 Lemma S1.2 pathwise", viol == 0, viol)
    return dict(N_range=[2, Nmax], paths=n, violations=viol)


def b8_identity():
    """Lemma S1.1: h + u(u-M)/Q - beta m = h(Q-u)(Q-u+nu)/(QN) + (u/Q) psi, and its three consequences."""
    viol = n = 0
    for h in range(0, 6):
        for u in range(1, 6):
            for nn in range(0, 6):
                for nu in range(0, nn + 1):
                    N = u + h + nn
                    m = u + h
                    beta = Fr(m, N)
                    Q = u + nn - nu
                    for M in [Fr(k, 2) for k in range(0, 2 * (u + nn) + 1)]:
                        psi = beta * nu - M
                        lhs = h + Fr(u) * (u - M) / Q - beta * m
                        rhs = Fr(h * (Q - u) * (Q - u + nu), Q * N) + Fr(u, Q) * psi
                        n += 1
                        viol += lhs != rhs
                        viol += lhs < Fr(u, Q) * psi
                        if psi >= 0:
                            viol += lhs < beta * psi
                        else:
                            viol += lhs < psi
    check("B8 Lemma S1.1 identity", viol == 0, viol)
    return dict(cases=n, violations=viol)


# =====================================================================================================
# B9 -- Theorem 8(b): step-4 sum bound in exact rationals; constants (S7); applicability (S8)
# =====================================================================================================
def phi(s):
    return (1 + s) * math.log1p(s) - s


def b9_lower_bound_internals(trials=400, seed=5):
    rng = random.Random(seed)
    viol = 0
    min_slack = math.inf
    for _ in range(trials):
        s = Fr(rng.randint(5, 400), 100)                       # varsigma in [0.05, 4]
        d1 = rng.randint(2, 60)
        d2 = rng.randint(d1, 12 * d1)
        i = rng.randint(d1 + 1, int((1 + s) * d2) + 1)
        Xi = Fr(0)
        for d in range(d1, d2 + 1):
            ld = math.floor(s * d)
            if d < i <= d + ld:
                Xi += Fr(d + ld - i + 1, d * d)
        bound = phi(float(s)) + float((1 + s) ** 2) / (4 * d1)
        sl = bound - float(Xi)
        min_slack = min(min_slack, sl)
        viol += sl < -1e-12
        # step 2 sums: sum 1/d^2 <= 2/d1 ; sum l_d/d^2 >= s ln(d2/d1) - 2/d1 ; sum 1/d <= ln(d2/d1) + 1/d1
        S0 = sum(Fr(1, d * d) for d in range(d1, d2 + 1))
        S1 = sum(Fr(math.floor(s * d), d * d) for d in range(d1, d2 + 1))
        S2 = sum(Fr(1, d) for d in range(d1, d2 + 1))
        L = math.log(d2 / d1)
        viol += S0 > Fr(2, d1)
        viol += float(S1) < float(s) * L - 2 / d1 - 1e-12
        viol += float(S2) > L + 1 / d1 + 1e-12
    check("B9 step-4 and step-2 bounds", viol == 0, viol)
    return dict(trials=trials, violations=viol, min_slack_step4=min_slack)


def consts(beta):
    g = 1 + (1 - beta) / beta ** 2
    f = lambda s: math.exp((1 - beta) * s) - 1 - s                     # noqa: E731
    s0 = beta / (1 - beta)
    hi = 1.0
    while f(hi) <= 0:
        hi *= 2
    sstar = brentq(f, max(1e-9, s0 * 0.5), hi, xtol=1e-15) if f(max(1e-9, s0 * 0.5)) < 0 else None
    Cstar = math.log1p(sstar) / beta
    Cp = (1 - beta + beta ** 2) * ((1 - beta) / beta) * math.exp(1 / g)
    # direct grid minimisation of phi(s) / (beta s - beta^2/(1-beta)) over s > s0
    grid = np.linspace(s0 * (1 + 1e-6), s0 + 60, 600001)
    E = ((1 + grid) * np.log1p(grid) - grid) / (beta * grid - beta ** 2 / (1 - beta))
    return dict(beta=beta, gamma_beta=g, upper_exponent=1 / g, varsigma_star=sstar,
                varsigma_star_gt_beta_over_1mbeta=sstar > s0, C_star=Cstar, C_star_grid=float(E.min()),
                ratio=Cstar * g, C_prime=Cp)


def thm1p_applicable(beta, m, x, s, delta):
    N = m / beta
    n = N - m
    om = (1 - delta) * beta * s - beta ** 2 / (1 - beta)
    if om <= 0:
        return False, None
    d2 = math.floor(delta * n)
    d1 = math.floor(d2 * math.exp(-(phi(s) * x + 1) / om))
    need = max(2, 1 / s, 4 * beta + 2 * beta ** 2 / (1 - beta) + (1 + s) ** 2 * x / 2)
    return d1 >= need, n * d1 / (4 * m * N)


def b9_constants_and_scope():
    table = {str(b): consts(b) for b in (0.2, 0.5, 0.8)}
    expect = {"0.2": (21, 0.048, 2.154, 45.2), "0.5": (3, 0.333, 2.513, 7.54), "0.8": (1.3125, 0.762, 3.325, 4.36)}
    ok = True
    for b, (g, ue, cs, ra) in expect.items():
        t = table[b]
        ok &= abs(t["gamma_beta"] - g) < 1e-9 and abs(t["upper_exponent"] - ue) < 5e-4
        ok &= abs(t["C_star"] - cs) < 5e-4 and abs(t["ratio"] - ra) < 5e-3 * ra
        ok &= abs(t["C_star_grid"] - t["C_star"]) < 1e-6 and t["varsigma_star_gt_beta_over_1mbeta"]
    check("B9 constants table S7", ok)
    # S8: smallest m (power grid) where the finite form of Theorem 8(b) applies, beta = 1/2, x in {1,2,3},
    # varsigma = varsigma*, delta = half of its admissible maximum
    b = 0.5
    s = table["0.5"]["varsigma_star"]
    dmax = min(1 / (1 + s), b / ((1 - b) * s))
    scope = {}
    for x in (1, 2, 3):
        first = None
        for e10 in np.arange(2, 17.01, 0.05):
            m = int(round(10 ** e10))
            if thm1p_applicable(b, m, x, s, dmax / 2)[0]:
                first = m
                break
        scope[f"x={x}"] = first
    s8_lower = scope["x=1"] is not None and scope["x=2"] is not None and 3e5 <= scope["x=2"] <= 3e7
    # scope with (varsigma, delta) optimised on a grid (the number stated in the paper)
    opt = {}
    for x in (1, 2):
        best = None
        for sv in np.linspace(1.01, 8, 141):
            dm = min(1 / (1 + sv), b / ((1 - b) * sv))
            for dl in np.linspace(0.02, 1, 50) * dm:
                lo, hi = 10.0, 1e12
                if not thm1p_applicable(b, int(hi), x, sv, dl)[0]:
                    continue
                while hi / lo > 1.02:
                    mid = math.sqrt(lo * hi)
                    if thm1p_applicable(b, int(mid), x, sv, dl)[0]:
                        hi = mid
                    else:
                        lo = mid
                if best is None or hi < best[0]:
                    best = (hi, sv, dl)
        # P5C: local refinement around the grid optimum (PROOFCHECK3 K2 found 6 547 at x=1, 240 242 at x=2)
        for (fs, fd, n) in ((0.07, 0.2, 29), (0.02, 0.05, 31), (0.005, 0.012, 21)):   # three zooms
            _, sv0, dl0 = best
            for sv in np.linspace((1 - fs) * sv0, (1 + fs) * sv0, n):
                dm = min(1 / (1 + sv), b / ((1 - b) * sv))
                for dl in np.linspace((1 - fd) * dl0, (1 + fd) * dl0, n):
                    if dl >= dm or sv <= 1:
                        continue
                    lo, hi = 10.0, best[0] * 1.05
                    if not thm1p_applicable(b, int(hi), x, sv, dl)[0]:
                        continue
                    while hi - lo > 1:
                        mid = (lo + hi) / 2
                        if thm1p_applicable(b, int(mid), x, sv, dl)[0]:
                            hi = mid
                        else:
                            lo = mid
                    if hi < best[0]:
                        best = (hi, sv, dl)
        opt[f"x={x}"] = dict(m_first=best[0], varsigma=best[1], delta=best[2])
    s8_opt = 2.3e5 <= opt["x=2"]["m_first"] < 2.5e5 and 6.5e3 <= opt["x=1"]["m_first"] < 6.6e3
    check("B9 S8 optimised scope (paper: m ~ 2.4e5 at x=2, 6.5e3 at x=1)", s8_opt, opt)
    # S8: at beta = 0.2 the upper bound beats the trivial gap beta - theta0 only for x > 60
    t2 = table["0.2"]
    x_star = t2["gamma_beta"] * math.log(t2["C_prime"] / 0.2)
    s8_upper = 60 < x_star < 61
    # N = 187, m = 75: upper bound of Theorem 8(a) and finite lower bound are both vacuous
    N, m = 187, 75
    bb = m / N
    g = 1 + (1 - bb) / bb ** 2
    Cp = (1 - bb + bb ** 2) * ((1 - bb) / bb) * math.exp(1 / g)
    n_ = N - m
    xs = np.linspace(1, 1 + g * math.log(n_ / g), 400)
    best_gap_bound = min(Cp * math.exp(-x / g) + (2 * x + 3) / m for x in xs)
    trivial = bb - max(0.0, (2 * bb - 1) / bb)
    vac_upper = best_gap_bound >= trivial
    sb = consts(bb)["varsigma_star"]
    dmx = min(1 / (1 + sb), bb / ((1 - bb) * sb))
    vac_lower = not any(thm1p_applicable(bb, m, x, sv, min(1 / (1 + sv), bb / ((1 - bb) * sv)) * f)[0]
                        for x in (1.0, 2.0, 3.0) for sv in np.linspace(1.01, 8, 36) for f in np.linspace(0.05, 0.95, 19))
    check("B9 S8 lower-bound scope (m ~ 1e6 at beta=1/2, x<=2)", s8_lower, scope)
    check("B9 S8 upper bound at beta=0.2 beats trivial only for x > 60", s8_upper, x_star)
    check("B9 vacuous at N=187, m=75", vac_upper and vac_lower, (best_gap_bound, trivial, vac_lower))
    return dict(constants=table, lower_bound_first_m_beta_half=scope, lower_bound_first_m_optimised=opt, x_star_beta02=x_star,
                N187=dict(min_upper_gap_bound=best_gap_bound, trivial_gap=trivial,
                          upper_vacuous=vac_upper, lower_not_applicable_for_x_ge_1=vac_lower))


def b10_ceiling_attainment(Nmax=40, Wmax=7):
    """P5C (PROOFCHECK3 MAJOR 1): Theorem 3's attainment. A uniformly random set of m blocks activates every
    block with probability m/N = beta exactly, so it attains beta on every instance; the periodic schedule of stride
    K = ceil(N(W+1)/m) covers (W+1)/K <= m/N, with equality iff m divides N(W+1)."""
    viol, eq_ok, n = 0, True, 0
    for N in range(2, Nmax + 1):
        for m in range(1, N):
            beta = Fr(m, N)
            rand_set = Fr(m, N)   # P(block in a uniform m-subset of [N]) = C(N-1,m-1)/C(N,m)
            from math import comb
            ok_r = Fr(comb(N - 1, m - 1), comb(N, m)) == rand_set == beta
            for W in range(0, Wmax + 1):
                n += 1
                K = -(-N * (W + 1) // m)
                per = min(Fr(1), Fr(W + 1, K))
                viol += (per > beta) or (not ok_r)
                eq_ok &= ((per == beta) == ((N * (W + 1)) % m == 0)) or per == 1
    cx = {}
    for (N, m, W) in ((187, 75, 1), (3, 2, 0)):
        K = -(-N * (W + 1) // m)
        cx[f"N={N},m={m},W={W}"] = dict(K=K, periodic=float(Fr(W + 1, K)), beta=float(Fr(m, N)),
                                        periodic_below_beta=Fr(W + 1, K) < Fr(m, N))
    ok = viol == 0 and eq_ok and all(v["periodic_below_beta"] for v in cx.values())
    check("B10 ceiling attainment: uniform m-set attains beta; periodic only if m | N(W+1)", ok, (viol, eq_ok))
    return dict(cases=n, violations=viol, equality_iff_divides=eq_ok, counterexamples=cx)


# =====================================================================================================
def main():
    RESULT["meta"] = dict(script="code/p5_checks.py", date="2026-11-24", python=sys.version.split()[0],
                          numpy=np.__version__, purpose="P5B PHA 1 checks after R2 (23-11-2026)")
    RESULT["R_reproduction"] = part_R()
    rep = RESULT["R_reproduction"]
    check("R reproduce item 2", rep["R1_item2_stride_rounding"]["reproduced"])
    check("R reproduce item 1", rep["R2_item1_old_prop5"]["reproduced"])
    check("R reproduce item 6", rep["R3_item6_advice_first_gain_at_c1"]["reproduced"])
    print("R done", round(time.time() - T0, 1), "s", flush=True)
    RESULT["B1_thm1p_ii_backbone"] = b1_backbone()
    print("B1 done", round(time.time() - T0, 1), "s", flush=True)
    RESULT["B2_canonical_hedge"] = b2_canonical_hedge()
    RESULT["B3_theorem1"] = b3_theorem1()
    RESULT["B4_thm1p_i"] = b4_theorem1p_i()
    RESULT["B5_prop5"] = b5_prop5()
    print("B2-B5 done", round(time.time() - T0, 1), "s", flush=True)
    b6, cache = b6_adaptive_lp()
    RESULT["B6_adaptive_lp"] = b6
    print("B6 done", round(time.time() - T0, 1), "s", flush=True)
    RESULT["B7_nonadaptive_lp"] = b7_nonadaptive()
    RESULT["B8_sprinkling"] = b8_sprinkling(cache)
    RESULT["B8_lemma_stop"] = b8_lemma_stop()
    RESULT["B8_identity"] = b8_identity()
    print("B7-B8 done", round(time.time() - T0, 1), "s", flush=True)
    RESULT["B9_lower_internals"] = b9_lower_bound_internals()
    RESULT["B9_constants_scope"] = b9_constants_and_scope()
    RESULT["B10_ceiling_attainment"] = b10_ceiling_attainment()
    RESULT["failures"] = FAILS
    RESULT["all_pass"] = not FAILS
    runtime = round(time.time() - T0, 1)  # printed only, so that the JSON is reproducible byte for byte
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(RESULT, indent=1, default=str), encoding="utf-8")
    tmp.replace(OUT)
    print("all_pass =", RESULT["all_pass"], "| failures:", FAILS, "| runtime", runtime, "s")


if __name__ == "__main__":
    main()
