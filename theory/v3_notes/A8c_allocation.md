# A8(c) — Optimal budget allocation across a device–edge tier (Proposition 6 → Theorem 7)

Status (21-10-2026): **Theorem 7 (parts A–C) proved below.** It extends Proposition 6 (uniform tiers cannot escape $r^\star$) and Remark `rem:skew` (skewed allocation can help) into a full optimisation result: the optimal allocation, its structure, an $O(n\log n)$ algorithm with an additive gap bound, an exact pseudo-polynomial DP for integer strides, and the boundary of the skew lever (it exists only when each node's worst case scales with its own budget). Numbers: `code/v3_a8c_alloc.py` (N = 8/32/128 nodes, 5 seeds for random weights). Old statements unchanged.

---

## 0. Setting

A tier of $n$ nodes shares an activation envelope $\sum_i a_i\le A$. Node $i$ has importance weight $w_i>0$, $\sum_iw_i=1$ (its share of the tier's events, as in §Composition), runs a non-adaptive gate at per-node consistency $c\in(0,1]$ (a tier-wide service level), and has budget share $\beta_i=\beta(a_i)$. Continuous model: $\beta(a)=\min\{1,(W{+}1)a\}$; integer-stride model (deployment): $a=1/K$, $\beta=\min\{1,(W{+}1)/K\}$, $K\in\{1,2,\dots\}\cup\{\infty\}$. Write $b:=(W{+}1)A$ for the total budget share in the continuous model.

Tier robustness aggregates node guarantees linearly, $r_{\rm tier}=\sum_iw_ir_i$ (proof of Proposition 6; no independence needed). What a node can guarantee at budget share $\beta_i$ depends on the worst case it must face; we treat the two natural models.

**Model T1 (budget-scaled worst case — the canonical family of Theorem 2, used by Proposition 6 and Remark `rem:skew`).** Each node faces its own canonical worst case at its own budget ($m_i=B_i$), so its best certified robustness is the frontier value
$$V_c(\beta):=r^\star(c;\beta)\quad(\text{Theorem 2}),\qquad V_c(0):=0 .$$
Recall is a fraction, so $V_c$ is scale-free.

**Model T2 (fixed loads).** Node $i$'s worst-case event density $\mu_i=m_i/N_i\in(0,1)$ is fixed independently of its budget. Repeating the converse of Theorem 2 with $m\neq B$ (advised mass $mc$, floor $g=(B-mc)/(N-m)$, adversary on the $m$ smallest $p_b$) gives the frontier
$$V^{(\mu)}_c(\beta)=\min\{c,\ \ell_\mu(\beta)\},\quad \ell_\mu(\beta)=\begin{cases}\dfrac{\beta-\mu c}{1-\mu},&\mu\le\tfrac12,\\[4pt]\dfrac{\beta-(1-\mu)c}{\mu},&\mu>\tfrac12,\end{cases}\qquad\beta\ge\mu c,$$
(consistency $c$ is infeasible below $\beta=\mu c$, the cost of covering the advised mass), attained by the symmetric configuration (SAT-GATE with floor $g$). With $\mu=\beta$ it reduces to Theorem 2.

**Problem.** $\mathrm{OPT}=\max\{\sum_iw_iV_i(\beta_i):\ \sum_i\beta_i\le b,\ \beta_i\in[0,1]\}$ (continuous) or the same with $\beta_i=\beta(1/K_i)$, $\sum_i1/K_i\le A$ (integer strides).

---

## 1. Shape of $V_c$ in Model T1

> **Lemma 7.1.** For fixed $c\in(0,1]$, $V_c$ is continuous, non-decreasing, convex on $[0,\min\{c,\tfrac12\}]$, concave on $[\tfrac12,c]$ (when $c>\tfrac12$), constant $=c$ on $[c,1]$; it satisfies $V_c(\beta)\le\min\{\beta,c\}$ with equality iff $\beta\in\{0\}\cup[c,1]$ (for $c<1$; for $c=1$ also at no other point than $0,1$). Its concave envelope on $[0,1]$ is $\widetilde V_c(\beta)=\min\{\beta,c\}$.

*Proof.* From (eq. frontier) with roles of $c$ fixed and $\beta$ varying: for $\beta\ge c$, $V=c$. For $\beta<c$, $\beta\le\frac12$: $V=\beta(1-c)/(1-\beta)$, whose second derivative $2(1-c)/(1-\beta)^3\ge0$. For $\frac12<\beta<c$: $V=1+c-c/\beta$, second derivative $-2c/\beta^3<0$. Continuity at $\frac12$: both branches give $1-c$; at $\beta=c$: both give $c$. Bound: on the convex branch $V-\beta=\beta(\beta-c)/(1-\beta)\le0$; on the concave branch $V-\beta=-(1-\beta)(c-\beta)/\beta\le0$; equality only at $\beta=c$ (or $0$, or $1$). $\min\{\beta,c\}$ is concave, majorises $V_c$ and touches it at $0$, at $c$ and on $[c,1]$; any concave majorant lies above the chord from $(0,0)$ to $(c,c)$ and above $c$ on $[c,1]$, hence above $\min\{\beta,c\}$. $\square$

Define the *non-concavity gap* $\gamma(c):=\max_{\beta\in[0,c]}\big(\beta-V_c(\beta)\big)$; e.g. $\gamma(1)=\tfrac12$ (at $\beta=\tfrac12$).

---

## 2. Theorem 7 (Optimal tier allocation)

> **Theorem 7.** **(A) Model T1, continuous.** Sort nodes so that $w_{(1)}\ge w_{(2)}\ge\cdots$, let $k=\min\{n,\lfloor b/c\rfloor\}$ and $\rho=b-kc$ (if $k<n$).
> (A1) *Upper bound:* $\mathrm{OPT}\le U:=c\sum_{j\le k}w_{(j)}+w_{(k+1)}\rho$ (with $U=c$ if $b\ge nc$).
> (A2) *Bang-bang allocation:* $\beta_{(j)}=c$ for $j\le k$, $\beta_{(k+1)}=\rho$, all other nodes $0$, attains $U-w_{(k+1)}(\rho-V_c(\rho))\ge U-w_{\max}\gamma(c)$. It is computed in $O(n\log n)$.
> (A3) *Structure:* there is an optimal allocation in which no node exceeds $c$ (unless $b>nc$), at most one node lies in the open convex region $(0,\min\{c,\tfrac12\})$, and all nodes in the concave region $[\tfrac12,c)$ have equal weighted marginals $w_iV_c'(\beta_i)$; for equal weights the latter share a common level.
> (A4) *Value of skew:* with equal weights $w_i=1/n$ and uniform provisioning $\beta_i=\bar\beta:=b/n$ (Proposition 6), $\mathrm{OPT}-V_c(\bar\beta)\ \ge\ \min\{\bar\beta,c\}-V_c(\bar\beta)-\gamma(c)/n$, and $\mathrm{OPT}\le\min\{\bar\beta,c\}$; at $c=1$, $\bar\beta\le\frac12$ the uniform tier certifies $0$ while the optimum certifies at least $\bar\beta-\frac1{2n}$ (exactly $\bar\beta$ when $n\bar\beta$ is an integer).
> **(B) Model T2 (fixed loads).** Each $V^{(\mu_i)}_c$ is concave piecewise-linear on its feasible range $[\mu_ic,1]$ with slope $s_i=1/\max\{1-\mu_i,\mu_i\}$ up to the saturation point $\beta_i=c$. If $b\ge c\sum_i\mu_i$ (otherwise consistency $c$ is infeasible for the tier), the greedy *water-filling* — give every node its feasibility floor $\mu_ic$, then raise nodes in decreasing order of $w_is_i$ to $c$ until the budget is exhausted — is optimal, and for identical nodes every feasible allocation with $\beta_i\in[\mu c,c]$ (in particular the uniform one) is optimal: **the skew lever of Remark `rem:skew` vanishes under fixed loads.**
> **(C) Integer strides.** With $\beta_i\in\{\min(1,(W{+}1)/K):K\in\mathbb N\}\cup\{0\}$ and costs $1/K$, OPT is a multiple-choice knapsack solved exactly by dynamic programming over a budget grid in $O(n\,L\,|\mathcal K|)$ time ($L$ grid cells, $|\mathcal K|$ stride levels; costs rounded up, so the DP allocation is always feasible).

*Proof.* (A1) By Lemma 7.1, $\sum_iw_iV_c(\beta_i)\le\sum_iw_i\min\{\beta_i,c\}$; maximising the right side subject to $\sum\beta_i\le b$ is a fractional knapsack with per-unit value $w_i$ and capacity $c$ per item, whose optimum is the greedy $U$.
(A2) The stated allocation is feasible ($kc+\rho=b$); nodes at $c$ contribute $w_{(j)}c$ (Lemma 7.1, equality at $\beta=c$), the residual node contributes $w_{(k+1)}V_c(\rho)$, so the value is $U-w_{(k+1)}(\rho-V_c(\rho))\ge U-w_{(k+1)}\gamma(c)$ since $\rho<c$. Sorting dominates the cost.
(A3) Raising a node above $c$ adds nothing (flat), so excess can be moved to any node below $c$ without loss. If two nodes $i,j$ lie in $(0,\beta_c)$, $\beta_c:=\min\{c,\frac12\}$, consider $f(t)=w_iV_c(\beta_i+t)+w_jV_c(\beta_j-t)$ on the interval of $t$ keeping both in $[0,\beta_c]$: $f$ is convex (sum of convex functions), so its maximum is at an endpoint, where one of the two nodes reaches $0$ or $\beta_c$; repeating strictly decreases the number of nodes in the open region, without decreasing the objective. In the concave region the objective is differentiable; if $w_iV'(\beta_i)>w_jV'(\beta_j)$ for two interior nodes, moving a small mass from $j$ to $i$ strictly increases the objective, contradicting optimality.
(A4) Upper: concavity of $\widetilde V_c$ and Jensen, $\sum\frac1n V_c(\beta_i)\le\sum\frac1n\widetilde V_c(\beta_i)\le\widetilde V_c(\bar\beta)$. Lower: (A2) with $w_{\max}=1/n$ gives $\mathrm{OPT}\ge U-\gamma(c)/n$ and $U=\min\{\bar\beta,c\}$ for equal weights. At $c=1$, $\bar\beta\le\frac12$: $V_1(\bar\beta)=0$ and $\gamma(1)=\frac12$; if $n\bar\beta$ is an integer the residual is $0$ and the bound is exact.
(B) On $[\mu_ic,c]$, $V^{(\mu_i)}_c=\ell_{\mu_i}$ is affine with slope $s_i$ and constant $c$ afterwards, hence concave; the problem is then a continuous knapsack with concave piecewise-linear returns, for which greedy by marginal value per unit budget ($w_is_i$, then $0$) is optimal (exchange argument). For identical nodes all marginal values are equal on $[\mu c,c]$, so any split of the budget within these ranges attains the optimum.
(C) Each node picks one stride from a finite menu (a multiple-choice knapsack); the DP $D_i(\ell)=\max_K\{D_{i-1}(\ell-\lceil(1/K)/\delta\rceil)+w_iV(\beta(1/K))\}$ over grid cells of width $\delta$ is exact for the rounded-up costs. $\square$

**Corollary 7.2 (operator rule).** Under budget-scaled worst cases (T1), certify a *subset*: give the heaviest nodes exactly $\beta=c$ — where the node's frontier reaches its consistency level and the periodic backbone alone already certifies $c$ — and starve the rest; spreading the budget uniformly certifies only $V_c(\bar\beta)$, which is $0$ whenever $c=1$ and $\bar\beta\le\frac12$. Under fixed loads (T2), budget beyond the feasibility floors goes first to the nodes with the largest $w_i/\max\{1-\mu_i,\mu_i\}$ (with weights proportional to load, the busiest nodes first).

**Worked check (Remark `rem:skew`, $W=7$, $n=2$, $w=\frac12$, $\bar a=0.0625$, $c=1$):** $b=1$; uniform $\beta_i=0.5$ gives $0$ (Proposition 6); the Remark's split $(0.2,0.8)$ gives $0.375$; Theorem 7(A2) gives $(1,0)$, value $\frac12\cdot1=0.5=U$ — the optimum, since $U$ is an upper bound.

**Which model is right?** T1 is the paper's worst case: the adversary picks, at every node, the hardest instance for that node's budget, and Theorem 2's tightness is with respect to exactly this family. T2 is the natural model when per-node event loads are known and fixed. The two give opposite operator rules (concentrate vs. water-fill); the paper states both and the limitation explicitly.

---
### P3C addendum (22–24-10-2026) — Lemma 3: the Model T2 frontier, converse and attainment (as now in the paper's Appendix)
Canonical family, $N$ blocks, $m=\mu N$ events, budget $B=\beta N$, advised set $|A|=m$; non-adaptive gate with marginals $p_b\in[0,1]$, $\sum p_b\le B$; consistency $c=\frac1m\sum_{A}p_b$; robustness $r=\min_{|E|=m}\frac1m\sum_{E}p_b$ (full adversary, as in §6).
*Converse.* $mc\le B$ (feasibility). $S:=\sum_{b\notin A}p_b\le B-mc$. $E=A$ gives $r\le c$. If $\mu\le\frac12$: $E$ = the $m$ smallest non-advised, sum $\le\frac{m}{N-m}S$ ⇒ $r\le\frac{\beta-\mu c}{1-\mu}$. If $\mu>\frac12$: $E$ = all $N-m$ non-advised + the $2m-N$ smallest advised (sum $\le\frac{2m-N}{m}mc$) ⇒ $mr\le B-(N-m)c$ ⇒ $r\le\frac{\beta-(1-\mu)c}{\mu}$.
*Attainment.* $p_b=c$ on $A$, $p_b=\min\{1,g\}$ off $A$, $g=\frac{B-mc}{N-m}$; the $m$ smallest values give exactly the two branches (and $r=c$ when $g\ge c$). Realised with $\le B$ activations surely by dependent rounding. For $\mu=\beta$ this is Theorem 2.
This replaces the earlier "repeat the converse of Theorem 2 with $m\neq B$" sketch: with the full-adversary robustness the converse needs no instance-pair (mislabelled-$\eta$) argument, so the question whether that argument used $m=B$ no longer arises.
