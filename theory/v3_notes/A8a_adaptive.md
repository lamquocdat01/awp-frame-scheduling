# A8(a) — Adaptive gates: ceiling, the price of consistency, and an exact separation

Status (21-10-2026): **Theorem 4, Theorem 5, Theorem 6 and Corollary 7 are proved below** (full proofs, no conjecture needed for the asymptotic statements). One quantitative gap is left open and is stated as **Conjecture 8** (the exact finite-$m$ adaptive frontier between the Theorem 5 upper bound and the Theorem 6 lower bound); it is supported numerically by `code/v3_a8a_mc.py` (5 seeds) and labelled as such. Old results (Lemma 1, Thm 1, 1′, 2, 3, Cor., Prop 4–6) are untouched.

---

## 0. Setting (canonical block family, temporal order)

We work on the canonical block family of the paper: $N$ disjoint blocks of length $W{+}1$ in temporal order $1,\dots,N$; a block is *recalled* if at least one fire lands in it (block = staleness window of the event at its first slot); budget $B<N$ block-fires; $m=B$ events; $\beta=B/N=m/N\in(0,1)$. The cue marks an *advised set* $A\subseteq[N]$, $|A|=m$. An **instance** is any event set $E\subseteq[N]$ with $|E|=m$; its prediction error is $\eta(E)=|E\setminus A|/m$. Recall $R(\sigma;E)=|\sigma\cap E|/m$.

**Feedback (Assumption F).** When the gate fires block $t$ it observes $y_t=\mathbf 1[t\in E]$ (detector hit/miss). It observes nothing about blocks it does not fire.

**Definition (adaptive gate).** A randomized *adaptive* (online) gate decides at each block $t=1,\dots,N$, in temporal order, whether to fire $t$, as a function of (i) the cue — we even allow it to know the whole advised set $A$ in advance, which only strengthens the impossibility results —, (ii) the history $\{(s,y_s): s<t,\ s\text{ fired}\}$, (iii) its remaining budget, and (iv) internal randomness; it fires at most $B$ blocks surely. Non-adaptive gates (Def. in the paper) are the special case that ignores (ii).

**Consistency / robustness (general adversary).** $c(G)=\mathbb E[R(G;A)]$ and $r(G)=\min_{E:|E|=m}\mathbb E[R(G;E)]$, the expectation over the gate's coins. *Remark on compatibility with Theorem 2:* for a non-adaptive gate with per-block firing probabilities $p_b$, $\mathbb E[R(G;E)]=\frac1m\sum_{b\in E}p_b$, so $r(G)$ is $\frac1m$ times the sum of the $m$ smallest $p_b$ — exactly the quantity bounded in the converse of Theorem 2, which therefore gives $r(G)\le r^\star(c(G))$ for this definition too; SAT-GATE attains it (its $m$ smallest values are those used in the proof of Theorem 2). So the non-adaptive frontier $r^\star(c)$ is unchanged, and all comparisons below are like-for-like.

Write $\theta_0:=\max\{0,(2\beta-1)/\beta\}=r^\star(1)$ (the full-consistency endpoint of Theorem 2) and
$$\Delta(\beta):=\beta-r^\star(1)=\begin{cases}\beta,&\beta\le\tfrac12,\\ (1-\beta)^2/\beta,&\beta>\tfrac12.\end{cases}$$

---

## 1. Theorem 4 (Adaptive ceiling)

> **Theorem 4.** For every adaptive gate $G$ — whatever its consistency — $r(G)\le\beta$. Hence the oblivious random-phase periodic schedule, which attains $\beta$ on every instance (Lemma 1(b)), is worst-case optimal among *all* online gates, adaptive or not: detector feedback can never raise the worst-case recall guarantee above the oblivious rate.

*Proof.* Draw $E$ uniformly among the $m$-subsets of $[N]$, independently of $A$ and of the gate's coins (Yao's principle: $r(G)=\min_E\mathbb E[R(G;E)]\le\mathbb E_{E}\mathbb E[R(G;E)]$). Fix the coins, so $G$ is deterministic. Let $G^+$ run $G$ and, after $G$'s last fire, fire further not-yet-fired blocks (any fixed rule) until exactly $B$ blocks are fired; $G^+$ need not respect temporal order — it is a bookkeeping device used only for the bound. Pathwise $|\sigma(G)\cap E|\le|\sigma(G^+)\cap E|$.
Let $\sigma_1,\sigma_2,\dots,\sigma_B$ be the blocks fired by $G^+$ in firing order and $H_k$ the number of hits among the first $k$. The choice of $\sigma_{k+1}$ is a function of $(y_{\sigma_1},\dots,y_{\sigma_k})$ only. Conditional on these outcomes, $E\setminus\{\sigma_1,\dots,\sigma_k\}$ is a uniformly random $(m-H_k)$-subset of the $N-k$ unfired blocks (the prior is exchangeable and the observations only concern fired blocks). Since $\sigma_{k+1}$ is unfired, $\Pr[\sigma_{k+1}\in E\mid y_{\sigma_1..\sigma_k}]=\frac{m-H_k}{N-k}$, i.e. the outcome sequence of $G^+$ has exactly the law of $B$ draws without replacement from an urn with $m$ red balls among $N$. Hence $\mathbb E[H_B]=Bm/N$ and $\mathbb E[R(G;E)]\le\mathbb E[H_B]/m=B/N=\beta$. Averaging over coins and taking the minimum over $E$ gives $r(G)\le\beta$. Attainment by the random-phase periodic schedule is Lemma 1(b) on the canonical family ($K>W{+}1$, $(W{+}1)/K=\beta$). $\square$

*Comment.* Theorem 4 is a genuine lower bound for the adaptive class on the same worst-case family as Theorems 1/1′/2: it bounds what any feedback-aware gate can *guarantee*. It is flat in $c$; what feedback can change is only *how much consistency one keeps while staying near the ceiling* — Theorems 5–6.

---

## 2. Theorem 5 (Consistency is not free for adaptive gates)

> **Theorem 5.** Let the cue place the advised set after all non-advised blocks ($A=\{N-m+1,\dots,N\}$). For every adaptive gate $G$ with $c(G)\ge1-\varepsilon$,
> $$r(G)\;\le\;\theta_0+(1-\theta_0)\,\min\Big\{1,\ (1-\theta_0)\,\frac{\beta}{1-\beta}\,\varepsilon m\Big\}.$$
> In particular, at full consistency ($\varepsilon=0$) $r(G)\le r^\star(1)$: **feedback buys nothing at $c=1$** — the endpoint of the non-adaptive frontier (Theorem 2) is tight for the whole adaptive class. The quantity $x=\varepsilon m$ is the expected number of fires the gate can spare from the advised blocks; with $x<\frac{1-\beta}{\beta(1-\theta_0)}$ spare fires the guarantee stays strictly below the ceiling $\beta$ (for $\beta\le\frac12$: $r\le\beta x/(1-\beta)$).

*Proof.* Call the first $N-m$ blocks phase 1 (all non-advised) and the last $m$ blocks phase 2 ($=A$). Fix the coins. The *$A$-run* is the run of $G$ on the instance $E=A$; on it every fire in phase 1 is a miss and every fire in phase 2 a hit, so $R(G;A)=(\#\text{phase-2 fires})/m$. Let $p_b$ be the probability (over coins) that the $A$-run fires block $b$. Since at most $B=m$ blocks are fired, $\sum_{b\le N-m}p_b\le m-\mathbb E[\#\text{phase-2 fires on the }A\text{-run}]=m(1-c(G))\le\varepsilon m$.
Let $k:=(1-\theta_0)m=\min\{m,N-m\}$ and let the adversary choose $E$ as $k$ phase-1 blocks of smallest $p_b$ together with any $m-k=\theta_0 m$ blocks of $A$. Because the $k$ smallest of $N-m$ non-negative numbers summing to at most $\varepsilon m$ sum to at most $k\varepsilon m/(N-m)$,
$$P_{\rm div}:=\Pr[\text{the }A\text{-run fires some block of }E\setminus A]\le \frac{k\,\varepsilon m}{N-m}=(1-\theta_0)\,\varepsilon m\,\frac{\beta}{1-\beta}$$
(union bound; $m/(N-m)=\beta/(1-\beta)$). With the same coins, the run on $E$ coincides with the $A$-run until the first fired block whose outcome differs between $E$ and $A$; during phase 1 the only such blocks are those of $E\setminus A$. On the complement of the divergence event the run on $E$ therefore fires no event block in phase 1 and recalls at most $|E\cap A|=\theta_0 m$ events in phase 2; on the divergence event it recalls at most $m$. Hence $\mathbb E[R(G;E)]\le\theta_0+(1-\theta_0)\min\{1,P_{\rm div}\}$, and $r(G)\le\mathbb E[R(G;E)]$. For $\varepsilon=0$, $P_{\rm div}=0$ and $r(G)\le\theta_0=r^\star(1)$. $\square$

*Why the cue placement matters.* If the advised blocks come first, a gate can test the cue before any non-advised event passes (this is the regime of Proposition 5). Theorem 5 shows that against an adversary that also controls *when* the advice is checkable, perfect consistency forbids any adaptive gain.

---

## 3. Theorem 6 (Sprinkle-and-verify reaches the ceiling)

**Algorithm SV($x$)** (sprinkle-and-verify), $x\in(0,N-m]$, $g:=x/(N-m)$.
*Trust mode* (initial): at an advised block, fire it (if budget remains); at a non-advised block, fire it with probability $g$ using a fresh coin (if budget remains). *Switch:* at the first fired outcome that contradicts the cue — an advised miss or a non-advised hit — at time $\tau$, enter *fallback mode*: draw a uniformly random set $S$ of $\min\{b_\tau,N-\tau\}$ blocks among $\{\tau{+}1,\dots,N\}$, where $b_\tau$ is the remaining budget, and fire exactly $S$. (Online-implementable; $O(1)$ state in trust mode.)

> **Theorem 6.** For every advised set $A$ (any placement) and every $x\in(0,N-m]$, SV($x$) satisfies
> $$c\;\ge\;1-\frac{x}{m},\qquad r\;\ge\;\beta-\frac{1-\beta}{\beta\,x}-\frac{x+2}{m}.$$
> With $x=\sqrt{(1-\beta)m/\beta}$: $c\ge1-\sqrt{(1-\beta)/(\beta m)}$ and $r\ge\beta-2\sqrt{(1-\beta)/(\beta m)}-2/m$. For any fixed $\delta>0$, choosing $x\ge 2(1-\beta)/(\beta\delta)$ gives $r\ge\beta-\delta$ and $c\ge1-\delta$ for all $m\ge 2(x+2)/\delta$.

*Proof.* Consistency: on $E=A$ no contradiction ever occurs; every advised block is fired unless sprinkle fires exhausted the budget, so $R(\mathrm{SV};A)\ge1-s/m$ with $s$ the number of sprinkle fires, and $\mathbb E[s]\le g(N-m)=x$.
Robustness: fix any $E$, $|E|=m$. Let $s$ be the number of sprinkle fires before $\tau$, $h$ the number of advised events in blocks $\le\tau$, and $M$ the number of non-advised events in blocks $<\tau$ that were not sprinkled. Let $X=|\sigma\cap E|$. We show $X\ge\beta m-(s+M+2)$ pathwise (in expectation over $S$).
*(a) Budget exhausted in trust mode, or no contradiction ever.* In trust mode every fired advised block was a hit (else a contradiction), so $X\ge(\#\text{advised fires})=(\text{fires})-s$. If the budget was exhausted, fires $=m$ and $X\ge m-s$. If no contradiction occurred and budget remained, all $m$ advised blocks were fired and hit, so $A\subseteq E$, $E=A$, $X=m$. Either way $X\ge m-s\ge\beta m-s$.
*(b) Switch at $\tau$ with $b_\tau>0$.* Budget never ran out before $\tau$, so every advised block $\le\tau$ was fired; all were hits except possibly block $\tau$ itself. Hence the $h$ advised events $\le\tau$ are recalled, the advised fires number at most $h+1$, and $b_\tau\ge m-h-1-s$. The events after $\tau$ number $R_\tau\ge m-h-M-1$ (the only other events $\le\tau$ are the $M$ missed non-advised ones and possibly the triggering non-advised hit). Since $S$ is a uniform subset independent of $E$, $\mathbb E|S\cap E|=\min\{1,b_\tau/(N-\tau)\}R_\tau$.
 If $b_\tau\ge N-\tau$: $X\ge h+R_\tau\ge m-M-1$.
 Otherwise, with $u:=m-h$ and $N-\tau\le N-h$ (the $h$ advised events lie in $[1,\tau]$): if $u\ge s+1$ and $u\ge M+1$,
 $X\ge h+\frac{(u-s-1)(u-M-1)}{N-h}\ge h+\frac{u^2}{N-h}-\frac{u}{N-h}(s+M+2)\ge \beta m-\beta(s+M+2),$
 using (i) $\phi(h):=h+\frac{(m-h)^2}{N-h}$ is non-decreasing on $[0,m]$ because $\phi'(h)=\frac{(N-m)^2}{(N-h)^2}\ge0$, so $\phi(h)\ge\phi(0)=m^2/N=\beta m$; and (ii) $\frac{m-h}{N-h}\le\frac mN=\beta$. If $u<s+1$ or $u<M+1$, then $X\ge h=m-u>\beta m-(s+M+2)$.
So $X\ge\beta m-(s+M+2)$ in all cases. Finally $\mathbb E[s]\le x$, and $M$ is dominated by the number of non-advised events preceding the first $g$-coin success among them (coins are independent of $E$), so $\mathbb E[M]\le(1-g)/g<(N-m)/x=m(1-\beta)/(\beta x)$. Dividing by $m$: $\mathbb E[R]\ge\beta-\frac{x+2}{m}-\frac{1-\beta}{\beta x}$. The parameter choices follow by substitution. $\square$

*Relation to Proposition 5.* Proposition 5's test-then-fallback is SV with no sprinkle ($x=0$) evaluated on the two-instance family ($\eta\in\{0,1\}$) with slack $B>m$. Theorem 6 makes the "verify before committing" rule robust against *all* instances and all cue placements, and Theorem 5 shows the sprinkle is necessary: without spare fires ($x=0$) the guarantee collapses to $r^\star(1)$.

---

## 4. Corollary 7 (Exact price of adaptivity)

Define the *gain of feedback* at consistency level $c$:
$\mathrm{Gain}_m(c):=\sup\{r(G):G\text{ adaptive},\,c(G)\ge c\}-\sup\{r(G):G\text{ non-adaptive},\,c(G)\ge c\}$.

> **Corollary 7.** (i) $\mathrm{Gain}_m(c)\le\beta-r^\star(\max\{c,\beta\})\le\Delta(\beta)$ for every $m$ and $c$. (ii) $\mathrm{Gain}_m(1)=0$ (Theorem 5). (iii) $\lim_{c\uparrow1}\liminf_{m\to\infty}\mathrm{Gain}_m(c)=\Delta(\beta)$. So adaptivity buys **at most** $\Delta(\beta)=\beta-r^\star(1)$ — i.e. $\beta$ when the budget is scarce and only $(1-\beta)^2/\beta$ when it is ample ($0.05$ at the deployed $\beta=0.8$) —, buys it only in the limit of near-full consistency, and buys **nothing** at exact full consistency.

*Proof.* (i) Theorem 4 bounds the adaptive supremum by $\beta$; the non-adaptive supremum at consistency $\ge c$ is $r^\star(\max\{c,\beta\})$ (Theorem 2: $r^\star$ increases to $\beta$ on $[0,\beta]$ and decreases on $[\beta,1]$; the periodic point $(\beta,\beta)$ is attained), and $r^\star(\max\{c,\beta\})\ge r^\star(1)$. (ii) is Theorem 5 with $\varepsilon=0$ together with SAT-GATE$(1)$ attaining $r^\star(1)$. (iii) Lower bound from Theorem 6 (for $\delta>0$ pick $x=2(1-\beta)/(\beta\delta)$; for $m$ large SV($x$) has $c\ge1-\delta$, $r\ge\beta-\delta$) and continuity of $r^\star$ at $1$; upper bound from (i). $\square$

**Numbers (exact, from $r^\star$):** $\Delta(0.2)=0.20$, $\Delta(0.4)=0.40$, $\Delta(0.5)=0.50$ (the maximum), $\Delta(0.8)=0.05$, $\Delta(1)=0$.

---

## 5. Conjecture 8 (finite-$m$ adaptive frontier) — open, numerically supported

Theorems 5 and 6 sandwich the best adaptive robustness with $x$ spare fires between $\beta-\frac{1-\beta}{\beta x}-\frac{x+2}m$ and $\min\{\beta,\theta_0+(1-\theta_0)^2\frac{\beta}{1-\beta}x\}$; the transition therefore happens at $x=\Theta(1)$ spare fires, independent of the horizon. **Conjecture 8:** the optimal adaptive robustness with consistency $1-x/m$ converges, as $m\to\infty$, to a function $\rho_\beta(x)$ with $\rho_\beta(0)=r^\star(1)$, $\rho_\beta(x)\uparrow\beta$, and $\beta-\rho_\beta(x)=\Theta(1/x)$. Status: **conjecture** — the $\Theta(1/x)$ upper side is proved (Theorem 6), the matching lower side is not. `code/v3_a8a_mc.py` reports SV($x$)'s empirical worst case over a structured adversary family (5 seeds), which lies between the two bounds; this is exploratory evidence, not a proof.

---
### P3C addendum (22–24-10-2026) — Corollary 7(iii), both directions written out (as now in the paper)
Let $c\in[\beta,1)$. *Upper side:* by (i), $\mathrm{Gain}_m(c)\le\beta-r^\star(c)$ for every $m$, so $\liminf_m\mathrm{Gain}_m(c)\le\beta-r^\star(c)$. *Lower side:* put $\delta=1-c$, $x=2(1-\beta)/(\beta\delta)$. By Theorem 6, for every $m\ge2(x+2)/\delta$ with $N-m\ge x$, SV($x$) has consistency $\ge1-\delta=c$ and robustness $\ge\beta-\delta$, while by (i) the best non-adaptive robustness at consistency $\ge c$ is $r^\star(c)$; hence $\liminf_m\mathrm{Gain}_m(c)\ge\beta-(1-c)-r^\star(c)$. Both bounds tend to $\Delta(\beta)$ as $c\uparrow1$ because $r^\star$ is affine on $[\beta,1]$, hence continuous at 1. $\square$
