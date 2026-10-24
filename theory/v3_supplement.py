# -*- coding: utf-8 -*-
"""v3_supplement.py -- build the LaTeX sources of the supplementary material of paper_v3 from the CSV/JSON
outputs (no number typed by hand):
  Revision_v3/S1_theory_supplement.tex  -- A8(a)/(c) numerics, Theorem 3 audit, frontier cross-check
  Revision_v3/S2_empirical_supplement.tex -- per-category table, thermal control, learned cue, synthetic sweep
"""
import csv
import json
import re
from pathlib import Path
from v3_io import atomic_write_text  # noqa: E402  (atomic writes, C7)

TOPIC = Path(__file__).resolve().parents[1]
RES = TOPIC / "results"
V3 = RES / "v3"
REV = TOPIC / "Revision_v3"

PRE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.65in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,longtable}
\usepackage[hidelinks]{hyperref}
\emergencystretch=3em
\renewcommand{\thetable}{%s.\arabic{table}}
\renewcommand{\thefigure}{%s.\arabic{figure}}
\begin{document}
"""


def f3(x):
    return f"{float(x):.3f}"


def s1():
    # P4: S1 carries the complete proofs (EJOR 30-page limit); xr-hyper resolves equation/definition labels of the paper
    pre1 = (PRE % ("S1", "S1")).replace(
        r"\usepackage[hidelinks]{hyperref}",
        "\\usepackage{amsthm}\n\\usepackage{xr-hyper}\n\\usepackage[hidelinks]{hyperref}\n\\externaldocument{paper_v3}\n"
        "\\newcommand{\\bbeta}{\\beta}")
    L = [pre1,
         r"\begin{center}{\Large Supplementary material S1: complete proofs and theory supplement}\\[4pt]",
         r"\emph{Learning-augmented scheduling under hard activation budgets} --- L.~Q.~Dat\end{center}",
         r"\input{S1_proofs}",
         r"\section*{S1-B\quad Numerical checks of the theory}",
         r"\input{S1_tab_beta}",
         r"\section*{S1.1 Robustness against the full adversary}",
         r"For a non-adaptive gate with per-block firing probabilities $p_b$, $\mathbb E[R(G;E)]=\frac1m\sum_{b\in E}p_b$ for every instance $E$, so the worst case over all $m$-subsets is the average of the $m$ smallest $p_b$. This is the quantity bounded in the converse of Theorem~2, so $r\le r^\star(c)$ holds for the full adversary, and \textsc{SAT-GATE} attains it. Theorems~4--6 therefore compare adaptive and non-adaptive gates under one and the same robustness notion.",
         r"\section*{S1.2 Theorem 4 (adaptive ceiling): numerical check}",
         r"Instances drawn uniformly ($E$ uniform among $m$-subsets), $W{=}7$, $N{=}187$, $m{=}B$, seeds $42$--$46$, $400$ runs per seed. Theorem~4 predicts mean coverage $\le\beta$ for every gate; gates that spend their whole budget have mean exactly $\beta$.",
         r"\begin{table}[h]\centering\caption{Mean coverage on uniform instances ($n=2000$ runs; $95\%$ interval).}",
         r"\begin{tabular}{llccc}\toprule $\beta$ & gate & mean & CI$_{95}$ & ceiling $\beta$\\\midrule"]
    for r in csv.DictReader(open(V3 / "a8a_ceiling.csv", encoding="utf-8")):
        L.append(rf"{f3(r['beta'])} & {r['gate']} & {f3(r['mean_recall'])} & [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}] & {f3(r['ceiling_beta'])}\\")
    L += [r"\bottomrule\end{tabular}\end{table}",
          r"\section*{S1.3 Theorems 5--6: sprinkle-and-verify against the adversary family}",
          r"The adversary family crosses: advised-set placement (last, first, interleaved, random); overlap $\theta=|E\cap A|/m$ on a grid of step $0.05$ together with $\theta_0=r^\star(1)$ and $\beta$; the order of the advised events (first, last, random); and the placement of the non-advised events (front, back, spread, random). Each instance is run $400$ times for each of $5$ seeds. The worst instance's mean is an \emph{upper} estimate of SV's true robustness (the family is not exhaustive); Theorem~6's bound is a proved lower bound; Theorem~5's bound is a proved upper bound for \emph{any} gate with the measured consistency when the advised set comes last.",
          r"\begin{longtable}{rrrccccl}\caption{SV($x$): full results.}\\\toprule $\beta$ & $x$ & $c$ & worst [CI$_{95}$] & Thm~6 & Thm~5 & $r^\star(c)$ & worst instance\\\midrule\endhead"]
    for r in csv.DictReader(open(V3 / "a8a_sv_worstcase.csv", encoding="utf-8")):
        lb = r["thm6_lower"]
        lb = "---" if lb == "" or float(lb) <= 0 else f3(lb)
        wi = r["worst_instance"].replace("_", r"\_").replace("|", "/")
        L.append(rf"{f3(r['beta'])} & {float(r['x']):g} & {f3(r['c_emp'])} & {f3(r['r_worst_emp'])} [{f3(r['ci_lo'])},{f3(r['ci_hi'])}] & {lb} & {f3(r['thm5_upper(A last)'])} & {f3(r['r_star_c_emp'])} & \scriptsize {wi}\\")
    L += [r"\bottomrule\end{longtable}"]
    # Theorem 7 extra tables
    rows = list(csv.DictReader(open(V3 / "a8c_alloc.csv", encoding="utf-8")))
    L += [r"\section*{S1.4 Theorem 7: allocation results for all weight models}",
          r"Uniform provisioning (Proposition~6) versus the exact integer-stride optimum (Theorem~7(C)), the continuous bang-bang allocation (Theorem~7(A2)) and the envelope bound $U$ (Theorem~7(A1)); Dirichlet(1) weights averaged over seeds $42$--$46$ (min--max of the optimum in brackets).",
          r"\begin{longtable}{rlccccccc}\caption{Model T1 allocation, all configurations.}\\\toprule $n$ & weights & $c$ & $\bar a$ & uniform & optimum & bang-bang & $U$ & opt. range\\\midrule\endhead"]
    for r in rows:
        rng = f"[{f3(r['min_dp_over_seeds'])},{f3(r['max_dp_over_seeds'])}]" if r["weights"] == "dirichlet" else ""
        L.append(rf"{r['n']} & {r['weights']} & {r['c']} & {r['a_bar']} & {f3(r['uniform'])} & {f3(r['dp_opt'])} & {f3(r['greedy_cont'])} & {f3(r['U_cont'])} & {rng}\\")
    L += [r"\bottomrule\end{longtable}",
          r"\noindent The bang-bang column uses the continuous model $\beta=\min(1,8a)$ and can exceed the integer-stride optimum when $\beta=c$ is not an admissible stride; the optimum never exceeds $U$.",
          r"\begin{longtable}{rrlccc}\caption{Model T2 (fixed loads, $c=0.9$): uniform provisioning vs.\ water-filling (Theorem~7(B)). Identical loads $\mu=0.2$; heterogeneous loads $\mu_i\sim U(0.05,0.45)$, weights $\propto\mu_i$.}\\\toprule $n$ & seed & loads & $\bar a$ & uniform & water-filling\\\midrule\endhead"]
    for r in csv.DictReader(open(V3 / "a8c_T2.csv", encoding="utf-8")):
        L.append(rf"{r['n']} & {r['seed']} & {r['loads']} & {r['a_bar']} & {r['uniform']} & {r['waterfill_opt']}\\")
    L += [r"\bottomrule\end{longtable}"]
    # Theorem 3 audit
    L += [r"\section*{S1.5 Theorem 3: block-aware versus naive budget split}",
          r"Monte Carlo estimate of per-block activation probabilities ($N{=}200$, $2000$ draws per $\lambda$, seed $42$; \texttt{theory/v3\_check\_thm3\_hedge.py}) for HEDGE-GATE whose advice arm skips blocks already activated by the backbone and draws uniformly among the rest (Algorithm~1, block-aware; backbone a uniformly random set of $\lambda B$ blocks) and for a naive split in which both arms may activate the same block. The block-aware split meets the frontier up to Monte Carlo error at both budgets; the naive split does not.",
          r"The final block (EXACT) enumerates every backbone draw in rational arithmetic on small instances and isolates the two symmetry conditions used in the proof of Theorem~3 (Remark \emph{What the two uniform draws buy}): with a uniformly random backbone set and a uniform choice among the free advised blocks the gap to $r^\star(c)$ is exactly $0$; a fixed tie-breaking order, or a period-$5$ backbone whose residue classes the advised set meets unequally, makes the advised marginals unequal, and the robustness falls below $r^\star(c)$ at $\beta=0.8$; at $\beta=0.4$ neither matters.",
          r"{\scriptsize\begin{verbatim}"]
    L += open(V3 / "thm3_hedge_audit.txt", encoding="utf-8").read().splitlines()
    L += [r"\end{verbatim}}",
          r"\section*{S1.6 Numerical cross-check of the frontier}",
          r"\texttt{theory/validate\_frontier.py} evaluates \textsc{SAT-GATE} on the canonical family ($N=2000$) for $\beta\in\{0.2,\allowbreak 0.4,\allowbreak 0.6,\allowbreak 0.8,\allowbreak 0.9\}$; the maximum deviation of its $(c,r)$ from $r^\star(c)$ is below $10^{-3}$ at every $\beta$.",
          r"\end{document}"]
    atomic_write_text(REV / "S1_theory_supplement.tex", "\n".join(L) + "\n", encoding="utf-8")


def s2():
    N = json.loads((REV / "numbers.json").read_text(encoding="utf-8"))
    L = [PRE % ("S2", "S2"),
         r"\begin{center}{\Large Supplementary material S2: empirical supplement}\\[4pt]",
         r"\emph{Learning-augmented scheduling under hard activation budgets} --- L.~Q.~Dat\end{center}",
         r"\section*{S2.1 Per-category breakdown of the real-data study}",
         rf"Windowed coverage ($W{{=}}7$), uncapped, ROI-aligned; {N['c121_videos_ge2']} videos with at least two events. Values are recomputed from the cached per-video cues by \texttt{{theory/v3\_numbers.py}} and match \texttt{{results/percategory.csv}} row by row.",
         r"\begin{table}[h]\centering\caption{Per-category breakdown: motion gate vs.\ periodic at $a\in\{0.1,0.2\}$; AUC is the cue onset-AUC.}\label{tab:S2percat}",
         r"\begin{tabular}{lrrcccccc}\toprule group & videos & events & AUC & gate$_{.1}$ & per.$_{.1}$ & gate$_{.2}$ & per.$_{.2}$ & gap$_{.2}$\\\midrule"]
    for r in csv.DictReader(open(REV / "S2_percategory_v3.csv", encoding="utf-8")):
        g = r["group"].replace("CDnet:", "CDnet: ")
        if g == "ALL":
            L.append(r"\midrule")
        L.append(rf"{g} & {r['n_videos']} & {r['n_events']} & {f3(r['onset_auc'])} & {f3(r['gate_a010'])} & {f3(r['periodic_a010'])} & {f3(r['gate_a020'])} & {f3(r['periodic_a020'])} & ${f3(r['gap_a020'])}$\\")
    L += [r"\bottomrule\end{tabular}\end{table}",
          r"\begin{figure}[h]\centering\includegraphics[width=2.94in]{figures/fig6.png}\caption{Per-category gap at $a{=}0.20$ (gate $-$ periodic); every group is negative.}\end{figure}",
          r"\section*{S2.2 Energy measurements: table, thermal control and instrument reproducibility}",
          r"\input{S2_tab_energy}",
          r"Because the activation sweep ran in ascending order, temperature co-varies with $a$, so we measured the temperature term directly instead of regressing it out. Immediately after a saturating $a{=}0.5$ block we ran an $a{=}0$ block with no cooldown, giving two detector-off measurements at the \emph{same} activation but different board temperature. Between conditions the power difference is $+0.011$\,W over $+0.67^\circ$C; regressing power on temperature \emph{within} the hot detector-off block, where activation is fixed and the board cools from $63.6$ to $59.6^\circ$C, gives $+0.0133\pm0.0014$\,W$/^\circ$C. Over the $2.53^\circ$C span of the sweep this accounts for $0.040$\,W of a $3.44$\,W total rise, i.e.\ $1.2\%$ of $\kappa$. (A regression that instead adds temperature as a covariate is not usable here: at $r=0.996$ the design is near-collinear, and it returns a physically impossible negative intercept, $-6.1\pm7.1$\,W, with the standard error on $\kappa$ inflated tenfold.) An earlier campaign on the same board and rail fits $W(a)=4.938+6.916\,a$; our slope reproduces it to $0.10\%$---evidence that the instrument is reproducible, not independent evidence about temperature, since that campaign shares the ascending design.",
          r"\section*{S2.3 A stronger learned cue does not rescue the gate}"]
    lp = json.loads((RES / "learned_cue_probe.json").read_text(encoding="utf-8"))
    gv = {round(r["a"], 3): r for r in lp["gate_vs_periodic"]}
    L += [rf"A gradient-boosted classifier on seven cheap, ground-truth-free per-frame features (motion density and its dispersion, connected-component counts, two background-model disagreement signals, illumination variance) predicts onset windows and is evaluated cross-scene by leave-one-CDnet-category-out ({lp['n_videos']} clips, {lp['n_cats']} categories). Its cross-scene onset-AUC is {lp['onset_auc']['cross']['learned']:.2f} versus {lp['onset_auc']['cross']['baseline']:.2f} for the raw motion feature, yet it trails the periodic schedule at every budget (Table~\ref{{tab:S2learned}}).",
          r"\begin{table}[h]\centering\caption{Learned-cue gate vs.\ periodic (windowed coverage, $W{=}7$; $95\%$ bootstrap intervals).}\label{tab:S2learned}",
          r"\begin{tabular}{lccc}\toprule & $a{=}0.1$ & $a{=}0.2$ & $a{=}0.333$\\\midrule",
          "periodic & " + " & ".join(f3(gv[a]["periodic"]) for a in (0.1, 0.2, 0.333)) + r"\\",
          "learned-cue gate & " + " & ".join(f"{f3(gv[a]['learned_gate'])} [{f3(gv[a]['gate_ci'][0])},{f3(gv[a]['gate_ci'][1])}]" for a in (0.1, 0.2, 0.333)) + r"\\",
          "gap & " + " & ".join(f"${f3(gv[a]['gap'])}$" for a in (0.1, 0.2, 0.333)) + r"\\",
          r"\bottomrule\end{tabular}\end{table}",
          r"\section*{S2.4 Calibrated synthetic corruption sweep}",
          r"Synthetic event streams calibrated to the measured sparse duration law (median $1$, mean $\approx6$, CV $\approx4.7$; $40$ videos of $1500$ frames, seed $42$) with permutation noise $p\in\{0,0.2,\dots,1\}$ traverse the full range of $\eta$. Simulated periodic coverage matches $\beta$ within bootstrap noise ($0.78/1.00/1.00$ at $a=0.1/0.2/0.333$); under uninformative noise HEDGE-GATE coverage is affine in $\eta$ (Pearson $r$ up to $-0.99$; Proposition~4); under adversarial advice the pure gate collapses toward $0$ while HEDGE-GATE holds its floor $\lambda\beta$ (Theorem~1$'$). (The Pearson coefficient is that of the earlier version's frame-level split.) " + syn_text(),
          r"\begin{figure}[h]\centering\includegraphics[width=7.14in]{figures/fig24.png}\caption{Two failure modes (rows: $a{=}0.1,0.2,0.333$). Left, noise regime: coverage vs.\ $\eta$ for each $\lambda$. Right, adversarial regime: the pure gate collapses below the oblivious level while HEDGE-GATE holds $\lambda\beta$. Solid with markers: block-aware split of Algorithm~1; thin dashed (same colour, $0<\lambda<1$): frame-level split of the earlier version. Grey dashed: periodic certificate; dotted: oblivious-random level; orange: the deployed cue's error $\eta=1-\mathrm{AUC}_{\text{onset}}$ on the 121-video corpus.}\end{figure}",
          ] + s2_sweep() + [r"\end{document}"]
    atomic_write_text(REV / "S2_empirical_supplement.tex", "\n".join(L) + "\n", encoding="utf-8")


def syn_text():
    """S2.4: the synthetic sweep re-run with the block-aware split (v3_sweep_blockaware.py --synthetic)."""
    S = json.loads((V3 / "sweep_blockaware_synthetic_summary.json").read_text(encoding="utf-8"))
    n, a = S["intermediate_lambda"]["noise"], S["intermediate_lambda"]["adversarial"]
    return (r"The sweep is re-run with the block-aware split of Algorithm~1 (\texttt{theory/v3\_sweep\_blockaware.py} with option \texttt{--synthetic}; same generator, order and seeds); "
            r"replaying the frame-level split reproduces \texttt{results/corruption\_sweep.csv} cell for cell, and at $\lambda\in\{0,1\}$ the two splits coincide (%d invariant failures). "
            r"At $0<\lambda<1$, under uninformative noise the block-aware split is higher in %d, equal in %d and lower in %d of %d cells (by at most %.3f); "
            r"under adversarial advice it is higher in %d (by up to %.3f), equal in %d and lower in %d (by at most %.3f). "
            r"The losses occur where the cue is informative (small corruption): the uniform draw among the advised epochs gives up the cue's ranking inside the advised set, a ranking the canonical family of Theorem~3 does not have. "
            r"The gains occur under adversarial corruption $p\ge0.4$; at $p\ge0.6$ the block-aware split is never lower (15 cells higher, 12 equal), the largest gain at $a=0.333$, $\lambda=0.25$; the floor $\lambda\beta$ of Theorem~1$'$ comes from the backbone alone and holds under both splits."
            % (S["invariant_failures"], n["n_ba_better"], n["n_equal"], n["n_ba_worse"], n["n_cells"], n["max_loss"],
               a["n_ba_better"], a["max_gain"], a["n_equal"], a["n_ba_worse"], a["max_loss"]))


def s2_sweep():
    """S2.5: the real-data trust-dial sweep (fig5) under both splits."""
    ba = list(csv.DictReader(open(V3 / "corruption_sweep_REAL_blockaware.csv", encoding="utf-8")))
    nv = list(csv.DictReader(open(RES / "corruption_sweep_REAL.csv", encoding="utf-8")))
    S = json.loads((V3 / "sweep_blockaware_summary.json").read_text(encoding="utf-8"))
    I = S["intermediate_lambda"]
    diffs = [v["diff_ba_minus_naive"] for v in S["rows"].values()]

    def cell(rows, a, lam, p):
        return next(r for r in rows if abs(float(r["a"]) - a) < 1e-9 and abs(float(r["lam"]) - lam) < 1e-9
                    and abs(float(r["corruption_p"]) - p) < 1e-9)

    L = [r"\section*{S2.5 Real-data trust-dial sweep: block-aware versus frame-level split}",
         r"Fig.~7 of the paper sweeps HEDGE-GATE$(\lambda)$, $\lambda\in\{0,0.25,0.5,0.75,1\}$, on the 77-video corpus with the cue corrupted by permuting a fraction $p$ of its values. \emph{Block-aware} is Algorithm~1 at frame level (\texttt{theory/v3\_sweep\_blockaware.py}): the advised epochs are the $B$ epochs of highest cue score, an epoch is free if its window $[t-W,t]$ holds no backbone activation, and the advice arm draws uniformly among the free advised epochs (then, if short, the free epochs of highest score). \emph{Frame-level} is the split of the earlier (TPDS) version, used there for this figure: its advice arm takes the highest-score frames not fired by the backbone, whether or not their window is already covered. Both use the same backbone, the periodic schedule at phase $0$ (round$(\lambda B)$ activations evenly spaced), as Table~5 of the paper does at $\lambda=1$; the cue and the events do not depend on the phase. Seeds are those of the earlier version: corruption $42+7+\lfloor100p\rfloor$, $\eta$ estimate $42+\lfloor100p\rfloor$, video-level bootstrap $42+\lfloor1000a\rfloor+\lfloor10\lambda\rfloor$ ($B=1000$); the uniform draw uses its own stream seeded by $(42+\lfloor1000a\rfloor+\lfloor10\lambda\rfloor,\ \mathrm{round}(100p))$.",
         r"\emph{Invariants} (%d failures). Replaying the frame-level split reproduces all %s$\times$%s cells of \texttt{results/corruption\_sweep\_REAL.csv}; at $\lambda\in\{0,1\}$ the block-aware split activates the same frames as the frame-level split in all %s per-video runs, so its %s endpoint rows are identical."
         % ((S["invariant_failures"],) + re.search(r"(\d+) rows x (\d+) cells", S["invariants"][0]).groups()
            + (re.search(r"(\d+) per-video", S["invariants"][1]).group(1), re.search(r"(\d+) rows", S["invariants"][1]).group(1))),
         r"\emph{Result.} At $0<\lambda<1$ the block-aware split is higher in %d of %d cells (by up to %.3f), equal in %d (both at coverage $1$) and lower in %d (by at most %.3f, always with overlapping $95\%%$ intervals); every $\lambda<1$ stays at or below the periodic schedule under both splits. Table~\ref{tab:S2sweep} gives the deployed cue ($p=0$) and the fully permuted cue ($p=1$)."
         % (I["n_ba_better"], I["n_cells"], max(diffs), I["n_equal"], I["n_ba_worse"], -min(diffs)),
         r"\begin{table}[h]\centering\caption{Real-data sweep: coverage block-aware (Algorithm~1) / frame-level (earlier version), $95\%$ video-level bootstrap intervals in brackets.}\label{tab:S2sweep}",
         r"\small\begin{tabular}{llcc}\toprule $a$ & $\lambda$ & $p=0$ ($\eta=%.3f$) & $p=1$ ($\eta=%.3f$)\\\midrule"
         % (float(cell(ba, 0.1, 0.0, 0.0)["eta"]), float(cell(ba, 0.1, 0.0, 1.0)["eta"]))]
    for a in (0.1, 0.2, 0.333):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = []
            for p in (0.0, 1.0):
                b, n = cell(ba, a, lam, p), cell(nv, a, lam, p)
                out.append(f"{f3(b['recall_mean'])} [{f3(b['ci_lo'])},{f3(b['ci_hi'])}] / "
                           f"{f3(n['recall_mean'])} [{f3(n['ci_lo'])},{f3(n['ci_hi'])}]")
            L.append(f"{a:g} & {lam:g} & " + " & ".join(out) + r"\\")
        if a != 0.333:
            L.append(r"\midrule")
    L += [r"\bottomrule\end{tabular}\end{table}",
          r"\noindent All $90$ cells of both splits: \texttt{results/v3/corruption\_sweep\_REAL\_blockaware.csv} and \texttt{results/corruption\_sweep\_REAL.csv}; side by side: \texttt{results/v3/sweep\_blockaware\_summary.json}."]
    return L


if __name__ == "__main__":
    s2()
    if (V3 / "a8a_sv_worstcase.csv").exists():
        s1()
    print("supplement sources written")
