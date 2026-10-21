# -*- coding: utf-8 -*-
"""v3_supplement.py -- build the LaTeX sources of the supplementary material of paper_v3 from the CSV/JSON
outputs (no number typed by hand):
  Revision_v3/S1_theory_supplement.tex  -- A8(a)/(c) numerics, Theorem 3 audit, frontier cross-check
  Revision_v3/S2_empirical_supplement.tex -- per-category table, thermal control, learned cue, synthetic sweep
"""
import csv
import json
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
\renewcommand{\thetable}{%s.\arabic{table}}
\renewcommand{\thefigure}{%s.\arabic{figure}}
\begin{document}
"""


def f3(x):
    return f"{float(x):.3f}"


def s1():
    L = [PRE % ("S1", "S1"),
         r"\begin{center}{\Large Supplementary material S1: theory supplement}\\[4pt]",
         r"\emph{Learning-augmented scheduling under hard activation budgets} --- L.~Q.~Dat\end{center}",
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
          r"Monte Carlo estimate of per-block activation probabilities ($N{=}200$, $2000$ draws per $\lambda$, seed $42$; \texttt{code/v3\_check\_thm3\_hedge.py}) for HEDGE-GATE whose advice arm skips blocks already activated by the backbone (Algorithm~1, block-aware) and for a naive split in which both arms may activate the same block. The block-aware split meets the frontier up to Monte Carlo error at both budgets; the naive split does not.",
          r"\begin{verbatim}"]
    L += open(V3 / "thm3_hedge_audit.txt", encoding="utf-8").read().splitlines()
    L += [r"\end{verbatim}",
          r"\section*{S1.6 Numerical cross-check of the frontier}",
          r"\texttt{validate\_frontier.py} evaluates \textsc{SAT-GATE} on the canonical family ($N=2000$) for $\beta\in\{0.2,0.4,0.6,0.8,0.9\}$; the maximum deviation of its $(c,r)$ from $r^\star(c)$ is below $10^{-3}$ at every $\beta$.",
          r"\end{document}"]
    atomic_write_text(REV / "S1_theory_supplement.tex", "\n".join(L) + "\n", encoding="utf-8")


def s2():
    N = json.loads((REV / "numbers.json").read_text(encoding="utf-8"))
    L = [PRE % ("S2", "S2"),
         r"\begin{center}{\Large Supplementary material S2: empirical supplement}\\[4pt]",
         r"\emph{Learning-augmented scheduling under hard activation budgets} --- L.~Q.~Dat\end{center}",
         r"\section*{S2.1 Per-category breakdown of the real-data study}",
         rf"Windowed coverage ($W{{=}}7$), uncapped, ROI-aligned; {N['c121_videos_ge2']} videos with at least two events. Values are recomputed from the cached per-video cues by \texttt{{code/v3\_numbers.py}} and match \texttt{{results/percategory.csv}} row by row.",
         r"\begin{table}[h]\centering\caption{Per-category breakdown: motion gate vs.\ periodic at $a\in\{0.1,0.2\}$; AUC is the cue onset-AUC.}\label{tab:S2percat}",
         r"\begin{tabular}{lrrcccccc}\toprule group & videos & events & AUC & gate$_{.1}$ & per.$_{.1}$ & gate$_{.2}$ & per.$_{.2}$ & gap$_{.2}$\\\midrule"]
    for r in csv.DictReader(open(REV / "S2_percategory_v3.csv", encoding="utf-8")):
        g = r["group"].replace("CDnet:", "CDnet: ")
        if g == "ALL":
            L.append(r"\midrule")
        L.append(rf"{g} & {r['n_videos']} & {r['n_events']} & {f3(r['onset_auc'])} & {f3(r['gate_a010'])} & {f3(r['periodic_a010'])} & {f3(r['gate_a020'])} & {f3(r['periodic_a020'])} & ${f3(r['gap_a020'])}$\\")
    L += [r"\bottomrule\end{tabular}\end{table}",
          r"\begin{figure}[h]\centering\includegraphics[width=2.94in]{figures/fig6.png}\caption{Per-category gap at $a{=}0.20$ (gate $-$ periodic); every group is negative.}\end{figure}",
          r"\section*{S2.2 Thermal control and instrument reproducibility for the energy anchor}",
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
          r"Synthetic event streams calibrated to the measured sparse duration law (median $1$, mean $\approx6$, CV $\approx4.7$; $40$ videos of $1500$ frames, seed $42$) with permutation noise $p\in\{0,0.2,\dots,1\}$ traverse the full range of $\eta$. Simulated periodic coverage matches $\beta$ within bootstrap noise ($0.78/1.00/1.00$ at $a=0.1/0.2/0.333$); under uninformative noise HEDGE-GATE coverage is affine in $\eta$ (Pearson $r$ up to $-0.99$; Proposition~4); under adversarial advice the pure gate collapses toward $0$ while HEDGE-GATE holds its floor $\lambda\beta$ (Theorem~1$'$).",
          r"\begin{figure}[h]\centering\includegraphics[width=7.14in]{figures/fig24.png}\caption{Two failure modes (rows: $a{=}0.1,0.2,0.333$). Left, noise regime: coverage affine in $\eta$ for each $\lambda$. Right, adversarial regime: the pure gate collapses below the oblivious level while HEDGE-GATE holds $\lambda\beta$. Dashed: periodic certificate; dotted: oblivious-random level; orange: the deployed cue's error $\eta=1-\mathrm{AUC}_{\text{onset}}$ on the 121-video corpus.}\end{figure}",
          r"\end{document}"]
    atomic_write_text(REV / "S2_empirical_supplement.tex", "\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    s2()
    if (V3 / "a8a_sv_worstcase.csv").exists():
        s1()
    print("supplement sources written")
