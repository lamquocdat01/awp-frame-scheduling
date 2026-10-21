# -*- coding: utf-8 -*-
"""v3_tables.py -- LaTeX tables for A8(a) and A8(c) from results/v3/*.csv (single source for paper_v3).
Writes Revision_v3/tab_a8a.tex and Revision_v3/tab_a8c.tex.
"""
import csv
from pathlib import Path
from v3_io import atomic_write_text  # noqa: E402  (atomic writes, C7)

TOPIC = Path(__file__).resolve().parents[1]
RES = TOPIC / "results" / "v3"
REV = TOPIC / "Revision_v3"


def f3(x):
    return f"{float(x):.3f}"


def tab_a8a():
    rows = list(csv.DictReader(open(RES / "a8a_sv_worstcase.csv", encoding="utf-8")))
    out = [r"\begin{table}[!tb]\centering",
           r"\caption{Sprinkle-and-verify SV($x$) on the canonical family ($W{=}7$, $N{=}187$, $m{=}B$): consistency, empirical worst case over the adversary family ($5$ seeds $\times$ $400$ runs per instance; $95\%$ interval over the $2000$ runs of the worst instance), the proved lower bound of Theorem~6, the best non-adaptive robustness $r^\star(c)$ at the same consistency (Theorem~2), and the adaptive ceiling $\beta$ (Theorem~4). $x{=}0$ is the full-consistency case of Theorem~5.}\label{tab:a8a}",
           r"\small", r"\begin{tabular}{rrcccc}", r"\toprule",
           r"$x$ & $c$ & SV worst case [CI$_{95}$] & Thm~6 bound & $r^\star(c)$ & ceiling $\beta$\\", r"\midrule"]
    for beta in sorted({r["beta"] for r in rows}, key=float):
        sub = [r for r in rows if r["beta"] == beta and float(r["x"]) in (0, 1, 5, 10, 20)]
        m = sub[0]["m"]
        out.append(rf"\multicolumn{{6}}{{l}}{{\emph{{$\beta={float(beta):.3f}$ ($m={m}$)}}}}\\")
        for r in sub:
            lb = r["thm6_lower"]
            lb = "---" if lb == "" or float(lb) <= 0 else f3(lb)
            out.append(rf"{float(r['x']):g} & {f3(r['c_emp'])} & {f3(r['r_worst_emp'])}\,[{f3(r['ci_lo'])},{f3(r['ci_hi'])}] & {lb} & {f3(r['r_star_c_emp'])} & {f3(r['ceiling_beta'])}\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    atomic_write_text(REV / "tab_a8a.tex", "\n".join(out) + "\n", encoding="utf-8")


def tab_a8c():
    rows = list(csv.DictReader(open(RES / "a8c_alloc.csv", encoding="utf-8")))
    get = lambda n, w, c, a: next(r for r in rows if r["n"] == str(n) and r["weights"] == w and r["c"] == c and r["a_bar"] == a)
    out = [r"\begin{table}[!tb]\centering",
           r"\caption{Certified tier robustness in Model T1 ($W{=}7$, integer strides): uniform provisioning (Proposition~6) $\to$ exact optimum (Theorem~7(C)), for $n$ equal-weight nodes and for Zipf weights $w_i\propto1/i$. Dirichlet(1) weights ($5$ seeds) are in the supplement.}\label{tab:a8c}",
           r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{cc ccc c}", r"\toprule",
           r"& & \multicolumn{3}{c}{equal weights} & Zipf\\ \cmidrule(lr){3-5}\cmidrule(lr){6-6}",
           r"$c$ & $\bar a$ ($\bar\beta$) & $n{=}8$ & $n{=}32$ & $n{=}128$ & $n{=}32$\\", r"\midrule"]
    for c in ("1.0", "0.9"):
        for a in ("0.03", "0.05", "0.0625", "0.08", "0.1"):
            cells = []
            for n, w in ((8, "equal"), (32, "equal"), (128, "equal"), (32, "zipf")):
                r = get(n, w, c, a)
                cells.append(rf"${f3(r['uniform'])}\to{f3(r['dp_opt'])}$")
            r0 = get(8, "equal", c, a)
            out.append(rf"{float(c):.1f} & {float(a):g} ({float(r0['beta_bar']):.2f}) & " + " & ".join(cells) + r"\\")
        if c == "1.0":
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    atomic_write_text(REV / "tab_a8c.tex", "\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tab_a8c()
    if (RES / "a8a_sv_worstcase.csv").exists():
        tab_a8a()
    print("tables written")
