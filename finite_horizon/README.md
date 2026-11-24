# finite_horizon/ — checks for the statements revised in v3.1.0

`p5_checks.py` checks, numerically and mostly in exact rational arithmetic, every statement of the paper that
was changed or added in v3.1.0:

| part | statement (label in the paper) | method |
|---|---|---|
| R | the errors fixed in v3.1.0 (reproduced first) | exact rationals / enumeration |
| B1 | Proposition `prop:advcommit`(ii), Algorithm 1: thinned stride-K backbone, budget <= B surely, E[R] >= lambda*beta | exhaustive grid, T <= 30, exact |
| B2 | Theorem `thm:pareto`: randomised rounding of lambda*B on the canonical family | exhaustive enumeration, N <= 8 |
| B3 | Proposition `prop:commit`: R <= 1 - eta + min{eta, delta}, pathwise | exhaustive, N <= 8 |
| B4 | Proposition `prop:advcommit`(i) with the single definition of eta | random instances |
| B5 | Proposition `prop:early` (advice first, test-then-fallback) | exhaustive enumeration, N <= 10 |
| B6 | Theorems `thm:ceiling`, `thm:notfree`, Corollary `cor:price`(ii), Propositions `prop:early` (optimality) and `prop:zerogap`, Lemma `lem:window` | own sequence-form LP over the full history tree (HiGHS), N <= 8 |
| B7 | Theorem `thm:frontier` for every position of the advised set | LP over marginals, N <= 7 |
| B8 | Theorem `thm:sv`, Theorem `thm:rate`(a) (general and consequence form), Lemmas S1.1-S1.2 of supplement S1 | exact evaluation over all event sets, N <= 9 |
| B9 | Theorem `thm:rate`(b): sum bounds of the proof, constants table, scope of the asymptotic statements | exact rationals / root finding |
| B10 | Theorem `thm:ceiling`: a uniform m-set attains beta; periodic only if m divides N(W+1) (counterexample N = 187, m = 75) | exact rationals |

Run: `python finite_horizon/p5_checks.py` (one process, about 4 minutes, < 1 GB RAM; numpy + scipy) →
`results/v3/p5_checks.json`, field `all_pass`. The JSON is reproducible byte for byte (the runtime is printed, not stored).
`theory/v3_reproduce.py --full` runs it and requires `all_pass`.

`results/v3/ext69_p1b_m5_table.json` is the finite-size table of worst-case gaps of uniform and harmonic
sprinkling (exact dynamic-programming brackets) used for Table `tab:svh` of the paper; `theory/v3_numbers.py`
records its sha256 in `Revision_v3/numbers.json`.
