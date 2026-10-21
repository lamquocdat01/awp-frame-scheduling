#!/usr/bin/env python3
"""Analyse the AwP activation-energy sweep.

Every number printed traces to run.log / raw/*.csv -- nothing is hand-entered.

Temperature confound: the campaign ran levels in ASCENDING a, and the 60 C
cooldown gate is a near no-op on a board that idles at 59-60 C, so T_start is
correlated with a by construction. We therefore fit BOTH
    M1: W = W0 + kappa*a                (the model the paper uses)
    M2: W = W0 + kappa*a + gamma*T_start (temperature as nuisance covariate)
and report corr(T_start, a) plus the kappa disagreement.
"""
import json, math, os, re, sys, csv
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "run.log"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
FAN = sys.argv[3] if len(sys.argv) > 3 else None
FPS = 30.0
C1_INT, C1_SLOPE, C1_R2 = 4.9380, 6.9155, 0.99901   # Deep3 C1_REPORT.md §3


def load(path):
    """run.log or results.jsonl. Activation re-derived from (mode,k), exact."""
    out, pre = [], {}
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        m = re.match(r"\[(\d+)/(\d+)\]\s+a=([\d.]+)\s+k=(\d+)\s+run=(\d+)\s+\(pre-temp\s+([\d.]+)C\)", s)
        if m:
            pre = {"pre_temp": float(m.group(6))}
            continue
        if s.startswith("SUMMARY {") or s.startswith("{"):
            j = json.loads(s[len("SUMMARY "):] if s.startswith("SUMMARY ") else s)
            j["a_nominal"] = 0.0 if j["mode"] == "idle" else 1.0 / j["k"]
            j.update(pre)
            out.append(j)
    return out


def mean(x): return sum(x) / len(x)
def pstd(x):
    if len(x) < 2: return 0.0
    m = mean(x); return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))
def corr(x, y):
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x)); sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0: return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ols(Xcols, y):
    """OLS with explicit intercept. Xcols = list of predictor columns.
    Returns (betas, ses, r2). Normal equations solved by Gauss-Jordan."""
    n = len(y); p = len(Xcols) + 1
    X = [[1.0] + [c[i] for c in Xcols] for i in range(n)]
    XtX = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
    Xty = [sum(X[i][r] * y[i] for i in range(n)) for r in range(p)]
    A = [XtX[r][:] + [0.0] * p + [Xty[r]] for r in range(p)]
    for r in range(p): A[r][p + r] = 1.0
    for c in range(p):                                    # Gauss-Jordan
        piv = max(range(c, p), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-15: raise ValueError("singular design matrix")
        A[c], A[piv] = A[piv], A[c]
        d = A[c][c]
        A[c] = [v / d for v in A[c]]
        for r in range(p):
            if r != c and A[r][c] != 0.0:
                f = A[r][c]
                A[r] = [vr - f * vc for vr, vc in zip(A[r], A[c])]
    beta = [A[r][2 * p] for r in range(p)]
    inv = [[A[r][p + c] for c in range(p)] for r in range(p)]
    yhat = [sum(b * xv for b, xv in zip(beta, X[i])) for i in range(n)]
    ss_res = sum((y[i] - yhat[i]) ** 2 for i in range(n))
    my = mean(y); ss_tot = sum((v - my) ** 2 for v in y)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    s2 = ss_res / (n - p) if n > p else float("nan")
    ses = [math.sqrt(s2 * inv[r][r]) for r in range(p)]
    return beta, ses, r2


rows = load(SRC)
print("runs loaded: %d from %s" % (len(rows), SRC))
if len(rows) < 4:
    print("too few runs to fit"); sys.exit(0)

by_a = defaultdict(list)
for r in rows: by_a[round(r["a_nominal"], 4)].append(r)

print("\n=== per-activation aggregate ===")
hdr = ("%7s %3s %4s %9s %7s %7s %8s %11s %8s %7s %7s" %
       ("a", "k", "runs", "W_mean", "W_sd", "fps", "inf_fps", "E_mJ/frame", "T_start", "T_end", "drift%"))
print(hdr)
agg, drift_flags = [], []
for a in sorted(by_a):
    rs = by_a[a]; w = [r["watts_mean"] for r in rs]
    wm, wsd = mean(w), pstd(w)
    d = 100.0 * (w[-1] - w[0]) / w[0] if w[0] else 0.0
    if abs(d) > 5.0: drift_flags.append((a, round(d, 2)))
    agg.append(dict(a=a, k=rs[0]["k"], n=len(rs), W=wm, Wsd=wsd,
                    fps=mean([r["fps"] for r in rs]), infer_fps=mean([r["infer_fps"] for r in rs]),
                    E_mJ=wm / FPS * 1000.0, t0=mean([r["temp_start"] for r in rs]),
                    t1=mean([r["temp_end"] for r in rs]), drift=d))
    print("%7.4f %3d %4d %9.4f %7.4f %7.2f %8.3f %11.2f %8.2f %7.2f %7.2f" %
          (a, rs[0]["k"], len(rs), wm, wsd, agg[-1]["fps"], agg[-1]["infer_fps"],
           agg[-1]["E_mJ"], agg[-1]["t0"], agg[-1]["t1"], d))

A = [r["a_nominal"] for r in rows]
Y = [r["watts_mean"] for r in rows]
T = [r["temp_start"] for r in rows]

rTA = corr(T, A)
print("\n=== CONFOUND DIAGNOSTIC ===")
print("  corr(T_start, a) = %+.4f   (campaign ran a in ASCENDING order)" % rTA)
print("  T_start range    = %.2f .. %.2f C" % (min(T), max(T)))

b1, se1, r2_1 = ols([A], Y)
print("\n=== M1: W = W0 + kappa*a ===")
print("  W0    = %.4f +- %.4f W" % (b1[0], se1[0]))
print("  kappa = %.4f +- %.4f W per unit a" % (b1[1], se1[1]))
print("  R^2   = %.5f" % r2_1)

k2 = g2 = r2_2 = float("nan"); b2 = se2 = None
try:
    b2, se2, r2_2 = ols([A, T], Y)
    k2, g2 = b2[1], b2[2]
    print("\n=== M2: W = W0 + kappa*a + gamma*T_start ===")
    print("  W0    = %.4f +- %.4f W" % (b2[0], se2[0]))
    print("  kappa = %.4f +- %.4f W per unit a" % (b2[1], se2[1]))
    print("  gamma = %+.4f +- %.4f W per C" % (b2[2], se2[2]))
    print("  R^2   = %.5f" % r2_2)
except Exception as e:
    print("\n=== M2 could not be fitted: %s ===" % e)

dk = 100.0 * (k2 - b1[1]) / b1[1] if b1[1] and k2 == k2 else float("nan")
print("\n  kappa disagreement M1 vs M2 = %+.2f %%" % dk)

print("\n=== STOP-RULE CHECKS ===")
r2ok = r2_1 >= 0.99
print("  R^2(M1) >= 0.99                : %s (%.5f)" % ("PASS" if r2ok else "*** FAIL -> STOP ***", r2_1))
print("  per-level drift <= 5%%          : %s %s" % ("PASS" if not drift_flags else "*** FAIL ***", drift_flags or ""))
conf = (abs(rTA) > 0.7) and (abs(dk) > 3.0)
print("  |corr|>0.7 AND |dkappa|>3%%     : %s" % ("*** TRIPPED -> STOP & REPORT ***" if conf else "not tripped"))

print("\n=== EXTERNAL ARBITER: Deep3 C1 (different thermal history) ===")
print("  C1  : W(a) = %.4f + %.4f a  (R^2=%.5f)" % (C1_INT, C1_SLOPE, C1_R2))
print("  M1  : W(a) = %.4f + %.4f a  (R^2=%.5f)   slope %+.2f%% vs C1" %
      (b1[0], b1[1], r2_1, 100 * (b1[1] - C1_SLOPE) / C1_SLOPE))
if k2 == k2:
    print("  M2  : kappa = %.4f                            slope %+.2f%% vs C1" %
          (k2, 100 * (k2 - C1_SLOPE) / C1_SLOPE))

fan_note, fan_cost = None, None
if FAN and os.path.exists(FAN):
    fr = [r for r in csv.DictReader(open(FAN)) if r.get("pwm")]
    if fr:
        pw = sorted({int(r["pwm"]) for r in fr})
        rp = [int(r["rpm"]) for r in fr if r["rpm"] not in ("-1", "")]
        tp = [float(r["temp_max_c"]) for r in fr]
        print("\n=== FAN CHECK (rule III; logger covers runs from its start onward) ===")
        print("  distinct PWM values: %s   (of 255)" % pw)
        if rp: print("  RPM range: %d .. %d" % (min(rp), max(rp)))
        print("  temp during logging: %.2f .. %.2f C" % (min(tp), max(tp)))

        # How much board power does a PWM step actually cost? Compare mean W at
        # each PWM level *within a narrow temperature band*, so the comparison is
        # not just re-reading the activation effect.
        band = [r for r in fr if 59.5 <= float(r["temp_max_c"]) <= 61.5]
        byp = defaultdict(list)
        for r in band: byp[int(r["pwm"])].append(float(r["watt"]))
        byp = {k: v for k, v in byp.items() if len(v) >= 20}
        if len(byp) >= 2:
            ks = sorted(byp)
            print("  mean board W per PWM level, restricted to 59.5-61.5 C:")
            for k in ks:
                print("    pwm=%3d  n=%5d  W=%.4f" % (k, len(byp[k]), mean(byp[k])))
            fan_cost = mean(byp[ks[-1]]) - mean(byp[ks[0]])
            print("  => PWM %d->%d costs %+.4f W  (%.2f%% of kappa=%.3f)"
                  % (ks[0], ks[-1], fan_cost, 100 * abs(fan_cost) / b1[1], b1[1]))
        fan_note = ("CONSTANT PWM (%d): fan is a constant offset, cannot create steps" % pw[0]) \
            if len(pw) == 1 else \
            ("PWM VARIES %s: fan ramps with temperature, so part of the measured "
             "slope may be fan power rather than detector power" % pw)
        print("  ->", fan_note)

os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, "E_of_a.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["a_nominal", "k", "n_runs", "watts_mean", "watts_sd", "fps_mean",
                "infer_fps_mean", "E_mJ_per_fed_frame", "temp_start_mean_C",
                "temp_end_mean_C", "run_to_run_drift_pct"])
    for r in agg:
        w.writerow(["%.4f" % r["a"], r["k"], r["n"], "%.4f" % r["W"], "%.4f" % r["Wsd"],
                    "%.3f" % r["fps"], "%.3f" % r["infer_fps"], "%.2f" % r["E_mJ"],
                    "%.2f" % r["t0"], "%.2f" % r["t1"], "%.2f" % r["drift"]])
json.dump(dict(n_runs=len(rows), fps=FPS,
               M1=dict(W0=b1[0], W0_se=se1[0], kappa_W_per_a=b1[1], kappa_se=se1[1], r2=r2_1),
               M2=(dict(W0=b2[0], W0_se=se2[0], kappa_W_per_a=b2[1], kappa_se=se2[1],
                        gamma_W_per_C=b2[2], gamma_se=se2[2], r2=r2_2) if b2 else None),
               kappa_disagreement_pct=dk, corr_Tstart_a=rTA,
               E_idle_mJ_per_frame=b1[0] / FPS * 1000, kappa_mJ_per_frame_per_a=b1[1] / FPS * 1000,
               mJ_per_invocation=b1[1] / FPS * 1000,
               drift_flags=drift_flags, confound_tripped=conf, fan_note=fan_note,
               fan_step_cost_W=fan_cost,
               c1=dict(intercept=C1_INT, slope=C1_SLOPE, r2=C1_R2)),
          open(os.path.join(OUTDIR, "fit.json"), "w"), indent=2)
print("\nwrote E_of_a.csv and fit.json to", OUTDIR)
