#!/usr/bin/env python3
"""
awp_energy_orchestrate.py -- AwP-specific activation-energy sweep on Jetson Orin Nano.

PURPOSE (distinct from Deep3 C1): C1 measured (E_idle, E_active) per detector to
certify worst-case latency of an oblivious schedule.  THIS run measures the
*marginal board power per unit activation* (kappa) so that a deployment energy
budget can be converted into an admissible activation a, hence K=ceil(1/a) and
beta=min(1,(W+1)/K).  Denser activation grid, >=3 repetitions per level.

Reuses the C1 measurement harness (c1_run.py) unchanged -- same board, same rail
(VDD_IN via INA3221 hwmon1), same power mode, same detector/backend/feed, so the
result is directly cross-checkable against C1's E(a) fit.
"""
import json, os, subprocess, sys, time, glob

CODE   = "/home/dat/topic30/code"
MODEL  = "/home/dat/topic30/models/yolo26s.onnx"
FRAMES = "/home/dat/topic30/data/highway/input"
OUT    = "/home/dat/awp_energy/raw"
IMGSZ  = 640
FPS    = 30.0
RUNS   = 3
WARMUP = 45.0
COOL_C = 60.0          # thermal gate, same as C1
COOL_MAX = 240.0

# a = 1/k ; a=0 is the idle-skip point (no inference at all)
LEVELS = [
    # (a,      mode,    k,   measure_s)
    (0.0,     "idle",   1,   300.0),   # prompt requires idle >= 5 min
    (0.05,    "sweep",  20,  150.0),
    (0.1,     "sweep",  10,  150.0),
    (0.2,     "sweep",  5,   150.0),
    (1.0/3.0, "sweep",  3,   150.0),
    (0.5,     "sweep",  2,   150.0),
]

ZONES = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
def temp_max():
    m = 0
    for z in ZONES:
        try:
            v = int(open(z).read())
            if v > m: m = v
        except Exception:
            pass
    return m / 1000.0

def cooldown():
    t0 = time.time()
    while temp_max() > COOL_C and (time.time() - t0) < COOL_MAX:
        time.sleep(5)
    return round(temp_max(), 2)

def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    jl = open("/home/dat/awp_energy/results.jsonl", "a")
    total = len(LEVELS) * RUNS
    n = 0
    for (a, mode, k, meas) in LEVELS:
        tag = "awp_a%s" % ("%0.3f" % a).replace(".", "p")
        for run in range(1, RUNS + 1):
            n += 1
            pre = cooldown()
            cmd = [sys.executable, os.path.join(CODE, "c1_run.py"),
                   "--mode", mode, "--detector", "yolo26s", "--runtime", "ort",
                   "--model", MODEL, "--imgsz", str(IMGSZ),
                   "--frames", FRAMES, "--k", str(k),
                   "--warmup", str(WARMUP), "--measure", str(meas),
                   "--fps", str(FPS), "--power_hz", "10",
                   "--run", str(run), "--tag", tag, "--outdir", OUT]
            print("[%d/%d] a=%.4f k=%d run=%d  (pre-temp %.1fC)" % (n, total, a, k, run, pre), flush=True)
            t0 = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/dat/topic30")
            if p.returncode != 0:
                print("  FAILED rc=%d\n%s\n%s" % (p.returncode, p.stdout[-1500:], p.stderr[-1500:]), flush=True)
                continue
            line = [l for l in p.stdout.strip().split("\n") if l.strip().startswith("{")]
            if not line:
                print("  NO JSON SUMMARY\n%s" % p.stdout[-1000:], flush=True); continue
            s = json.loads(line[-1])
            s["a_nominal"] = a
            s["pre_temp"] = pre
            s["wall_s"] = round(time.time() - t0, 1)
            results.append(s)
            jl.write(json.dumps(s) + "\n"); jl.flush()
            print("  W=%.4f+-%.4f fps=%.2f infer_fps=%.2f temp %s->%s" %
                  (s["watts_mean"], s["watts_std"], s["fps"], s["infer_fps"],
                   s.get("temp_start"), s.get("temp_end")), flush=True)
    jl.close()
    json.dump(results, open("/home/dat/awp_energy/summary.json", "w"), indent=2)
    print("DONE %d runs" % len(results), flush=True)

if __name__ == "__main__":
    main()
