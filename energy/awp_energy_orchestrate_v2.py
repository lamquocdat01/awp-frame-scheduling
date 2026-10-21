#!/usr/bin/env python3
"""
awp_energy_orchestrate_v2.py -- AwP activation-energy sweep, Jetson Orin Nano.

Fixes over v1 (all three found during the v1 campaign):
  I.   RANDOMISED (a, repeat) ORDER with a fixed seed. v1 ran a ascending, which
       confounded T_start with a (measured corr = +0.81).
  III. Per-run FAN telemetry (PWM + RPM). VDD_IN includes fan power, and a PWM
       ramp is a STEP, not a smooth drift -- it can break linearity outright.
  IV.  RELATIVE thermal gate. An absolute 60 C gate is a no-op on a board that
       idles at 59-60 C. Gate on T <= T_baseline_idle + COOL_DELTA, with
       T_baseline_idle measured immediately before the campaign.
  (+)  v1's summary parser missed c1_run.py's "SUMMARY " prefix; fixed here.

Purpose (distinct from Deep3 C1): measure marginal board power per unit
activation (kappa) so a deployment energy budget converts into an admissible a,
hence K=ceil(1/a) and beta=min(1,(W+1)/K).
"""
import json, os, random, subprocess, sys, time, glob

CODE   = "/home/dat/topic30/code"
MODEL  = "/home/dat/topic30/models/yolo26s.onnx"
FRAMES = "/home/dat/topic30/data/highway/input"
BASE   = "/home/dat/awp_energy_v2"
OUT    = BASE + "/raw"
IMGSZ, FPS, RUNS, WARMUP = 640, 30.0, 3, 45.0

SEED       = 20260816     # fixed, recorded in metadata
COOL_DELTA = 1.5          # gate = T_baseline_idle + COOL_DELTA
COOL_MAX   = 240.0
T_BASELINE = None

FAN_PWM = "/sys/class/hwmon/hwmon0/pwm1"
FAN_RPM = "/sys/class/hwmon/hwmon2/rpm"

LEVELS = [
    (0.0,     "idle",   1,   300.0),   # idle >= 5 min
    (0.05,    "sweep",  20,  150.0),
    (0.1,     "sweep",  10,  150.0),
    (0.2,     "sweep",  5,   150.0),
    (1.0/3.0, "sweep",  3,   150.0),
    (0.5,     "sweep",  2,   150.0),
]

ZONES = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))

def _rd(p, d=-1):
    try:    return int(open(p).read())
    except Exception: return d

def temp_max():
    return max([_rd(z, 0) for z in ZONES] or [0]) / 1000.0

def fan():
    return _rd(FAN_PWM), _rd(FAN_RPM)

def cooldown():
    gate = (T_BASELINE + COOL_DELTA) if T_BASELINE is not None else 60.0
    t0 = time.time()
    while temp_max() > gate and (time.time() - t0) < COOL_MAX:
        time.sleep(5)
    return round(temp_max(), 2), round(time.time() - t0, 1)

def main():
    global T_BASELINE
    os.makedirs(OUT, exist_ok=True)

    print("measuring idle thermal baseline (60 s) ...", flush=True)
    samples = []
    for _ in range(30):
        samples.append(temp_max()); time.sleep(2)
    T_BASELINE = min(samples)
    gate = T_BASELINE + COOL_DELTA
    p0, r0 = fan()
    meta = dict(seed=SEED, T_baseline_idle=T_BASELINE, cool_delta=COOL_DELTA,
                gate_C=gate, fan_pwm_at_baseline=p0, fan_rpm_at_baseline=r0,
                model=MODEL, imgsz=IMGSZ, fps=FPS, runs_per_level=RUNS,
                warmup_s=WARMUP, levels=[dict(a=a, mode=m, k=k, measure_s=s)
                                         for (a, m, k, s) in LEVELS])
    json.dump(meta, open(BASE + "/metadata.json", "w"), indent=2)
    print("T_baseline_idle=%.2f C  gate=%.2f C  seed=%d  fan pwm=%s rpm=%s"
          % (T_BASELINE, gate, SEED, p0, r0), flush=True)

    plan = [(a, mode, k, meas, run)
            for (a, mode, k, meas) in LEVELS for run in range(1, RUNS + 1)]
    random.Random(SEED).shuffle(plan)          # break the a-vs-temperature confound
    json.dump([dict(a=a, k=k, run=r) for (a, _, k, _, r) in plan],
              open(BASE + "/run_order.json", "w"), indent=2)

    jl = open(BASE + "/results.jsonl", "a")
    results = []
    for i, (a, mode, k, meas, run) in enumerate(plan, 1):
        pre_t, waited = cooldown()
        pwm_b, rpm_b = fan()
        tag = "awp_a%s" % ("%0.3f" % a).replace(".", "p")
        cmd = [sys.executable, os.path.join(CODE, "c1_run.py"),
               "--mode", mode, "--detector", "yolo26s", "--runtime", "ort",
               "--model", MODEL, "--imgsz", str(IMGSZ), "--frames", FRAMES,
               "--k", str(k), "--warmup", str(WARMUP), "--measure", str(meas),
               "--fps", str(FPS), "--power_hz", "10", "--run", str(run),
               "--tag", tag, "--outdir", OUT]
        print("[%d/%d] a=%.4f k=%d run=%d  pre-temp %.1fC (waited %.0fs) fan %s/%s"
              % (i, len(plan), a, k, run, pre_t, waited, pwm_b, rpm_b), flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/dat/topic30")
        pwm_a, rpm_a = fan()
        if p.returncode != 0:
            print("  FAILED rc=%d\n%s\n%s" % (p.returncode, p.stdout[-1200:], p.stderr[-1200:]), flush=True)
            continue
        cand = [l.strip() for l in p.stdout.splitlines()
                if l.strip().startswith("{") or l.strip().startswith("SUMMARY {")]
        if not cand:
            print("  NO JSON SUMMARY\n%s" % p.stdout[-800:], flush=True); continue
        ln = cand[-1]
        s = json.loads(ln[len("SUMMARY "):] if ln.startswith("SUMMARY ") else ln)
        s.update(a_nominal=a, pre_temp=pre_t, cooldown_wait_s=waited,
                 fan_pwm_before=pwm_b, fan_rpm_before=rpm_b,
                 fan_pwm_after=pwm_a, fan_rpm_after=rpm_a,
                 order_index=i, wall_s=round(time.time() - t0, 1))
        results.append(s); jl.write(json.dumps(s) + "\n"); jl.flush()
        print("  W=%.4f+-%.4f fps=%.2f inf_fps=%.2f temp %s->%s fan %s->%s"
              % (s["watts_mean"], s["watts_std"], s["fps"], s["infer_fps"],
                 s.get("temp_start"), s.get("temp_end"), pwm_b, pwm_a), flush=True)
    jl.close()
    json.dump(results, open(BASE + "/summary.json", "w"), indent=2)
    print("DONE %d/%d runs" % (len(results), len(plan)), flush=True)

if __name__ == "__main__":
    main()
