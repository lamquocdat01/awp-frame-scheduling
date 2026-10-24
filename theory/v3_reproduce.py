# -*- coding: utf-8 -*-
"""v3_reproduce.py -- one-command reproduction of every number of AwP v3 (C7(h)). Works in the repository
layout (scripts in theory/) and in the working folder (scripts in code/).

  python theory/v3_reproduce.py            # numbers + allocation + TPDS artefacts (~5 min)
  python theory/v3_reproduce.py --full     # + A8(a) Monte Carlo (~40 min on a laptop)

Phase 0  SNAPSHOT  copy every committed reference output to a temporary folder (before anything is rewritten).
Phase 1  GENERATE  run the generators to completion, one after the other:
                   [v3_a8a_mc.py] -> v3_a8c_alloc.py -> v3_sweep_blockaware.py -> v3_numbers.py
                   -> v3_check_thm3_hedge.py (stdout -> results/v3/thm3_hedge_audit.txt);
                   all outputs are written atomically (v3_io).
Phase 2  CHECK     only after phase 1 has finished:
                   (a) v3_regen_tpds_results.py: TPDS real-data/synthetic artefacts regenerated from source and
                       compared with results/ (sha256 / key by key);
                   (b) numbers.json key by key vs the snapshot (floats to 1e-9, everything else exact) and the
                       in-script re-verification list _verification_failures must be empty;
                       v3_sweep_blockaware.py exits 0 only if its lambda in {0,1} invariants hold;
                   (c) a8c_summary.json (and a8a_summary.json with --full) key by key; every regenerated CSV
                       cell by cell (exact string match); numbers.tex byte for byte.
Exit code 0 iff phase 2 reports 0 differences.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
REV = TOPIC / "Revision_v3"
RES = TOPIC / "results"
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

JSON_ALWAYS = [REV / "numbers.json", RES / "v3" / "a8c_summary.json", RES / "v3" / "sweep_blockaware_summary.json"]
JSON_FULL = [RES / "v3" / "a8a_summary.json"]
CSV_ALWAYS = [RES / "v3" / "a8c_alloc.csv", RES / "v3" / "a8c_T2.csv", REV / "S2_percategory_v3.csv",
              RES / "v3" / "corruption_sweep_REAL_blockaware.csv"]
CSV_FULL = [RES / "v3" / n for n in ("a8a_ceiling.csv", "a8a_sv_worstcase.csv", "a8a_sv_family_all.csv")]
BYTES_ALWAYS = [REV / "numbers.tex", RES / "v3" / "thm3_hedge_audit.txt"]


def run(script, *args, check=True):
    cmd = [sys.executable, str(HERE / script), *args]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, env=ENV).returncode


def flat(d, p=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}/{k}"))
        return out
    if isinstance(d, list):
        out = {}
        for i, v in enumerate(d):
            out.update(flat(v, f"{p}[{i}]"))
        return out
    return {p: d}


def cmp_json(ref, new):
    old, cur = flat(json.loads(ref.read_text(encoding="utf-8"))), flat(json.loads(new.read_text(encoding="utf-8")))
    diffs = []
    for k in sorted(set(old) | set(cur)):
        x, y = old.get(k, "<missing>"), cur.get(k, "<missing>")
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool)
        if not ((abs(x - y) <= 1e-9) if num else x == y):
            diffs.append((k, x, y))
    return len(cur), diffs


def cmp_csv(ref, new):
    a = list(csv.reader(open(ref, encoding="utf-8", newline="")))
    b = list(csv.reader(open(new, encoding="utf-8", newline="")))
    diffs = [(f"row {i}", "<missing>" if i >= len(a) else a[i], "<missing>" if i >= len(b) else b[i])
             for i in range(max(len(a), len(b))) if i >= len(a) or i >= len(b) or len(a[i]) != len(b[i])]
    for i in range(min(len(a), len(b))):
        if len(a[i]) == len(b[i]):
            diffs += [(f"row {i} col {j}", x, y) for j, (x, y) in enumerate(zip(a[i], b[i])) if x != y]
    return sum(len(r) for r in b), diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    jsons, csvs = JSON_ALWAYS + (JSON_FULL if a.full else []), CSV_ALWAYS + (CSV_FULL if a.full else [])

    # ---- phase 0: snapshot of the committed references ------------------------------------------
    snap = Path(tempfile.mkdtemp(prefix="v3_reproduce_"))
    ref = {}
    for p in jsons + csvs + BYTES_ALWAYS:
        ref[p] = snap / f"{len(ref):02d}_{p.name}"
        shutil.copy2(p, ref[p])
    print(f"[phase 0] snapshot of {len(ref)} committed outputs -> {snap}", flush=True)

    # ---- phase 1: generate (to completion) ------------------------------------------------------
    print("[phase 1] GENERATE", flush=True)
    if a.full:
        run("v3_a8a_mc.py")
    run("v3_a8c_alloc.py")
    sweep_rc = run("v3_sweep_blockaware.py", check=False)
    run("v3_numbers.py")
    audit = subprocess.run([sys.executable, str(HERE / "v3_check_thm3_hedge.py")], check=True, env=ENV,
                           capture_output=True).stdout
    tmp = RES / "v3" / "thm3_hedge_audit.txt.tmp"
    tmp.write_bytes(audit)
    os.replace(tmp, RES / "v3" / "thm3_hedge_audit.txt")
    print("[phase 1] done", flush=True)

    # ---- phase 2: check -------------------------------------------------------------------------
    print("[phase 2] CHECK", flush=True)
    total = 0
    rc = run("v3_regen_tpds_results.py", "--out", str(snap / "tpds_regen"), "--ref", str(RES), check=False)
    print(f"  TPDS artefacts regenerated from source: {'0 differences' if rc == 0 else 'DIFFERENCES (see above)'}")
    total += rc != 0
    fails = json.loads((REV / "numbers.json").read_text(encoding="utf-8")).get("_verification_failures", ["<missing>"])
    print(f"  numbers.json in-script re-verification of the TPDS numbers: {len(fails)} failure(s)")
    for f in fails:
        print("    FAIL", f)
    total += len(fails)
    print(f"  v3_sweep_blockaware.py invariants (lambda in {{0,1}} = TPDS rows; naive replay = TPDS file): "
          f"{'0 failures' if sweep_rc == 0 else 'FAILED'}")
    total += sweep_rc != 0
    for p in jsons:
        n, diffs = cmp_json(ref[p], p)
        print(f"  {p.relative_to(TOPIC)}: {n} keys, {len(diffs)} differences vs committed")
        for k, x, y in diffs[:30]:
            print("    DIFF", k, x, "->", y)
        total += len(diffs)
    for p in csvs:
        n, diffs = cmp_csv(ref[p], p)
        print(f"  {p.relative_to(TOPIC)}: {n} cells, {len(diffs)} differences vs committed")
        for k, x, y in diffs[:30]:
            print("    DIFF", k, x, "->", y)
        total += len(diffs)
    for p in BYTES_ALWAYS:
        same = ref[p].read_bytes() == p.read_bytes()
        print(f"  {p.relative_to(TOPIC)}: {'byte-identical' if same else 'DIFFERENT'}")
        total += not same
    print(f"TOTAL: {total} differences")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
