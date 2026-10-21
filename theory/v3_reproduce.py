# -*- coding: utf-8 -*-
"""v3_reproduce.py -- one-command reproduction of every number of AwP v3 and key-by-key comparison with the
committed Revision_v3/numbers.json (C7(h)). Works in the repository layout (scripts in theory/) and in the
working folder (scripts in code/).

  python theory/v3_reproduce.py            # numbers + allocation (fast, ~5-10 min)
  python theory/v3_reproduce.py --full     # + A8(a) Monte Carlo (~40 min on a laptop)

Exit code 0 iff the regenerated numbers.json equals the committed one key by key (floats to 1e-9) and the
in-script re-verification of the TPDS numbers reports zero failures.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOPIC = HERE.parent
REV = TOPIC / "Revision_v3"


def run(script, *args):
    cmd = [sys.executable, str(HERE / script), *args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    ref = REV / "numbers.json"
    keep = Path(tempfile.mkdtemp()) / "numbers_committed.json"
    shutil.copy(ref, keep)
    if a.full:
        run("v3_a8a_mc.py")
    run("v3_a8c_alloc.py")
    run("v3_numbers.py", "--check")
    old, new = flat(json.loads(keep.read_text(encoding="utf-8"))), flat(json.loads(ref.read_text(encoding="utf-8")))
    diffs = []
    for k in sorted(set(old) | set(new)):
        x, y = old.get(k, "<missing>"), new.get(k, "<missing>")
        same = (abs(x - y) <= 1e-9) if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) else x == y
        if not same:
            diffs.append((k, x, y))
    print(f"numbers.json: {len(new)} keys, {len(diffs)} differences vs committed")
    for k, x, y in diffs[:30]:
        print("  DIFF", k, x, "->", y)
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
