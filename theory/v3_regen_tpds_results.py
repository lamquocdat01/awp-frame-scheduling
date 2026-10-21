# -*- coding: utf-8 -*-
"""v3_regen_tpds_results.py -- regenerate the TPDS result artefacts from source and compare (C7 root fix).

The public TPDS snapshot (66e48c2) shipped three truncated result files (metrics_summary.json,
metrics_summary_REAL.json, corruption_sweep_REAL.csv); C7 run 1 failed on the second one. Mechanism
(reproduced 21-10-2026): the TPDS scripts write with json.dump(obj, open(path, "w")) -- not atomic and, without
encoding=, in the Windows locale code page -- so an encode error on a Vietnamese note leaves a half-written file.

  real-data study : re-runs the *unchanged* main() of awp_real_data.py on results/awp_cache/*.npz (no raw video:
                    the video list is rebuilt from the cache in the original order BMC -> LASIESTA -> CDnet)
                    -> metrics_summary_REAL.json, corruption_sweep_REAL.csv must be byte-identical (sha256).
  synthetic study : re-runs main() of awp_experiments.py -> theory_bounds/consistency_robustness/corruption_sweep
                    .csv byte-identical; metrics_summary.json key by key. Its 12 sanity_vs_pa/*/pa_* keys come from
                    the P-A anchor files of the external evaluation repository; they are checked against the
                    committed results/anchors_pa.csv instead.

  python code/v3_regen_tpds_results.py --out <dir> [--ref <results dir>]      (exit 0 iff everything matches)
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(HERE))
try:
    import cv2  # noqa: F401  (only used for frame extraction, which the cache makes unnecessary)
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

import awp_experiments as axp  # noqa: E402
import awp_real_data as ard  # noqa: E402

REAL_FILES = ("metrics_summary_REAL.json", "corruption_sweep_REAL.csv")
SYN_FILES = ("theory_bounds.csv", "consistency_robustness.csv", "corruption_sweep.csv")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flat(d, p=""):
    if isinstance(d, dict):
        return {k2: v2 for k, v in d.items() for k2, v2 in flat(v, f"{p}/{k}").items()}
    if isinstance(d, list):
        return {k2: v2 for i, v in enumerate(d) for k2, v2 in flat(v, f"{p}[{i}]").items()}
    return {p: d}


def videos_from_cache():
    tail = "__%s.npz" % ard.TAG
    names = sorted(f[: -len(tail)] for f in os.listdir(ard.CACHE_DIR) if f.endswith(tail))
    out = []
    for pre, ds in (("BMC_", "BMC"), ("LAS_", "LASIESTA"), ("CDN_", "CDnet")):
        out += [(ds, v, ["cached"], ["cached"]) for v in names if v.startswith(pre)]
    return out


def same_bytes(names, new_dir, ref_dir):
    bad = 0
    for nm in names:
        new, ref = new_dir / nm, ref_dir / nm
        hn, hr = sha(new), (sha(ref) if ref.exists() else "<missing>")
        print(f"  {nm}: regenerated {hn}  reference {hr}  {'IDENTICAL' if hn == hr else 'DIFFERENT'}")
        bad += hn != hr
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    ref = Path(a.ref)
    real, syn = Path(a.out) / "real", Path(a.out) / "synthetic"
    real.mkdir(parents=True, exist_ok=True)
    syn.mkdir(parents=True, exist_ok=True)
    bad = 0

    print("== real-data study (awp_real_data.main on the cache)")
    ard.enumerate_videos = videos_from_cache
    ard.RESULTS_DIR = str(real)
    if ard.main() is None:
        sys.exit("cache incomplete")
    bad += same_bytes(REAL_FILES, real, ref)

    print("== synthetic study (awp_experiments.main)")
    axp.RESULTS_DIR = str(syn)
    axp.main()
    bad += same_bytes(SYN_FILES, syn, ref)
    old = flat(json.loads((ref / "metrics_summary.json").read_text(encoding="utf-8")))
    new = flat(json.loads((syn / "metrics_summary.json").read_text(encoding="utf-8")))
    pa = {k: v for k, v in old.items() if k not in new and k.split("/")[-1].startswith("pa_")}
    diffs = [k for k in sorted(set(old) | set(new)) if k not in pa and old.get(k, "<missing>") != new.get(k, "<missing>")]
    anchors = {float(r["a"]): r for r in csv.DictReader(open(ref / "anchors_pa.csv", encoding="utf-8"))}
    for k, v in pa.items():
        a_key = float(old[k.rsplit("/", 1)[0] + "/pa_nearest_a"])
        col = {"pa_periodic_nearest": "periodic", "pa_gateS_nearest": "S", "pa_gateRF_nearest": "RF"}.get(k.split("/")[-1])
        if col and abs(float(anchors[a_key][col]) - v) > 1e-9:
            diffs.append(k)
    print(f"  metrics_summary.json: {len(new)} anchor-free keys + {len(pa)} pa_* keys (vs anchors_pa.csv): "
          f"{len(diffs)} differences")
    for k in diffs[:20]:
        print("    DIFF", k, old.get(k, "<missing>"), "->", new.get(k, "<missing>"))
    bad += len(diffs)
    print(f"TOTAL differences: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
