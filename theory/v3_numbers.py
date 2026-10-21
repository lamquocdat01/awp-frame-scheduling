# -*- coding: utf-8 -*-
"""v3_numbers.py -- single source of truth for every number printed in paper_v3 (AwP v3).

Recomputes from cached derived data (no raw video needed):
  * C121 uncapped real-data study from results/awp_cache/*.npz (per-video cue + GT event mask)
    -> Table III, onset-AUC, per-category table (re-verified row by row against results/percategory.csv),
       paired video-level bootstrap CI of the gate - periodic gap (B = 1000, seed 42).
  * Energy anchor from the measured fit (energy/fit.json in the repo; copied path below).
  * Theory constants from closed forms (Theorem 2, Corollary, Remark rem:skew, Corollary 7, Theorem 7).
  * A8(a) / A8(c) outputs from results/v3/a8a_summary.json and a8c_summary.json.
Writes Revision_v3/numbers.json and Revision_v3/numbers.tex (\newcommand macros).
Usage: python code/v3_numbers.py [--check]   (--check: exit 1 if any re-verification fails)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from decimal import ROUND_HALF_UP, Decimal

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC / "code"))
sys.path.insert(0, str(TOPIC / "theory"))  # repository layout
from awp_experiments import bootstrap_ci, hedge_gate_fire, onset_auc, recall_windowed  # noqa: E402

RES = TOPIC / "results"
CACHE = RES / "awp_cache"
REV = TOPIC / "Revision_v3"
FIT = TOPIC / "energy" / "fit.json"  # repository layout
if not FIT.exists():
    FIT = Path(os.environ.get("AWP_REPO", r"D:\Repos\awp-frame-scheduling")) / "energy" / "fit.json"
W, B, SEED = 7, 1000, 42
A_GRID = [0.10, 0.20, 0.333]


def r3(x):
    # TPDS convention: values were stored at 4 decimals, then printed half-up at 3 (double rounding)
    return float(Decimal(str(round(float(x), 4))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def extract_events(gt, min_duration=2):
    ev, T, t = [], len(gt), 0
    while t < T:
        if gt[t]:
            s = t
            while t < T and gt[t]:
                t += 1
            if t - s >= min_duration:
                ev.append({"onset": s, "end": t - 1, "duration": t - s})
        else:
            t += 1
    return ev


def group_of(fname: str) -> tuple[str, str]:
    vid = os.path.basename(fname).split("__")[0]
    if vid.startswith("BMC_"):
        return "BMC", "BMC"
    if vid.startswith("LAS_"):
        return "LASIESTA", "LASIESTA"
    return "CDnet", "CDnet:" + vid.split("_", 2)[1]


def load_corpus():
    # same order as the original uncapped run (BMC -> LASIESTA -> CDnet, each sorted) so that the
    # video-level bootstrap resamples identically and reproduces the TPDS CIs to the 4th decimal
    allf = sorted(glob.glob(str(CACHE / "*__64x48_fALL.npz")))
    files = [f for pre in ("BMC_", "LAS_", "CDN_") for f in allf if os.path.basename(f).startswith(pre)]
    corpus, n_frames = [], 0
    for f in files:
        z = np.load(f)
        cue, gt = z["cue"], z["gt"].astype(bool)
        n_frames += len(gt)
        ev = extract_events(gt)
        if len(ev) >= 2:
            win = np.zeros(len(gt), bool)
            for e in ev:
                win[e["onset"]:min(len(gt), e["onset"] + W + 1)] = True
            ds, grp = group_of(f)
            corpus.append(dict(ds=ds, grp=grp, cue=cue, events=ev, win=win, T=len(gt)))
    return files, corpus, n_frames


def r_star(c, beta):
    if beta <= 0:
        return 0.0
    if c <= beta:
        return c
    if beta <= 0.5:
        return beta * (1 - c) / (1 - beta)
    return 1 - (1 - beta) * c / beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    fails: list[str] = []
    N: dict = {}

    # ---------------- C121 uncapped -----------------------------------------------------------
    files, corpus, n_frames = load_corpus()
    N["c121_videos"] = len(files)
    N["c121_frames_all121"] = n_frames
    N["c121_frames"] = sum(c["T"] for c in corpus)  # frames of the videos with >= 2 events (TPDS "131,325")
    N["c121_videos_ge2"] = len(corpus)
    N["c121_events"] = sum(len(c["events"]) for c in corpus)
    aucs = [onset_auc(c["cue"], c["win"]) for c in corpus]
    N["auc_all"] = round(float(np.mean(aucs)), 3)  # AUCs were printed with Python round in TPDS
    for ds in ("BMC", "LASIESTA", "CDnet"):
        N[f"auc_{ds}"] = round(float(np.mean([x for x, c in zip(aucs, corpus) if c["ds"] == ds])), 3)
    N["eta_deployed"] = round(1 - float(np.mean(aucs)), 3)
    old = json.loads((RES / "metrics_summary_REAL.json").read_text(encoding="utf-8"))
    for k_new, k_old in (("c121_videos", "n_videos_selected"), ("c121_videos_ge2", "n_videos_with_events"),
                         ("c121_events", "n_events"), ("c121_frames", "n_frames")):
        if N[k_new] != old["config"][k_old]:
            fails.append(f"{k_new}: {N[k_new]} != {old['config'][k_old]}")
    if abs(N["auc_all"] - old["cue_onset_auc"]["mean"]) > 6e-4:
        fails.append("auc_all mismatch")
    table3 = {}
    for av in A_GRID:
        g = np.array([recall_windowed(hedge_gate_fire(c["cue"], c["T"], av, 0.0), c["events"], W) for c in corpus])
        p = np.array([recall_windowed(hedge_gate_fire(c["cue"], c["T"], av, 1.0), c["events"], W) for c in corpus])
        gm, glo, ghi = bootstrap_ci(g, B, SEED + int(av * 1000))
        pm, plo, phi = bootstrap_ci(p, B, SEED + int(av * 1000) + 10)  # awp_real_data.py convention: + int(lam*10)
        dm, dlo, dhi = bootstrap_ci(g - p, B, SEED)  # v3: paired video-level bootstrap of the gap
        key = f"a{av:.3f}"
        o = old["deployed_gate_vs_periodic"][key]
        for nm, new, ref in (("gate", gm, o["real_cue_gate"]), ("periodic", pm, o["periodic"]),
                             ("gate_lo", glo, o["gate_ci"][0]), ("gate_hi", ghi, o["gate_ci"][1]),
                             ("per_lo", plo, o["periodic_ci"][0]), ("per_hi", phi, o["periodic_ci"][1])):
            if abs(round(new, 4) - ref) > 1e-4:
                fails.append(f"table3 {key} {nm}: {round(new, 4)} != {ref}")
        table3[key] = dict(gate=r3(gm), gate_ci=[r3(glo), r3(ghi)], periodic=r3(pm),
                           periodic_ci=[r3(plo), r3(phi)], gap=r3(gm - pm),
                           gap_paired_ci=[r3(dlo), r3(dhi)], gap_ci_excludes_0=bool(dhi < 0),
                           n_videos=len(corpus))
    N["table3"] = table3
    N["trails_at_every_budget"] = all(v["gap_ci_excludes_0"] for v in table3.values())

    # per-category re-verification (every row)
    ref_rows = {r["group"]: r for r in csv.DictReader(open(RES / "percategory.csv", encoding="utf-8"))}
    groups = sorted(set(c["grp"] for c in corpus)) + ["ALL"]
    pc = []
    for grp in groups:
        rows = corpus if grp == "ALL" else [c for c in corpus if c["grp"] == grp]
        rec = {"group": grp, "n_videos": len(rows), "n_events": sum(len(c["events"]) for c in rows),
               "onset_auc": round(float(np.mean([onset_auc(c["cue"], c["win"]) for c in rows])), 3)}
        for av, tag in ((0.10, "a010"), (0.20, "a020")):
            gg = float(np.mean([recall_windowed(hedge_gate_fire(c["cue"], c["T"], av, 0.0), c["events"], W) for c in rows]))
            pp = float(np.mean([recall_windowed(hedge_gate_fire(c["cue"], c["T"], av, 1.0), c["events"], W) for c in rows]))
            rec[f"gate_{tag}"], rec[f"periodic_{tag}"], rec[f"gap_{tag}"] = round(gg, 3), round(pp, 3), round(gg - pp, 3)
        pc.append(rec)
        ref = ref_rows.get(grp)
        if ref is None:
            fails.append(f"percategory: group {grp} missing in results/percategory.csv")
            continue
        for k, v in rec.items():
            if k == "group":
                continue
            if abs(float(ref[k]) - float(v)) > 1.01e-3:
                fails.append(f"percategory {grp} {k}: {v} != {ref[k]}")
    if len(ref_rows) != len(groups):
        fails.append(f"percategory rows {len(ref_rows)} != {len(groups)}")
    N["percategory_rows"] = len(pc)
    N["percategory_all_negative_a020"] = all(r["gap_a020"] < 0 for r in pc if r["group"] != "ALL")
    sub = [r for r in pc if r["group"] != "ALL"]
    N["percategory_gap_a020_min"] = min(r["gap_a020"] for r in sub)
    N["percategory_gap_a020_max"] = max(r["gap_a020"] for r in sub)
    N["percategory_groups"] = len(sub)
    with open(REV / "S2_percategory_v3.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pc[0]))
        w.writeheader()
        w.writerows(pc)

    # ---------------- energy anchor --------------------------------------------------------------
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    N["W0"] = round(fit["M1"]["W0"], 4)
    N["kappa"] = round(fit["M1"]["kappa_W_per_a"], 4)
    N["kappa_se"] = round(fit["M1"]["kappa_se"], 3)
    N["R2"] = round(fit["M1"]["r2"], 5)
    N["mJ_invocation"] = round(fit["mJ_per_invocation"], 1)
    N["mJ_floor"] = round(fit["E_idle_mJ_per_frame"], 1)
    for nm, ref in (("W0", 4.9571), ("kappa", 6.9089), ("R2", 0.99899)):
        if abs(N[nm] - ref) > 1e-4:
            fails.append(f"anchor {nm}: {N[nm]} != {ref}")

    # ---------------- theory constants ---------------------------------------------------------
    N["rstar1_beta08"] = round(r_star(1.0, 0.8), 4)              # 0.75
    N["worked_c09_beta02"] = round(r_star(0.9, 0.2), 4)          # 0.025
    N["remark_skew"] = round(0.5 * (r_star(1, 0.2) + r_star(1, 0.8)), 4)  # 0.375
    fg = list(csv.DictReader(open(RES / "frontier_general.csv", encoding="utf-8")))
    hg = [float(r["frontier"]) - float(r["r"]) for r in fg if r["gate"].upper().startswith("HEDGE") and abs(float(r["beta"]) - 0.8) < 1e-9]
    N["hedge_gap_beta08"] = round(max(hg), 2) if hg else None
    N["delta"] = {f"{b:.1f}": round(b - r_star(1, b), 4) for b in (0.2, 0.4, 0.5, 0.8)}

    # ---------------- A8 outputs ---------------------------------------------------------------
    for nm in ("a8a_summary.json", "a8c_summary.json"):
        p = RES / "v3" / nm
        N[nm.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    lp = json.loads((RES / "learned_cue_probe.json").read_text(encoding="utf-8"))
    N["learned_cue"] = {"auc_cross_baseline": lp["onset_auc"]["cross"]["baseline"],
                        "auc_cross_learned": lp["onset_auc"]["cross"]["learned"],
                        "gate": {str(r["a"]): r["learned_gate"] for r in lp["gate_vs_periodic"]}}

    N["_verification_failures"] = fails
    (REV / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False), encoding="utf-8")
    write_tex(N)
    print(f"numbers.json written; {len(fails)} verification failure(s)")
    for f in fails:
        print("  FAIL", f)
    if a.check and fails:
        sys.exit(1)


def write_tex(N: dict) -> None:
    def f3(x):
        return f"{x:.3f}"
    t3 = N["table3"]
    mac = {
        "Videos": f"{N['c121_videos']}", "Frames": f"{N['c121_frames']:,}".replace(",", "{,}"),
        "VideosEv": f"{N['c121_videos_ge2']}", "Events": f"{N['c121_events']}",
        "aucAll": f3(N["auc_all"]), "aucBMC": f3(N["auc_BMC"]), "aucLAS": f3(N["auc_LASIESTA"]),
        "aucCDN": f3(N["auc_CDnet"]),
        "Wzero": "4.957", "kappa": "6.909", "kappaSE": "0.055", "Rsq": "0.99899",
        "mJinv": f"{N['mJ_invocation']:.1f}", "mJfloor": f"{N['mJ_floor']:.1f}",
        "Groups": f"{N['percategory_groups']}",
    }
    for key, tag in (("a0.100", "A"), ("a0.200", "B"), ("a0.333", "C")):
        r = t3[key]
        mac[f"gate{tag}"] = f3(r["gate"])
        mac[f"gateLo{tag}"], mac[f"gateHi{tag}"] = f3(r["gate_ci"][0]), f3(r["gate_ci"][1])
        mac[f"per{tag}"] = f3(r["periodic"])
        mac[f"perLo{tag}"], mac[f"perHi{tag}"] = f3(r["periodic_ci"][0]), f3(r["periodic_ci"][1])
        mac[f"gap{tag}"] = f3(r["gap"])
        mac[f"gapLo{tag}"], mac[f"gapHi{tag}"] = f3(r["gap_paired_ci"][0]), f3(r["gap_paired_ci"][1])
    lines = ["% numbers.tex -- generated by code/v3_numbers.py; do not edit by hand"]
    for k, v in mac.items():
        lines.append(f"\\newcommand{{\\n{k}}}{{{v}}}")
    (REV / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
