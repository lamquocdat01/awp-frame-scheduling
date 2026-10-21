"""
T5 -- Reality check: sim vs real, 4 decision checks.

Two-phase workflow:

  Phase 1 (run now -- no engine needed):
    python scripts/t5_eval.py --sim-only
      Evaluates DQN + CALIB on the 26 test videos in sim.
      Writes outputs/rl/t5_sim_results.csv.

  Phase 2 (after real engine runs on Jetson):
    python scripts/t5_eval.py --real-dir path/to/real_results
      Loads phase-1 sim CSV + real engine CSVs, runs 4 checks,
      writes outputs/rl/t5_report.csv and prints comparison table.

Real engine CSV format (one file per config, named by tag):
  Filename : p4_calib_raw.csv | p5_calib_guard.csv
             p6_dqn05_raw.csv | p7_dqn05_guard.csv
             p6_dqn20_raw.csv | p7_dqn20_guard.csv
  Columns  : video,ef1,activation,energy_j
  Example  : PTZ/intermittentPan,0.72,0.55,12.3

Matched pairs (never mix raw vs guard):
  P6 vs P4  -- raw-DQN vs raw-CALIB
  P7 vs P5  -- DQN+GUARD vs CALIB+GUARD

Policies used: T4b fair-shot (val-selected, weight_decay=1e-4).
  lam=0.05 seed=2  (repr seed, T4b test EF1=0.8963)
  lam=0.20 seed=2  (repr seed, T4b test EF1=0.8282 -- matched-activation vs CALIB 0.515)

Decision checks (4):
  (a) sim-real gap DQN ~= CALIB (~+0.10)?
      DQN gap >> CALIB gap (+0.20) -> sim over-credits DQN alone -> artifact
  (b) lam=0.20 parity holds at REAL?
      T4b sim: DQN EF1 +0.011 vs CALIB at matched activation (+0.010).
      Verdict is PARITY (not dominance -- margin within noise).
      T5 asks: does parity survive real engine, or does gap flip?
      Gap of 0.02-0.03 either way changes the verdict.
  (c) RAW activation sanity: real_act approx sim_act for DQN?
      Gap > 0.10 -> wrapper bug (z-score/fail-open mismatch)
  (d) GUARD override rate high?
      override_rate = mean(act_P7 - act_P6)
      > 0.10 -> GUARD overrides majority of DQN FAST decisions -> consider GUARD-in-loop for T6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.agents import DQNAgent
from rl.fast_eval import fast_simulate, dqn_actions, evaluate_videos_fast
from asmag_core.controllers import ACTION_SPACE

# ---------------------------------------------------------------------------
# Constants -- frozen from T3.7 / T4
# ---------------------------------------------------------------------------

W = 7
P_MISS = 0.5
R_HIT = 1.0
R_MAINTAIN = 0.2
R_QUIET = 0.1

_SPLIT_FILE   = _REPO / "splits" / "S1_fold0.json"
_NPZ_DIR      = _REPO / "data" / "features"
_TRAIN_STATS  = _REPO / "data" / "features" / "train_stats.json"
_CALIB_PKL    = _REPO / "models" / "calib_S1_fold0_trainonly.pkl"
_RL_OUT       = _REPO / "outputs" / "rl"
_T4B_OUT      = _REPO / "outputs" / "rl" / "t4b"   # T4b fair-shot policies
_SIM_CSV      = _RL_OUT / "t5_sim_results_t4b.csv"

# Representative seeds -- nearest-to-mean from T4b test EF1 (3 seeds):
#   lam=0.05: mean=0.8984; seed0=0.9064(d0.008), seed1=0.8926(d0.006), seed2=0.8963(d0.002) <- seed2
#   lam=0.20: mean=0.8387; seed0=0.8742(d0.036), seed1=0.8137(d0.025), seed2=0.8282(d0.011) <- seed2
# Using T4b policies (val-selected, weight_decay=1e-4) -- NOT T4 final-epoch policies.
_REPR_SEED = {0.05: 2, 0.20: 2}

_REAL_FILE_TAGS = {
    "p4_calib_raw":   {"config": "calib",    "lambda": None,  "guard": False},
    "p5_calib_guard": {"config": "calib",    "lambda": None,  "guard": True},
    "p6_dqn05_raw":   {"config": "dqn",      "lambda": 0.05,  "guard": False},
    "p7_dqn05_guard": {"config": "dqn",      "lambda": 0.05,  "guard": True},
    "p6_dqn20_raw":   {"config": "dqn",      "lambda": 0.20,  "guard": False},
    "p7_dqn20_guard": {"config": "dqn",      "lambda": 0.20,  "guard": True},
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_split():
    with open(str(_SPLIT_FILE)) as f:
        return json.load(f)


def _load_train_stats():
    with open(str(_TRAIN_STATS)) as f:
        s = json.load(f)
    cols = s["feature_columns"]
    mean = np.array([s["mean"][k] for k in cols], dtype=np.float32)
    std  = np.array([s["std"][k]  for k in cols], dtype=np.float32)
    std  = np.where(std < 1e-7, 1.0, std)
    return mean, std


def _load_calib_model():
    with open(str(_CALIB_PKL), "rb") as f:
        obj = pickle.load(f)
    # pkl is a dict with model at key "model"
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _load_dqn_agent(lam: float, seed: int) -> DQNAgent:
    # T4b fair-shot policy (val-selected, weight_decay) takes priority over T4 final-epoch
    tag_t4b = f"fair_dqn_lam{lam:.2f}_seed{seed}"
    path = _T4B_OUT / f"policy_{tag_t4b}.pt"
    if not path.exists():
        # fallback to T4 policy if T4b not available
        tag_t4 = f"dqn_lam{lam:.2f}_seed{seed}"
        path = _RL_OUT / f"policy_{tag_t4}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Policy not found (tried T4b and T4): {tag_t4b}")
    agent = DQNAgent(seed=seed)
    agent.load(str(path))
    agent.eps = 0.0
    return agent


# ---------------------------------------------------------------------------
# Sim evaluation on test split
# ---------------------------------------------------------------------------

def compute_sim_results(test_videos: list, lam: float = 0.20) -> dict:
    """
    Evaluates CALIB-DT3, DQN lam=0.05 seed=2, DQN lam=0.20 seed=2 on test split.
    Returns dict[config_tag] -> {ef1, ef1_std, act, energy, p3fb, n, per_video}.

    lam: lambda used for CALIB reward simulation (must match T4 baseline).
         CALIB decisions are policy-independent of lambda, but reward/energy
         are lambda-dependent.  For T5 we report at lam=0.20 (headline config).
         Call again with lam=0.05 to get CALIB gap at that lambda.
    """
    z_mean, z_std = _load_train_stats()
    results = {}

    # --- CALIB-DT3 (raw features, bypass z-score) -------------------------
    calib_model = None
    if _CALIB_PKL.exists():
        calib_model = _load_calib_model()
    else:
        print(f"  [warn] CALIB pkl not found at {_CALIB_PKL}, skipping CALIB.")

    if calib_model is not None:
        ef1_l, act_l, eng_l, per_vid = [], [], [], {}
        for vid in test_videos:
            cat, name = vid.split("/")
            npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
            roi = npz["temporal_roi"].astype(bool)
            feat_raw = npz["features"][roi].astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = calib_model.predict(feat_raw)
            actions = np.array([ACTION_SPACE.index(p) for p in preds], dtype=np.int32)
            ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            ef1_l.append(ep["event_f1"])
            act_l.append(ep["activation"])
            eng_l.append(ep["energy"])
            per_vid[vid] = {"ef1": ep["event_f1"], "act": ep["activation"], "energy": ep["energy"]}
        results["calib_raw"] = {
            "ef1": float(np.mean(ef1_l)), "ef1_std": float(np.std(ef1_l)),
            "act": float(np.mean(act_l)), "energy": float(np.mean(eng_l)),
            "n": len(ef1_l), "per_video": per_vid,
        }

    # --- DQN configs (T4b fair-shot policies) --------------------------------
    for dqn_lam in [0.05, 0.20]:
        seed = _REPR_SEED[dqn_lam]
        tag = f"t4b_dqn_lam{dqn_lam:.2f}_seed{seed}"   # "t4b_" prefix marks fair-shot
        try:
            agent = _load_dqn_agent(dqn_lam, seed)
        except FileNotFoundError as e:
            print(f"  [warn] {e}, skipping.")
            continue

        fn = lambda obs, a=agent: dqn_actions(a, obs, greedy=True)
        m = evaluate_videos_fast(fn, test_videos, _NPZ_DIR, z_mean, z_std,
                                 lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)

        # Also collect per-video for sim-real comparison
        per_vid = {}
        for vid in test_videos:
            cat, name = vid.split("/")
            npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
            roi = npz["temporal_roi"].astype(bool)
            feat = npz["features"][roi].astype(np.float32)
            obs = (feat - z_mean) / z_std
            actions = dqn_actions(agent, obs, greedy=True)
            ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            per_vid[vid] = {"ef1": ep["event_f1"], "act": ep["activation"], "energy": ep["energy"]}

        results[tag] = {**m, "per_video": per_vid}

    return results


def save_sim_csv(results: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, r in results.items():
        rows.append({
            "config": tag,
            "sim_ef1": f"{r['ef1']:.4f}",
            "sim_ef1_std": f"{r['ef1_std']:.4f}",
            "sim_act": f"{r['act']:.4f}",
            "sim_energy": f"{r['energy']:.1f}",
            "n": r["n"],
        })
    with open(str(out_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Per-video detail CSV
    pv_path = out_path.parent / "t5_sim_per_video.csv"
    with open(str(pv_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "video", "sim_ef1", "sim_act", "sim_energy"])
        for tag, r in results.items():
            for vid, m in r["per_video"].items():
                w.writerow([tag, vid, f"{m['ef1']:.4f}", f"{m['act']:.4f}", f"{m['energy']:.1f}"])

    print(f"  Saved: {out_path}")
    print(f"  Saved: {pv_path}")


# ---------------------------------------------------------------------------
# Real engine result loading
# ---------------------------------------------------------------------------

def load_real_results(real_dir: Path) -> dict:
    """
    Load per-config real engine CSVs.
    Expected filenames: p4_calib_raw.csv, p6_dqn05_raw.csv, etc.
    Returns dict[tag] -> {"mean_ef1": ..., "mean_act": ..., "per_video": {...}}
    """
    results = {}
    for tag in _REAL_FILE_TAGS:
        fpath = real_dir / f"{tag}.csv"
        if not fpath.exists():
            print(f"  [warn] Missing real result file: {fpath}")
            continue
        rows, per_vid = [], {}
        with open(str(fpath)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row["video"].strip()
                ef1 = float(row["ef1"])
                act = float(row["activation"])
                energy = float(row.get("energy_j", 0.0))
                rows.append((ef1, act, energy))
                per_vid[vid] = {"ef1": ef1, "act": act, "energy": energy}
        if rows:
            ef1s, acts, engs = zip(*rows)
            results[tag] = {
                "mean_ef1": float(np.mean(ef1s)),
                "mean_act": float(np.mean(acts)),
                "mean_energy": float(np.mean(engs)),
                "n": len(rows),
                "per_video": per_vid,
            }
    return results


def load_sim_csv(path: Path) -> dict:
    result = {}
    with open(str(path)) as f:
        for row in csv.DictReader(f):
            result[row["config"]] = {
                "ef1": float(row["sim_ef1"]),
                "ef1_std": float(row["sim_ef1_std"]),
                "act": float(row["sim_act"]),
                "energy": float(row["sim_energy"]),
                "n": int(row["n"]),
            }
    return result


# ---------------------------------------------------------------------------
# Decision checks
# ---------------------------------------------------------------------------

def run_checks(sim: dict, real: dict):
    """
    Runs T5 decision checks and prints a structured report.
    Matched pairs enforced: P6 vs P4 (raw), P7 vs P5 (+guard).
    """
    print("\n" + "=" * 72)
    print("T5 REALITY CHECK REPORT")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Sim vs real comparison table
    # -----------------------------------------------------------------------
    print("\n-- Sim vs Real (test split, macro-avg EF1) --")
    print(f"{'Config':<25} {'Sim EF1':>9} {'Real EF1':>9} {'Gap':>7} {'Sim act':>8} {'Real act':>9}")
    print("-" * 72)

    gap_info = {}
    for real_tag, meta in _REAL_FILE_TAGS.items():
        if real_tag not in real:
            continue
        r = real[real_tag]
        # Map real_tag to sim key (t4b_ prefix for DQN)
        if meta["config"] == "calib":
            sim_key = "calib_raw"
        else:
            lam = meta["lambda"]
            seed = _REPR_SEED[lam]
            sim_key = f"t4b_dqn_lam{lam:.2f}_seed{seed}"

        if sim_key not in sim:
            continue
        s = sim[sim_key]
        gap = s["ef1"] - r["mean_ef1"]
        act_gap = r["mean_act"] - s["act"]

        guard_label = "+GUARD" if meta["guard"] else "  RAW"
        label = f"{meta['config'].upper()} lam={meta['lambda'] or 'N/A'} {guard_label}"
        print(f"{label:<25} {s['ef1']:>9.4f} {r['mean_ef1']:>9.4f} {gap:>+7.4f} {s['act']:>8.4f} {r['mean_act']:>9.4f}")

        gap_info[real_tag] = {
            "sim_ef1": s["ef1"], "real_ef1": r["mean_ef1"],
            "gap": gap, "sim_act": s["act"], "real_act": r["mean_act"],
            "act_delta": act_gap,
        }

    # -----------------------------------------------------------------------
    # Check (a): sim-real gap DQN vs CALIB
    # -----------------------------------------------------------------------
    print("\n-- Check (a): Sim-real gap -- DQN vs CALIB --")
    calib_raw_gap = gap_info.get("p4_calib_raw", {}).get("gap")
    for dqn_tag, lam_str in [("p6_dqn05_raw", "0.05"), ("p6_dqn20_raw", "0.20")]:
        if dqn_tag not in gap_info or calib_raw_gap is None:
            continue
        dqn_gap = gap_info[dqn_tag]["gap"]
        delta = dqn_gap - calib_raw_gap
        flag = ""
        if dqn_gap > calib_raw_gap + 0.20:
            flag = " <- ALERT: DQN over-credit (sim artifact)"
        elif dqn_gap > calib_raw_gap + 0.10:
            flag = " <- WARNING: DQN gap noticeably larger than CALIB"
        else:
            flag = " <- OK: DQN gap ~= CALIB gap"
        print(f"  DQN lam={lam_str}: gap={dqn_gap:+.4f}, CALIB gap={calib_raw_gap:+.4f}, d={delta:+.4f}{flag}")

    # -----------------------------------------------------------------------
    # Check (b): Parity at REAL -- lam=0.20 matched-activation comparison.
    # T4b sim: DQN EF1 +0.011 at matched activation (+0.010). Claim is PARITY
    # not dominance -- +0.011 is within inter-seed noise. T5 resolves whether parity holds.
    # -----------------------------------------------------------------------
    print("\n-- Check (b): Parity lam=0.20 at REAL (matched-activation head-to-head) --")
    if "p6_dqn20_raw" in gap_info and "p4_calib_raw" in gap_info:
        dqn20  = gap_info["p6_dqn20_raw"]
        calib4 = gap_info["p4_calib_raw"]
        ef1_delta = dqn20["real_ef1"] - calib4["real_ef1"]
        act_delta = dqn20["real_act"]  - calib4["real_act"]
        print(f"  DQN lam=0.20  real EF1={dqn20['real_ef1']:.4f}  real act={dqn20['real_act']:.4f}")
        print(f"  CALIB         real EF1={calib4['real_ef1']:.4f}  real act={calib4['real_act']:.4f}")
        print(f"  Delta: EF1 {ef1_delta:+.4f}  act {act_delta:+.4f}")
        if abs(ef1_delta) <= 0.03 and abs(act_delta) <= 0.03:
            print("  RESULT: PARITY holds at real engine (both deltas within +-0.03 noise).")
        elif ef1_delta > 0.03:
            print("  RESULT: DQN clearly outperforms CALIB at real (EF1 gap > noise threshold).")
        elif ef1_delta < -0.03:
            print("  RESULT: CALIB outperforms DQN at real (parity lost, DQN EF1 clearly lower).")
        else:
            print("  RESULT: Mixed -- one metric outside noise band; check individual seeds.")

    # -----------------------------------------------------------------------
    # Check (c): RAW activation sanity (wrapper correctness)
    # -----------------------------------------------------------------------
    print("\n-- Check (c): Activation sim ~= real (wrapper correctness) --")
    for dqn_tag, lam_str in [("p6_dqn05_raw", "0.05"), ("p6_dqn20_raw", "0.20")]:
        if dqn_tag not in gap_info:
            continue
        info = gap_info[dqn_tag]
        act_diff = abs(info["real_act"] - info["sim_act"])
        flag = ""
        if act_diff > 0.10:
            flag = " <- FAIL: wrapper bug suspected (z-score / fail-open mismatch)"
        elif act_diff > 0.05:
            flag = " <- marginal"
        else:
            flag = " <- OK"
        print(f"  DQN lam={lam_str}: sim_act={info['sim_act']:.4f}, real_act={info['real_act']:.4f}, |d|={act_diff:.4f}{flag}")

    # -----------------------------------------------------------------------
    # Check (d): GUARD override rate
    # -----------------------------------------------------------------------
    print("\n-- Check (d): GUARD override rate --")
    for raw_tag, guard_tag, lam_str in [
        ("p6_dqn05_raw", "p7_dqn05_guard", "0.05"),
        ("p6_dqn20_raw", "p7_dqn20_guard", "0.20"),
    ]:
        if raw_tag not in real or guard_tag not in real:
            continue
        raw_act   = real[raw_tag]["mean_act"]
        guard_act = real[guard_tag]["mean_act"]
        override_rate = guard_act - raw_act
        flag = ""
        if override_rate > 0.15:
            flag = " <- HIGH: GUARD overrides frequently; consider GUARD-in-loop for T6"
        elif override_rate > 0.05:
            flag = " <- moderate"
        else:
            flag = " <- low"
        print(f"  DQN lam={lam_str}: raw_act={raw_act:.4f}, guard_act={guard_act:.4f}, "
              f"override_rate={override_rate:+.4f}{flag}")

    # Also CALIB
    if "p4_calib_raw" in real and "p5_calib_guard" in real:
        raw_act   = real["p4_calib_raw"]["mean_act"]
        guard_act = real["p5_calib_guard"]["mean_act"]
        override_rate = guard_act - raw_act
        print(f"  CALIB:      raw_act={raw_act:.4f}, guard_act={guard_act:.4f}, "
              f"override_rate={override_rate:+.4f}")

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-only", action="store_true",
                    help="Compute sim EF1 on test split and save. No real engine needed.")
    ap.add_argument("--real-dir", type=str, default=None,
                    help="Directory containing p4/p5/p6/p7 real engine CSVs.")
    ap.add_argument("--lam-sim", type=float, default=0.20,
                    help="Lambda used for reward simulation (default 0.20, matches headline config).")
    args = ap.parse_args()

    split = _load_split()
    test_videos = split["test"]

    if args.sim_only or args.real_dir is None:
        print(f"Computing sim results on {len(test_videos)} test videos (lam={args.lam_sim})...")
        sim_results = compute_sim_results(test_videos, lam=args.lam_sim)

        # Print summary table
        print(f"\n{'Config':<30} {'EF1':>7} {'+-std':>6} {'act':>7} {'energy':>8} {'n':>3}")
        print("-" * 62)
        for tag, r in sim_results.items():
            print(f"{tag:<30} {r['ef1']:>7.4f} {r['ef1_std']:>6.4f} {r['act']:>7.4f} {r['energy']:>8.1f} {r['n']:>3}")

        save_sim_csv(sim_results, _SIM_CSV)

    if args.real_dir is not None:
        if not _SIM_CSV.exists():
            print(f"Sim results not found at {_SIM_CSV}. Run --sim-only first.")
            sys.exit(1)
        sim_results_flat = load_sim_csv(_SIM_CSV)
        real_results = load_real_results(Path(args.real_dir))
        if not real_results:
            print("No real result files found. Check filenames match p4/p5/p6/p7 tags.")
            sys.exit(1)
        run_checks(sim_results_flat, real_results)

        # Save combined report
        report_path = _RL_OUT / "t5_report.csv"
        rows = []
        for real_tag, meta in _REAL_FILE_TAGS.items():
            if real_tag not in real_results:
                continue
            r = real_results[real_tag]
            sim_key = "calib_raw" if meta["config"] == "calib" else \
                f"t4b_dqn_lam{meta['lambda']:.2f}_seed{_REPR_SEED[meta['lambda']]}"
            s = sim_results_flat.get(sim_key, {})
            rows.append({
                "tag": real_tag, "config": meta["config"],
                "lambda": meta.get("lambda", ""),
                "guard": meta["guard"],
                "sim_ef1": s.get("ef1", ""), "real_ef1": r["mean_ef1"],
                "gap_ef1": (s["ef1"] - r["mean_ef1"]) if s.get("ef1") else "",
                "sim_act": s.get("act", ""), "real_act": r["mean_act"],
                "mean_energy_j": r["mean_energy"], "n": r["n"],
            })
        if rows:
            with open(str(report_path), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
