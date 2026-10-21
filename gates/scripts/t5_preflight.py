"""
T5 Pre-flight smoke test --MUST PASS before full engine run.

Validates on 1-2 test videos (RAW mode, no GUARD):
  1. Canonical feature contract: the 3 gating-decision features are 0 in the NPZ
     -- reuse_success_rate (7), active_frame_rate (8), gate_closed_rate (9).
  2. Activation sanity: DQN + CALIB activation on smoke videos ~= sim values.
     Tolerance: |real_act - sim_act| < 0.05 per-video.

Why this matters:
  x_t during training is canonical (policy-independent) -> features[7/8/9] are
  always 0.  features[8]=active_frame_rate especially MUST be 0: feeding live GT
  there (old A3 bug) makes it a per-frame oracle.  If the engine feeds LIVE values
  for any of the three, the DQN sees OOD input -> activation skew != sim.
  Pre-flight catches this BEFORE a full 6x26 run wastes hours.

Usage:
  python scripts/t5_preflight.py                    # default: 2 smoke videos
  python scripts/t5_preflight.py --videos 4         # more videos for confidence
  python scripts/t5_preflight.py --all              # all 26 test videos

Exit code:
  0 = PASS --proceed to full run
  1 = FAIL --fix canonical feature feeding before full run
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.rl_gate import DQNGate
from rl.agents import DQNAgent
from rl.fast_eval import fast_simulate, dqn_actions
from asmag_core.controllers import ACTION_SPACE
from asmag_core.features import (
    ONLINE_MODE_FEATURE_COLUMNS,
    CANONICAL_ZERO_INDICES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SPLIT_FILE  = _REPO / "splits" / "S1_fold0.json"
_NPZ_DIR     = _REPO / "data" / "features"
_TRAIN_STATS = _REPO / "data" / "features" / "train_stats.json"
_CALIB_PKL   = _REPO / "models" / "calib_S1_fold0_trainonly.pkl"
_T4B_OUT     = _REPO / "outputs" / "rl" / "t4b"
_SIM_PV_CSV  = _REPO / "outputs" / "rl" / "t5_sim_per_video.csv"

# Sim reference values (macro-avg from t5_sim_results_t4b.csv)
_SIM_REF = {
    "calib_raw":              {"act": 0.5146},
    "t4b_dqn_lam0.05_seed2": {"act": 0.6545},
    "t4b_dqn_lam0.20_seed2": {"act": 0.5299},
}

W = 7
P_MISS = 0.5
R_HIT = 1.0
R_MAINTAIN = 0.2
R_QUIET = 0.1

# Tolerance for per-video activation match
_ACT_TOL = 0.05

# Canonical feature indices (single source of truth in asmag_core.features)
_CANONICAL_ZERO_INDICES = CANONICAL_ZERO_INDICES  # (7, 8, 9)


def load_sim_per_video() -> dict:
    """Load per-video sim results for comparison."""
    import csv
    result = {}
    if not _SIM_PV_CSV.exists():
        return result
    with open(str(_SIM_PV_CSV)) as f:
        for row in csv.DictReader(f):
            key = (row["config"], row["video"])
            result[key] = {
                "ef1": float(row["sim_ef1"]),
                "act": float(row["sim_act"]),
            }
    return result


def check_canonical_features(npz_data, video_id: str) -> list[str]:
    """Verify the 3 gating-decision features (idx 7/8/9) are zero (canonical contract)."""
    errors = []
    roi = npz_data["temporal_roi"].astype(bool)
    feat = npz_data["features"][roi].astype(np.float32)

    for idx in _CANONICAL_ZERO_INDICES:
        col = ONLINE_MODE_FEATURE_COLUMNS[idx]
        vals = feat[:, idx]
        if np.any(vals != 0.0):
            nz = int(np.sum(vals != 0.0))
            errors.append(
                f"  {video_id}: {col} (idx {idx}) has {nz}/{len(vals)} non-zero frames "
                f"(max={vals.max():.4f})"
            )
    return errors


def run_smoke_test(videos: list[str]) -> bool:
    """
    Run pre-flight on selected videos.  Returns True if all checks pass.
    """
    print(f"T5 PRE-FLIGHT: {len(videos)} video(s)")
    print("=" * 68)

    # Load sim per-video reference
    sim_pv = load_sim_per_video()

    # Load z-score stats
    with open(str(_TRAIN_STATS)) as f:
        s = json.load(f)
    cols = s["feature_columns"]
    z_mean = np.array([s["mean"][k] for k in cols], dtype=np.float32)
    z_std  = np.array([s["std"][k]  for k in cols], dtype=np.float32)
    z_std  = np.where(z_std < 1e-7, 1.0, z_std)

    # Load CALIB
    with open(str(_CALIB_PKL), "rb") as f:
        obj = pickle.load(f)
    calib_model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    # Load DQN gates via DQNGate (production interface)
    dqn_gates = {}
    for lam in [0.05, 0.20]:
        tag = f"fair_dqn_lam{lam:.2f}_seed2"
        dqn_gates[lam] = DQNGate.from_config(
            tag, str(_T4B_OUT), str(_TRAIN_STATS), lambda_cost=lam
        )

    # Also load DQN agents for fast_simulate comparison
    dqn_agents = {}
    for lam in [0.05, 0.20]:
        tag = f"fair_dqn_lam{lam:.2f}_seed2"
        path = _T4B_OUT / f"policy_{tag}.pt"
        agent = DQNAgent(seed=2)
        agent.load(str(path))
        agent.eps = 0.0
        dqn_agents[lam] = agent

    all_pass = True
    canonical_errors = []

    # -----------------------------------------------------------------------
    # Check 1: Canonical feature contract
    # -----------------------------------------------------------------------
    print("\n-- Check 1: Canonical feature contract --")
    for vid in videos:
        cat, name = vid.split("/")
        npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
        errs = check_canonical_features(npz, vid)
        canonical_errors.extend(errs)

    if canonical_errors:
        print("  FAIL --non-zero canonical features detected:")
        for e in canonical_errors:
            print(e)
        all_pass = False
    else:
        print(f"  PASS --reuse_success_rate=0 and gate_closed_rate=0 for all {len(videos)} videos.")

    # -----------------------------------------------------------------------
    # Check 2: Activation match (sim replay via fast_simulate)
    # -----------------------------------------------------------------------
    print("\n-- Check 2: Activation sanity (DQN gate -> sim) --")

    configs = [
        ("calib_raw", None, None),
        ("t4b_dqn_lam0.05_seed2", 0.05, 2),
        ("t4b_dqn_lam0.20_seed2", 0.20, 2),
    ]

    act_failures = []
    for config_tag, lam, seed in configs:
        print(f"\n  Config: {config_tag}")
        act_list = []
        for vid in videos:
            cat, name = vid.split("/")
            npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
            roi = npz["temporal_roi"].astype(bool)
            feat_raw = npz["features"][roi].astype(np.float32)

            if lam is None:
                # CALIB
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds = calib_model.predict(feat_raw)
                actions = np.array([ACTION_SPACE.index(p) for p in preds], dtype=np.int32)
                sim_lam = 0.20
            else:
                # DQN via DQNGate.predict (production path)
                preds = dqn_gates[lam].predict(feat_raw)
                actions = np.array([ACTION_SPACE.index(p) for p in preds], dtype=np.int32)
                sim_lam = lam

            ep = fast_simulate(npz, actions, sim_lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            act = ep["activation"]
            act_list.append(act)

            # Compare with sim per-video reference
            sim_ref = sim_pv.get((config_tag, vid))
            ref_act = sim_ref["act"] if sim_ref else None
            delta_str = ""
            if ref_act is not None:
                delta = abs(act - ref_act)
                delta_str = f"  d={delta:+.4f}"
                if delta > _ACT_TOL:
                    act_failures.append(
                        f"    {config_tag} / {vid}: gate_act={act:.4f} vs sim={ref_act:.4f} |d|={delta:.4f} > {_ACT_TOL}"
                    )
                    delta_str += " <- MISMATCH"
                else:
                    delta_str += " <- OK"

            print(f"    {vid:<45} act={act:.4f}{delta_str}")

        mean_act = float(np.mean(act_list))
        ref_macro = _SIM_REF.get(config_tag, {}).get("act")
        macro_str = ""
        if ref_macro is not None:
            macro_delta = abs(mean_act - ref_macro)
            macro_str = f"  (ref={ref_macro:.4f}, d={macro_delta:+.4f})"
        print(f"    -> mean act = {mean_act:.4f}{macro_str}")

    # -----------------------------------------------------------------------
    # Check 3: DQNGate vs DQNAgent consistency
    # -----------------------------------------------------------------------
    print("\n-- Check 3: DQNGate.predict == DQNAgent (batch) --")
    gate_agent_ok = True
    for lam in [0.05, 0.20]:
        gate = dqn_gates[lam]
        agent = dqn_agents[lam]
        for vid in videos[:1]:  # just first video
            cat, name = vid.split("/")
            npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
            roi = npz["temporal_roi"].astype(bool)
            feat_raw = npz["features"][roi].astype(np.float32)

            # DQNGate path (production: raw -> internal z-score)
            gate_preds = gate.predict(feat_raw)

            # DQNAgent path (sim: manual z-score -> dqn_actions)
            obs = (feat_raw - z_mean) / z_std
            agent_actions = dqn_actions(agent, obs, greedy=True)
            agent_preds = np.array([ACTION_SPACE[i] for i in agent_actions])

            match = np.sum(gate_preds == agent_preds)
            total = len(gate_preds)
            pct = 100.0 * match / total
            status = "PASS" if pct == 100.0 else "WARN" if pct > 99.0 else "FAIL"
            if status == "FAIL":
                gate_agent_ok = False
            print(f"  lam={lam:.2f} {vid}: {match}/{total} ({pct:.1f}%) --{status}")

    if not gate_agent_ok:
        all_pass = False

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    if act_failures:
        print("ACTIVATION MISMATCHES:")
        for f in act_failures:
            print(f)
        all_pass = False

    if all_pass:
        print("PRE-FLIGHT: PASS -- proceed to full engine run.")
        print("  Canonical features clean, activations match sim, gate consistent.")
    else:
        print("PRE-FLIGHT: FAIL -- DO NOT run full engine until fixed.")
        print("  If activation mismatch: feed canonical x_t (zero features[7,9]) to gate.")
        print("  Re-run pre-flight after fix.")

    return all_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=2,
                    help="Number of smoke-test videos (default 2, from diverse categories)")
    ap.add_argument("--all", action="store_true",
                    help="Run on all 26 test videos (thorough pre-flight)")
    args = ap.parse_args()

    with open(str(_SPLIT_FILE)) as f:
        split = json.load(f)
    test_videos = split["test"]

    if args.all:
        smoke_videos = test_videos
    else:
        # Pick diverse videos: one easy (baseline/highway), one hard (dynamicBackground/fall)
        diverse_picks = ["baseline/highway", "dynamicBackground/fall",
                         "nightVideos/bridgeEntry", "shadow/copyMachine"]
        smoke_videos = [v for v in diverse_picks if v in test_videos][:args.videos]
        if len(smoke_videos) < args.videos:
            for v in test_videos:
                if v not in smoke_videos:
                    smoke_videos.append(v)
                if len(smoke_videos) >= args.videos:
                    break

    ok = run_smoke_test(smoke_videos)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
