"""
T4b -- Fair generalization shot for DQN.

Addresses three fairness failures in T4:
  1. Checkpoint selected by val EF1 (not final epoch or train-monitored)
  2. Adam weight_decay=1e-4 (light L2)
  3. Val set carved stratified from train (1 per category)
     → val=11 videos, new_train=16, test=26 (unchanged)

Usage:
    python scripts/train_rl_fair.py                        # all 5 lambda, 3 seeds
    python scripts/train_rl_fair.py --lambdas 0.2          # single lambda
    python scripts/train_rl_fair.py --epochs 400 --seeds 0 1 2

Outputs:
    outputs/rl/t4b/policy_fair_dqn_lam{L}_seed{S}.pt      # best-val checkpoint
    outputs/rl/t4b/curve_fair_dqn_lam{L}_seed{S}.csv
    outputs/rl/t4b/t4b_results.csv                         # test EF1/act, all seeds
    outputs/rl/t4b/t4b_comparison.txt                      # fair vs current vs CALIB
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.agents import DQNAgent
from rl.fast_eval import fast_simulate, dqn_actions, evaluate_videos_fast
from asmag_core.controllers import ACTION_SPACE

import torch

# ---------------------------------------------------------------------------
# Constants — frozen from T3.7 / T4
# ---------------------------------------------------------------------------

W = 7
P_MISS = 0.5
R_HIT = 1.0
R_MAINTAIN = 0.2
R_QUIET = 0.1

LAMBDAS_DEFAULT = [0.05, 0.1, 0.2, 0.4, 0.8]
SEEDS_DEFAULT   = [0, 1, 2]
EPOCHS_DEFAULT  = 400        # slightly more than T4 (300) to compensate fewer train videos

_SPLIT_FILE   = _REPO / "splits" / "S1_fold0.json"
_NPZ_DIR      = _REPO / "data" / "features"
_TRAIN_STATS  = _REPO / "data" / "features" / "train_stats.json"
_CALIB_PKL    = _REPO / "models" / "calib_S1_fold0_trainonly.pkl"
_RL_OUT       = _REPO / "outputs" / "rl"
_T4B_OUT      = _REPO / "outputs" / "rl" / "t4b"
_T4_CURVE_DIR = _RL_OUT   # T4 curves live here

VAL_CARVE_SEED = 42   # frozen; determines which video goes to val per category


# ---------------------------------------------------------------------------
# Data loading
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
    if not _CALIB_PKL.exists():
        return None
    with open(str(_CALIB_PKL), "rb") as f:
        d = pickle.load(f)
    return d["model"] if isinstance(d, dict) else d


def carve_val_from_train(train_videos: list, by_category: dict) -> tuple[list, list]:
    """
    Stratified carve: 1 video per category → val; rest → new_train.
    Deterministic via VAL_CARVE_SEED=42.
    Returns (new_train, val).
    """
    rng = np.random.RandomState(VAL_CARVE_SEED)
    val_set, new_train = [], []
    for cat in sorted(by_category.keys()):
        tvids = by_category[cat]["train"]
        idx = rng.randint(len(tvids))
        val_set.append(tvids[idx])
        new_train.extend([v for i, v in enumerate(tvids) if i != idx])
    return new_train, val_set


# ---------------------------------------------------------------------------
# Fair DQN training (val-selected checkpoint + weight_decay)
# ---------------------------------------------------------------------------

def train_dqn_fair(
    new_train: list,
    val_videos: list,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    lambda_cost: float,
    n_epochs: int,
    seed: int,
    out_dir: Path,
    eval_freq: int = 25,
    weight_decay: float = 1e-4,
) -> tuple[DQNAgent, list]:
    """
    DQN with val-based checkpoint selection + L2 regularisation.
    Best checkpoint (by val EF1) is saved as the final policy file.
    """
    np.random.seed(seed)

    eps_start = 1.0
    eps_end = 0.05
    decay_epochs = max(1, int(n_epochs * 0.6))
    eps_per_epoch = (eps_start - eps_end) / decay_epochs

    agent = DQNAgent(
        obs_dim=10, n_actions=3, hidden=64, lr=1e-3,
        gamma=0.99, batch_size=128, buffer_capacity=100_000,
        target_update_freq=200, eps_start=eps_start, eps_end=eps_end,
        seed=seed,
    )
    # Patch Adam to add weight_decay
    import torch.optim as optim
    agent.opt = optim.Adam(agent.q.parameters(), lr=1e-3, weight_decay=weight_decay)

    all_npz = {v: np.load(str(_NPZ_DIR / v.split("/")[0] / f"{v.split('/')[1]}.npz"),
                           allow_pickle=True) for v in new_train}
    eval_log = []
    best_val_ef1 = -1.0
    best_ckpt = None
    best_epoch = -1
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        order = np.random.permutation(len(new_train))

        for idx in order:
            npz = all_npz[new_train[idx]]
            T_roi = int(npz["temporal_roi"].sum())
            n_updates = min(T_roi // 128, 20)
            from rl.fast_eval import dqn_episode_fast
            dqn_episode_fast(agent, npz, z_mean, z_std, lambda_cost, P_MISS, W,
                             R_HIT, R_MAINTAIN, R_QUIET, n_updates=n_updates)

        agent.decay_eps(eps_per_epoch)

        if epoch % eval_freq == 0 or epoch == n_epochs:
            train_m = evaluate_videos_fast(
                lambda obs: dqn_actions(agent, obs, greedy=True),
                new_train, _NPZ_DIR, z_mean, z_std, lambda_cost,
                P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET,
            )
            val_m = evaluate_videos_fast(
                lambda obs: dqn_actions(agent, obs, greedy=True),
                val_videos, _NPZ_DIR, z_mean, z_std, lambda_cost,
                P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET,
            )
            elapsed = time.time() - t0
            best_marker = ""
            if val_m["ef1"] > best_val_ef1:
                best_val_ef1 = val_m["ef1"]
                best_epoch = epoch
                best_ckpt = {k: v.cpu().clone() for k, v in agent.q.state_dict().items()}
                best_marker = " *"

            eval_log.append({
                "epoch": epoch, "eps": agent.eps,
                "train_ef1": train_m["ef1"], "train_act": train_m["act"],
                "val_ef1": val_m["ef1"],   "val_act": val_m["act"],
                "best_epoch": best_epoch,
            })
            print(f"  [{epoch:4d}/{n_epochs}] "
                  f"tr_ef1={train_m['ef1']:.4f} act={train_m['act']:.3f} | "
                  f"val_ef1={val_m['ef1']:.4f} act={val_m['act']:.3f}{best_marker} "
                  f"({elapsed:.0f}s)")

    # Restore best-val checkpoint before saving
    if best_ckpt is not None:
        agent.q.load_state_dict(best_ckpt)
        agent.q_target.load_state_dict(best_ckpt)
        print(f"  -> Restored best-val checkpoint from epoch {best_epoch} "
              f"(val_ef1={best_val_ef1:.4f})")

    tag = f"fair_dqn_lam{lambda_cost:.2f}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    agent.save(out_dir / f"policy_{tag}.pt")
    _save_curve(eval_log, out_dir / f"curve_{tag}.csv")
    return agent, eval_log


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_curve(log: list, path: Path):
    if not log:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)


def _calib_sim_ef1(videos: list, calib_model, lam: float) -> tuple[float, float, float, float]:
    ef1_l, act_l = [], []
    for vid in videos:
        cat, name = vid.split("/")
        npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
        roi = npz["temporal_roi"].astype(bool)
        feat = npz["features"][roi].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = calib_model.predict(feat)
        actions = np.array([ACTION_SPACE.index(p) for p in preds], dtype=np.int32)
        ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
        ef1_l.append(ep["event_f1"]); act_l.append(ep["activation"])
    return float(np.mean(ef1_l)), float(np.std(ef1_l)), float(np.mean(act_l)), float(np.std(act_l))


def _load_t4_final_ef1(lam: float, seed: int) -> tuple[float, float]:
    """Load final-epoch EF1 and act from T4 training curve (TRAIN eval, not test)."""
    path = _T4_CURVE_DIR / f"curve_dqn_lam{lam:.2f}_seed{seed}.csv"
    if not path.exists():
        return float("nan"), float("nan")
    with open(str(path)) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return float(last["ef1"]), float(last["act"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS_DEFAULT)
    ap.add_argument("--seeds",   type=int,   nargs="+", default=SEEDS_DEFAULT)
    ap.add_argument("--epochs",  type=int,   default=EPOCHS_DEFAULT)
    ap.add_argument("--eval-freq", type=int, default=25)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--compare-only", action="store_true",
                    help="Skip training; just load saved T4b policies and compare to T4.")
    args = ap.parse_args()

    split = _load_split()
    new_train, val_videos = carve_val_from_train(split["train"], split["by_category"])
    test_videos = split["test"]
    z_mean, z_std = _load_train_stats()
    calib_model = _load_calib_model()
    _T4B_OUT.mkdir(parents=True, exist_ok=True)

    print(f"T4b fair-shot DQN: new_train={len(new_train)}, val={len(val_videos)}, "
          f"test={len(test_videos)}, epochs={args.epochs}, wd={args.weight_decay}")
    print(f"  new_train: {sorted(new_train)}")
    print(f"  val:       {sorted(val_videos)}")

    # ------------------------------------------------------------------
    # CALIB reference (once, lambda-independent for EF1/act)
    # ------------------------------------------------------------------
    ref_lam = 0.20   # reference lambda for reward sim
    if calib_model is not None:
        calib_train_ef1, _, calib_train_act, _ = _calib_sim_ef1(split["train"], calib_model, ref_lam)
        calib_test_ef1,  _, calib_test_act,  _ = _calib_sim_ef1(test_videos,    calib_model, ref_lam)
        print(f"\nCALIB-DT3 (ref):  train_ef1={calib_train_ef1:.4f} act={calib_train_act:.4f} | "
              f"test_ef1={calib_test_ef1:.4f} act={calib_test_act:.4f}")

    # ------------------------------------------------------------------
    # Train / load T4b policies
    # ------------------------------------------------------------------
    all_results = []   # rows for t4b_results.csv

    for lam in args.lambdas:
        for seed in args.seeds:
            tag = f"fair_dqn_lam{lam:.2f}_seed{seed}"
            policy_path = _T4B_OUT / f"policy_{tag}.pt"

            if args.compare_only and not policy_path.exists():
                print(f"  [skip] {policy_path} not found")
                continue

            if not args.compare_only:
                print(f"\nTraining T4b: lam={lam}  seed={seed}  epochs={args.epochs}")
                agent, _ = train_dqn_fair(
                    new_train, val_videos, z_mean, z_std,
                    lam, args.epochs, seed, _T4B_OUT,
                    eval_freq=args.eval_freq,
                    weight_decay=args.weight_decay,
                )
            else:
                agent = DQNAgent(seed=seed)
                agent.load(str(policy_path))
                agent.eps = 0.0

            fn = lambda obs, a=agent: dqn_actions(a, obs, greedy=True)

            # Evaluate on full 27-video train, val subset, and test
            train_m = evaluate_videos_fast(fn, split["train"], _NPZ_DIR, z_mean, z_std,
                                           lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)
            val_m   = evaluate_videos_fast(fn, val_videos,  _NPZ_DIR, z_mean, z_std,
                                           lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)
            test_m  = evaluate_videos_fast(fn, test_videos, _NPZ_DIR, z_mean, z_std,
                                           lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)

            t4_train_ef1, t4_train_act = _load_t4_final_ef1(lam, seed)

            row = {
                "algo": "t4b_dqn", "lambda": lam, "seed": seed,
                "t4_train_ef1": f"{t4_train_ef1:.4f}",
                "t4b_train_ef1": f"{train_m['ef1']:.4f}",
                "t4b_val_ef1":   f"{val_m['ef1']:.4f}",
                "t4b_test_ef1":  f"{test_m['ef1']:.4f}",
                "t4b_test_act":  f"{test_m['act']:.4f}",
                "t4b_test_ef1_std": f"{test_m['ef1_std']:.4f}",
            }
            all_results.append(row)
            print(f"  Result lam={lam} seed={seed}: "
                  f"t4_tr={t4_train_ef1:.4f} | t4b_tr={train_m['ef1']:.4f} "
                  f"val={val_m['ef1']:.4f} TEST={test_m['ef1']:.4f} act={test_m['act']:.4f}")

    # ------------------------------------------------------------------
    # Save results CSV
    # ------------------------------------------------------------------
    if all_results:
        out_csv = _T4B_OUT / "t4b_results.csv"
        with open(str(out_csv), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {out_csv}")

    # ------------------------------------------------------------------
    # Comparison report
    # ------------------------------------------------------------------
    if all_results and calib_model is not None:
        report_lines = []
        report_lines.append("T4b Fair-Shot vs T4 vs CALIB -- Test Sim EF1 (lam_sim=0.20)")
        report_lines.append("=" * 70)
        report_lines.append(f"{'Config':<32} {'TEST EF1':>9} {'TEST act':>9} {'vs CALIB':>9}")
        report_lines.append("-" * 62)

        # CALIB line
        report_lines.append(f"{'CALIB-DT3 (train=0.7883)':<32} {calib_test_ef1:>9.4f} {calib_test_act:>9.4f} {'---':>9}")

        # T4 vs T4b per lambda
        for lam in args.lambdas:
            t4b_rows = [r for r in all_results if float(r["lambda"]) == lam]
            if not t4b_rows:
                continue
            t4b_test_ef1s = [float(r["t4b_test_ef1"]) for r in t4b_rows]
            t4b_test_acts = [float(r["t4b_test_act"]) for r in t4b_rows]
            t4_train_ef1s = [float(r["t4_train_ef1"]) for r in t4b_rows]

            # T4 test sim (from Phase 1 t5_sim_results if available)
            t4_test_csv = _RL_OUT / "t5_sim_results.csv"
            t4_test_ef1 = float("nan")
            if t4_test_csv.exists():
                seed_rep = 2   # representative seed from T4
                tag = f"dqn_lam{lam:.2f}_seed{seed_rep}"
                with open(str(t4_test_csv)) as ff:
                    for rr in csv.DictReader(ff):
                        if rr["config"] == tag:
                            t4_test_ef1 = float(rr["sim_ef1"])

            mean_t4b_test = float(np.mean(t4b_test_ef1s))
            mean_t4b_act  = float(np.mean(t4b_test_acts))
            vs_calib = mean_t4b_test - calib_test_ef1 if calib_model else float("nan")

            report_lines.append(f"  DQN lam={lam:.2f} T4  (seed=2 repr) {'':>5} {t4_test_ef1:>9.4f} {'---':>9} {t4_test_ef1-calib_test_ef1:>+9.4f}")
            report_lines.append(f"  DQN lam={lam:.2f} T4b (mean 3s){'':>7} {mean_t4b_test:>9.4f} {mean_t4b_act:>9.4f} {vs_calib:>+9.4f}")
            report_lines.append("")

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        report_path = _T4B_OUT / "t4b_comparison.txt"
        with open(str(report_path), "w") as f:
            f.write(report_text + "\n")
        print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
