"""
T4 -- Train RL policy (simplified mode, W=7, matched to CALIB-DT3 baseline).

Policies
--------
linear  REINFORCE with softmax-linear policy (sanity gate)
dqn     Deep Q-Network, MLP 10->64->64->3 (main)

Lambda sweep
------------
lambda in {0.05, 0.1, 0.2, 0.4, 0.8}

Seeds
-----
>= 3 seeds; reports mean+-std in Pareto table.

Arm-matching note (T7)
----------------------
This script trains in SIMPLIFIED mode (no GUARD), matching how CALIB-DT3
baseline is evaluated.  This is P4 (CALIB raw) and P6 (RL raw).
P5 (CALIB+GUARD) and P7 (RL+GUARD) are built in T6.

Usage
-----
cd asmag-rl-vs-deterministic
python scripts/train_rl.py [options]

python scripts/train_rl.py --algos linear --epochs-linear 300
python scripts/train_rl.py --algos dqn --epochs-dqn 300 --lambdas 0.2
python scripts/train_rl.py --quick   # 50 epochs each, useful for smoke-test
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
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.agents import LinearPolicy, DQNAgent
from rl.fast_eval import (
    fast_simulate, linear_actions, dqn_actions,
    reinforce_episode, reinforce_update,
    dqn_episode_fast, evaluate_videos_fast,
)
from asmag_core.controllers import ACTION_SPACE, CONTROLLER_CATEGORY_POLICY

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W = 7               # frozen from T3.7
P_MISS = 0.5
R_HIT = 1.0
R_MAINTAIN = 0.2
R_QUIET = 0.1

LAMBDAS_DEFAULT = [0.05, 0.1, 0.2, 0.4, 0.8]
SEEDS_DEFAULT = [0, 1, 2]

_SPLIT_FILE = _REPO / "splits" / "S1_fold0.json"
_NPZ_DIR = _REPO / "data" / "features"
_TRAIN_STATS = _REPO / "data" / "features" / "train_stats.json"
_CALIB_PKL = _REPO / "models" / "calib_S1_fold0_trainonly.pkl"
_OUT_DIR = _REPO / "outputs" / "rl"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_split():
    with open(str(_SPLIT_FILE)) as f:
        return json.load(f)


def _load_train_stats():
    with open(str(_TRAIN_STATS)) as f:
        s = json.load(f)
    mean = np.array([s["mean"][k] for k in s["feature_columns"]], dtype=np.float32)
    std  = np.array([s["std"][k]  for k in s["feature_columns"]], dtype=np.float32)
    std = np.where(std < 1e-7, 1.0, std)   # constants stay at ~0 after z-score
    return mean, std


def _load_npz(video_id: str):
    cat, vid = video_id.split("/")
    p = _NPZ_DIR / cat / f"{vid}.npz"
    return np.load(str(p), allow_pickle=True)


def _load_calib_model():
    if not _CALIB_PKL.exists():
        return None
    with open(str(_CALIB_PKL), "rb") as f:
        d = pickle.load(f)
    # pkl is a dict with 'model' key holding the sklearn pipeline
    return d["model"] if isinstance(d, dict) else d


# ---------------------------------------------------------------------------
# Evaluation helpers (fast path — no VideoEnv, vectorised simulate)
# ---------------------------------------------------------------------------

def _eval_fast(policy_or_fn, videos, lambda_cost, z_mean, z_std) -> dict:
    """Evaluate any policy on videos using vectorised fast_simulate."""
    if callable(policy_or_fn) and not hasattr(policy_or_fn, "W"):
        # raw callable(obs_matrix) -> int array
        fn = policy_or_fn
    elif hasattr(policy_or_fn, "W"):
        fn = lambda obs: linear_actions(policy_or_fn, obs, greedy=True)
    else:
        fn = lambda obs: dqn_actions(policy_or_fn, obs, greedy=True)

    return evaluate_videos_fast(fn, videos, _NPZ_DIR, z_mean, z_std, lambda_cost,
                                P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)


# ---------------------------------------------------------------------------
# Baseline computation (vectorised)
# ---------------------------------------------------------------------------

def compute_baselines(train_videos: list, calib_model, z_mean, z_std, lambdas: list) -> dict:
    """Compute all fixed baselines using fast_simulate. Returns dict keyed by (name, lam)."""
    results = {}

    for lam in lambdas:
        def fn_detect(obs):
            return np.ones(len(obs), dtype=np.int32)          # idx 1 = ACC

        def fn_skip(obs):
            return np.full(len(obs), 2, dtype=np.int32)       # idx 2 = P3_FALLBACK

        results[("all-detect", lam)] = evaluate_videos_fast(
            fn_detect, train_videos, _NPZ_DIR, z_mean, z_std, lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)
        results[("all-skip", lam)] = evaluate_videos_fast(
            fn_skip, train_videos, _NPZ_DIR, z_mean, z_std, lam, P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET)

        # fixed-AUTO needs video_id; use closure over NPZ video_id
        def fn_auto_factory(lam=lam):
            def fn_auto(obs, npz_ref=None):
                # obs shape (T_roi, 10); video_id from npz is passed via wrapper
                raise RuntimeError("use _eval_auto instead")
            return fn_auto

        # fixed-AUTO: evaluate per-video manually (need category)
        ef1_l, act_l, eng_l, p3_l, rew_l = [], [], [], [], []
        for vid in train_videos:
            cat, name = vid.split("/")
            npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
            T_roi = int(npz["temporal_roi"].sum())
            mode = CONTROLLER_CATEGORY_POLICY.get(cat, "P3_FALLBACK")
            idx = ACTION_SPACE.index(mode)
            actions = np.full(T_roi, idx, dtype=np.int32)
            ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            ef1_l.append(ep["event_f1"]); act_l.append(ep["activation"])
            eng_l.append(ep["energy"]);   p3_l.append(1-ep["activation"]); rew_l.append(ep["total_reward"])
        results[("fixed-auto", lam)] = {
            "ef1": float(np.mean(ef1_l)), "ef1_std": float(np.std(ef1_l)),
            "act": float(np.mean(act_l)), "energy": float(np.mean(eng_l)),
            "p3fb": float(np.mean(p3_l)), "reward": float(np.mean(rew_l)), "n": len(ef1_l),
        }

        # CALIB-DT3: use RAW features (pipeline has StandardScaler internally;
        # evaluate_videos_fast z-scores before calling fn, so CALIB must bypass it)
        if calib_model is not None:
            ef1_l, act_l, eng_l, p3_l, rew_l = [], [], [], [], []
            for vid in train_videos:
                cat, name = vid.split("/")
                npz = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
                roi = npz["temporal_roi"].astype(bool)
                feat_raw = npz["features"][roi].astype(np.float32)   # RAW — pipeline scales internally
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds = calib_model.predict(feat_raw)
                actions = np.array([ACTION_SPACE.index(p) for p in preds], dtype=np.int32)
                ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
                ef1_l.append(ep["event_f1"]); act_l.append(ep["activation"])
                eng_l.append(ep["energy"]);   p3_l.append(1-ep["activation"]); rew_l.append(ep["total_reward"])
            results[("calib-dt3", lam)] = {
                "ef1": float(np.mean(ef1_l)), "ef1_std": float(np.std(ef1_l)),
                "act": float(np.mean(act_l)), "energy": float(np.mean(eng_l)),
                "p3fb": float(np.mean(p3_l)), "reward": float(np.mean(rew_l)), "n": len(ef1_l),
            }

    return results


# ---------------------------------------------------------------------------
# Training: Linear (REINFORCE, vectorised via fast_eval)
# ---------------------------------------------------------------------------

def train_linear(
    train_videos: list,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    lambda_cost: float,
    n_epochs: int,
    seed: int,
    out_dir: Path,
    eval_freq: int = 25,
) -> tuple[LinearPolicy, list]:
    np.random.seed(seed)
    # lr=1e-3 + entropy_coeff=0.01 prevents entropy collapse (all-skip fixed point)
    policy = LinearPolicy(obs_dim=10, n_actions=3, lr=1e-3, seed=seed)

    all_npz = {v: np.load(str(_NPZ_DIR / v.split("/")[0] / f"{v.split('/')[1]}.npz"),
                           allow_pickle=True) for v in train_videos}
    eval_log = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        order = np.random.permutation(len(train_videos))
        ep_rewards = []

        for idx in order:
            npz = all_npz[train_videos[idx]]
            obs, actions, log_probs, returns = reinforce_episode(
                policy, npz, z_mean, z_std, lambda_cost, P_MISS, W,
                R_HIT, R_MAINTAIN, R_QUIET, gamma=0.99,
            )
            reinforce_update(policy, obs, actions, returns, entropy_coeff=0.01)
            ep_rewards.append(float((returns * 0 + 1).sum()))  # count frames as proxy

        if epoch % eval_freq == 0 or epoch == n_epochs:
            metrics = evaluate_videos_fast(
                lambda obs: linear_actions(policy, obs, greedy=True),
                train_videos, _NPZ_DIR, z_mean, z_std, lambda_cost,
                P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET,
            )
            elapsed = time.time() - t0
            eval_log.append({"epoch": epoch, **metrics})
            print(f"  [{epoch:4d}/{n_epochs}] "
                  f"ef1={metrics['ef1']:.4f}  act={metrics['act']:.3f}  "
                  f"p3fb={metrics['p3fb']:.3f}  rew={metrics['reward']:.1f}  "
                  f"({elapsed:.0f}s)")

    tag = f"linear_lam{lambda_cost:.2f}_seed{seed}"
    policy.save(out_dir / f"policy_{tag}.npz")
    _save_curve(eval_log, out_dir / f"curve_{tag}.csv")
    return policy, eval_log


# ---------------------------------------------------------------------------
# Training: DQN
# ---------------------------------------------------------------------------

def train_dqn(
    train_videos: list,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    lambda_cost: float,
    n_epochs: int,
    seed: int,
    out_dir: Path,
    eval_freq: int = 25,
) -> tuple[DQNAgent, list]:
    """
    Train DQNAgent on 27 train videos.
    Returns (trained_agent, eval_log).
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

    all_npz = {v: np.load(str(_NPZ_DIR / v.split("/")[0] / f"{v.split('/')[1]}.npz"),
                           allow_pickle=True) for v in train_videos}
    eval_log = []
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        order = np.random.permutation(len(train_videos))

        for idx in order:
            npz = all_npz[train_videos[idx]]
            T_roi = int(npz["temporal_roi"].sum())
            n_updates = min(T_roi // 128, 20)
            dqn_episode_fast(agent, npz, z_mean, z_std, lambda_cost, P_MISS, W,
                             R_HIT, R_MAINTAIN, R_QUIET, n_updates=n_updates)

        agent.decay_eps(eps_per_epoch)

        if epoch % eval_freq == 0 or epoch == n_epochs:
            metrics = evaluate_videos_fast(
                lambda obs: dqn_actions(agent, obs, greedy=True),
                train_videos, _NPZ_DIR, z_mean, z_std, lambda_cost,
                P_MISS, W, R_HIT, R_MAINTAIN, R_QUIET,
            )
            elapsed = time.time() - t0
            eval_log.append({"epoch": epoch, "eps": agent.eps, **metrics})
            print(f"  [{epoch:4d}/{n_epochs}] eps={agent.eps:.3f}  "
                  f"ef1={metrics['ef1']:.4f}  act={metrics['act']:.3f}  "
                  f"p3fb={metrics['p3fb']:.3f}  rew={metrics['reward']:.1f}  "
                  f"({elapsed:.0f}s)")

    tag = f"dqn_lam{lambda_cost:.2f}_seed{seed}"
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


def _ae(ef1: float, act: float) -> float:
    return ef1 / max(act, 0.001)


def _fmt(v: float, s: float) -> str:
    return f"{v:.4f}+-{s:.4f}"


def _print_baseline_row(name, lam, m):
    ae = _ae(m["ef1"], m["act"])
    print(f"  {name:<18}  lam={lam:.2f}  ef1={m['ef1']:.4f}  act={m['act']:.3f}  "
          f"p3fb={m['p3fb']:.3f}  energy={m['energy']:.1f}  AE={ae:.4f}")


# ---------------------------------------------------------------------------
# Pareto table
# ---------------------------------------------------------------------------

def build_pareto_table(
    run_results: dict,      # (algo, lam, seed) -> metrics dict
    baselines: dict,        # (name, lam) -> metrics dict
    train_videos: list,
    lambdas: list,
    algos: list,
    seeds: list,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print()
    print("=" * 100)
    print("  LAMBDA-SWEEP PARETO TABLE (sim, train, simplified mode)")
    print("=" * 100)
    header = f"  {'Algo':<8}  {'lam':>5}  {'sim_ef1':>14}  {'activation':>12}  {'energy':>10}  {'AE':>10}  {'p3fb':>10}  vs_calib  vs_auto"
    print(header)
    print(f"  {'-'*8}  {'-'*5}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*7}")

    for algo in algos:
        for lam in lambdas:
            seed_metrics = []
            for seed in seeds:
                key = (algo, lam, seed)
                if key in run_results:
                    seed_metrics.append(run_results[key])

            if not seed_metrics:
                continue

            ef1s  = [m["ef1"]    for m in seed_metrics]
            acts  = [m["act"]    for m in seed_metrics]
            engs  = [m["energy"] for m in seed_metrics]
            p3s   = [m["p3fb"]   for m in seed_metrics]
            rews  = [m["reward"] for m in seed_metrics]

            mean_ef1, std_ef1 = np.mean(ef1s), np.std(ef1s)
            mean_act = np.mean(acts)
            mean_eng = np.mean(engs)
            mean_p3  = np.mean(p3s)
            ae       = _ae(mean_ef1, mean_act)

            calib_ef1 = baselines.get(("calib-dt3", lam), {}).get("ef1", float("nan"))
            auto_ef1  = baselines.get(("fixed-auto", lam), {}).get("ef1", float("nan"))
            d_calib = mean_ef1 - calib_ef1
            d_auto  = mean_ef1 - auto_ef1

            row = {
                "algo": algo, "lambda": lam, "n_seeds": len(seed_metrics),
                "sim_ef1_mean": mean_ef1, "sim_ef1_std": std_ef1,
                "activation_mean": mean_act, "energy_mean": mean_eng,
                "p3fb_mean": mean_p3, "AE": ae,
                "reward_mean": np.mean(rews),
                "vs_calib_ef1": d_calib, "vs_auto_ef1": d_auto,
            }
            rows.append(row)

            print(f"  {algo:<8}  {lam:>5.2f}  {_fmt(mean_ef1,std_ef1):>14}  "
                  f"{mean_act:>12.4f}  {mean_eng:>10.1f}  {ae:>10.4f}  "
                  f"{mean_p3:>10.4f}  {d_calib:>+8.4f}  {d_auto:>+7.4f}")

    # Baseline rows
    print()
    print("  --- Baselines (lam=0.2 representative) ---")
    lam_rep = 0.2
    for name in ["calib-dt3", "fixed-auto", "all-detect", "all-skip"]:
        m = baselines.get((name, lam_rep))
        if m:
            _print_baseline_row(name, lam_rep, m)

    # Save CSV
    if rows:
        csv_path = out_dir / "pareto_table.csv"
        with open(str(csv_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Pareto table saved: {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# Pareto plot
# ---------------------------------------------------------------------------

def plot_pareto(rows: list, baselines: dict, lambdas: list, algos: list, out_dir: Path):
    if not _HAS_MPL or not rows:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis
    lam_norm = {l: i / (len(lambdas) - 1) if len(lambdas) > 1 else 0.5
                for i, l in enumerate(lambdas)}
    markers = {"linear": "o", "dqn": "^", "ppo": "s"}

    for r in rows:
        color = cmap(lam_norm.get(r["lambda"], 0.5))
        marker = markers.get(r["algo"], "D")
        ax.scatter(r["activation_mean"], r["sim_ef1_mean"],
                   c=[color], marker=marker, s=80, zorder=3,
                   label=f"{r['algo']} lam={r['lambda']}")
        ax.errorbar(r["activation_mean"], r["sim_ef1_mean"],
                    yerr=r["sim_ef1_std"], fmt="none", ecolor=color, alpha=0.5)

    # Baseline points at lam=0.2
    lam_rep = 0.2
    bsl_style = {"calib-dt3": ("*", "red", 120), "fixed-auto": ("P", "blue", 100),
                 "all-detect": ("x", "gray", 80), "all-skip": ("+", "gray", 80)}
    for name, (mk, col, sz) in bsl_style.items():
        m = baselines.get((name, lam_rep))
        if m:
            ax.scatter(m["act"], m["ef1"], marker=mk, c=col, s=sz, zorder=5,
                       label=f"{name} (lam=0.2)", edgecolors="black" if mk not in "x+" else "none")

    # Colorbar for lambda
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(lambdas), vmax=max(lambdas)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="lambda")

    ax.set_xlabel("Activation (YOLO call rate)", fontsize=10)
    ax.set_ylabel("Sim EF1 (train, simplified mode)", fontsize=10)
    ax.set_title("T4 Lambda-Sweep Pareto: RL vs Baselines (P6 raw, no GUARD)", fontsize=11)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2,
              loc="lower right", framealpha=0.8)

    path = out_dir / "pareto_plot.png"
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot saved: {path}")


# ---------------------------------------------------------------------------
# Sanity gate
# ---------------------------------------------------------------------------

def check_sanity_gate(run_results: dict, baselines: dict, lambdas: list, seeds: list):
    """
    Sanity gate: for each lambda, check that Linear beats fixed-AUTO on reward
    and has EF1 > all-SKIP.  Must pass for at least one lambda.
    """
    print()
    print("=" * 70)
    print("  SANITY GATE: Linear policy vs baselines")
    print("=" * 70)

    any_pass = False
    for lam in lambdas:
        linear_metrics = [run_results.get(("linear", lam, s)) for s in seeds
                          if ("linear", lam, s) in run_results]
        if not linear_metrics:
            continue

        mean_ef1 = np.mean([m["ef1"] for m in linear_metrics])
        mean_rew = np.mean([m["reward"] for m in linear_metrics])
        mean_act = np.mean([m["act"] for m in linear_metrics])
        mean_p3  = np.mean([m["p3fb"] for m in linear_metrics])

        skip_ef1  = baselines.get(("all-skip",   lam), {}).get("ef1", 0.0)
        auto_rew  = baselines.get(("fixed-auto", lam), {}).get("reward", float("-inf"))
        auto_ef1  = baselines.get(("fixed-auto", lam), {}).get("ef1", 0.0)
        skip_rew  = baselines.get(("all-skip",   lam), {}).get("reward", float("-inf"))
        calib_ef1 = baselines.get(("calib-dt3",  lam), {}).get("ef1", float("nan"))
        detect_ef1= baselines.get(("all-detect", lam), {}).get("ef1", float("nan"))

        pass_ef1 = mean_ef1 > skip_ef1 + 0.01    # EF1 must beat all-SKIP
        pass_rew = mean_rew > auto_rew            # reward must beat fixed-AUTO

        gate = pass_ef1 and pass_rew
        if gate:
            any_pass = True

        status = "PASS" if gate else "FAIL"
        print(f"\n  lam={lam:.2f}  [{status}]")
        print(f"    Linear:      ef1={mean_ef1:.4f}  act={mean_act:.3f}  p3fb={mean_p3:.3f}  reward={mean_rew:.2f}")
        print(f"    all-SKIP:    ef1={skip_ef1:.4f}  reward={skip_rew:.2f}")
        print(f"    fixed-AUTO:  ef1={auto_ef1:.4f}  reward={auto_rew:.2f}")
        print(f"    CALIB-DT3:   ef1={calib_ef1:.4f}")
        print(f"    all-DETECT:  ef1={detect_ef1:.4f}")
        print(f"    EF1 > all-SKIP+0.01?  {'PASS' if pass_ef1 else 'FAIL'}  ({mean_ef1:.4f} vs {skip_ef1+0.01:.4f})")
        print(f"    reward > fixed-AUTO?  {'PASS' if pass_rew else 'FAIL'}  ({mean_rew:.2f} vs {auto_rew:.2f})")

    if any_pass:
        print("\n  SANITY GATE: PASS -- Linear beats fixed-AUTO+all-SKIP on at least one lambda.")
        print("  Proceeding to DQN is justified.")
    else:
        print("\n  SANITY GATE: FAIL -- Linear cannot beat fixed-AUTO on any lambda.")
        print("  STOP. Check reward/setup before running DQN.")

    return any_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T4 RL training for ASMAG gate control")
    parser.add_argument("--algos", nargs="+", default=["linear", "dqn"],
                        choices=["linear", "dqn"],
                        help="Algorithms to train (default: linear dqn)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS_DEFAULT,
                        help="Lambda values to sweep (default: 0.05 0.1 0.2 0.4 0.8)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT,
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--epochs-linear", type=int, default=300,
                        help="Training epochs for linear (default: 300)")
    parser.add_argument("--epochs-dqn", type=int, default=300,
                        help="Training epochs for DQN (default: 300)")
    parser.add_argument("--eval-freq", type=int, default=25,
                        help="Eval every N epochs (default: 25)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 50 epochs each, seeds=[0], lambdas=[0.1,0.2,0.4]")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (algo, lam, seed) if output file already exists")
    args = parser.parse_args()

    if args.quick:
        args.epochs_linear = 50
        args.epochs_dqn = 50
        args.seeds = [0]
        args.lambdas = [0.1, 0.2, 0.4]
        args.eval_freq = 10

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("  T4 -- RL Policy Training")
    print(f"  W={W}  P_miss={P_MISS}  (simplified mode, no GUARD)")
    print(f"  Algos: {args.algos}")
    print(f"  Lambdas: {args.lambdas}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Epochs linear={args.epochs_linear}  dqn={args.epochs_dqn}")
    print("=" * 70)

    # Load data
    split = _load_split()
    train_videos = split["train"]
    z_mean, z_std = _load_train_stats()
    calib_model = _load_calib_model()
    if calib_model is None:
        print("  WARNING: CALIB pkl not found; CALIB-DT3 baseline will be skipped.")
    else:
        print(f"  Loaded CALIB model: {_CALIB_PKL.name}")

    print(f"\n  Train videos: {len(train_videos)}")
    print(f"  Output dir: {_OUT_DIR}")

    # ------------------------------------------------------------------
    # Baselines
    print("\n  Computing baselines...")
    baselines = compute_baselines(train_videos, calib_model, z_mean, z_std, args.lambdas)
    print("  Baselines at lam=0.2 (representative):")
    lam_rep = 0.2 if 0.2 in args.lambdas else args.lambdas[0]
    for name in ["calib-dt3", "fixed-auto", "all-detect", "all-skip"]:
        m = baselines.get((name, lam_rep))
        if m:
            _print_baseline_row(name, lam_rep, m)

    # Save baselines
    bsl_rows = []
    for (name, lam), m in baselines.items():
        bsl_rows.append({"name": name, "lambda": lam, **{k: v for k, v in m.items()}})
    bsl_rows.sort(key=lambda r: (r["name"], r["lambda"]))
    bsl_path = _OUT_DIR / "baselines.csv"
    if bsl_rows:
        with open(str(bsl_path), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(bsl_rows[0].keys()))
            w.writeheader()
            w.writerows(bsl_rows)
    print(f"  Baselines saved: {bsl_path}")

    # ------------------------------------------------------------------
    # Training
    run_results: dict = {}

    for algo in args.algos:
        for lam in args.lambdas:
            for seed in args.seeds:
                tag = f"{algo}_lam{lam:.2f}_seed{seed}"
                out_tag = (
                    _OUT_DIR / f"policy_{tag}.npz"
                    if algo == "linear" else
                    _OUT_DIR / f"policy_{tag}.pt"
                )

                if args.skip_existing and out_tag.exists():
                    print(f"\n  SKIP (already exists): {tag}")
                    # Load final eval from curve
                    curve_path = _OUT_DIR / f"curve_{tag}.csv"
                    if curve_path.exists():
                        with open(str(curve_path)) as f:
                            rows_ = list(csv.DictReader(f))
                        if rows_:
                            last = rows_[-1]
                            run_results[(algo, lam, seed)] = {
                                k: float(last[k]) for k in ["ef1", "act", "energy", "p3fb", "reward"]
                                if k in last
                            }
                    continue

                print(f"\n  Training: {tag}")
                print(f"  lambda={lam}  seed={seed}  epochs={args.epochs_linear if algo=='linear' else args.epochs_dqn}")

                if algo == "linear":
                    _, log = train_linear(
                        train_videos, z_mean, z_std, lam,
                        n_epochs=args.epochs_linear, seed=seed,
                        out_dir=_OUT_DIR, eval_freq=args.eval_freq
                    )
                else:
                    _, log = train_dqn(
                        train_videos, z_mean, z_std, lam,
                        n_epochs=args.epochs_dqn, seed=seed,
                        out_dir=_OUT_DIR, eval_freq=args.eval_freq
                    )

                if log:
                    last = log[-1]
                    run_results[(algo, lam, seed)] = {
                        k: last[k] for k in ["ef1", "act", "energy", "p3fb", "reward"]
                        if k in last
                    }

    # ------------------------------------------------------------------
    # Sanity gate (check after linear is done)
    if "linear" in args.algos:
        sanity_ok = check_sanity_gate(run_results, baselines, args.lambdas, args.seeds)
        if not sanity_ok:
            print("\n  Sanity gate FAILED. Review reward setup before interpreting DQN results.")

    # ------------------------------------------------------------------
    # Pareto table
    pareto_rows = build_pareto_table(
        run_results, baselines, train_videos, args.lambdas, args.algos, args.seeds, _OUT_DIR
    )

    # Skip-rate summary for T5
    print()
    print("=" * 70)
    print("  P3_FALLBACK RATE SUMMARY (for T5 skip-rate vs DT3 check)")
    print("=" * 70)
    calib_p3 = baselines.get(("calib-dt3", lam_rep), {}).get("p3fb", float("nan"))
    print(f"  CALIB-DT3 p3fb rate (lam={lam_rep}): {calib_p3:.4f}")
    for algo in args.algos:
        for lam in args.lambdas:
            seed_p3 = [run_results.get((algo, lam, s), {}).get("p3fb", float("nan"))
                       for s in args.seeds]
            seed_p3 = [v for v in seed_p3 if not np.isnan(v)]
            if seed_p3:
                mean_p3 = np.mean(seed_p3)
                delta = mean_p3 - calib_p3
                flag = " *** >+0.20 -- CHECK MechB" if delta > 0.20 else ""
                print(f"  {algo:<8} lam={lam:.2f}  p3fb={mean_p3:.4f}  vs_calib={delta:+.4f}{flag}")

    # Pareto plot
    if not args.no_plot:
        plot_pareto(pareto_rows, baselines, args.lambdas, args.algos, _OUT_DIR)

    print("\n  T4 complete.")


if __name__ == "__main__":
    main()
