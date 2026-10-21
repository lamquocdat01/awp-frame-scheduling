"""
Vectorized simulation and policy evaluation for T4 training speed.

fast_simulate() replicates simulate() via numpy ops (no Python per-frame loop).
Speedup over simulate(): ~30-80x for typical CDnet videos.

Correctness: matches simulate() output exactly, including:
  - W=7 staleness window
  - MechA T3.7 fix (quiet chain FP via is_fresh)
  - Simplified mode (gate_open=True for cost)
  - event_caught state machine (caught_before vectorized per span)
"""

from __future__ import annotations
import numpy as np

from asmag_core.controllers import ACTION_SPACE
from asmag_core.metrics import event_f1_from_rows


# ---------------------------------------------------------------------------
# Core vectorized simulator
# ---------------------------------------------------------------------------

def _spans_np(mask: np.ndarray):
    """Vectorized span finder — returns (starts, ends) arrays (inclusive)."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return starts, ends


def fast_simulate(
    npz_data,
    actions_roi,           # int array (T_roi,): 0=FAST 1=ACC 2=P3_FALLBACK
    lambda_cost: float,
    P_miss: float,
    R_hit: float = 1.0,
    R_maintain: float = 0.2,
    R_quiet: float = 0.1,
    W: int = 7,
) -> dict:
    """
    Vectorized episode simulation.  Drops the Python per-frame loop in simulate().

    Parameters identical to simulate() except:
      - actions_roi is T_roi-length int array (not T_total action sequence)
      - No yolo_called_sequence support (simplified mode only)
      - Returns same episode dict as simulate() + 'per_frame_reward' array
    """
    roi = npz_data["temporal_roi"].astype(bool)
    gt_event = npz_data["gt_event"][roi].astype(bool)
    fidx = npz_data["frame_idx"][roi].astype(np.int64)
    T = len(gt_event)

    actions_roi = np.asarray(actions_roi, dtype=np.int32)
    is_det = (actions_roi == 0) | (actions_roi == 1)   # FAST=0 or ACC=1

    # ------------------------------------------------------------------
    # 1. Vectorised last_detect_fidx and is_fresh
    # ------------------------------------------------------------------
    if W is None:
        is_fresh = np.ones(T, dtype=bool)
    elif T == 0:
        is_fresh = np.empty(0, dtype=bool)
    else:
        initial_fdx = int(fidx[0]) - W - 1
        det_pos = np.where(is_det)[0]
        if len(det_pos) == 0:
            last_det_fidx = np.full(T, initial_fdx, dtype=np.int64)
        else:
            # For each frame t, find index of last detect in det_pos that is <= t
            idx = np.searchsorted(det_pos, np.arange(T), side="right") - 1
            # idx[t] == -1 means no prior detect
            last_det_fidx = np.where(
                idx >= 0,
                fidx[det_pos[np.maximum(idx, 0)]],
                initial_fdx,
            )
        is_fresh = (fidx - last_det_fidx) <= W

    # ------------------------------------------------------------------
    # 2. Per-frame R_acc and detection_available
    # ------------------------------------------------------------------
    detect_avail = np.zeros(T, dtype=bool)
    R_acc = np.full(T, -P_miss, dtype=np.float32)

    # Quiet frames
    quiet = ~gt_event
    detect_avail[quiet] = is_det[quiet] | is_fresh[quiet]   # MechA T3.7
    R_acc[quiet] = np.where(is_det[quiet], np.float32(0.0), np.float32(R_quiet))

    # Event spans — vectorised within each span
    starts, ends = _spans_np(gt_event)
    for s, e in zip(starts, ends):
        sd = is_det[s:e + 1]        # span detects
        sf = is_fresh[s:e + 1]      # span freshness
        cum = np.cumsum(sd)
        # caught_before[t] = True if any detect in span strictly before t
        caught_before = np.concatenate([[False], cum[:-1] > 0])

        detect_avail[s:e + 1] = sd | (caught_before & sf)

        # R_acc: first_catch | re-detect | maintain | miss
        first_catch = sd & ~caught_before
        redet       = sd & caught_before
        maintain    = (~sd) & caught_before & sf
        # everything else → -P_miss (already initialised)
        R_acc[s:e + 1] = np.where(first_catch, np.float32(R_hit),
                          np.where(redet, np.float32(0.0),
                          np.where(maintain, np.float32(R_maintain),
                                   np.float32(-P_miss))))

    # ------------------------------------------------------------------
    # 3. EF1
    # ------------------------------------------------------------------
    event_tp = int(np.sum(gt_event & detect_avail))
    event_fn = int(np.sum(gt_event & ~detect_avail))
    event_fp = int(np.sum(quiet & detect_avail))

    ef1_dict = event_f1_from_rows([{
        "Event_TP": event_tp, "Event_FP": event_fp,
        "Event_FN": event_fn, "Event_TN": 0,
    }])

    # ------------------------------------------------------------------
    # 4. Energy and reward (simplified: gate_open=True)
    # ------------------------------------------------------------------
    costs = np.where(is_det, np.float32(1.0), np.float32(0.5))
    per_frame_reward = R_acc - lambda_cost * costs
    activation = float(is_det.mean()) if T > 0 else 0.0
    energy = float(costs.sum())
    total_reward = float(per_frame_reward.sum())

    return {
        "per_frame_reward": per_frame_reward,
        "episode": {
            "event_f1": ef1_dict["event_f1"],
            "event_tp": event_tp, "event_fp": event_fp, "event_fn": event_fn,
            "event_precision": ef1_dict["event_precision"],
            "event_recall":    ef1_dict["event_recall"],
            "activation": activation,
            "energy": energy,
            "total_reward": total_reward,
        },
    }


# ---------------------------------------------------------------------------
# Vectorised action selection
# ---------------------------------------------------------------------------

def linear_actions(policy, obs_matrix: np.ndarray, greedy: bool = True) -> np.ndarray:
    """Return int action array for a LinearPolicy on pre-z-scored obs_matrix."""
    logits = obs_matrix @ policy.W.T + policy.b   # (T_roi, n_actions)
    if greedy:
        return logits.argmax(axis=1).astype(np.int32)
    lg = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(lg)
    probs /= probs.sum(axis=1, keepdims=True)
    cum = probs.cumsum(axis=1)
    u = np.random.uniform(size=len(obs_matrix))
    return (u[:, np.newaxis] < cum).argmax(axis=1).astype(np.int32)


def dqn_actions(agent, obs_matrix: np.ndarray, greedy: bool = True) -> np.ndarray:
    """Batch Q-network forward pass → action array."""
    import torch
    with torch.no_grad():
        q = agent.q(torch.FloatTensor(obs_matrix))   # (T_roi, 3)
        greedy_acts = q.argmax(dim=1).numpy().astype(np.int32)
    if greedy:
        return greedy_acts
    rand_acts = np.random.randint(0, agent.n_actions, size=len(obs_matrix), dtype=np.int32)
    use_rand = np.random.uniform(size=len(obs_matrix)) < agent.eps
    return np.where(use_rand, rand_acts, greedy_acts).astype(np.int32)


# ---------------------------------------------------------------------------
# Episode rollout for REINFORCE (LinearPolicy, fully vectorised)
# ---------------------------------------------------------------------------

def reinforce_episode(policy, npz_data, z_mean, z_std, lambda_cost, P_miss, W,
                      R_hit=1.0, R_maintain=0.2, R_quiet=0.1, gamma=0.99):
    """
    Vectorised REINFORCE episode for LinearPolicy.
    Returns (obs_matrix, actions, log_probs, returns) — no VideoEnv needed.
    """
    roi = npz_data["temporal_roi"].astype(bool)
    feat = npz_data["features"][roi].astype(np.float32)
    obs = (feat - z_mean) / z_std
    T = len(obs)

    # Stochastic action sampling
    logits = obs @ policy.W.T + policy.b
    lg = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(lg)
    probs /= probs.sum(axis=1, keepdims=True)
    u = np.random.uniform(size=T)
    actions = (u[:, np.newaxis] < probs.cumsum(axis=1)).argmax(axis=1).astype(np.int32)
    log_probs = np.log(probs[np.arange(T), actions] + 1e-8)

    # Fast simulate → per_frame_reward
    result = fast_simulate(npz_data, actions, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W)
    rewards = result["per_frame_reward"]

    # Discounted returns
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    for t in reversed(range(T)):
        G = float(rewards[t]) + gamma * G
        returns[t] = G
    if returns.std() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return obs, actions, log_probs, returns


def reinforce_update(policy, obs, actions, returns, entropy_coeff: float = 0.01):
    """
    Vectorised REINFORCE gradient step for LinearPolicy.

    entropy_coeff: entropy regularisation weight.  Prevents entropy collapse
    (all-skip fixed point where score → 0).  0.01 is a safe default.
    """
    T = len(obs)
    # Batch softmax
    logits = obs @ policy.W.T + policy.b        # (T, K)
    lg = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(lg)
    probs /= probs.sum(axis=1, keepdims=True)   # (T, K)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(T), actions] = 1.0

    # Policy gradient
    score = (one_hot - probs) * returns[:, np.newaxis]   # (T, K)
    dW = score.T @ obs / T          # (K, obs_dim)
    db = score.sum(axis=0) / T      # (K,)

    # Entropy gradient: ∂H/∂θ via -(log π(a) + 1) * score
    if entropy_coeff > 0.0:
        log_taken = np.log(probs[np.arange(T), actions] + 1e-8)   # (T,)
        H_weight = -(log_taken + 1.0)                              # (T,)
        H_score = H_weight[:, np.newaxis] * (one_hot - probs)      # (T, K)
        dW += entropy_coeff * (H_score.T @ obs) / T
        db += entropy_coeff * H_score.sum(axis=0) / T

    policy.W += policy.lr * dW
    policy.b += policy.lr * db


# ---------------------------------------------------------------------------
# Fast DQN episode (vectorised action selection + batch replay push)
# ---------------------------------------------------------------------------

def dqn_episode_fast(agent, npz_data, z_mean, z_std, lambda_cost, P_miss, W,
                     R_hit=1.0, R_maintain=0.2, R_quiet=0.1, n_updates=10):
    """
    One DQN episode using batch Q-network inference + fast_simulate.
    Pushes all transitions to replay buffer then does n_updates gradient steps.
    Returns (total_reward, losses).
    """
    roi = npz_data["temporal_roi"].astype(bool)
    feat = npz_data["features"][roi].astype(np.float32)
    obs = (feat - z_mean) / z_std
    T = len(obs)

    # Vectorised ε-greedy
    import torch
    with torch.no_grad():
        q = agent.q(torch.FloatTensor(obs))
        greedy = q.argmax(dim=1).numpy().astype(np.int32)
    rand = np.random.randint(0, agent.n_actions, size=T, dtype=np.int32)
    use_rand = np.random.uniform(size=T) < agent.eps
    actions = np.where(use_rand, rand, greedy)

    # Fast simulate → per_frame_reward
    result = fast_simulate(npz_data, actions, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W)
    rewards = result["per_frame_reward"]

    # Vectorised batch push (~10x faster than per-frame loop)
    s_next = np.vstack([obs[1:], np.zeros((1, obs.shape[1]), dtype=np.float32)])
    dones = np.zeros(T, dtype=np.float32); dones[-1] = 1.0
    agent.push_batch(obs, actions, rewards.astype(np.float32), s_next, dones)

    # Gradient updates
    losses = []
    for _ in range(n_updates):
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

    return float(rewards.sum()), losses


# ---------------------------------------------------------------------------
# Batch evaluation over multiple videos
# ---------------------------------------------------------------------------

def evaluate_videos_fast(
    get_actions_fn,    # callable(obs_matrix) -> int array (T_roi,)
    videos: list,
    npz_dir,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    lambda_cost: float,
    P_miss: float = 0.5,
    W: int = 7,
    R_hit: float = 1.0,
    R_maintain: float = 0.2,
    R_quiet: float = 0.1,
) -> dict:
    """Evaluate policy on a list of video IDs; return mean metrics."""
    import json
    from pathlib import Path

    ef1_l, act_l, eng_l, p3_l, rew_l = [], [], [], [], []

    for vid in videos:
        cat, name = vid.split("/")
        p = Path(npz_dir) / cat / f"{name}.npz"
        npz = np.load(str(p), allow_pickle=True)

        roi = npz["temporal_roi"].astype(bool)
        feat = npz["features"][roi].astype(np.float32)
        obs = (feat - z_mean) / z_std

        actions = get_actions_fn(obs)
        res = fast_simulate(npz, actions, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W)
        ep = res["episode"]

        ef1_l.append(ep["event_f1"])
        act_l.append(ep["activation"])
        eng_l.append(ep["energy"])
        p3_l.append(1.0 - ep["activation"])
        rew_l.append(ep["total_reward"])

    return {
        "ef1":    float(np.mean(ef1_l)),
        "ef1_std":float(np.std(ef1_l)),
        "act":    float(np.mean(act_l)),
        "energy": float(np.mean(eng_l)),
        "p3fb":   float(np.mean(p3_l)),
        "reward": float(np.mean(rew_l)),
        "n":      len(ef1_l),
    }
