"""
T2/T3.5 -- Offline reward simulator (event-coverage model with staleness).

Assumption: DETECT (ACC/FAST action in simplified model, or yolo_called=1 in gate-aware
mode) in a gt-event-active frame catches the event (recall~=1 given YOLO runs).

Staleness (T3.5): reuse is valid only if frames_since_last_detect <= W.
Stale REUSE in event-active => -P_miss (treated as miss).
W=None disables staleness (original T2 behavior).

MechA fix (T3.7): quiet frames within the W-frame reuse window also count as FP
in EF1, matching the real system (chain detection persists into quiet frames).
Per-frame R_acc is unchanged; only detection_available (and EF1) is affected.

Usage:
    simulate(npz_or_path, action_sequence, lambda_cost, P_miss, W=7)
    -> {"per_frame_reward": ndarray[T_roi], "episode": {...}}

CLI:
    python -m rl.reward_simulator --npz data/features/baseline/highway.npz
        --lambda 0.3 --P-miss 0.5 --W 7 --sanity
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

# Add repo root to path when run directly
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from asmag_core.controllers import ACTION_SPACE, compute_cost
from asmag_core.metrics import event_f1_from_rows

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous True runs in a 1-D boolean array.
    Returns list of (start, end) INCLUSIVE index pairs.
    """
    spans = []
    in_span = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_span:
            start = i
            in_span = True
        elif not v and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, len(mask) - 1))
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    npz_or_path: Union[str, Path, dict],
    action_sequence: Sequence[Union[str, int]],
    lambda_cost: float,
    P_miss: float,
    R_hit: float = 1.0,
    R_maintain: float = 0.2,
    R_quiet: float = 0.1,
    yolo_called_sequence: Optional[Sequence[int]] = None,
    W: Optional[int] = None,
) -> dict:
    """
    Offline reward simulation for one video episode.

    Parameters
    ----------
    npz_or_path : str | Path | dict
        NPZ file path or pre-loaded dict with keys:
        features[T,10], temporal_roi[T], gt_event[T], frame_idx[T]
    action_sequence : array-like, length T (all frames, aligned with NPZ arrays)
        Action per frame: "ACC", "FAST", or "P3_FALLBACK" (or int index).
        Pre-ROI frames should be P3_FALLBACK (they are masked out of reward).
    lambda_cost : float
        Energy penalty coefficient. lambda=0 => pure event-F1 maximizer.
    P_miss : float
        Penalty for REUSE on an uncaught or stale event frame.
    R_hit : float
        Reward for first DETECT in an event span (catches event).
    R_maintain : float
        Reward for valid REUSE in a caught event span (fresh, frames_since_detect <= W).
    R_quiet : float
        Reward for REUSE in a quiet (non-event) frame.
    yolo_called_sequence : array-like, length T, optional
        If provided, overrides action-based DETECT inference.
        Use for gate-aware mode: pass yolo_called column from frame_metrics.csv.
        When provided, also uses gate_open=yolo_called for compute_cost.
    W : int or None
        Staleness window in real CDnet frames. Reuse is valid only when
        (current_frame_idx - last_detect_frame_idx) <= W.
        Stale REUSE in event-active frame => -P_miss.
        None = no staleness (infinite fresh window, original T2 behavior).

    Returns
    -------
    dict with:
        "per_frame_reward"  : float32 ndarray [T_roi]   rewards for ROI frames only
        "episode"           : dict
            event_f1, event_tp, event_fp, event_fn,
            event_precision, event_recall,
            activation,     # fraction of ROI frames where YOLO ran
            energy,         # sum C_compute over ROI frames
            total_reward    # sum per_frame_reward
    """
    # --- Load NPZ ---
    if isinstance(npz_or_path, (str, Path)):
        data = np.load(str(npz_or_path), allow_pickle=True)
    else:
        data = npz_or_path

    temporal_roi = data["temporal_roi"].astype(bool)
    gt_event = data["gt_event"].astype(bool)
    T = len(temporal_roi)

    # --- Resolve action sequence ---
    actions = np.asarray(list(action_sequence))
    if len(actions) != T:
        raise ValueError(
            f"action_sequence length {len(actions)} != NPZ length {T}. "
            "action_sequence must cover all frames (use P3_FALLBACK for pre-ROI)."
        )
    # Convert int actions to strings
    if actions.dtype.kind in ("i", "u"):
        actions = np.array([ACTION_SPACE[int(a)] for a in actions])

    # --- ROI views ---
    actions_roi = actions[temporal_roi]
    gt_event_roi = gt_event[temporal_roi]
    T_roi = int(temporal_roi.sum())

    # --- Frame indices for staleness tracking ---
    if "frame_idx" in data:
        frame_idx_roi = data["frame_idx"][temporal_roi].astype(np.int64)
    else:
        frame_idx_roi = np.arange(T_roi, dtype=np.int64)

    # --- Resolve is_detect per ROI frame ---
    if yolo_called_sequence is not None:
        yolo_called = np.asarray(list(yolo_called_sequence), dtype=bool)
        if len(yolo_called) != T:
            raise ValueError("yolo_called_sequence must be same length as action_sequence")
        yolo_called_roi = yolo_called[temporal_roi]
        is_detect_roi = yolo_called_roi
    else:
        yolo_called_roi = np.array([a in ("ACC", "FAST") for a in actions_roi])
        is_detect_roi = yolo_called_roi

    # --- Per-frame reward, cost, and detection_available ---
    per_frame_reward = np.zeros(T_roi, dtype=np.float32)
    per_frame_cost = np.zeros(T_roi, dtype=np.float32)
    detection_available = np.zeros(T_roi, dtype=bool)

    # Staleness: initialize so first frame is already "past W" if W is set
    # (system starts cold -- no prior detection)
    if W is not None and T_roi > 0:
        last_detect_fidx = int(frame_idx_roi[0]) - W - 1
    else:
        last_detect_fidx = -1  # unused when W is None

    event_caught = False
    prev_in_event = False

    for t in range(T_roi):
        action = actions_roi[t]
        in_event = bool(gt_event_roi[t])
        frame_fidx = int(frame_idx_roi[t])
        frame_is_detect = bool(is_detect_roi[t])

        # Reset event_caught at event span onset
        if in_event and not prev_in_event:
            event_caught = False

        # Update staleness tracker before freshness check
        if frame_is_detect:
            last_detect_fidx = frame_fidx

        # Freshness: True if W is disabled or within W frames of last detect
        if W is None:
            is_fresh = True
        else:
            is_fresh = (frame_fidx - last_detect_fidx) <= W

        # detection_available for EF1 (TP/FP/FN):
        #   event frame: DETECT always covers; valid REUSE covers if caught+fresh
        #   quiet frame: direct DETECT is FP; chain (is_fresh) also FP (MechA fix T3.7)
        if in_event:
            detection_available[t] = frame_is_detect or (event_caught and is_fresh)
        else:
            detection_available[t] = frame_is_detect or is_fresh

        # R_acc
        if in_event:
            if frame_is_detect:
                if not event_caught:
                    R_acc = R_hit
                    event_caught = True
                else:
                    R_acc = 0.0  # re-detect in already-caught span, no bonus
            elif event_caught and is_fresh:
                R_acc = R_maintain
            else:  # stale (> W since last detect) or onset never caught
                R_acc = -P_miss
        else:  # quiet
            R_acc = R_quiet if not frame_is_detect else 0.0

        # C_compute
        if yolo_called_sequence is not None:
            gate_open_for_cost = bool(yolo_called_roi[t])
        else:
            gate_open_for_cost = True  # simplified: gate always open for chosen action
        C = compute_cost(action, gate_open=gate_open_for_cost)
        per_frame_cost[t] = C

        per_frame_reward[t] = float(R_acc - lambda_cost * C)
        prev_in_event = in_event

    # --- Episode metrics ---
    event_tp = int(np.sum(gt_event_roi & detection_available))
    event_fn = int(np.sum(gt_event_roi & ~detection_available))
    event_fp = int(np.sum(~gt_event_roi & detection_available))
    event_tn = int(np.sum(~gt_event_roi & ~detection_available))

    ef1_dict = event_f1_from_rows([{
        "Event_TP": event_tp,
        "Event_FP": event_fp,
        "Event_FN": event_fn,
        "Event_TN": event_tn,
    }])

    # Activation = fraction of ROI frames where YOLO actually ran
    if yolo_called_sequence is not None:
        activation = float(yolo_called_roi.mean())
    else:
        activation = float((actions_roi == "ACC").mean() + (actions_roi == "FAST").mean())

    energy = float(per_frame_cost.sum())
    total_reward = float(per_frame_reward.sum())

    return {
        "per_frame_reward": per_frame_reward,
        "episode": {
            "event_f1": ef1_dict["event_f1"],
            "event_tp": ef1_dict["event_tp"],
            "event_fp": ef1_dict["event_fp"],
            "event_fn": ef1_dict["event_fn"],
            "event_precision": ef1_dict["event_precision"],
            "event_recall": ef1_dict["event_recall"],
            "activation": activation,
            "energy": energy,
            "total_reward": total_reward,
        },
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _load_npz_for_check(npz_path: Path) -> dict:
    return np.load(str(npz_path), allow_pickle=True)


def _print_episode(label: str, result: dict) -> None:
    ep = result["episode"]
    print(
        f"  {label:<35}"
        f"  EF1={ep['event_f1']:.4f}"
        f"  TP={ep['event_tp']:5d}  FP={ep['event_fp']:4d}  FN={ep['event_fn']:4d}"
        f"  activation={ep['activation']:.3f}"
        f"  energy={ep['energy']:.1f}"
        f"  total_reward={ep['total_reward']:.2f}"
    )


def run_sanity_checks(
    npz_path: Union[str, Path],
    lambda_cost: float,
    P_miss: float,
    R_hit: float,
    R_maintain: float,
    R_quiet: float,
    calib_exp_dir: Optional[str] = None,
    W: Optional[int] = None,
) -> None:
    """
    4 mandatory sanity checks printed to stdout.

    Check 1: all-DETECT (all ACC)       -> FN=0, activation=1.0
    Check 2: all-SKIP  (all P3_FALLBACK) -> EF1=0.0, activation=0.0
    Check 3: real ONLINE_CALIBRATED actions from frame_metrics.csv
             -> EF1 & activation should match logged summary values
    Check 4: loophole test (only when W is set)
             -> ACC at t=0 then all-SKIP -> EF1 must be low (loophole closed)
    """
    npz_path = Path(npz_path)
    data = _load_npz_for_check(npz_path)
    T = len(data["temporal_roi"])
    T_roi = int(data["temporal_roi"].sum())
    video_id = str(data["video_id"]) if "video_id" in data else npz_path.stem

    print()
    print("=" * 80)
    print(f"  SANITY CHECKS  |  video={video_id}  |  lambda={lambda_cost}  P_miss={P_miss}  W={W}")
    print(f"  NPZ: {npz_path}")
    gt_event_roi = data["gt_event"][data["temporal_roi"].astype(bool)]
    print(f"  T_total={T}  T_roi={T_roi}  gt_event_roi={int(gt_event_roi.sum())}")
    print("=" * 80)

    # --- Check 1: all-DETECT ---
    seq_all_detect = ["ACC"] * T
    r1 = simulate(data, seq_all_detect, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W=W)
    print("\n[1] all-DETECT (action=ACC for all frames):")
    _print_episode("all-ACC", r1)
    ep1 = r1["episode"]
    assert ep1["activation"] == 1.0, f"Sanity #1 FAIL: activation={ep1['activation']} != 1.0"
    assert ep1["event_fn"] == 0, f"Sanity #1 FAIL: {ep1['event_fn']} FN even with all-DETECT"
    print(f"  [1] PASS  activation=1.0  FN=0  (FP={ep1['event_fp']} from quiet-DETECT frames)")

    # --- Check 2: all-SKIP ---
    seq_all_skip = ["P3_FALLBACK"] * T
    r2 = simulate(data, seq_all_skip, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W=W)
    print("\n[2] all-SKIP (action=P3_FALLBACK for all frames):")
    _print_episode("all-P3_FALLBACK", r2)
    ep2 = r2["episode"]
    assert ep2["activation"] == 0.0, f"Sanity #2 FAIL: activation={ep2['activation']} != 0.0"
    assert ep2["event_tp"] == 0, f"Sanity #2 FAIL: event_tp={ep2['event_tp']} should be 0"
    assert ep2["event_f1"] == 0.0, f"Sanity #2 FAIL: EF1={ep2['event_f1']:.4f} should be 0.0"
    print("  [2] PASS  activation=0.0  TP=0  EF1=0.0")

    # --- Check 4: loophole test (run before check 3 to highlight the W effect) ---
    if W is not None:
        print(f"\n[4] Loophole test (W={W}): ACC at first ROI frame, then all P3_FALLBACK:")
        # ACC must be placed at the first ROI frame (global index), not at t=0
        # (most CDnet videos have pre-ROI frames at the start)
        temporal_roi_arr = data["temporal_roi"].astype(bool)
        first_roi_idx = int(np.argmax(temporal_roi_arr))
        seq_loophole = ["P3_FALLBACK"] * T
        seq_loophole[first_roi_idx] = "ACC"
        r4 = simulate(data, seq_loophole, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W=W)
        _print_episode("detect-once-then-skip", r4)
        ep4 = r4["episode"]
        n_gt_event = int(gt_event_roi.sum())
        loophole_ef1 = ep4["event_f1"]
        # TP should be at most W+1 (1 DETECT + W fresh REUSEs) if event starts at t=0
        print(f"  n_gt_event={n_gt_event}  TP={ep4['event_tp']}  "
              f"(max ~{W+1} if first span starts at t=0)")
        if loophole_ef1 < 0.5:
            print(f"  [4] PASS: EF1={loophole_ef1:.4f} < 0.5 (loophole closed with W={W})")
        else:
            print(f"  [4] WARNING: EF1={loophole_ef1:.4f} >= 0.5 -- W may be too large")

    # --- Check 3: real CALIB (ONLINE_CALIBRATED) action sequence ---
    print("\n[3] ONLINE_CALIBRATED real action sequence:")

    if calib_exp_dir is None:
        calib_exp_dir = (
            r"D:\THS Programing\06.6 ASMAG Project\ASMAG_Project_Code\outputs"
            r"\asmag_tr_controller_online_guarded_cdnet_full_2q2_dryrun\raw_results"
        )

    parts = video_id.replace("\\", "/").split("/")
    if len(parts) >= 2:
        category, video = parts[-2], parts[-1]
    else:
        category, video = "unknown", video_id
        print(f"  WARNING: cannot parse category/video from video_id={video_id!r}")
        print("  Skipping check #3.")
        return

    csv_path = (
        Path(calib_exp_dir) / category / video / "ONLINE_CALIBRATED" / "frame_metrics.csv"
    )
    summary_path = Path(calib_exp_dir).parent / "summary_event_metrics.csv"

    if not csv_path.exists():
        print(f"  WARNING: frame_metrics.csv not found at:\n    {csv_path}")
        print("  Skipping check #3.")
        return

    # Load frame_metrics.csv
    frame_rows = []
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Is_Active", "0") == "1":
                frame_rows.append(row)

    if not frame_rows:
        print("  WARNING: No Is_Active=1 frames found. Skipping check #3.")
        return

    # Build aligned sequences (length=T) for all NPZ frames
    frame_idx = data["frame_idx"]
    frame_id_to_row = {int(r["frame_id"]): r for r in frame_rows}

    actions_calib = np.array(["P3_FALLBACK"] * T, dtype=object)
    yolo_called = np.zeros(T, dtype=np.int32)
    sampled_mask_roi = np.zeros(T_roi, dtype=bool)
    sampled_roi_count = 0
    missing_frames = 0
    roi_position = -1

    for i, fid in enumerate(frame_idx):
        if not data["temporal_roi"][i]:
            continue
        roi_position += 1
        row = frame_id_to_row.get(int(fid))
        if row is None:
            missing_frames += 1
            continue
        actions_calib[i] = str(row.get("selected_mode", "P3_FALLBACK"))
        yolo_called[i] = int(row.get("yolo_called", row.get("activation", 0)) or 0)
        sampled_mask_roi[roi_position] = True
        sampled_roi_count += 1

    coverage_pct = 100.0 * sampled_roi_count / max(T_roi, 1)
    print(
        f"  CSV coverage: {sampled_roi_count}/{T_roi} ROI frames "
        f"({coverage_pct:.1f}%)  |  {missing_frames} unsampled (default P3_FALLBACK)"
    )

    # Run simulator: gate-aware (uses actual yolo_called from CSV)
    r3_gateaware = simulate(
        data, actions_calib, lambda_cost, P_miss, R_hit, R_maintain, R_quiet,
        yolo_called_sequence=yolo_called, W=W,
    )
    print("  Gate-aware model (yolo_called from CSV):")
    _print_episode("CALIB gate-aware", r3_gateaware)

    # Sampled-subset activation for apples-to-apples comparison
    yolo_called_roi = yolo_called[data["temporal_roi"].astype(bool)]
    sampled_activation = (
        float(yolo_called_roi[sampled_mask_roi].sum()) / max(sampled_roi_count, 1)
    )

    if summary_path.exists():
        ref_ef1 = None
        ref_tp = ref_fp = ref_fn = None
        with open(str(summary_path), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("pipeline") == "ONLINE_CALIBRATED"
                        and row.get("category") == category
                        and row.get("video") == video):
                    ref_ef1 = float(row["Event_F1"])
                    ref_tp = int(row["Event_TP"])
                    ref_fp = int(row["Event_FP"])
                    ref_fn = int(row["Event_FN"])
                    break

        ref_activation = sum(
            int(r.get("yolo_called", r.get("activation", 0)) or 0) for r in frame_rows
        ) / max(len(frame_rows), 1)

        if ref_ef1 is not None:
            print(f"\n  Reference (ONLINE_CALIBRATED summary):")
            print(
                f"  {'[logged]':<35}"
                f"  EF1={ref_ef1:.4f}"
                f"  TP={ref_tp:5d}  FP={ref_fp:4d}  FN={ref_fn:4d}"
                f"  activation={ref_activation:.3f}"
                f"  ({len(frame_rows)} sampled frames)"
            )

            act_gap = abs(sampled_activation - ref_activation)
            print(
                f"\n  Sampled-frame activation: sim={sampled_activation:.3f}  "
                f"ref={ref_activation:.3f}  delta={act_gap:.4f}"
            )
            ga_ef1 = r3_gateaware["episode"]["event_f1"]
            ef1_gap = abs(ga_ef1 - ref_ef1)
            print(
                f"  Full-video EF1 (gate-aware, W={W}):  sim={ga_ef1:.4f}  "
                f"ref={ref_ef1:.4f}  delta={ef1_gap:.4f}"
            )
            if act_gap < 0.001:
                print("  [3] Activation exact match on sampled frames. PASS.")
            else:
                print(f"  WARNING: Sampled activation gap {act_gap:.4f}.")
            if ef1_gap <= 0.05:
                print("  [3] EF1 gap <=0.05. PASS.")
            elif coverage_pct < 50:
                print(
                    f"  NOTE: EF1 gap {ef1_gap:.4f} likely explained by low coverage "
                    f"({coverage_pct:.0f}% sampled) -> chain-propagation to unsampled frames."
                )
            else:
                print(f"  WARNING: EF1 gap {ef1_gap:.4f} > 0.05.")

            if coverage_pct < 95:
                print(
                    f"  NOTE: {100-coverage_pct:.0f}% ROI frames unsampled; "
                    "unsampled frames default to P3_FALLBACK and propagate coverage from "
                    "prior DETECT, inflating full-video EF1. This is expected for stride experiments."
                )
        else:
            print(
                f"  WARNING: No ONLINE_CALIBRATED row for {category}/{video} "
                f"in {summary_path.name}"
            )
    else:
        print(f"  WARNING: summary_event_metrics.csv not found at {summary_path}")

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline reward simulator -- T2/T3.5 sanity checks and per-video reward."
    )
    parser.add_argument(
        "--npz",
        required=True,
        help="Path to NPZ feature file (e.g. data/features/baseline/highway.npz)",
    )
    parser.add_argument("--lambda", dest="lambda_cost", type=float, default=0.3)
    parser.add_argument("--P-miss", dest="P_miss", type=float, default=0.5)
    parser.add_argument("--R-hit", type=float, default=1.0)
    parser.add_argument("--R-maintain", type=float, default=0.2)
    parser.add_argument("--R-quiet", type=float, default=0.1)
    parser.add_argument(
        "--W", dest="W", type=int, default=None,
        help="Staleness window in real frames. None = no staleness (default)."
    )
    parser.add_argument(
        "--sanity", action="store_true",
        help="Run sanity checks (4 checks when --W is set, 3 otherwise)"
    )
    parser.add_argument(
        "--calib-exp-dir",
        default=None,
        help=(
            "Path to raw_results/ directory of an ONLINE_CALIBRATED experiment. "
            "Default: hard-coded path to asmag_tr_controller_online_guarded_cdnet_full_2q2_dryrun"
        ),
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = Path.cwd() / npz_path

    if args.sanity:
        run_sanity_checks(
            npz_path=npz_path,
            lambda_cost=args.lambda_cost,
            P_miss=args.P_miss,
            R_hit=args.R_hit,
            R_maintain=args.R_maintain,
            R_quiet=args.R_quiet,
            calib_exp_dir=args.calib_exp_dir,
            W=args.W,
        )
    else:
        data = np.load(str(npz_path), allow_pickle=True)
        T = len(data["temporal_roi"])
        seq = ["ACC"] * T
        result = simulate(data, seq, args.lambda_cost, args.P_miss,
                          args.R_hit, args.R_maintain, args.R_quiet, W=args.W)
        ep = result["episode"]
        print(f"[simulate] all-ACC | W={args.W} | EF1={ep['event_f1']:.4f} | "
              f"activation={ep['activation']:.3f} | "
              f"total_reward={ep['total_reward']:.2f}")


if __name__ == "__main__":
    main()
