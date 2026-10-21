"""
T2b -- Validate reward simulator against real CALIB actions across all train videos.

For each train video in splits/S1_fold0.json:
  1. Load ONLINE_CALIBRATED frame_metrics.csv (real CALIB action sequence, sampled frames)
  2. Run reward_simulator in gate-aware mode on the sampled subset (apples-to-apples)
  3. Compare sim EF1 to logged reference EF1 from summary_event_metrics.csv

Prints per-video and per-category table. Flags dEF1 > 0.05.

Purpose: verify whether recall~=1 assumption overstates EF1 in hard categories
(PTZ/nightVideos/badWeather). Large gap -> T4 needs per-category recall calibration.

Usage:
    python scripts/validate_reward.py [--split splits/S1_fold0.json] [--lambda 0.3] [--P-miss 0.5]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.reward_simulator import simulate

# Default CALIB experiment directory (raw_results/)
_DEFAULT_EXP_DIR = (
    r"D:\THS Programing\06.6 ASMAG Project\ASMAG_Project_Code\outputs"
    r"\asmag_tr_controller_online_guarded_cdnet_full_2q2_dryrun\raw_results"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_video_id(video_id: str):
    parts = video_id.replace("\\", "/").split("/")
    return parts[-2], parts[-1]


def _load_summary(exp_dir: Path) -> dict:
    """Load summary_event_metrics.csv -> {(category, video, pipeline): row_dict}."""
    summary_path = exp_dir.parent / "summary_event_metrics.csv"
    lookup = {}
    if not summary_path.exists():
        return lookup
    with open(str(summary_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row.get("category", ""), row.get("video", ""), row.get("pipeline", ""))
            lookup[key] = row
    return lookup


def _load_sampled_frames(csv_path: Path, frame_idx_npz: np.ndarray, gt_event_npz: np.ndarray):
    """
    Load ONLINE_CALIBRATED frame_metrics.csv and align to NPZ frame_idx.
    Returns (n_sampled, gt_event_sampled, yolo_called_sampled, selected_mode_sampled)
    or None if CSV not found / no ROI frames.
    """
    if not csv_path.exists():
        return None

    frame_rows = []
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Is_Active", "0") == "1":
                frame_rows.append(row)

    if not frame_rows:
        return None

    fid_to_row = {int(r["frame_id"]): r for r in frame_rows}
    fid_to_npz_idx = {int(fid): i for i, fid in enumerate(frame_idx_npz)}

    gt_event_s, yolo_s, mode_s = [], [], []
    matched = 0
    for fid, row in sorted(fid_to_row.items()):
        npz_i = fid_to_npz_idx.get(fid)
        if npz_i is None:
            continue
        gt_event_s.append(bool(gt_event_npz[npz_i]))
        yolo_s.append(int(row.get("yolo_called", row.get("activation", 0)) or 0))
        mode_s.append(str(row.get("selected_mode", "P3_FALLBACK")))
        matched += 1

    if matched == 0:
        return None

    return (
        matched,
        np.array(gt_event_s, dtype=bool),
        np.array(yolo_s, dtype=np.int32),
        np.array(mode_s, dtype=object),
    )


def _sim_on_sampled(gt_event_s, yolo_called_s, mode_s, lambda_cost, P_miss, R_hit, R_maintain, R_quiet):
    """
    Run simulate() on sampled frames only (mock NPZ with temporal_roi = all True).
    Returns episode dict.
    """
    n = len(gt_event_s)
    mock_npz = {
        "temporal_roi": np.ones(n, dtype=bool),
        "gt_event": gt_event_s,
        "frame_idx": np.arange(1, n + 1),
    }
    result = simulate(
        mock_npz,
        mode_s,  # length T = n (all ROI)
        lambda_cost=lambda_cost,
        P_miss=P_miss,
        R_hit=R_hit,
        R_maintain=R_maintain,
        R_quiet=R_quiet,
        yolo_called_sequence=yolo_called_s,
    )
    return result["episode"]


def _check_video(
    video_id: str,
    npz_dir: Path,
    exp_dir: Path,
    summary: dict,
    lambda_cost: float,
    P_miss: float,
    R_hit: float,
    R_maintain: float,
    R_quiet: float,
) -> dict | None:
    """
    Process one video. Returns result dict or None if data unavailable.
    """
    category, video = _parse_video_id(video_id)

    npz_path = npz_dir / category / f"{video}.npz"
    if not npz_path.exists():
        return {"video_id": video_id, "status": "NPZ_MISSING"}

    data = np.load(str(npz_path), allow_pickle=True)
    frame_idx = data["frame_idx"]
    gt_event = data["gt_event"].astype(bool)
    temporal_roi = data["temporal_roi"].astype(bool)
    T_roi = int(temporal_roi.sum())

    # Only look at ROI frames
    frame_idx_roi = frame_idx[temporal_roi]
    gt_event_roi = gt_event[temporal_roi]

    csv_path = exp_dir / category / video / "ONLINE_CALIBRATED" / "frame_metrics.csv"
    sampled = _load_sampled_frames(csv_path, frame_idx_roi, gt_event_roi)
    if sampled is None:
        return {"video_id": video_id, "status": "CSV_MISSING", "T_roi": T_roi}

    n_sampled, gt_event_s, yolo_s, mode_s = sampled
    coverage_pct = 100.0 * n_sampled / max(T_roi, 1)

    # Reference EF1 from summary
    ref_row = summary.get(("ONLINE_CALIBRATED", category, video)) or \
              summary.get((category, video, "ONLINE_CALIBRATED"))
    ref_ef1 = float(ref_row["Event_F1"]) if ref_row else None
    ref_tp = int(ref_row["Event_TP"]) if ref_row else None
    ref_fp = int(ref_row["Event_FP"]) if ref_row else None
    ref_fn = int(ref_row["Event_FN"]) if ref_row else None

    # Reference activation (from CSV sampled frames directly)
    ref_activation = float(yolo_s.sum()) / max(n_sampled, 1)

    # Simulate on sampled frames (gate-aware)
    ep = _sim_on_sampled(
        gt_event_s, yolo_s, mode_s,
        lambda_cost, P_miss, R_hit, R_maintain, R_quiet
    )

    sim_ef1 = ep["event_f1"]
    sim_activation = ep["activation"]

    d_ef1 = abs(sim_ef1 - ref_ef1) if ref_ef1 is not None else None
    d_act = abs(sim_activation - ref_activation)
    flag = ""
    if d_ef1 is not None and d_ef1 > 0.05:
        flag = "!"

    return {
        "video_id": video_id,
        "category": category,
        "video": video,
        "status": "OK",
        "T_roi": T_roi,
        "n_sampled": n_sampled,
        "coverage_pct": coverage_pct,
        "gt_event_rate": float(gt_event_s.mean()),
        "ref_ef1": ref_ef1,
        "ref_tp": ref_tp,
        "ref_fp": ref_fp,
        "ref_fn": ref_fn,
        "sim_ef1": sim_ef1,
        "sim_tp": ep["event_tp"],
        "sim_fp": ep["event_fp"],
        "sim_fn": ep["event_fn"],
        "d_ef1": d_ef1,
        "ref_activation": ref_activation,
        "sim_activation": sim_activation,
        "d_activation": d_act,
        "flag": flag,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="T2b: validate reward simulator dEF1 by category across train videos."
    )
    parser.add_argument("--split", default="splits/S1_fold0.json")
    parser.add_argument("--npz-dir", default="data/features")
    parser.add_argument(
        "--exp-dir", default=_DEFAULT_EXP_DIR,
        help="Path to raw_results/ dir of ONLINE_CALIBRATED experiment"
    )
    parser.add_argument("--lambda", dest="lambda_cost", type=float, default=0.3)
    parser.add_argument("--P-miss", dest="P_miss", type=float, default=0.5)
    parser.add_argument("--R-hit", type=float, default=1.0)
    parser.add_argument("--R-maintain", type=float, default=0.2)
    parser.add_argument("--R-quiet", type=float, default=0.1)
    parser.add_argument("--test-set", action="store_true",
                        help="Also run on test videos (default: train only)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    with open(str(split_path)) as f:
        split = json.load(f)

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_absolute():
        npz_dir = repo_root / npz_dir

    exp_dir = Path(args.exp_dir)
    summary = _load_summary(exp_dir)

    videos = split["train"]
    if args.test_set:
        videos = videos + split["test"]

    print(f"\nT2b -- Reward Simulator Validation")
    print(f"Split: {split_path.name}  |  videos: {len(videos)}  |  "
          f"lambda={args.lambda_cost}  P_miss={args.P_miss}")
    print(f"Exp:   {exp_dir}")
    print()

    results = []
    for vid in sorted(videos):
        r = _check_video(
            vid, npz_dir, exp_dir, summary,
            args.lambda_cost, args.P_miss,
            args.R_hit, args.R_maintain, args.R_quiet,
        )
        results.append(r)
        status = r.get("status", "?")
        if status != "OK":
            print(f"  SKIP  {vid:<45}  [{status}]")

    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        print("No OK results. Check experiment directory and NPZ paths.")
        return

    # --- Per-video table ---
    print("=" * 115)
    print(f"{'Video':<48}  {'Cov%':>5}  {'GT%':>5}  "
          f"{'RefEF1':>7}  {'SimEF1':>7}  {'dEF1':>6}  "
          f"{'RefAct':>6}  {'SimAct':>6}  {'dAct':>6}  {'!':>2}")
    print("-" * 115)

    by_category = defaultdict(list)
    for r in sorted(ok, key=lambda x: x["video_id"]):
        category = r["category"]
        by_category[category].append(r)
        ref_ef1_s = f"{r['ref_ef1']:.4f}" if r["ref_ef1"] is not None else "  N/A "
        sim_ef1_s = f"{r['sim_ef1']:.4f}" if r["sim_ef1"] is not None else "  N/A "
        d_ef1_s = f"{r['d_ef1']:.4f}" if r["d_ef1"] is not None else "  N/A "
        print(
            f"  {r['video_id']:<46}  {r['coverage_pct']:>5.1f}  "
            f"{100*r['gt_event_rate']:>5.1f}  "
            f"{ref_ef1_s:>7}  {sim_ef1_s:>7}  {d_ef1_s:>6}  "
            f"{r['ref_activation']:>6.3f}  {r['sim_activation']:>6.3f}  "
            f"{r['d_activation']:>6.4f}  {r['flag']:>2}"
        )

    print("=" * 115)

    # --- Per-category summary ---
    print()
    print(f"{'Category':<28}  {'n':>2}  {'avgCov%':>7}  "
          f"{'avg_dEF1':>8}  {'max_dEF1':>8}  "
          f"{'avg_dAct':>8}  {'max_dAct':>8}  {'!':>2}")
    print("-" * 95)

    flagged_categories = []
    all_d_ef1 = []
    all_d_act = []

    for cat in sorted(by_category):
        rows = by_category[cat]
        d_ef1_vals = [r["d_ef1"] for r in rows if r["d_ef1"] is not None]
        d_act_vals = [r["d_activation"] for r in rows]
        cov_vals = [r["coverage_pct"] for r in rows]
        avg_d_ef1 = float(np.mean(d_ef1_vals)) if d_ef1_vals else float("nan")
        max_d_ef1 = float(np.max(d_ef1_vals)) if d_ef1_vals else float("nan")
        avg_d_act = float(np.mean(d_act_vals))
        max_d_act = float(np.max(d_act_vals))
        avg_cov = float(np.mean(cov_vals))
        cat_flag = "!" if max_d_ef1 > 0.05 else ""
        if cat_flag:
            flagged_categories.append(cat)
        all_d_ef1.extend(d_ef1_vals)
        all_d_act.extend(d_act_vals)
        print(
            f"  {cat:<26}  {len(rows):>2}  {avg_cov:>7.1f}  "
            f"{avg_d_ef1:>8.4f}  {max_d_ef1:>8.4f}  "
            f"{avg_d_act:>8.4f}  {max_d_act:>8.4f}  {cat_flag:>2}"
        )

    print("-" * 95)
    print(
        f"  {'OVERALL':<26}  {len(ok):>2}  {np.mean([r['coverage_pct'] for r in ok]):>7.1f}  "
        f"{np.mean(all_d_ef1) if all_d_ef1 else float('nan'):>8.4f}  "
        f"{np.max(all_d_ef1) if all_d_ef1 else float('nan'):>8.4f}  "
        f"{np.mean(all_d_act):>8.4f}  {np.max(all_d_act):>8.4f}"
    )

    print()
    if flagged_categories:
        print(f"FLAGGED categories (max_dEF1 > 0.05): {', '.join(flagged_categories)}")
        print("  -> T4 may need per-category recall calibration for these.")
    else:
        print("All categories: max_dEF1 <= 0.05. Reward model usable as-is for T4.")

    n_act_match = sum(1 for r in ok if r["d_activation"] < 0.001)
    print(f"Activation exact match (delta < 0.001): {n_act_match}/{len(ok)} videos")

    # Activation gap note
    if any(r["d_activation"] > 0.01 for r in ok):
        print()
        print("NOTE: activation delta > 0.01 on some videos may indicate:")
        print("  - Frame ID alignment mismatch (CSV frame_id vs NPZ frame_idx)")
        print("  - Videos processed at different stride in experiment vs NPZ")

    # EF1 gap note
    print()
    print("NOTE: dEF1 sources (in decreasing importance):")
    print("  1. Coverage gap: low % sampled -> chain-propagation inflates sim EF1")
    print("     (check 'Cov%' column; gap expected for Cov% < 50%)")
    print("  2. Chain breaks: real system breaks detection chain (staleness);")
    print("     sim assumes perfect chain after first DETECT. Gap = ~0.03-0.05 typical.")
    print("  3. Recall < 1: YOLO misses small/distant objects; sim assumes recall=1.")
    print("     Visible as gap that persists even at high coverage in hard categories.")


if __name__ == "__main__":
    main()
