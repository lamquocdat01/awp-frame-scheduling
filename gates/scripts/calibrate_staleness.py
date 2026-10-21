"""
T3.5 -- Calibrate staleness window W for the reward simulator.

Uses the FULL-FRAME CALIB run (all ROI frames, no stride sampling) to find
the W that minimises |sim_EF1 - ref_EF1| across all 27 train videos.

Algorithm:
  1. For each train video, load the full-frame frame_metrics.csv
     (pipeline ASMAG_TR_CONTROLLER_ONLINE_CALIBRATED, every ROI frame present)
  2. Run simulate(W=W) with real CALIB actions for W in W_SWEEP
  3. Compute dEF1 = |sim_ef1 - ref_ef1| per video per W
  4. Print W-vs-dEF1 table; choose optimal W = argmin mean_dEF1
  5. Print loophole test: "detect once, skip forever" must give EF1 << 1.0

Usage:
    cd asmag-rl-vs-deterministic
    python scripts/calibrate_staleness.py
    python scripts/calibrate_staleness.py --full-exp-dir <path> --lambda 0.3 --P-miss 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.reward_simulator import simulate

_DEFAULT_FULL_EXP_DIR = (
    r"D:\THS Programing\06.6 ASMAG Project\ASMAG_Project_Code\outputs"
    r"\full_cdnet2014_official_edge_profile_pc\raw_results"
)

_PIPELINE = "ASMAG_TR_CONTROLLER_ONLINE_CALIBRATED"

W_SWEEP = [1, 2, 3, 5, 7, 8, 10, 12, 15, 20, 30, 50, None]


def _parse_video_id(video_id: str):
    parts = video_id.replace("\\", "/").split("/")
    return parts[-2], parts[-1]


def _load_summary(exp_dir: Path) -> dict:
    """Load summary_event_metrics.csv -> {(category, video): row_dict} for our pipeline."""
    summary_path = exp_dir.parent / "summary_event_metrics.csv"
    lookup = {}
    if not summary_path.exists():
        print(f"  WARNING: summary not found at {summary_path}")
        return lookup
    with open(str(summary_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("pipeline", "") == _PIPELINE:
                key = (row.get("category", ""), row.get("video", ""))
                lookup[key] = row
    return lookup


def _load_full_frame_csv(csv_path: Path, frame_idx_roi: np.ndarray):
    """
    Load full-frame frame_metrics.csv, aligning to ROI frame indices.
    Returns (modes_roi[T_roi], yolo_roi[T_roi], n_matched, coverage_pct)
    where T_roi = len(frame_idx_roi).
    Unmatched ROI frames (should not happen for full-frame run) default to P3_FALLBACK/0.

    NOTE: We do NOT filter on Is_Active here. In the full-frame run, Is_Active=1 means
    YOLO produced a positive detection (objects found), while Is_Active=0 means YOLO ran
    but found nothing (or gate was closed). Both are valid ROI frames; we need yolo_called
    from all of them to correctly model FP from quiet frames with open gates.
    """
    if not csv_path.exists():
        return None

    fid_to_row = {}
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # All rows are ROI frames in the full-frame run (no Is_Active filter)
            fid_to_row[int(row["frame_id"])] = row

    if not fid_to_row:
        return None

    T_roi = len(frame_idx_roi)
    modes_roi = np.array(["P3_FALLBACK"] * T_roi, dtype=object)
    yolo_roi = np.zeros(T_roi, dtype=np.int32)
    n_matched = 0

    for i, fid in enumerate(frame_idx_roi):
        row = fid_to_row.get(int(fid))
        if row is None:
            continue
        modes_roi[i] = str(row.get("selected_mode", "P3_FALLBACK"))
        yolo_roi[i] = int(row.get("yolo_called", row.get("activation", 0)) or 0)
        n_matched += 1

    coverage_pct = 100.0 * n_matched / max(T_roi, 1)
    return modes_roi, yolo_roi, n_matched, coverage_pct


def _process_video(
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
    Load data and sweep W for one video.
    Returns result dict with per-W sim_ef1 and ref_ef1, or error dict.
    """
    category, video = _parse_video_id(video_id)

    npz_path = npz_dir / category / f"{video}.npz"
    if not npz_path.exists():
        return {"status": "NPZ_MISSING", "video_id": video_id}

    data = np.load(str(npz_path), allow_pickle=True)
    temporal_roi = data["temporal_roi"].astype(bool)
    T_total = len(temporal_roi)
    T_roi = int(temporal_roi.sum())
    frame_idx_roi = data["frame_idx"][temporal_roi]

    csv_path = exp_dir / category / video / _PIPELINE / "frame_metrics.csv"
    loaded = _load_full_frame_csv(csv_path, frame_idx_roi)
    if loaded is None:
        return {"status": "CSV_MISSING", "video_id": video_id}

    modes_roi, yolo_roi, n_matched, coverage_pct = loaded

    # Reference EF1 from summary
    ref_row = summary.get((category, video))
    if ref_row is None:
        return {"status": "SUMMARY_MISSING", "video_id": video_id}

    ref_ef1 = float(ref_row["Event_F1"])
    ref_tp = int(ref_row["Event_TP"])
    ref_fp = int(ref_row["Event_FP"])
    ref_fn = int(ref_row["Event_FN"])

    # Build T_total-length sequences for simulate()
    # simulate() expects action_sequence of length T_total
    roi_global_indices = np.where(temporal_roi)[0]
    modes_full = np.array(["P3_FALLBACK"] * T_total, dtype=object)
    yolo_full = np.zeros(T_total, dtype=np.int32)
    for i, g_idx in enumerate(roi_global_indices):
        modes_full[g_idx] = modes_roi[i]
        yolo_full[g_idx] = yolo_roi[i]

    # Sweep W
    w_results = {}
    for W in W_SWEEP:
        r = simulate(
            data, modes_full, lambda_cost, P_miss, R_hit, R_maintain, R_quiet,
            yolo_called_sequence=yolo_full, W=W,
        )
        ep = r["episode"]
        w_results[W] = {
            "ef1": ep["event_f1"],
            "tp": ep["event_tp"],
            "fp": ep["event_fp"],
            "fn": ep["event_fn"],
            "d_ef1": abs(ep["event_f1"] - ref_ef1),
            "signed_d_ef1": ep["event_f1"] - ref_ef1,
        }

    return {
        "status": "OK",
        "video_id": video_id,
        "category": category,
        "video": video,
        "T_roi": T_roi,
        "n_matched": n_matched,
        "coverage_pct": coverage_pct,
        "ref_ef1": ref_ef1,
        "ref_tp": ref_tp,
        "ref_fp": ref_fp,
        "ref_fn": ref_fn,
        "w_results": w_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="T3.5: calibrate staleness window W for the reward simulator."
    )
    parser.add_argument("--split", default="splits/S1_fold0.json")
    parser.add_argument("--npz-dir", default="data/features")
    parser.add_argument(
        "--full-exp-dir", default=_DEFAULT_FULL_EXP_DIR,
        help="raw_results/ dir of full-frame CALIB run"
    )
    parser.add_argument("--lambda", dest="lambda_cost", type=float, default=0.3)
    parser.add_argument("--P-miss", dest="P_miss", type=float, default=0.5)
    parser.add_argument("--R-hit", type=float, default=1.0)
    parser.add_argument("--R-maintain", type=float, default=0.2)
    parser.add_argument("--R-quiet", type=float, default=0.1)
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

    exp_dir = Path(args.full_exp_dir)
    summary = _load_summary(exp_dir)
    if not summary:
        print("ERROR: Could not load summary. Check --full-exp-dir.")
        sys.exit(1)

    train_videos = split["train"]

    print(f"\nT3.5 -- Staleness calibration")
    print(f"  Videos : {len(train_videos)} train")
    print(f"  lambda={args.lambda_cost}  P_miss={args.P_miss}")
    print(f"  Full-frame exp: {exp_dir}")
    print(f"  W sweep: {W_SWEEP}")
    print()

    all_results = []
    for vid in sorted(train_videos):
        r = _process_video(
            vid, npz_dir, exp_dir, summary,
            args.lambda_cost, args.P_miss,
            args.R_hit, args.R_maintain, args.R_quiet,
        )
        all_results.append(r)
        status = r.get("status", "?")
        if status != "OK":
            print(f"  SKIP  {vid:<50}  [{status}]")
        else:
            cov = r["coverage_pct"]
            if cov < 99:
                print(f"  NOTE  {vid:<50}  coverage={cov:.1f}% (expected 100%)")

    ok = [r for r in all_results if r.get("status") == "OK"]
    if not ok:
        print("No OK results. Check exp dir and NPZ paths.")
        sys.exit(1)

    print(f"  Processed {len(ok)}/{len(train_videos)} videos OK\n")

    # ---------------------------------------------------------------------------
    # W vs mean_dEF1 table
    # ---------------------------------------------------------------------------
    w_mean_d = {}
    w_max_d = {}
    for W in W_SWEEP:
        d_vals = [r["w_results"][W]["d_ef1"] for r in ok]
        w_mean_d[W] = float(np.mean(d_vals))
        w_max_d[W] = float(np.max(d_vals))

    optimal_W = min(W_SWEEP, key=lambda w: w_mean_d[w])

    # signed dEF1: positive = sim > ref (over-credit); negative = sim < ref (under-credit)
    w_mean_signed = {}
    w_n_pos = {}
    w_n_neg = {}
    for W in W_SWEEP:
        s_vals = [r["w_results"][W]["signed_d_ef1"] for r in ok]
        w_mean_signed[W] = float(np.mean(s_vals))
        w_n_pos[W] = int(sum(1 for v in s_vals if v > 0.005))
        w_n_neg[W] = int(sum(1 for v in s_vals if v < -0.005))

    print("=" * 75)
    print(f"  W vs dEF1  (n={len(ok)} train videos, after MechA fix)")
    print("=" * 75)
    print(f"  {'W':>8}  {'mean|dEF1|':>10}  {'max|dEF1|':>10}  "
          f"{'mean_sign':>10}  {'n_pos':>6}  {'n_neg':>6}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*6}")
    for W in W_SWEEP:
        marker = "  <-- OPTIMAL" if W == optimal_W else ""
        print(f"  {str(W):>8}  {w_mean_d[W]:>10.4f}  {w_max_d[W]:>10.4f}  "
              f"{w_mean_signed[W]:>+10.4f}  {w_n_pos[W]:>6}  {w_n_neg[W]:>6}{marker}")
    print("=" * 75)
    print(f"\n  Optimal W = {optimal_W}  "
          f"(mean|dEF1|={w_mean_d[optimal_W]:.4f}  "
          f"mean_signed={w_mean_signed[optimal_W]:+.4f}  "
          f"n_pos={w_n_pos[optimal_W]} n_neg={w_n_neg[optimal_W]})")

    # ---------------------------------------------------------------------------
    # Per-category breakdown at optimal W
    # ---------------------------------------------------------------------------
    print()
    print(f"  Per-category dEF1 at W={optimal_W}:")
    print(f"  {'Category':<30}  {'n':>2}  {'mean_dEF1':>10}  {'max_dEF1':>10}")
    print(f"  {'-'*30}  {'-'*2}  {'-'*10}  {'-'*10}")

    by_cat = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)

    for cat in sorted(by_cat):
        rows = by_cat[cat]
        d_vals = [r["w_results"][optimal_W]["d_ef1"] for r in rows]
        flag = "  *" if np.max(d_vals) >= 0.05 else ""
        print(f"  {cat:<30}  {len(rows):>2}  {np.mean(d_vals):>10.4f}  "
              f"{np.max(d_vals):>10.4f}{flag}")

    all_d_opt = [r["w_results"][optimal_W]["d_ef1"] for r in ok]
    all_d_none = [r["w_results"][None]["d_ef1"] for r in ok]
    n_pass = sum(1 for d in all_d_opt if d < 0.05)
    print(f"  {'OVERALL':<30}  {len(ok):>2}  {np.mean(all_d_opt):>10.4f}  "
          f"{np.max(all_d_opt):>10.4f}")
    print(f"\n  Videos dEF1 < 0.05: {n_pass}/{len(ok)}")
    print(f"  Improvement vs W=None: "
          f"mean {np.mean(all_d_none):.4f} -> {np.mean(all_d_opt):.4f}  "
          f"max {np.max(all_d_none):.4f} -> {np.max(all_d_opt):.4f}")

    # ---------------------------------------------------------------------------
    # Per-video table: W=None vs optimal W
    # ---------------------------------------------------------------------------
    print()
    print("=" * 108)
    print(f"  Per-video: W=None vs W={optimal_W}")
    print("=" * 108)
    print(f"  {'Video':<50}  {'RefEF1':>7}  {'Cov%':>5}  "
          f"{'Sim_Opt':>8}  {'|dOpt|':>6}  {'sign_d':>7}  {'FP_sim':>7}  {'FP_ref':>7}")
    print(f"  {'-'*50}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")

    for r in sorted(ok, key=lambda x: x["video_id"]):
        ref = r["ref_ef1"]
        d_opt = r["w_results"][optimal_W]["d_ef1"]
        s_opt = r["w_results"][optimal_W]["signed_d_ef1"]
        sim_opt = r["w_results"][optimal_W]["ef1"]
        sim_fp = r["w_results"][optimal_W]["fp"]
        flag = "  *" if d_opt >= 0.05 else ""
        print(f"  {r['video_id']:<50}  {ref:>7.4f}  {r['coverage_pct']:>5.1f}  "
              f"{sim_opt:>8.4f}  {d_opt:>6.4f}  {s_opt:>+7.4f}  "
              f"{sim_fp:>7d}  {r['ref_fp']:>7d}{flag}")

    print("=" * 108)

    # ---------------------------------------------------------------------------
    # Loophole test at optimal W
    # ---------------------------------------------------------------------------
    print(f"\n  Loophole test at W={optimal_W}: 'DETECT once at span onset, skip forever'")
    print(f"  Expected: EF1 << 1.0  (W caps chain at {optimal_W} fresh frames per DETECT)")
    print()

    # Test on first 5 OK videos (mix of categories)
    sample_videos = ok[:5]
    loophole_pass = 0

    for r in sample_videos:
        category, video = r["category"], r["video"]
        npz_path_l = npz_dir / category / f"{video}.npz"
        data_l = np.load(str(npz_path_l), allow_pickle=True)
        T = len(data_l["temporal_roi"])
        temporal_roi_arr = data_l["temporal_roi"].astype(bool)
        first_roi_idx = int(np.argmax(temporal_roi_arr))
        T_roi_l = int(temporal_roi_arr.sum())

        seq_loop = ["P3_FALLBACK"] * T
        seq_loop[first_roi_idx] = "ACC"

        # With optimal W
        r_opt = simulate(
            data_l, seq_loop, args.lambda_cost, args.P_miss,
            args.R_hit, args.R_maintain, args.R_quiet, W=optimal_W,
        )
        ep_opt = r_opt["episode"]

        # Without staleness (W=None) for comparison
        r_none = simulate(
            data_l, seq_loop, args.lambda_cost, args.P_miss,
            args.R_hit, args.R_maintain, args.R_quiet, W=None,
        )
        ep_none = r_none["episode"]

        passed = ep_opt["event_f1"] < 0.5
        if passed:
            loophole_pass += 1
        status = "PASS" if passed else "WARN"

        print(f"  [{status}] {r['video_id']:<48}")
        print(f"         W=None : EF1={ep_none['event_f1']:.4f}  "
              f"TP={ep_none['event_tp']:5d}  FN={ep_none['event_fn']:5d}")
        print(f"         W={str(optimal_W):<4} : EF1={ep_opt['event_f1']:.4f}  "
              f"TP={ep_opt['event_tp']:5d}  FN={ep_opt['event_fn']:5d}  "
              f"(max_fresh={optimal_W+1 if optimal_W else 'inf'} per DETECT)")

    print()
    print(f"  Loophole PASS: {loophole_pass}/{len(sample_videos)}")

    # ---------------------------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------------------------
    print()
    print("=" * 55)
    print(f"  RECOMMENDATION")
    print("=" * 55)
    print(f"  Optimal W = {optimal_W}")
    print(f"  mean_dEF1 = {w_mean_d[optimal_W]:.4f}  (W=None was {w_mean_d[None]:.4f})")
    print(f"  Videos dEF1 < 0.05: {n_pass}/{len(ok)}")
    print()
    print(f"  Use --W {optimal_W} in all simulate() calls during T4 training.")
    print(f"  The staleness window closes the 'detect-once-skip-forever' loophole.")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
