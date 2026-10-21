"""
T5 Feature Divergence Diagnostic.

Dumps live-extracted features (SharedGuardedGateBank on real frames) alongside
NPZ offline features for direct per-feature comparison.

For each video:
  - Loads NPZ offline features (features[temporal_roi])
  - Runs real frames through SharedGuardedGateBank + OnlineSceneDifficultyEstimator
    (same pipeline as t5_engine_runner.py)
  - Saves per-frame comparison CSV + summary stats

Output: t5_diag/<video_id>.csv (per-frame, 10 features x 2 sources)
        t5_diag/summary.csv (per-feature mean/std/max divergence across videos)

Usage:
  python scripts/t5_feature_diag.py --dataset-root /data/cdnet2014 --npz-dir data/features
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from asmag_core.adaptive_gate import SharedGuardedGateBank
from asmag_core.features import OnlineSceneDifficultyEstimator, ONLINE_MODE_FEATURE_COLUMNS, compute_is_active

# Same GUARD config as t5_engine_runner.py (256px resize)
_GUARD_CFG = {
    "p4_asmag_plus": {
        "min_area_ratio": 0.001,
        "entropy_threshold": 8.5,
        "open_threshold": 0.55,
        "close_threshold": 0.30,
        "watchdog_closed_frames": 12,
        "watchdog_interval": 6,
        "recent_object_memory": 30,
    },
    "p4_efficient": {},
    "online_controller_guarded": {"resize_width_for_gating": 256},
    "edge_profile": {"resize_width_for_gating": 256},
}

# 8 diverse diagnostic videos (1 per category, biased toward high-drift categories)
_DIAG_VIDEOS = [
    "PTZ/intermittentPan",
    "nightVideos/bridgeEntry",
    "lowFramerate/turnpike_0_5fps",
    "badWeather/snowFall",
    "baseline/highway",
    "dynamicBackground/fall",
    "thermal/library",
    "turbulence/turbulence3",
]


def extract_live_features(video_id, dataset_root):
    """
    Run real frames through SharedGuardedGateBank + OnlineSceneDifficultyEstimator.
    Returns (T_roi,) list of 10-dim feature dicts, one per ROI frame.
    """
    cat, name = video_id.split("/")
    vid_dir = dataset_root / cat / name
    input_dir = vid_dir / "input"
    gt_dir = vid_dir / "groundtruth"

    roi_file = vid_dir / "temporalROI.txt"
    with open(str(roi_file)) as f:
        parts = f.read().strip().split()
        roi_start, roi_end = int(parts[0]), int(parts[1])

    bank = SharedGuardedGateBank(_GUARD_CFG)
    estimator = OnlineSceneDifficultyEstimator({})
    estimator.reset()

    prev_frame = None
    all_frames_count = 0
    roi_features = []

    frame_idx = 1
    while True:
        img_path = input_dir / f"in{frame_idx:06d}.jpg"
        if not img_path.exists():
            img_path = input_dir / f"in{frame_idx:06d}.png"
        if not img_path.exists():
            break

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            frame_idx += 1
            continue

        in_roi = roi_start <= frame_idx <= roi_end
        all_frames_count += 1

        # Gate processing (every frame for BG warm-up)
        candidates, telemetry = bank.process(
            bgr, prev_frame,
            state={"is_evaluation": True, "evaluated_index": all_frames_count - 1}
        )

        # GT view kept ONLY for debug/diagnostic comparison — NEVER fed to the
        # estimator (feeding GT was the A3 oracle bug). active_frame_rate is zeroed
        # in canonicalization; the estimator gets compute_is_active() == False.
        gt = None
        if in_roi:
            gt_path = gt_dir / f"gt{frame_idx:06d}.png"
            if gt_path.exists():
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        is_active_gt_oracle_DEBUG = bool(gt is not None and np.any(gt > 127))  # noqa: do NOT feed

        # Feature estimation (every frame for rolling window)
        features_dict = estimator.update(
            gate_info=telemetry,
            gate_open=True,
            reused_prediction=False,
            is_active=compute_is_active(),
            evaluated_index=all_frames_count - 1,
        )

        if in_roi:
            x_t = estimator.get_feature_vector(features_dict)
            roi_features.append(x_t.copy())

        prev_frame = bgr
        frame_idx += 1

    return np.array(roi_features, dtype=np.float32) if roi_features else np.empty((0, 10), dtype=np.float32)


def diagnose_video(video_id, dataset_root, npz_dir, out_dir):
    """Compare live vs NPZ features for one video."""
    cat, name = video_id.split("/")

    # Load NPZ offline features
    npz_path = npz_dir / cat / f"{name}.npz"
    if not npz_path.exists():
        print(f"  SKIP {video_id}: NPZ not found at {npz_path}")
        return None

    npz = np.load(str(npz_path), allow_pickle=True)
    roi = npz["temporal_roi"].astype(bool)
    npz_feat = npz["features"][roi].astype(np.float32)  # (T_roi, 10)

    # Extract live features
    live_feat = extract_live_features(video_id, dataset_root)

    T_npz = len(npz_feat)
    T_live = len(live_feat)
    T = min(T_npz, T_live)

    if T == 0:
        print(f"  SKIP {video_id}: 0 ROI frames")
        return None

    if T_npz != T_live:
        print(f"  WARN {video_id}: T_npz={T_npz} != T_live={T_live}, using min={T}")

    npz_feat = npz_feat[:T]
    live_feat = live_feat[:T]

    # Per-feature stats
    diff = live_feat - npz_feat  # (T, 10)
    abs_diff = np.abs(diff)

    stats = {}
    for i, col in enumerate(ONLINE_MODE_FEATURE_COLUMNS):
        d = diff[:, i]
        ad = abs_diff[:, i]
        stats[col] = {
            "mean_diff": float(np.mean(d)),
            "std_diff": float(np.std(d)),
            "mean_abs_diff": float(np.mean(ad)),
            "max_abs_diff": float(np.max(ad)),
            "npz_mean": float(np.mean(npz_feat[:, i])),
            "live_mean": float(np.mean(live_feat[:, i])),
            "npz_std": float(np.std(npz_feat[:, i])),
            "live_std": float(np.std(live_feat[:, i])),
        }

    # Save per-frame CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{cat}_{name}.csv"
    with open(str(csv_path), "w", newline="") as f:
        header = ["frame_t"]
        for col in ONLINE_MODE_FEATURE_COLUMNS:
            header.extend([f"npz_{col}", f"live_{col}", f"diff_{col}"])
        w = csv.writer(f)
        w.writerow(header)
        for t in range(T):
            row = [t]
            for i in range(10):
                row.extend([
                    f"{npz_feat[t, i]:.6f}",
                    f"{live_feat[t, i]:.6f}",
                    f"{diff[t, i]:.6f}",
                ])
            w.writerow(row)

    # Print summary for this video
    print(f"\n  {video_id} (T={T} ROI frames)")
    print(f"  {'Feature':<30s} {'NPZ mean':>10s} {'Live mean':>10s} {'mean_diff':>10s} {'|max_diff|':>10s}")
    print(f"  {'-'*72}")
    for col in ONLINE_MODE_FEATURE_COLUMNS:
        s = stats[col]
        flag = ""
        if s["mean_abs_diff"] > 0.01:
            flag = " <-- DIVERGES"
        if col in ("reuse_success_rate", "gate_closed_rate"):
            if s["live_mean"] > 0.001:
                flag = " <-- NON-ZERO (canonical violation!)"
        print(f"  {col:<30s} {s['npz_mean']:>10.5f} {s['live_mean']:>10.5f} "
              f"{s['mean_diff']:>+10.5f} {s['max_abs_diff']:>10.5f}{flag}")

    return {"video": video_id, "T": T, "stats": stats}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", type=str, required=True)
    ap.add_argument("--npz-dir", type=str, default=str(_REPO / "data" / "features"))
    ap.add_argument("--out-dir", type=str, default=str(_REPO / "t5_diag"))
    ap.add_argument("--videos", type=str, nargs="*", default=None,
                    help="Override video list (default: 8 diverse)")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    videos = args.videos or _DIAG_VIDEOS

    print(f"T5 Feature Divergence Diagnostic: {len(videos)} videos")
    print(f"Dataset: {dataset_root}")
    print(f"NPZ dir: {npz_dir}")
    print(f"Output: {out_dir}")

    all_results = []
    for vid in videos:
        result = diagnose_video(vid, dataset_root, npz_dir, out_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo results. Check paths.")
        return

    # Aggregate summary across all videos
    print(f"\n{'='*80}")
    print(f"AGGREGATE FEATURE DIVERGENCE ({len(all_results)} videos)")
    print(f"{'='*80}")
    print(f"{'Feature':<30s} {'mean|diff|':>10s} {'max|diff|':>10s} {'NPZ avg':>10s} {'Live avg':>10s} {'verdict':>12s}")
    print(f"{'-'*84}")

    summary_rows = []
    for i, col in enumerate(ONLINE_MODE_FEATURE_COLUMNS):
        all_mean_abs = [r["stats"][col]["mean_abs_diff"] for r in all_results]
        all_max_abs = [r["stats"][col]["max_abs_diff"] for r in all_results]
        all_npz_mean = [r["stats"][col]["npz_mean"] for r in all_results]
        all_live_mean = [r["stats"][col]["live_mean"] for r in all_results]

        agg_mean_abs = float(np.mean(all_mean_abs))
        agg_max_abs = float(np.max(all_max_abs))
        agg_npz = float(np.mean(all_npz_mean))
        agg_live = float(np.mean(all_live_mean))

        if col in ("reuse_success_rate", "gate_closed_rate"):
            verdict = "CANONICAL" if agg_live < 0.001 else "VIOLATION"
        elif agg_mean_abs < 0.005:
            verdict = "MATCH"
        elif agg_mean_abs < 0.02:
            verdict = "MINOR"
        elif agg_mean_abs < 0.05:
            verdict = "MODERATE"
        else:
            verdict = "MAJOR"

        print(f"  {col:<30s} {agg_mean_abs:>10.5f} {agg_max_abs:>10.5f} "
              f"{agg_npz:>10.5f} {agg_live:>10.5f} {verdict:>12s}")

        summary_rows.append({
            "feature": col, "mean_abs_diff": f"{agg_mean_abs:.6f}",
            "max_abs_diff": f"{agg_max_abs:.6f}",
            "npz_avg": f"{agg_npz:.6f}", "live_avg": f"{agg_live:.6f}",
            "verdict": verdict,
        })

    # Save summary CSV
    summary_path = out_dir / "summary.csv"
    with open(str(summary_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSaved: {summary_path}")

    # Per-video activation comparison (to correlate with T5 drift)
    print(f"\nPer-video activation proxy (feature[8] = active_frame_rate):")
    for r in all_results:
        s = r["stats"]["active_frame_rate"]
        print(f"  {r['video']:<35s} npz={s['npz_mean']:.4f} live={s['live_mean']:.4f} "
              f"diff={s['mean_diff']:+.4f}")


if __name__ == "__main__":
    main()
