"""
T1 -- Per-frame canonical state x_t dump.

For each video in splits/S1_fold0.json:
  - Run SharedGuardedGateBank per frame (shared CV feature extraction)
  - Feed into OnlineSceneDifficultyEstimator with canonical inputs:
      gate_open=True          (fixes gate_closed_rate = 0 -- policy-agnostic)
      reused_prediction=False (fixes reuse_success_rate = 0 -- policy-agnostic)
      is_active=False         (fixes active_frame_rate = 0 -- removes GT oracle)
  All three gating-decision features (idx 7/8/9) are then re-zeroed via the shared
  canonicalize_feature_vector() so the dump is byte-identical to engine inference.
  - Output: data/features/{category}/{video}.npz

Performance note:
  resize_width_for_gating (default 256) downscales frames before gate processing.
  On 704x480 PTZ frames: no-resize=0.8fps, resize=256 gives 17fps (21x speedup).
  Features are normalized ratios so scale-invariant; component_count may differ
  slightly at lower resolution. ALL videos must use the same setting for consistency.
  Current default=256 → estimated runtime ~2h for 159k CDnet frames.

NPZ keys per video:
  features     float32[T, 10]  -- ONLINE_MODE_FEATURE_COLUMNS, RAW (not z-scored)
  frame_idx    int32[T]        -- 1-based frame number from filename
  temporal_roi bool[T]         -- True if frame in [roi_start, roi_end]
  gt_event     bool[T]         -- True if GT mask has any foreground (value==255) pixel

Also saved:
  data/features/train_stats.json  -- mean+std (ddof=0) from TRAIN-set ROI frames only
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from asmag_core.features import (
    ONLINE_MODE_FEATURE_COLUMNS,
    OnlineSceneDifficultyEstimator,
    canonicalize_feature_vector,
    compute_is_active,
)
from asmag_core.adaptive_gate import SharedGuardedGateBank
from asmag_core.cdnet_loader import (
    frame_number, list_frames, read_temporal_roi,
    build_gt_index, read_gt_for_frame, gt_has_foreground,
)

# ---------------------------------------------------------------------------
# Pipeline-independence analysis
# ---------------------------------------------------------------------------

_ANALYSIS_ROWS = [
    ("motion_density_mean",    "No",       "rolling mean of motion_density (MOG2+KNN area / image_area)",
     "n/a"),
    ("motion_density_std",     "No",       "rolling std of motion_density",
     "n/a"),
    ("component_count_mean",   "No",       "rolling mean of kept_components (shape filter output)",
     "n/a"),
    ("component_count_std",    "No",       "rolling std of kept_components",
     "n/a"),
    ("fd_mog_disagreement",    "No",       "rolling mean of |fd_area - mog_area| / image_area",
     "n/a"),
    ("knn_mog_disagreement",   "No",       "rolling mean of |knn_area - mog_area| / image_area",
     "n/a"),
    ("illumination_variance",  "No",       "rolling var of illumination_diff",
     "n/a"),
    ("reuse_success_rate",     "YES",      "rolling mean of reused_prediction (policy decision)",
     "fix: reused_prediction=False  =>  always 0.0"),
    ("active_frame_rate",      "YES",      "rolling mean of is_active (policy/GT decision)",
     "fix: is_active=False          =>  always 0.0 (removes GT oracle)"),
    ("gate_closed_rate",       "YES",      "rolling mean of (not gate_open) (policy decision)",
     "fix: gate_open=True           =>  always 0.0"),
]


def print_analysis():
    print()
    print("=" * 76)
    print("  Pipeline-independence analysis -- ONLINE_MODE_FEATURE_COLUMNS")
    print("=" * 76)
    fmt = "  {:<28s}  {:<10s}  {}"
    print(fmt.format("Column", "Pipeline?", "Canonical fix"))
    print("  " + "-" * 73)
    n_indep = 0
    for col, dep, src, fix in _ANALYSIS_ROWS:
        print(fmt.format(col, dep, fix if dep != "No" else "(pipeline-independent)"))
        if dep == "No":
            n_indep += 1
    print("  " + "-" * 73)
    print(f"  {n_indep} pipeline-independent  |  3 fixed to 0.0 (policy-agnostic, leakage-free)")
    print("  x_t is reproducible and comparable across CALIB and RL policies.")
    print("=" * 76)
    print()


# ---------------------------------------------------------------------------
# Per-video feature extraction
# ---------------------------------------------------------------------------

def _process_video(video_id, input_dir, gt_dir, roi_start, roi_end, cfg,
                   seed=0, resize_width=256):
    """
    Extract canonical x_t for all frames of one video.

    Returns dict:
      features     np.float32[T, 10]
      frame_idx    np.int32[T]
      temporal_roi np.bool_[T]
      gt_event     np.bool_[T]
    """
    # MOG2 and KNN use OpenCV's global RNG for stochastic updates (sample replacement,
    # GMM initialisation).  Fix the seed so x_t is reproducible across runs.
    cv2.setRNGSeed(seed)

    frames, gts = list_frames(input_dir, gt_dir)
    if not frames:
        print(f"  [WARN] No frames found: {input_dir}")
        return None

    gt_index = build_gt_index(gts)

    gate_cfg = dict(cfg)
    if resize_width and resize_width > 0:
        gate_cfg.setdefault("online_controller_guarded", {})
        gate_cfg["online_controller_guarded"]["resize_width_for_gating"] = resize_width

    bank = SharedGuardedGateBank(cfg=gate_cfg)
    estimator = OnlineSceneDifficultyEstimator(cfg={})

    all_x = []
    all_idx = []
    all_roi = []
    all_gt = []

    prev_frame = None
    for i, frame_path in enumerate(frames):
        fnum = frame_number(frame_path)
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            print(f"  [WARN] Cannot read frame: {frame_path}")
            prev_frame = None
            continue

        # GT mask — zeros if GT not available (pre-ROI warmup frames sometimes have no GT)
        gt_mask = read_gt_for_frame(gt_index, frame_path, frame_bgr.shape)
        gt_ev = gt_has_foreground(gt_mask)

        # Gate bank: extract raw CV features
        bank_state = {"evaluated_index": i, "is_evaluation": True}
        candidates, _ = bank.process(frame_bgr, prev_frame, state=bank_state)
        gate_info = candidates["ACC"]["info"]

        # Estimator: canonical inputs. is_active=False -- the estimator must NEVER
        # see ground truth (in original ASMAG live it was hardcoded 0; feeding GT
        # here was the A3 oracle bug). gt_ev is still recorded separately below for
        # labels/metrics. reused_prediction=False / gate_open=True zero idx 7/9.
        feat_dict = estimator.update(
            gate_info,
            gate_open=True,
            reused_prediction=False,
            is_active=compute_is_active(),
            evaluated_index=i,
        )
        # Single source of truth: re-zero the 3 gating-decision features (idx 7/8/9)
        # so the dumped x_t is identical to what the engine sees at inference (T5).
        x_t = canonicalize_feature_vector(estimator.get_feature_vector(feat_dict))

        # Temporal ROI membership (1-based frame numbers)
        in_roi = False
        if fnum is not None and roi_start is not None and roi_end is not None:
            in_roi = roi_start <= fnum <= roi_end

        all_x.append(x_t)
        all_idx.append(fnum if fnum is not None else i + 1)
        all_roi.append(in_roi)
        all_gt.append(gt_ev)
        prev_frame = frame_bgr

    if not all_x:
        return None

    return {
        "features":      np.stack(all_x, axis=0).astype(np.float32),
        "frame_idx":     np.array(all_idx, dtype=np.int32),
        "temporal_roi":  np.array(all_roi, dtype=bool),
        "gt_event":      np.array(all_gt, dtype=bool),
        "resize_width":  resize_width,
    }


def save_npz(out_path, data_dict, video_id):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        features=data_dict["features"],
        frame_idx=data_dict["frame_idx"],
        temporal_roi=data_dict["temporal_roi"],
        gt_event=data_dict["gt_event"],
        video_id=np.array(video_id),
        feature_columns=np.array(ONLINE_MODE_FEATURE_COLUMNS),
        resize_width=np.array(data_dict.get("resize_width", 256)),
    )


# ---------------------------------------------------------------------------
# Train stats (anti-leakage: train videos, within-ROI frames only)
# ---------------------------------------------------------------------------

def compute_train_stats(train_ids, npz_dir):
    """
    Compute per-column mean and std (ddof=0) from train videos, ROI frames only.
    Returns dict ready to be written as train_stats.json.
    """
    all_vecs = []
    for video_id in train_ids:
        npz_path = npz_dir / (video_id + ".npz")
        if not npz_path.exists():
            print(f"  [WARN] Missing NPZ for train video: {video_id}")
            continue
        data = np.load(str(npz_path), allow_pickle=True)
        feats = data["features"]      # [T, 10]
        roi   = data["temporal_roi"]  # [T]
        roi_feats = feats[roi]
        if roi_feats.shape[0] > 0:
            all_vecs.append(roi_feats)

    if not all_vecs:
        print("  [WARN] No train frames found -- train_stats.json will be all zeros.")
        mean = [0.0] * len(ONLINE_MODE_FEATURE_COLUMNS)
        std  = [1.0] * len(ONLINE_MODE_FEATURE_COLUMNS)
    else:
        combined = np.concatenate(all_vecs, axis=0)   # [N, 10]
        mean = combined.mean(axis=0).tolist()
        std  = combined.std(axis=0, ddof=0).tolist()
        # Avoid zero-std (constant columns reuse_success_rate, gate_closed_rate)
        std = [max(s, 1e-8) for s in std]

    # Read resize_width from first available NPZ (self-documenting config)
    resize_w = 256
    for v in train_ids:
        p = npz_dir / (v + ".npz")
        if p.exists():
            try:
                resize_w = int(np.load(str(p), allow_pickle=True).get("resize_width", 256))
            except Exception:
                pass
            break

    return {
        "feature_columns": ONLINE_MODE_FEATURE_COLUMNS,
        "n_frames": int(sum(v.shape[0] for v in all_vecs)) if all_vecs else 0,
        "n_train_videos": len(train_ids),
        "resize_width_for_gating": resize_w,
        "mean": {col: mean[i] for i, col in enumerate(ONLINE_MODE_FEATURE_COLUMNS)},
        "std":  {col: std[i]  for i, col in enumerate(ONLINE_MODE_FEATURE_COLUMNS)},
        "note": (
            "Z-score params computed on TRAIN videos only, within temporalROI frames only. "
            "Apply to BOTH train and test to avoid leakage."
        ),
    }


# ---------------------------------------------------------------------------
# Dataset discovery helpers
# ---------------------------------------------------------------------------

def _load_split(split_path):
    with open(split_path, encoding="utf-8") as f:
        return json.load(f)


def _find_video_sequences(video_ids, dataset_root):
    """
    Given a list of video_ids ("category/video"), locate each video's
    input_dir, gt_dir, roi_start, roi_end under dataset_root.
    Returns {video_id: seq_dict} for found videos.
    """
    root = Path(dataset_root)
    seqs = {}
    for vid_id in video_ids:
        parts = vid_id.split("/")
        if len(parts) != 2:
            print(f"  [WARN] Unexpected video_id format (expected cat/vid): {vid_id}")
            continue
        cat, vid = parts
        video_dir = root / cat / vid
        input_dir = video_dir / "input"
        gt_dir    = video_dir / "groundtruth"
        if not input_dir.exists():
            print(f"  [WARN] Not found: {input_dir}")
            continue
        roi_start, roi_end = read_temporal_roi(video_dir)
        seqs[vid_id] = {
            "input_dir": str(input_dir),
            "gt_dir":    str(gt_dir),
            "roi_start": roi_start,
            "roi_end":   roi_end,
        }
    return seqs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CDNET_ROOT = r"D:\THS Programing\06.6 ASMAG Project\dataset cdnet2014\archive\dataset"


def main():
    ap = argparse.ArgumentParser(description="T1 -- canonical per-frame x_t dump")
    ap.add_argument("--cdnet-root",    default="", help="CDnet2014 dataset root (contains category folders)")
    ap.add_argument("--split",         default="", help="Path to split JSON (default: splits/S1_fold0.json)")
    ap.add_argument("--out-dir",       default="", help="Output directory (default: data/features/)")
    ap.add_argument("--overwrite",     action="store_true", help="Re-process even if NPZ already exists")
    ap.add_argument("--dry-run",       action="store_true", help="Print videos to be processed, then exit")
    ap.add_argument("--resize-width",  type=int, default=256,
                    help="Downscale frames to this width before gate processing (0=no resize). "
                         "Default=256 gives ~17fps on 704x480 vs 0.8fps without resize. "
                         "ALL videos MUST use the same value for feature consistency.")
    args = ap.parse_args()

    print_analysis()

    # Resolve paths
    cdnet_root = args.cdnet_root or DEFAULT_CDNET_ROOT
    if not Path(cdnet_root).exists():
        sys.exit(f"[ERROR] CDnet root not found: {cdnet_root}\nPass --cdnet-root <path>")

    split_path = args.split or str(REPO_ROOT / "splits" / "S1_fold0.json")
    if not Path(split_path).exists():
        sys.exit(f"[ERROR] Split file not found: {split_path}")

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = _load_split(split_path)
    train_ids = split["train"]
    test_ids  = split["test"]
    all_ids   = train_ids + test_ids

    print(f"Split: {split_path}")
    print(f"  train={len(train_ids)}  test={len(test_ids)}  total={len(all_ids)}")
    print(f"CDnet root: {cdnet_root}")
    print(f"Output dir: {out_dir}")
    print(f"resize_width_for_gating: {args.resize_width} ({'no resize' if args.resize_width == 0 else 'downscale before gate'})")
    print()

    seqs = _find_video_sequences(all_ids, cdnet_root)
    if not seqs:
        sys.exit("[ERROR] No video sequences found.")
    missing = [v for v in all_ids if v not in seqs]
    if missing:
        print(f"[WARN] {len(missing)} video(s) in split not found in dataset root:")
        for v in missing:
            print(f"  {v}")

    if args.dry_run:
        print("\n[DRY-RUN] Would process:")
        for vid_id in sorted(seqs):
            split_label = "TRAIN" if vid_id in train_ids else "TEST "
            print(f"  [{split_label}] {vid_id}")
        print(f"\n{len(seqs)} total  ({len(train_ids)} train, {len(test_ids)} test)")
        return

    # Process all videos
    processed = 0
    skipped   = 0
    errors    = []

    for vid_id in tqdm(sorted(seqs), desc="Dump x_t", unit="video"):
        seq    = seqs[vid_id]
        npz_path = out_dir / (vid_id + ".npz")

        if npz_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            result = _process_video(
                video_id     = vid_id,
                input_dir    = seq["input_dir"],
                gt_dir       = seq["gt_dir"],
                roi_start    = seq["roi_start"],
                roi_end      = seq["roi_end"],
                cfg          = {},
                resize_width = args.resize_width,
            )
        except Exception as exc:
            errors.append((vid_id, str(exc)))
            print(f"\n  [ERROR] {vid_id}: {exc}")
            continue

        if result is None:
            errors.append((vid_id, "no frames extracted"))
            continue

        save_npz(npz_path, result, vid_id)
        n_roi = int(result["temporal_roi"].sum())
        n_gt  = int(result["gt_event"].sum())
        tqdm.write(
            f"  {vid_id:40s}  frames={result['features'].shape[0]:5d}  "
            f"roi={n_roi:5d}  gt_events={n_gt:5d} -> {npz_path.relative_to(REPO_ROOT)}"
        )
        processed += 1

    print(f"\nDone.  processed={processed}  skipped(existing)={skipped}  errors={len(errors)}")
    if errors:
        print("Errors:")
        for v, e in errors:
            print(f"  {v}: {e}")

    # Compute and save train_stats.json (always recompute after a run)
    found_train = [v for v in train_ids if (out_dir / (v + ".npz")).exists()]
    if found_train:
        print(f"\nComputing train_stats.json from {len(found_train)} train videos...")
        stats = compute_train_stats(found_train, out_dir)
        stats_path = out_dir / "train_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved -> {stats_path}")
        print(f"  n_frames (train, within-ROI): {stats['n_frames']}")
        print()
        print("  Column                      mean       std")
        print("  " + "-" * 46)
        for col in ONLINE_MODE_FEATURE_COLUMNS:
            print(f"  {col:<28s}  {stats['mean'][col]:8.4f}   {stats['std'][col]:8.4f}")
    else:
        print("\n[WARN] No train NPZs found -- skipping train_stats.json")


if __name__ == "__main__":
    main()
