"""
CDnet2014 frame/GT loader — extracted from ASMAG src/data/cdnet_loader.py.

No production dependencies.
"""
import os
import glob
from pathlib import Path

import cv2
import numpy as np


def frame_number(path):
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    return int(digits) if digits else None


def read_temporal_roi(video_dir):
    """Read temporalROI.txt → (roi_start, roi_end) 1-based frame numbers, or (None, None)."""
    roi_path = Path(video_dir) / "temporalROI.txt"
    if not roi_path.exists():
        return None, None
    parts = roi_path.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2:
        return None, None
    return int(parts[0]), int(parts[1])


def scan_cdnet(dataset_root, categories, videos="auto"):
    """Discover CDnet sequences → list of dicts with category/video/input_dir/gt_dir/roi."""
    sequences = []
    root = Path(dataset_root)
    for cat in categories:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue
        if videos == "auto":
            video_names = [p.name for p in sorted(cat_dir.iterdir()) if p.is_dir()]
        else:
            video_names = videos.get(cat, [])
        for vid in video_names:
            video_dir = cat_dir / vid
            input_dir = video_dir / "input"
            gt_dir = video_dir / "groundtruth"
            if input_dir.exists():
                roi_start, roi_end = read_temporal_roi(video_dir)
                sequences.append({
                    "category": cat,
                    "video": vid,
                    "video_id": f"{cat}/{vid}",
                    "input_dir": str(input_dir),
                    "gt_dir": str(gt_dir),
                    "roi_start": roi_start,
                    "roi_end": roi_end,
                })
    return sequences


def list_frames(input_dir, gt_dir):
    """Return sorted lists of frame paths and GT paths."""
    frames = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    gts = sorted(glob.glob(os.path.join(gt_dir, "*.png"))) if os.path.exists(gt_dir) else []
    return frames, gts


def read_frame(path):
    return cv2.imread(path)


def read_gt(gt_files, idx, shape):
    if idx < len(gt_files):
        gt = cv2.imread(gt_files[idx], cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            return gt
    return np.zeros(shape[:2], dtype=np.uint8)


def read_gt_for_frame(gt_by_number, frame_file, shape):
    """Load GT mask for a specific frame file. Returns (H,W) uint8 array."""
    num = frame_number(frame_file)
    gt_path = gt_by_number.get(num)
    if gt_path:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            return gt
    return np.zeros(shape[:2], dtype=np.uint8)


def build_gt_index(gt_files):
    """Build {frame_number: path} dict from a sorted list of GT paths."""
    return {frame_number(p): p for p in gt_files if frame_number(p) is not None}


# Canonical CDnet2014 GT value semantics, matching the published ASMAG
# src/metrics/cdnet_pixel_metrics.py (foreground=255; background=0; 50/85/170 are
# ignore/unknown). The per-frame EVENT label = "any foreground pixel present" =
# np.any(gt in foreground_values). This is exactly valid_gt_mask()'s foreground set
# (fg is a subset of valid, so the valid/ignore masking does not change np.any(fg)).
# Using `> 127` would wrongly fold value 170 ("unknown", e.g. PTZ non-ROI regions)
# into the event label, inflating the event-rate and scoring sim vs real against
# DIFFERENT ground truth.
DEFAULT_CDNET_GT_CONFIG = {
    "foreground_values": (255,),
    "background_values": (0,),
    "ignore_values": (50, 85, 170),
}


def gt_event_label(gt_mask, cdnet_gt_config=None):
    """Canonical per-frame event label: True iff GT mask has any foreground pixel.

    SINGLE SOURCE OF TRUTH shared by the offline feature dump (NPZ ``gt_event``) and
    the T5 engine (``has_event`` metric) so both sides score against the SAME ground
    truth. Mirrors the published valid_gt_mask() foreground definition.
    """
    if gt_mask is None:
        return False
    gt = gt_mask
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    fg_values = (cdnet_gt_config or DEFAULT_CDNET_GT_CONFIG).get(
        "foreground_values", DEFAULT_CDNET_GT_CONFIG["foreground_values"]
    )
    return bool(np.any(np.isin(gt, list(fg_values))))


def gt_has_foreground(gt_mask):
    """Back-compat alias -> canonical gt_event_label (foreground == 255)."""
    return gt_event_label(gt_mask)
