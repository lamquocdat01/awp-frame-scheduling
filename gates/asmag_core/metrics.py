"""
Evaluation metrics for ASMAG A3 comparison.

Sources:
  - compute_binary_mask_metrics, aggregate_binary_mask_metrics, event_f1_from_rows:
      ASMAG evaluation/binary_mask_metrics.py
  - normalize_prediction_mask, normalize_gt_mask:
      ASMAG evaluation/mask_normalization.py
  - frame_energy, summarize_edge_energy, DEFAULT_ENERGY_PROXY_CONFIG:
      ASMAG src/metrics/edge_energy_metrics.py

No production dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Mask normalisation (from evaluation/mask_normalization.py)
# ---------------------------------------------------------------------------

def to_grayscale(mask) -> np.ndarray:
    import cv2
    arr = np.asarray(mask)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] >= 3 else arr[:, :, 0]
    return arr


def normalize_prediction_mask(mask, threshold: int = 127) -> np.ndarray:
    gray = to_grayscale(mask)
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[gray > threshold] = 255
    return out


def normalize_gt_mask(
    mask,
    foreground_values: Optional[Iterable[int]] = (255,),
    foreground_threshold: int = 127,
    ignore_mask=None,
) -> np.ndarray:
    gray = to_grayscale(mask)
    if foreground_values is None:
        foreground = gray > int(foreground_threshold)
    else:
        foreground = np.isin(gray, [int(v) for v in foreground_values])
    out = np.zeros(gray.shape, dtype=np.uint8)
    out[foreground] = 255
    if ignore_mask is not None:
        out[np.asarray(ignore_mask).astype(bool)] = 0
    return out


# ---------------------------------------------------------------------------
# Pixel-level metrics (from evaluation/binary_mask_metrics.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinaryMaskCounts:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def as_dict(self) -> dict:
        return {"tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn}


def counts_to_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    fmeasure = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + tn + fp + fn
    pwc = 100.0 * (fp + fn) / total if total else 0.0
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "pwc": pwc,
        "fmeasure": fmeasure,
    }


def compute_binary_mask_metrics(pred_mask, gt_mask, ignore_mask: Optional[np.ndarray] = None) -> dict:
    """Per-frame pixel-level TP/TN/FP/FN and derived metrics."""
    pred = normalize_prediction_mask(pred_mask) > 0
    gt = normalize_prediction_mask(gt_mask) > 0
    if ignore_mask is None:
        valid = np.ones(gt.shape, dtype=bool)
    else:
        valid = ~np.asarray(ignore_mask).astype(bool)
    tp = int(np.sum(pred & gt & valid))
    tn = int(np.sum((~pred) & (~gt) & valid))
    fp = int(np.sum(pred & (~gt) & valid))
    fn = int(np.sum((~pred) & gt & valid))
    return counts_to_metrics(tp, tn, fp, fn)


def aggregate_binary_mask_metrics(rows: Iterable[Mapping]) -> dict:
    """Aggregate pixel metrics over a list of per-frame dicts."""
    rows = list(rows)
    tp = sum(int(row.get("tp", row.get("TP_pixel", 0)) or 0) for row in rows)
    tn = sum(int(row.get("tn", row.get("TN_pixel", 0)) or 0) for row in rows)
    fp = sum(int(row.get("fp", row.get("FP_pixel", 0)) or 0) for row in rows)
    fn = sum(int(row.get("fn", row.get("FN_pixel", 0)) or 0) for row in rows)
    return counts_to_metrics(tp, tn, fp, fn)


# ---------------------------------------------------------------------------
# Event-level metrics (from evaluation/binary_mask_metrics.py)
# ---------------------------------------------------------------------------

def event_f1_from_rows(rows: Iterable[Mapping]) -> dict:
    """
    Aggregate event-level TP/FP/FN from per-frame rows that have
    Event_TP / Event_FP / Event_FN columns (or snake_case variants).
    """
    rows = list(rows)
    tp = sum(int(row.get("Event_TP", row.get("event_tp", 0)) or 0) for row in rows)
    tn = sum(int(row.get("Event_TN", row.get("event_tn", 0)) or 0) for row in rows)
    fp = sum(int(row.get("Event_FP", row.get("event_fp", 0)) or 0) for row in rows)
    fn = sum(int(row.get("Event_FN", row.get("event_fn", 0)) or 0) for row in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "event_tp": int(tp), "event_tn": int(tn), "event_fp": int(fp), "event_fn": int(fn),
        "event_precision": precision,
        "event_recall": recall,
        "event_accuracy": accuracy,
        "event_f1": f1,
    }


# ---------------------------------------------------------------------------
# Energy proxy (from src/metrics/edge_energy_metrics.py)
# ---------------------------------------------------------------------------

DEFAULT_ENERGY_PROXY_CONFIG = {
    "E_base": 1.0,
    "E_gate": 0.2,
    "E_yolo": 5.0,
    "E_reuse": 0.1,
    "E_mog2": 0.5,
    "E_framediff": 0.1,
    "unit": "relative_energy_unit",
}


def normalize_energy_config(config=None) -> dict:
    energy = dict(DEFAULT_ENERGY_PROXY_CONFIG)
    if config:
        energy.update(config)
    for key in ["E_base", "E_gate", "E_yolo", "E_reuse", "E_mog2", "E_framediff"]:
        energy[key] = float(energy[key])
    return energy


def frame_energy(row: Mapping, config=None) -> float:
    """
    Compute per-frame energy proxy.

    row must contain:
        yolo_called (0/1), reused_prediction (0/1),
        mog2_used (0/1), framediff_used (0/1)
    """
    energy = normalize_energy_config(config)
    return (
        energy["E_base"]
        + energy["E_gate"]
        + int(row.get("yolo_called", 0)) * energy["E_yolo"]
        + int(row.get("reused_prediction", 0)) * energy["E_reuse"]
        + int(row.get("mog2_used", 0)) * energy["E_mog2"]
        + int(row.get("framediff_used", 0)) * energy["E_framediff"]
    )


def activation_rate(rows: Iterable[Mapping]) -> float:
    """Fraction of frames where YOLO was called (gate_open=1 / yolo_called=1)."""
    rows = list(rows)
    if not rows:
        return 0.0
    called = sum(int(r.get("yolo_called", r.get("activation", 0)) or 0) for r in rows)
    return called / len(rows)


def summarize_edge_energy(frame_rows: list, config=None) -> pd.DataFrame:
    """Group-level energy summary table."""
    if not frame_rows:
        return pd.DataFrame()
    energy = normalize_energy_config(config)
    df = pd.DataFrame(frame_rows).copy()
    if "energy_frame" not in df.columns:
        df["energy_frame"] = df.apply(lambda row: frame_energy(row, energy), axis=1)
    group_keys = [k for k in ("category", "video", "pipeline") if k in df.columns]
    if not group_keys:
        group_keys = ["video"] if "video" in df.columns else []
    if not group_keys:
        return df[["energy_frame"]].describe()
    grouped = df.groupby(group_keys, as_index=False).agg(
        Estimated_energy_total=("energy_frame", "sum"),
        processed_frames=("energy_frame", "count"),
        yolo_calls=("yolo_called", "sum") if "yolo_called" in df.columns else ("energy_frame", "count"),
    )
    grouped["Estimated_energy_per_frame"] = (
        grouped["Estimated_energy_total"] / grouped["processed_frames"].clip(lower=1)
    )
    return grouped
