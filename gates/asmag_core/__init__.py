"""
asmag_core — Publishable subset of ASMAG gate-control code.

Extracted from the ASMAG (Adaptive Surveillance with Multi-Architecture Gate)
project for use in the A3 RL-vs-Deterministic experiment.

Cite: [ASMAG JRTIP 2025 — placeholder]

Production components (AIPipelineEngine, PostgreSQL, Telegram, RTSP loop,
AAP sensing modules) are NOT included here.
"""
from .features import OnlineSceneDifficultyEstimator, ONLINE_MODE_FEATURE_COLUMNS
from .controllers import (
    CONTROLLER_CATEGORY_POLICY,
    CONTROLLER_MODE_TO_PIPELINE,
    controller_mode_for_category,
)
from .adaptive_gate import (
    ASMAGPlusGate,
    ASMAGPlusPrecisionGate,
    ASMAGPlusBalancedGate,
    ASMAGPlusEfficientGate,
    SharedGuardedGateBank,
)
from .arbitration import final_safety_arbitration_v2
from .cdnet_loader import (
    frame_number, read_temporal_roi, scan_cdnet, list_frames,
    read_frame, read_gt_for_frame, build_gt_index, gt_has_foreground,
)
from .metrics import (
    compute_binary_mask_metrics,
    aggregate_binary_mask_metrics,
    event_f1_from_rows,
    frame_energy,
    summarize_edge_energy,
)

__all__ = [
    "OnlineSceneDifficultyEstimator",
    "ONLINE_MODE_FEATURE_COLUMNS",
    "CONTROLLER_CATEGORY_POLICY",
    "CONTROLLER_MODE_TO_PIPELINE",
    "controller_mode_for_category",
    "ASMAGPlusGate",
    "ASMAGPlusPrecisionGate",
    "ASMAGPlusBalancedGate",
    "ASMAGPlusEfficientGate",
    "SharedGuardedGateBank",
    "final_safety_arbitration_v2",
    "compute_binary_mask_metrics",
    "aggregate_binary_mask_metrics",
    "event_f1_from_rows",
    "frame_energy",
    "summarize_edge_energy",
    "frame_number",
    "read_temporal_roi",
    "scan_cdnet",
    "list_frames",
    "read_frame",
    "read_gt_for_frame",
    "build_gt_index",
    "gt_has_foreground",
]
