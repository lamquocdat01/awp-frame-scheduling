"""
T3 -- Re-fit CALIB on train-only split (fix leakage).

Original CALIB was fitted on frame_metrics.csv from two full-CDnet experiments,
which included test videos. This script re-fits using only T1 NPZ features from
the 27 train videos in splits/S1_fold0.json.

Method: IDENTICAL to original train_online_mode_policy.py (no method change):
  - 4 candidate models: DT3, DT5, LogisticRegression, RandomForest
  - Pipeline(StandardScaler + classifier) for LR
  - Pseudo-labels: CONTROLLER_CATEGORY_POLICY per category (TR_CTRL oracle)
  - Selection: 0.40*accuracy + 0.35*macro_f1 + 0.25*p3_fallback_recall
  - Validation: GroupShuffleSplit within train videos

Input features:
  - Raw T1 NPZ features (features[temporal_roi, :]) -- canonical, pipeline-independent
  - reuse_success_rate=0 and gate_closed_rate=0 everywhere (canonical contract T1)
  - StandardScaler handles zero-variance columns (stays 0, no contribution to LR)

Output:
  - models/calib_S1_fold0_trainonly.pkl (model artifact, compatible with ASMAG's mode_policy_path)
  - models/calib_S1_fold0_trainonly_metrics.csv

Usage:
    python scripts/train_calib.py [--split splits/S1_fold0.json] [--npz-dir data/features]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from asmag_core.controllers import (
    ACTION_SPACE,
    CONTROLLER_CATEGORY_POLICY,
    controller_mode_for_category,
)

FEATURE_COLUMNS = [
    "motion_density_mean",
    "motion_density_std",
    "component_count_mean",
    "component_count_std",
    "fd_mog_disagreement",
    "knn_mog_disagreement",
    "illumination_variance",
    "reuse_success_rate",
    "active_frame_rate",
    "gate_closed_rate",
]

MODE_ORDER = ["FAST", "ACC", "P3_FALLBACK"]


# ---------------------------------------------------------------------------
# Dataset construction from T1 NPZs
# ---------------------------------------------------------------------------

def load_train_data(train_video_ids: list[str], npz_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load T1 NPZ features for all train videos (ROI frames only).

    Returns
    -------
    X : float64 ndarray [N, 10]
    y : object ndarray [N]   -- pseudo-mode labels
    groups : object ndarray [N] -- video_id per row (for GroupShuffleSplit)
    """
    X_parts, y_parts, g_parts = [], [], []

    for video_id in sorted(train_video_ids):
        parts = video_id.replace("\\", "/").split("/")
        category, video = parts[-2], parts[-1]
        npz_path = npz_dir / category / f"{video}.npz"

        if not npz_path.exists():
            print(f"  WARNING: NPZ not found: {npz_path}")
            continue

        data = np.load(str(npz_path), allow_pickle=True)
        temporal_roi = data["temporal_roi"].astype(bool)
        features = data["features"][temporal_roi].astype(float)  # [T_roi, 10]

        # Replace inf/nan -> 0 (same as original load_feature_rows)
        features = np.where(np.isfinite(features), features, 0.0)

        pseudo_mode = controller_mode_for_category(category)
        T_roi = features.shape[0]

        X_parts.append(features)
        y_parts.append(np.array([pseudo_mode] * T_roi, dtype=object))
        g_parts.append(np.array([video_id] * T_roi, dtype=object))

    if not X_parts:
        raise RuntimeError("No NPZ data loaded. Check --npz-dir and split file.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(g_parts, axis=0)
    return X, y, groups


# ---------------------------------------------------------------------------
# Model definitions (identical to original train_online_mode_policy.py)
# ---------------------------------------------------------------------------

def candidate_models(random_state: int) -> dict:
    return {
        "decision_tree_depth3": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=random_state
        ),
        "decision_tree_depth5": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=random_state
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            ),
        ),
        "random_forest_small": RandomForestClassifier(
            n_estimators=80,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def score_model(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, labels=MODE_ORDER, average="macro", zero_division=0)
    p3_recall = recall_score(
        y_val, pred, labels=["P3_FALLBACK"], average="macro", zero_division=0
    )
    selection_score = 0.40 * accuracy + 0.35 * macro_f1 + 0.25 * p3_recall
    return {
        "mode_accuracy": accuracy,
        "macro_f1": macro_f1,
        "p3_fallback_recall": p3_recall,
        "selection_score": selection_score,
    }


def best_val_split(X, y, groups, validation_size: float, random_state: int):
    """
    GroupShuffleSplit (n_splits=32): pick split that has all modes in both halves.
    Identical logic to original train_online_mode_policy.py:split_by_video().
    """
    unique_modes = set(np.unique(y))
    splitter = GroupShuffleSplit(
        n_splits=32, test_size=validation_size, random_state=random_state
    )
    best = None
    for train_idx, val_idx in splitter.split(X, y, groups):
        train_modes = set(np.unique(y[train_idx]))
        val_modes = set(np.unique(y[val_idx]))
        score = len(train_modes & unique_modes) + len(val_modes & unique_modes)
        candidate = (score, train_idx, val_idx)
        if best is None or score > best[0]:
            best = candidate
        if train_modes == unique_modes and val_modes == unique_modes:
            return train_idx, val_idx
    return best[1], best[2]


def feature_importance(model) -> pd.DataFrame:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]
    if hasattr(estimator, "feature_importances_"):
        vals = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        vals = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        vals = np.zeros(len(FEATURE_COLUMNS))
    total = float(np.sum(np.abs(vals)))
    vals = vals / total if total > 0 else vals
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": vals})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="T3: Re-fit CALIB on train-only split."
    )
    parser.add_argument("--split", default="splits/S1_fold0.json")
    parser.add_argument("--npz-dir", default="data/features")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--model-name", default="calib_S1_fold0_trainonly")
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    with open(str(split_path)) as f:
        split_data = json.load(f)

    train_videos = split_data["train"]
    test_videos = split_data["test"]
    assert len(train_videos) == split_data["n_train"]

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_absolute():
        npz_dir = repo_root / npz_dir

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Acceptance check: zero test videos in training data ---
    print(f"\nT3 -- CALIB re-fit (train-only, leakage fix)")
    print(f"Split: {split_path.name}")
    print(f"Train videos ({len(train_videos)}):")
    for v in sorted(train_videos):
        print(f"  {v}")
    print(f"\nTest videos ({len(test_videos)}) -- ZERO of these will be used for fitting:")
    for v in sorted(test_videos):
        print(f"  {v}")

    # --- Load data ---
    print(f"\nLoading T1 NPZ features from {npz_dir} ...")
    X, y, groups = load_train_data(train_videos, npz_dir)

    # Verify no test video leaked in
    train_video_set = set(train_videos)
    test_video_set = set(test_videos)
    in_groups = set(np.unique(groups))
    leaked = in_groups & test_video_set
    assert not leaked, f"LEAKAGE: test videos in training data: {leaked}"
    missing = train_video_set - in_groups
    if missing:
        print(f"  WARNING: {len(missing)} train videos had no NPZ: {missing}")

    print(f"  Loaded: {X.shape[0]:,} frames  |  {len(in_groups)} videos  |  features: {X.shape[1]}")
    label_counts = {m: int(np.sum(y == m)) for m in MODE_ORDER}
    print(f"  Label distribution: {label_counts}")

    # Verify canonical contract (pipeline-independent)
    col_names = FEATURE_COLUMNS
    reuse_idx = col_names.index("reuse_success_rate")
    gate_idx = col_names.index("gate_closed_rate")
    reuse_all_zero = np.allclose(X[:, reuse_idx], 0.0)
    gate_all_zero = np.allclose(X[:, gate_idx], 0.0)
    print(f"  Canonical contract: reuse_success_rate=0 everywhere: {reuse_all_zero}")
    print(f"  Canonical contract: gate_closed_rate=0 everywhere:   {gate_all_zero}")
    if not (reuse_all_zero and gate_all_zero):
        print("  WARNING: canonical contract violated -- re-check T1 dump!")

    # --- Train/val split (within train videos only) ---
    train_idx, val_idx = best_val_split(X, y, groups, args.validation_size, args.random_state)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    train_groups = set(np.unique(groups[train_idx]))
    val_groups = set(np.unique(groups[val_idx]))
    assert not (train_groups & val_groups), "Video leakage in train/val split!"
    assert not (val_groups & test_video_set), "Test video in val split!"
    print(f"\n  Train split: {len(X_train):,} frames  |  {len(train_groups)} videos")
    print(f"  Val split:   {len(X_val):,} frames  |  {len(val_groups)} videos")
    print(f"  Val videos: {sorted(val_groups)}")

    # --- Fit & evaluate all candidates ---
    print("\nFitting candidate models ...")
    metrics_rows = []
    fitted = {}
    for name, model in candidate_models(args.random_state).items():
        model.fit(X_train, y_train)
        scores = score_model(model, X_val, y_val)
        fitted[name] = model
        metrics_rows.append({"policy": name, **scores})
        print(
            f"  {name:<28}  acc={scores['mode_accuracy']:.4f}  "
            f"f1={scores['macro_f1']:.4f}  p3r={scores['p3_fallback_recall']:.4f}  "
            f"sel={scores['selection_score']:.4f}"
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("selection_score", ascending=False)
    best_name = str(metrics_df.iloc[0]["policy"])
    best_model = fitted[best_name]
    best_scores = metrics_df.iloc[0].to_dict()
    print(f"\n  Selected: {best_name}  (selection_score={best_scores['selection_score']:.4f})")

    # --- Final fit on ALL train data (not just the split) ---
    print(f"\nRe-fitting {best_name} on ALL {X.shape[0]:,} train frames ...")
    final_model = candidate_models(args.random_state)[best_name]
    final_model.fit(X, y)

    # --- Acceptance: test on a sample test video (should produce valid, non-degenerate output) ---
    if test_videos:
        test_vid_id = test_videos[0]
        test_parts = test_vid_id.replace("\\", "/").split("/")
        test_cat, test_vid = test_parts[-2], test_parts[-1]
        test_npz_path = npz_dir / test_cat / f"{test_vid}.npz"
        if test_npz_path.exists():
            test_data = np.load(str(test_npz_path), allow_pickle=True)
            test_X = test_data["features"][test_data["temporal_roi"].astype(bool)].astype(float)
            test_X = np.where(np.isfinite(test_X), test_X, 0.0)
            test_preds = final_model.predict(test_X)
            unique_pred = list(np.unique(test_preds))
            counts = {m: int(np.sum(test_preds == m)) for m in unique_pred}
            print(f"\nAcceptance check -- predict on TEST video {test_vid_id}:")
            print(f"  Predictions: {counts}")
            is_degenerate = len(unique_pred) == 1
            if is_degenerate:
                print(f"  WARNING: degenerate predictions (all '{unique_pred[0]}')")
            else:
                print(f"  Non-degenerate ({len(unique_pred)} distinct modes). PASS.")
        else:
            print(f"\nAcceptance check: test NPZ not found ({test_npz_path}), skipping.")

    # --- Confusion matrix on val set ---
    val_pred = best_model.predict(X_val)
    cm = confusion_matrix(y_val, val_pred, labels=MODE_ORDER)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{m}" for m in MODE_ORDER],
        columns=[f"pred_{m}" for m in MODE_ORDER],
    )
    print(f"\nValidation confusion matrix:\n{cm_df.to_string()}")

    # --- Feature importance ---
    imp = feature_importance(final_model)
    print(f"\nFeature importance ({best_name}, top 5):")
    for _, row in imp.head(5).iterrows():
        print(f"  {row['feature']:<30}  {row['importance']:.4f}")

    # --- Save artifact ---
    model_path = out_dir / f"{args.model_name}.pkl"
    artifact = {
        "model": final_model,
        "policy_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "mode_order": MODE_ORDER,
        "category_policy": dict(CONTROLLER_CATEGORY_POLICY),
        "split_name": split_data.get("scheme", "S1") + f"_fold{split_data.get('fold', 0)}",
        "train_videos": sorted(train_videos),
        "n_train_frames": int(X.shape[0]),
        "validation_metrics": metrics_df.to_dict("records"),
        "canonical_contract": {
            "reuse_success_rate": "always 0 (pipeline-independent)",
            "gate_closed_rate": "always 0 (pipeline-independent)",
        },
    }
    with open(str(model_path), "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved: {model_path}")

    metrics_csv = out_dir / f"{args.model_name}_metrics.csv"
    metrics_df.to_csv(str(metrics_csv), index=False)
    print(f"Metrics:     {metrics_csv}")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("T3 SUMMARY")
    print("=" * 60)
    print(f"Fit on: {len(in_groups)} train videos (0 test videos)")
    print(f"Method: {best_name} (identical to original CALIB)")
    print(f"Frames: {X.shape[0]:,} (train-ROI only, no test data)")
    print(f"Val accuracy: {best_scores['mode_accuracy']:.4f}")
    print(f"Val macro-F1: {best_scores['macro_f1']:.4f}")
    print(f"Val P3-recall:{best_scores['p3_fallback_recall']:.4f}")
    print(f"Model:        {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
