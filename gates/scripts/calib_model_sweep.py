"""
A3 -- CALIB model sweep on de-oracled features (sim-only, Windows-only).

Goal: make the deterministic baseline the STRONGEST reasonable supervised
classifier on x_t (the 7 informative features; idx 7/8/9 are constant 0), so the
RL-vs-deterministic comparison is fair and reviewer-proof (anti strawman).

Method (identical no-leakage protocol to scripts/train_calib.py / T3):
  - Pseudo-labels: CONTROLLER_CATEGORY_POLICY per category.
  - GroupShuffleSplit train/val WITHIN the 27 train videos (best_val_split).
  - Fit each candidate on train split, score on val split.
  - SELECTION CRITERION: val macro-F1 (per request). T3 composite selection_score
    (0.40*acc + 0.35*macroF1 + 0.25*p3_recall) also reported.
  - Test videos NEVER used for fitting or selection.
  - For EVERY candidate: compute de-oracled SIM point on the test split
    (EF1, activation) by running predicted modes through fast_simulate.
    NOTE: EF1 and activation are lambda-independent (lambda only scales the
    reward, not the action sequence), so each classifier is a single point.

Does NOT touch DQN or features. Read-only on data/features.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import json
from asmag_core.controllers import ACTION_SPACE
from rl.fast_eval import fast_simulate
from scripts.train_calib import (
    FEATURE_COLUMNS, MODE_ORDER, load_train_data, best_val_split, feature_importance,
)

# Frozen reward constants (same as T3.7/T4)
P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W = 0.5, 1.0, 0.2, 0.1, 7
RS = 42
NPZ_DIR = _REPO / "data" / "features"


def candidates(rs: int) -> dict:
    """Wide set with light per-family hyperparameter variation."""
    return {
        "logreg_C1.0":        make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, random_state=rs)),
        "logreg_C0.3":        make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.3, random_state=rs)),
        "linsvc_C1.0":        make_pipeline(StandardScaler(), LinearSVC(
            C=1.0, class_weight="balanced", random_state=rs)),
        "dtree_depth3":       DecisionTreeClassifier(max_depth=3,  class_weight="balanced", random_state=rs),
        "dtree_depth5":       DecisionTreeClassifier(max_depth=5,  class_weight="balanced", random_state=rs),
        "dtree_depth7":       DecisionTreeClassifier(max_depth=7,  class_weight="balanced", random_state=rs),
        "dtree_depth10":      DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=rs),
        "rf_n100_d5":         RandomForestClassifier(n_estimators=100, max_depth=5,  min_samples_leaf=5,
                                                     class_weight="balanced", random_state=rs, n_jobs=-1),
        "rf_n300_d12":        RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                                     class_weight="balanced", random_state=rs, n_jobs=-1),
        "rf_n400_dNone":      RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                                     class_weight="balanced", random_state=rs, n_jobs=-1),
        "extratrees_n300_d14":ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                                   class_weight="balanced", random_state=rs, n_jobs=-1),
        "histgb_lr0.1_it200": HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200,
                                                             class_weight="balanced", random_state=rs),
        "histgb_lr0.05_it400":HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                                             max_leaf_nodes=31, class_weight="balanced",
                                                             random_state=rs),
    }


def val_scores(model, Xv, yv) -> dict:
    pred = model.predict(Xv)
    acc = accuracy_score(yv, pred)
    mf1 = f1_score(yv, pred, labels=MODE_ORDER, average="macro", zero_division=0)
    p3r = recall_score(yv, pred, labels=["P3_FALLBACK"], average="macro", zero_division=0)
    return {"val_acc": acc, "val_macro_f1": mf1, "val_p3_recall": p3r,
            "t3_sel_score": 0.40 * acc + 0.35 * mf1 + 0.25 * p3r}


def sim_point(model, test_videos: list) -> tuple[float, float, float]:
    """Mean (EF1, act) over test videos; also EF1 std across videos."""
    ef1s, acts = [], []
    for vid in test_videos:
        cat, name = vid.split("/")
        npz = np.load(str(NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)
        roi = npz["temporal_roi"].astype(bool)
        feat = npz["features"][roi].astype(np.float32)
        feat = np.where(np.isfinite(feat), feat, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(feat)
        actions = np.array([ACTION_SPACE.index(str(p)) for p in preds], dtype=np.int32)
        ep = fast_simulate(npz, actions, 0.20, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
        ef1s.append(ep["event_f1"]); acts.append(ep["activation"])
    return float(np.mean(ef1s)), float(np.std(ef1s)), float(np.mean(acts))


def main():
    split = json.load(open(str(_REPO / "splits" / "S1_fold0.json")))
    train_videos, test_videos = split["train"], split["test"]

    X, y, groups = load_train_data(train_videos, NPZ_DIR)
    X = np.where(np.isfinite(X), X, 0.0)
    tr_idx, va_idx = best_val_split(X, y, groups, 0.25, RS)
    Xt, yt, Xv, yv = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]
    assert not (set(np.unique(groups[tr_idx])) & set(np.unique(groups[va_idx]))), "val leak!"
    assert not (set(np.unique(groups[va_idx])) & set(test_videos)), "test in val!"
    print(f"train frames={len(Xt):,} ({len(set(groups[tr_idx]))} vids) | "
          f"val frames={len(Xv):,} ({len(set(groups[va_idx]))} vids) | test vids={len(test_videos)}")

    rows = []
    fitted = {}
    for name, model in candidates(RS).items():
        model.fit(Xt, yt)
        vs = val_scores(model, Xv, yv)
        # final fit on ALL 27 train videos (as T3 does) before the test sim
        final = candidates(RS)[name]
        final.fit(X, y)
        fitted[name] = final
        ef1, ef1_std, act = sim_point(final, test_videos)
        rows.append({"model": name, **vs, "sim_ef1": ef1, "sim_ef1_std": ef1_std, "sim_act": act})
        print(f"  {name:<22} valF1={vs['val_macro_f1']:.4f} sel={vs['t3_sel_score']:.4f} "
              f"| TEST ef1={ef1:.4f}±{ef1_std:.3f} act={act:.4f}")

    df = pd.DataFrame(rows)
    best_name = df.sort_values("val_macro_f1", ascending=False).iloc[0]["model"]
    best = df[df.model == best_name].iloc[0]

    out = _REPO / "outputs" / "calib_sweep_deoracle.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("val_macro_f1", ascending=False).to_csv(str(out), index=False)

    print("\n=== FULL TABLE (sorted by val_macro_f1) ===")
    show = df.sort_values("val_macro_f1", ascending=False)[
        ["model", "val_acc", "val_macro_f1", "val_p3_recall", "t3_sel_score",
         "sim_ef1", "sim_ef1_std", "sim_act"]]
    print(show.to_string(index=False,
          formatters={c: (lambda v: f"{v:.4f}") for c in show.columns if c != "model"}))

    print(f"\nBEST CALIB by val_macro_f1: {best_name}  "
          f"(valF1={best['val_macro_f1']:.4f}) -> SIM ef1={best['sim_ef1']:.4f}±{best['sim_ef1_std']:.3f} "
          f"act={best['sim_act']:.4f}")
    print("\nFeature importance (best, if available):")
    try:
        imp = feature_importance(fitted[best_name])
        for _, r in imp.head(7).iterrows():
            print(f"  {r['feature']:<28} {r['importance']:.4f}")
    except Exception as e:
        print(f"  (n/a: {e})")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
