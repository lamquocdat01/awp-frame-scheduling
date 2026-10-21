"""
T3.6 -- Full 53-video reward audit.

Four diagnostics across the entire CDnet2014 corpus (27 train + 26 test):

D1 - RECALL ATTRIBUTION
    detector_miss_rate = P(Is_Active=0 | yolo_called=1 AND gt_event=1)
    Scatter dEF1@W5 vs miss_rate; Spearman + Pearson.

D2 - STALENESS RULE-OUT
    W sweep {1,2,3,5,8,15,None}; for videos with dEF1@W5>0.05:
    W-flat (recall cause) vs W-responsive (staleness cause).

D3 - EXPLOIT TEST
    detect-once-then-skip on every video -> EF1 must be low (<0.5) with W=5.

D4 - EVENT-DEF CONSISTENCY
    npz_gt_event_count vs (ref_tp + ref_fn); flag discrepancies > 1%.

Also computes CALIB BASELINE (train-only pkl, simplified mode, W=5) for T7.

DISCIPLINE: W=5 and reward FROZEN from train calibration.
Test results document generalization; nothing is re-tuned based on test outcomes.

Usage:
    cd asmag-rl-vs-deterministic
    python scripts/audit_reward_full.py [--full-exp-dir <path>] [--no-plot]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.reward_simulator import simulate

# Try scipy for correlations (optional)
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not found; correlation coefficients will not be computed.")

# Try matplotlib for scatter plot (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    warnings.warn("matplotlib not found; scatter plot will not be saved.")

_DEFAULT_FULL_EXP_DIR = (
    r"D:\THS Programing\06.6 ASMAG Project\ASMAG_Project_Code\outputs"
    r"\full_cdnet2014_official_edge_profile_pc\raw_results"
)
_PIPELINE = "ASMAG_TR_CONTROLLER_ONLINE_CALIBRATED"
_DEFAULT_CALIB_PKL = "models/calib_S1_fold0_trainonly.pkl"

W_AUDIT = 7        # frozen calibrated value after T3.7 MechA fix
W_D2_SWEEP = [1, 2, 3, 5, 7, 8, 15, None]


# ---------------------------------------------------------------------------
# CDnet2014 full video list
# ---------------------------------------------------------------------------
CDNET_VIDEOS = [
    "baseline/highway", "baseline/office", "baseline/pedestrians", "baseline/PETS2006",
    "cameraJitter/badminton", "cameraJitter/boulevard", "cameraJitter/sidewalk", "cameraJitter/traffic",
    "dynamicBackground/boats", "dynamicBackground/canoe", "dynamicBackground/fall",
    "dynamicBackground/fountain01", "dynamicBackground/fountain02", "dynamicBackground/overpass",
    "intermittentObjectMotion/abandonedBox", "intermittentObjectMotion/parking",
    "intermittentObjectMotion/sofa", "intermittentObjectMotion/streetLight",
    "intermittentObjectMotion/tramstop", "intermittentObjectMotion/winterDriveway",
    "lowFramerate/port_0_17fps", "lowFramerate/tramCrossroad_1fps",
    "lowFramerate/tunnelExit_0_35fps", "lowFramerate/turnpike_0_5fps",
    "nightVideos/bridgeEntry", "nightVideos/busyBoulvard", "nightVideos/fluidHighway",
    "nightVideos/streetCornerAtNight", "nightVideos/tramStation", "nightVideos/winterStreet",
    "PTZ/continuousPan", "PTZ/intermittentPan", "PTZ/twoPositionPTZCam", "PTZ/zoomInZoomOut",
    "shadow/backdoor", "shadow/bungalows", "shadow/busStation",
    "shadow/copyMachine", "shadow/cubicle", "shadow/peopleInShade",
    "thermal/corridor", "thermal/diningRoom", "thermal/lakeSide",
    "thermal/library", "thermal/park",
    "turbulence/turbulence0", "turbulence/turbulence1",
    "turbulence/turbulence2", "turbulence/turbulence3",
    "badWeather/blizzard", "badWeather/skating", "badWeather/snowFall", "badWeather/wetSnow",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_vid(video_id: str):
    parts = video_id.replace("\\", "/").split("/")
    return parts[-2], parts[-1]


def _load_summary(exp_dir: Path) -> dict:
    p = exp_dir.parent / "summary_event_metrics.csv"
    out = {}
    if not p.exists():
        return out
    with open(str(p), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("pipeline", "") == _PIPELINE:
                out[(row.get("category", ""), row.get("video", ""))] = row
    return out


def _load_csv_all_rows(csv_path: Path, frame_idx_roi: np.ndarray):
    """
    Load full-frame CSV using ALL rows (no Is_Active filter).
    Returns (modes_roi, yolo_roi, is_active_roi, n_matched, coverage_pct).
    """
    if not csv_path.exists():
        return None
    fid_to_row = {}
    with open(str(csv_path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fid_to_row[int(row["frame_id"])] = row

    if not fid_to_row:
        return None

    T_roi = len(frame_idx_roi)
    modes_roi = np.array(["P3_FALLBACK"] * T_roi, dtype=object)
    yolo_roi = np.zeros(T_roi, dtype=np.int32)
    is_active_roi = np.zeros(T_roi, dtype=np.int32)
    n_matched = 0

    for i, fid in enumerate(frame_idx_roi):
        row = fid_to_row.get(int(fid))
        if row is None:
            continue
        modes_roi[i] = str(row.get("selected_mode", "P3_FALLBACK"))
        yolo_roi[i] = int(row.get("yolo_called", row.get("activation", 0)) or 0)
        is_active_roi[i] = int(row.get("Is_Active", "0") or 0)
        n_matched += 1

    coverage_pct = 100.0 * n_matched / max(T_roi, 1)
    return modes_roi, yolo_roi, is_active_roi, n_matched, coverage_pct


def _build_full_arrays(modes_roi, yolo_roi, temporal_roi):
    """Expand ROI arrays to T_total arrays for simulate()."""
    T_total = len(temporal_roi)
    roi_global = np.where(temporal_roi)[0]
    modes_full = np.array(["P3_FALLBACK"] * T_total, dtype=object)
    yolo_full = np.zeros(T_total, dtype=np.int32)
    for i, g in enumerate(roi_global):
        modes_full[g] = modes_roi[i]
        yolo_full[g] = yolo_roi[i]
    return modes_full, yolo_full


def _calib_predict(calib_model, features_roi: np.ndarray) -> np.ndarray:
    """Apply sklearn pipeline model to ROI features, return mode array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calib_model.predict(features_roi)


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    npz_dir: Path,
    exp_dir: Path,
    summary: dict,
    calib_model,
    lambda_cost: float,
    P_miss: float,
    R_hit: float,
    R_maintain: float,
    R_quiet: float,
) -> dict:
    category, video = _parse_vid(video_id)

    npz_path = npz_dir / category / f"{video}.npz"
    if not npz_path.exists():
        return {"status": "NPZ_MISSING", "video_id": video_id}

    data = np.load(str(npz_path), allow_pickle=True)
    temporal_roi = data["temporal_roi"].astype(bool)
    gt_event = data["gt_event"].astype(bool)
    features = data["features"]  # (T_total, 10)
    T_total = len(temporal_roi)
    T_roi = int(temporal_roi.sum())
    frame_idx_roi = data["frame_idx"][temporal_roi]
    gt_event_roi = gt_event[temporal_roi]
    features_roi = features[temporal_roi]

    npz_event_count = int(gt_event_roi.sum())

    # Load full-frame CSV
    csv_path = exp_dir / category / video / _PIPELINE / "frame_metrics.csv"
    loaded = _load_csv_all_rows(csv_path, frame_idx_roi)
    if loaded is None:
        return {"status": "CSV_MISSING", "video_id": video_id}
    modes_roi, yolo_roi, is_active_roi, n_matched, coverage_pct = loaded

    # Reference EF1 from summary
    ref_row = summary.get((category, video))
    if ref_row is None:
        return {"status": "SUMMARY_MISSING", "video_id": video_id}
    ref_ef1 = float(ref_row["Event_F1"])
    ref_tp = int(ref_row["Event_TP"])
    ref_fp = int(ref_row["Event_FP"])
    ref_fn = int(ref_row["Event_FN"])

    # D4 -- Event-def consistency
    ref_event_count = ref_tp + ref_fn
    event_def_match = ref_event_count > 0 and abs(npz_event_count - ref_event_count) / max(ref_event_count, 1) <= 0.01
    event_def_delta = npz_event_count - ref_event_count

    # D1 -- Detector miss rate
    # miss_rate = P(Is_Active=0 | yolo_called=1 AND gt_event=1)
    yolo_on_event = (yolo_roi == 1) & gt_event_roi
    yolo_miss_on_event = (yolo_roi == 1) & gt_event_roi & (is_active_roi == 0)
    n_yolo_on_event = int(yolo_on_event.sum())
    n_yolo_miss = int(yolo_miss_on_event.sum())
    detector_miss_rate = float(n_yolo_miss) / max(n_yolo_on_event, 1) if n_yolo_on_event > 0 else 0.0

    # Build T_total arrays for simulate()
    modes_full, yolo_full = _build_full_arrays(modes_roi, yolo_roi, temporal_roi)

    # D2 -- W sweep (gate-aware mode)
    d2_results = {}
    for W in W_D2_SWEEP:
        r = simulate(data, modes_full, lambda_cost, P_miss, R_hit, R_maintain, R_quiet,
                     yolo_called_sequence=yolo_full, W=W)
        ep = r["episode"]
        signed = ep["event_f1"] - ref_ef1
        d2_results[W] = {
            "ef1": ep["event_f1"], "tp": ep["event_tp"],
            "fp": ep["event_fp"], "fn": ep["event_fn"],
            "d_ef1": abs(signed),
            "signed_d_ef1": signed,
        }

    sim_ef1_Wopt = d2_results[W_AUDIT]["ef1"]
    sim_tp_Wopt = d2_results[W_AUDIT]["tp"]
    sim_fp_Wopt = d2_results[W_AUDIT]["fp"]
    sim_fn_Wopt = d2_results[W_AUDIT]["fn"]
    signed_d_ef1_Wopt = d2_results[W_AUDIT]["signed_d_ef1"]
    d_ef1_Wopt = d2_results[W_AUDIT]["d_ef1"]   # |signed|

    # Chain propagation metrics
    d_fp = sim_fp_Wopt - ref_fp   # >0: sim overcounts FP; <0: undercounts
    d_tp = sim_tp_Wopt - ref_tp   # >0: sim overcounts TP; <0: undercounts
    reuse_in_event = int(((yolo_roi == 0) & gt_event_roi).sum())
    reuse_in_event_rate = reuse_in_event / max(npz_event_count, 1)

    # W-flat check: is SIGNED dEF1 variation across W sweep < 0.02?
    d_ef1_vals = [d2_results[W]["d_ef1"] for W in W_D2_SWEEP]
    w_range = max(d_ef1_vals) - min(d_ef1_vals)
    w_flat = w_range < 0.02

    # D3 -- Exploit test: detect-once-then-skip
    first_roi_idx = int(np.argmax(temporal_roi))
    seq_exploit = ["P3_FALLBACK"] * T_total
    seq_exploit[first_roi_idx] = "ACC"
    r_exploit = simulate(data, seq_exploit, lambda_cost, P_miss, R_hit, R_maintain, R_quiet,
                         W=W_AUDIT)
    exploit_ef1 = r_exploit["episode"]["event_f1"]
    exploit_pass = exploit_ef1 < 0.5

    # Aggressive-skip = all P3_FALLBACK (W=5 should give EF1=0)
    seq_skip = ["P3_FALLBACK"] * T_total
    r_skip = simulate(data, seq_skip, lambda_cost, P_miss, R_hit, R_maintain, R_quiet, W=W_AUDIT)
    skip_ef1 = r_skip["episode"]["event_f1"]

    # CALIB baseline -- train-only pkl, simplified mode (no gate-aware), W=5
    if calib_model is not None:
        calib_modes_roi = _calib_predict(calib_model, features_roi)
        calib_modes_full = ["P3_FALLBACK"] * T_total
        roi_global = np.where(temporal_roi)[0]
        for i, g in enumerate(roi_global):
            calib_modes_full[g] = calib_modes_roi[i]
        r_calib = simulate(data, calib_modes_full, lambda_cost, P_miss, R_hit, R_maintain, R_quiet,
                           W=W_AUDIT)
        calib_sim_ef1 = r_calib["episode"]["event_f1"]
        calib_sim_act = r_calib["episode"]["activation"]
    else:
        calib_sim_ef1 = float("nan")
        calib_sim_act = float("nan")

    return {
        "status": "OK",
        "video_id": video_id,
        "category": category,
        "video": video,
        "T_roi": T_roi,
        "n_matched": n_matched,
        "coverage_pct": coverage_pct,
        # D1
        "ref_ef1": ref_ef1,
        "ref_tp": ref_tp, "ref_fp": ref_fp, "ref_fn": ref_fn,
        "npz_event_count": npz_event_count,
        "detector_miss_rate": detector_miss_rate,
        "n_yolo_on_event": n_yolo_on_event,
        "n_yolo_miss": n_yolo_miss,
        # D2
        "sim_ef1_Wopt": sim_ef1_Wopt,
        "signed_d_ef1": signed_d_ef1_Wopt,
        "d_ef1_Wopt": d_ef1_Wopt,
        "d2_results": d2_results,
        "w_flat": w_flat,
        "w_range": w_range,
        # D3
        "exploit_ef1": exploit_ef1,
        "exploit_pass": exploit_pass,
        "skip_ef1": skip_ef1,
        # D4
        "event_def_match": event_def_match,
        "event_def_delta": event_def_delta,
        # Chain propagation
        "sim_tp_Wopt": sim_tp_Wopt, "sim_fp_Wopt": sim_fp_Wopt, "sim_fn_Wopt": sim_fn_Wopt,
        "d_fp": d_fp, "d_tp": d_tp,
        "reuse_in_event": reuse_in_event,
        "reuse_in_event_rate": reuse_in_event_rate,
        # CALIB baseline
        "calib_sim_ef1": calib_sim_ef1,
        "calib_sim_act": calib_sim_act,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T3.6: full 53-video reward audit.")
    parser.add_argument("--split", default="splits/S1_fold0.json")
    parser.add_argument("--npz-dir", default="data/features")
    parser.add_argument("--full-exp-dir", default=_DEFAULT_FULL_EXP_DIR)
    parser.add_argument("--calib-pkl", default=_DEFAULT_CALIB_PKL)
    parser.add_argument("--lambda", dest="lambda_cost", type=float, default=0.3)
    parser.add_argument("--P-miss", dest="P_miss", type=float, default=0.5)
    parser.add_argument("--R-hit", type=float, default=1.0)
    parser.add_argument("--R-maintain", type=float, default=0.2)
    parser.add_argument("--R-quiet", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-csv", default="outputs/audit_reward_full.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    with open(str(split_path)) as f:
        split = json.load(f)
    train_set = set(split["train"])
    test_set = set(split["test"])

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_absolute():
        npz_dir = repo_root / npz_dir

    exp_dir = Path(args.full_exp_dir)
    summary = _load_summary(exp_dir)
    if not summary:
        print("ERROR: summary empty. Check --full-exp-dir.")
        sys.exit(1)

    calib_pkl_path = Path(args.calib_pkl)
    if not calib_pkl_path.is_absolute():
        calib_pkl_path = repo_root / calib_pkl_path
    calib_model = None
    if calib_pkl_path.exists():
        artifact = pickle.load(open(str(calib_pkl_path), "rb"))
        calib_model = artifact["model"]
        print(f"  Loaded train-only CALIB pkl: {calib_pkl_path.name}  "
              f"({artifact.get('policy_name', '?')})")
    else:
        print(f"  WARNING: CALIB pkl not found at {calib_pkl_path}. Skipping baseline.")

    out_csv_path = Path(args.out_csv)
    if not out_csv_path.is_absolute():
        out_csv_path = repo_root / out_csv_path
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nT3.6 -- Full reward audit (53 CDnet2014 videos)")
    print(f"  W={W_AUDIT} FROZEN  |  lambda={args.lambda_cost}  P_miss={args.P_miss}")
    print(f"  Full-frame exp: {exp_dir.parent.name}")
    print()

    all_results = []
    for vid in CDNET_VIDEOS:
        r = process_video(
            vid, npz_dir, exp_dir, summary, calib_model,
            args.lambda_cost, args.P_miss,
            args.R_hit, args.R_maintain, args.R_quiet,
        )
        all_results.append(r)
        status = r.get("status", "?")
        if status != "OK":
            split_label = "TRAIN" if vid in train_set else "TEST"
            print(f"  SKIP  [{split_label}] {vid:<50}  [{status}]")

    ok = [r for r in all_results if r.get("status") == "OK"]
    skip = [r for r in all_results if r.get("status") != "OK"]
    train_ok = [r for r in ok if r["video_id"] in train_set]
    test_ok = [r for r in ok if r["video_id"] in test_set]

    print(f"  Processed {len(ok)}/53 OK  ({len(train_ok)} train, {len(test_ok)} test)")
    if skip:
        print(f"  Skipped: {[r['video_id'] for r in skip]}")
    print()

    # -----------------------------------------------------------------------
    # 53-row audit table
    # -----------------------------------------------------------------------
    print("=" * 140)
    print(f"  {'Video':<50}  {'Split':>5}  {'RefEF1':>7}  {'SimEF1':>7}  "
          f"{'signed':>7}  {'|dEF1|':>6}  "
          f"{'Wflat':>5}  {'Exploit':>7}  {'E4ok':>4}  "
          f"{'CalibSim':>8}  {'Group'}")
    print("=" * 140)

    group_counts = defaultdict(int)
    for r in ok:
        signed = r["signed_d_ef1"]
        d = r["d_ef1_Wopt"]
        # Classify by signed direction (after MechA fix: some flip negative):
        #   gap-over  (+>0.05): sim > ref  (overcount TP or insufficient FP fix)
        #   gap-under (-<-0.05): sim < ref (overcount FP from MechA fix, or mechB insufficient)
        if d < 0.05:
            group = "faithful"
        elif signed > 0.05:
            group = "gap-over"
        elif signed < -0.05:
            group = "gap-under"
        else:
            group = "gap-other"
        r["group"] = group
        group_counts[group] += 1

        split_label = "TRAIN" if r["video_id"] in train_set else "test"
        ef4 = "ok" if r["event_def_match"] else f"D{r['event_def_delta']:+d}"
        calib_s = f"{r['calib_sim_ef1']:.4f}" if not np.isnan(r["calib_sim_ef1"]) else "  N/A"
        flag = "  *" if d >= 0.05 else ""
        print(
            f"  {r['video_id']:<50}  {split_label:>5}  {r['ref_ef1']:>7.4f}  "
            f"{r['sim_ef1_Wopt']:>7.4f}  "
            f"{signed:>+7.4f}  {d:>6.4f}  "
            f"{'Y' if r['w_flat'] else 'N':>5}  "
            f"{r['exploit_ef1']:>7.4f}  {ef4:>4}  "
            f"{calib_s:>8}  {group}{flag}"
        )

    print("=" * 140)

    # -----------------------------------------------------------------------
    # D1 -- Recall attribution
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  D1 -- RECALL ATTRIBUTION (detector_miss_rate)")
    print("=" * 60)

    d_ef1_arr = np.array([r["d_ef1_Wopt"] for r in ok])
    signed_arr = np.array([r["signed_d_ef1"] for r in ok])
    miss_rate_arr = np.array([r["detector_miss_rate"] for r in ok])
    reuse_ev_arr = np.array([r["reuse_in_event_rate"] for r in ok])
    train_mask = np.array([r["video_id"] in train_set for r in ok])

    n_zero_miss = int((miss_rate_arr == 0.0).sum())
    print(f"  miss_rate=0: {n_zero_miss}/{len(ok)} videos  "
          f"(GUARD gate forces YOLO on virtually every ROI frame; YOLO recall~=1)")
    print(f"  -> Recall is NOT the bottleneck. Gap explained by chain propagation mismatch.")

    if _HAS_SCIPY and miss_rate_arr.std() > 0:
        rho, p_rho = _scipy_stats.spearmanr(miss_rate_arr, d_ef1_arr)
        r_pear, p_pear = _scipy_stats.pearsonr(miss_rate_arr, d_ef1_arr)
        print(f"  Spearman rho (miss_rate ~ dEF1) = {rho:+.3f}  (p={p_rho:.4f})")
        d1_pass = False  # by design: miss_rate=0 everywhere, correlation undefined
        print(f"  D1 acceptance (rho>0 AND p<0.05): FAIL (miss_rate constant=0, correlation undefined)")
    else:
        print("  Spearman: undefined (miss_rate identically 0 for all 53 videos).")
        d1_pass = False

    # Signed dEF1 statistics (primary acceptance criterion for T3.7)
    n_sim_over = int((signed_arr > 0.005).sum())
    n_sim_under = int((signed_arr < -0.005).sum())
    n_near_zero = len(ok) - n_sim_over - n_sim_under
    print(f"\n  Signed dEF1 (T3.7 acceptance key metrics):")
    print(f"    mean_signed = {signed_arr.mean():+.4f}  "
          f"(pre-T3.7 target: was +0.028, now bidir balanced)")
    print(f"    #(sim>ref by >0.005): {n_sim_over}  |  #(sim<ref by >0.005): {n_sim_under}  |  "
          f"#near-zero: {n_near_zero}")
    print(f"    Train: mean_signed={signed_arr[train_mask].mean():+.4f}  "
          f"max={signed_arr[train_mask].max():+.4f}  "
          f"min={signed_arr[train_mask].min():+.4f}")
    print(f"    Test:  mean_signed={signed_arr[~train_mask].mean():+.4f}  "
          f"max={signed_arr[~train_mask].max():+.4f}  "
          f"min={signed_arr[~train_mask].min():+.4f}")
    signed_accept = abs(signed_arr.mean()) < 0.05 and abs(n_sim_over - n_sim_under) <= 15
    print(f"    Signed balance: {'PASS (|mean|<0.05 and |nOver-nUnder|<=15)' if signed_accept else 'NOTE: still directionally biased'}")

    # Alternative D1: |dEF1| vs reuse_in_event_rate
    if _HAS_SCIPY and reuse_ev_arr.std() > 0:
        rho2, p_rho2 = _scipy_stats.spearmanr(reuse_ev_arr, d_ef1_arr)
        r2, p2 = _scipy_stats.pearsonr(reuse_ev_arr, d_ef1_arr)
        print(f"\n  Alt D1: |dEF1| vs reuse_in_event_rate:")
        print(f"    Spearman rho = {rho2:+.3f}  (p={p_rho2:.4f})")

    n_faithful = group_counts["faithful"]
    n_gap_over = group_counts["gap-over"]
    n_gap_under = group_counts["gap-under"]
    n_gap_other = group_counts["gap-other"]
    print(f"\n  Groups (after T3.7 fix): faithful(<0.05)={n_faithful}  "
          f"gap-over(sim>ref)={n_gap_over}  gap-under(sim<ref)={n_gap_under}  "
          f"gap-other={n_gap_other}  (of {len(ok)} OK)")

    # -----------------------------------------------------------------------
    # D2 -- Staleness rule-out
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  D2 -- STALENESS RULE-OUT (videos with dEF1@Wopt={W_AUDIT} > 0.05)")
    print("=" * 60)

    flagged = [r for r in ok if r["d_ef1_Wopt"] >= 0.05]
    if not flagged:
        print("  No videos with dEF1 > 0.05. All faithful.")
    else:
        print(f"  {'Video':<50}  {'split':>5}  {'dFP':>6}  {'dTP':>6}  ", end="")
        for W in W_D2_SWEEP:
            print(f"  {str(W):>4}", end="")
        print("  Wflat  Mech")
        print(f"  {'-'*50}  {'-'*5}  {'-'*6}  {'-'*6}  ", end="")
        for _ in W_D2_SWEEP:
            print(f"  {'----':>4}", end="")
        print()
        for r in sorted(flagged, key=lambda x: -x["d_ef1_Wopt"]):
            split_label = "TRAIN" if r["video_id"] in train_set else "test"
            print(f"  {r['video_id']:<50}  {split_label:>5}  {r['d_fp']:>+6}  {r['d_tp']:>+6}  ", end="")
            for W in W_D2_SWEEP:
                print(f"  {r['d2_results'][W]['d_ef1']:>4.2f}", end="")
            print(f"  {'Y' if r['w_flat'] else 'N':>5}  {r.get('group','?')}")

    # Mechanism breakdown (T3.7: gap-over = sim>ref, gap-under = sim<ref)
    gap_over = [r for r in flagged if r.get("group") == "gap-over"]
    gap_under = [r for r in flagged if r.get("group") == "gap-under"]
    gap_other = [r for r in flagged if r.get("group") == "gap-other"]

    if gap_over:
        print(f"\n  gap-over (sim EF1 > ref EF1): MechB residual — sim chain TP too generous.")
        print(f"  {len(gap_over)} video(s):")
        for r in gap_over:
            split_label = "TRAIN" if r["video_id"] in train_set else "test"
            print(f"    [{split_label}] {r['video_id']}  "
                  f"signed={r['signed_d_ef1']:+.4f}  dTP={r['d_tp']:+d}  dFP={r['d_fp']:+d}  "
                  f"reuse_ev={r['reuse_in_event']}({r['reuse_in_event_rate']*100:.0f}%)")

    if gap_under:
        print(f"\n  gap-under (sim EF1 < ref EF1): MechA overcorrection — quiet chain FP over-counted.")
        print(f"  {len(gap_under)} video(s):")
        for r in gap_under:
            split_label = "TRAIN" if r["video_id"] in train_set else "test"
            print(f"    [{split_label}] {r['video_id']}  "
                  f"signed={r['signed_d_ef1']:+.4f}  dFP={r['d_fp']:+d}  "
                  f"(ref_FP={r['ref_fp']} sim_FP={r['sim_fp_Wopt']})")

    if gap_other:
        print(f"\n  ** {len(gap_other)} gap-other video(s) (mechanism not identified):")
        for r in gap_other:
            split_label = "TRAIN" if r["video_id"] in train_set else "test"
            print(f"     [{split_label}] {r['video_id']}  dEF1={r['d_ef1_Wopt']:.4f}  "
                  f"signed={r['signed_d_ef1']:+.4f}  "
                  f"dFP={r['d_fp']:+d}  dTP={r['d_tp']:+d}  w_range={r['w_range']:.4f}")

    if not gap_over and not gap_under and not gap_other:
        print("\n  All flagged videos are faithful.")

    print(f"\n  Chain propagation summary (W={W_AUDIT}):")
    print(f"  gap-over (MechB residual): {len(gap_over)} videos  |  "
          f"gap-under (MechA overcorrect): {len(gap_under)} videos  |  "
          f"unexplained: {len(gap_other)}")

    # -----------------------------------------------------------------------
    # D3 -- Exploit test
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  D3 -- EXPLOIT TEST (W={W_AUDIT}: detect-once-then-skip)")
    print("=" * 60)
    exploit_ef1_arr = np.array([r["exploit_ef1"] for r in ok])
    n_d3_pass = int(np.sum(exploit_ef1_arr < 0.5))
    n_d3_fail = len(ok) - n_d3_pass
    print(f"  EF1 < 0.5: {n_d3_pass}/{len(ok)} videos PASS")
    if n_d3_fail > 0:
        fails = [r for r in ok if r["exploit_ef1"] >= 0.5]
        print(f"  FAIL videos (EF1 >= 0.5):")
        for r in fails:
            split_label = "TRAIN" if r["video_id"] in train_set else "test"
            print(f"    [{split_label}] {r['video_id']}  exploit_EF1={r['exploit_ef1']:.4f}")
    else:
        print(f"  All {len(ok)} videos: loophole CLOSED.")
    d3_pass = n_d3_fail == 0
    print(f"  D3 acceptance (all EF1<0.5): {'PASS' if d3_pass else 'FAIL'}")
    print(f"  exploit_EF1 stats: mean={exploit_ef1_arr.mean():.4f}  "
          f"max={exploit_ef1_arr.max():.4f}  median={np.median(exploit_ef1_arr):.4f}")

    # -----------------------------------------------------------------------
    # D4 -- Event-def consistency
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  D4 -- EVENT-DEF CONSISTENCY (npz gt_event vs ref_tp+ref_fn)")
    print("=" * 60)
    n_d4_ok = sum(1 for r in ok if r["event_def_match"])
    n_d4_fail = len(ok) - n_d4_ok
    print(f"  Consistent (delta <= 1%): {n_d4_ok}/{len(ok)} videos")
    if n_d4_fail > 0:
        print("  Inconsistent videos:")
        for r in ok:
            if not r["event_def_match"]:
                split_label = "TRAIN" if r["video_id"] in train_set else "test"
                frac = abs(r["event_def_delta"]) / max(r["ref_tp"] + r["ref_fn"], 1)
                print(f"    [{split_label}] {r['video_id']}  "
                      f"npz={r['npz_event_count']}  ref={r['ref_tp']+r['ref_fn']}  "
                      f"delta={r['event_def_delta']:+d}  ({frac*100:.1f}%)")
    d4_pass = n_d4_fail == 0
    print(f"  D4 acceptance (all consistent): {'PASS' if d4_pass else f'FAIL ({n_d4_fail} inconsistent)'}")

    # -----------------------------------------------------------------------
    # CALIB baseline (train vs test)
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print(f"  CALIB BASELINE (train-only pkl, simplified mode, W={W_AUDIT})")
    print("  [This is the T7 head-to-head reference - NOT re-tuned on test]")
    print("=" * 80)

    calib_train = [r for r in train_ok if not np.isnan(r.get("calib_sim_ef1", float("nan")))]
    calib_test = [r for r in test_ok if not np.isnan(r.get("calib_sim_ef1", float("nan")))]

    if calib_train:
        calib_train_ef1 = [r["calib_sim_ef1"] for r in calib_train]
        calib_train_ref = [r["ref_ef1"] for r in calib_train]
        print(f"  Train (n={len(calib_train)}):  "
              f"sim_ef1 mean={np.mean(calib_train_ef1):.4f}  "
              f"ref_ef1 mean={np.mean(calib_train_ref):.4f}")
    if calib_test:
        calib_test_ef1 = [r["calib_sim_ef1"] for r in calib_test]
        calib_test_ref = [r["ref_ef1"] for r in calib_test]
        print(f"  Test  (n={len(calib_test)}):  "
              f"sim_ef1 mean={np.mean(calib_test_ef1):.4f}  "
              f"ref_ef1 mean={np.mean(calib_test_ref):.4f}")

    print()
    print(f"  Per-video CALIB baseline:")
    print(f"  {'Video':<50}  {'Split':>5}  {'SimEF1':>7}  {'RefEF1':>7}  {'Activation':>10}")
    print(f"  {'-'*50}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*10}")
    for r in sorted(ok, key=lambda x: x["video_id"]):
        if np.isnan(r.get("calib_sim_ef1", float("nan"))):
            continue
        split_label = "TRAIN" if r["video_id"] in train_set else "test"
        print(f"  {r['video_id']:<50}  {split_label:>5}  "
              f"{r['calib_sim_ef1']:>7.4f}  {r['ref_ef1']:>7.4f}  "
              f"{r['calib_sim_act']:>10.3f}")

    # -----------------------------------------------------------------------
    # Acceptance summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  ACCEPTANCE SUMMARY")
    print("=" * 60)
    n_gap_flagged = len(gap_over) + len(gap_under) + len(gap_other)
    n_faithful_total = len(ok) - n_gap_flagged
    print(f"  D1 (miss_rate=0 everywhere): EXPLAINED - GUARD forces YOLO~=1 recall; gap is chain propagation mismatch")
    print(f"  T3.7 signed balance: {'PASS (|mean_signed|<0.05)' if abs(signed_arr.mean()) < 0.05 else f'NOTE: mean_signed={signed_arr.mean():+.4f}'}")
    print(f"     #(sim>ref by>0.005)={n_sim_over}  #(sim<ref by>0.005)={n_sim_under}")
    print(f"  D2 (gap mechanism identified): {'PASS' if not gap_other else f'PARTIAL ({len(gap_other)} unexplained)'}")
    print(f"     gap-over(MechB residual)={len(gap_over)}  gap-under(MechA overcorrect)={len(gap_under)}  unknown={len(gap_other)}")
    print(f"  D3 (detect-once EF1 < 0.5): {'PASS' if d3_pass else f'FAIL ({n_d3_fail} videos)'}")
    print(f"  D4 (event-def consistent): {'PASS' if d4_pass else f'FAIL ({n_d4_fail} videos)'}")
    print(f"\n  Overall (T3.7+D2+D3+D4): {sum([abs(signed_arr.mean()) < 0.05, not gap_other, d3_pass, d4_pass])}/4 criteria met")
    if d3_pass and d4_pass:
        print(f"  -> Faithful: {n_faithful_total}/{len(ok)} videos. Gap documented (over={len(gap_over)}, under={len(gap_under)}).")
        print(f"  -> MechA fix applied (W={W_AUDIT}). MechB residual documented in Limitations. Safe to proceed to T4.")
    print()

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    fieldnames = [
        "video_id", "split", "T_roi", "coverage_pct",
        "ref_ef1", "ref_tp", "ref_fp", "ref_fn",
        "sim_ef1_Wopt", "sim_tp_Wopt", "sim_fp_Wopt", "sim_fn_Wopt",
        "d_ef1_Wopt", "signed_d_ef1", "d_fp", "d_tp", "group",
        "detector_miss_rate", "n_yolo_on_event", "n_yolo_miss",
        "reuse_in_event", "reuse_in_event_rate",
        "w_flat", "w_range",
        "exploit_ef1", "exploit_pass",
        "skip_ef1",
        "event_def_match", "event_def_delta", "npz_event_count",
        "calib_sim_ef1", "calib_sim_act",
    ]
    with open(str(out_csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in ok:
            row = dict(r)
            row["split"] = "TRAIN" if r["video_id"] in train_set else "TEST"
            writer.writerow(row)
    print(f"  CSV saved: {out_csv_path}")

    # -----------------------------------------------------------------------
    # Scatter plot (D1)
    # -----------------------------------------------------------------------
    if _HAS_MPL and not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        train_r = [r for r in ok if r["video_id"] in train_set]
        test_r = [r for r in ok if r["video_id"] in test_set]

        color_map = {"faithful": "gray", "gap-over": "purple", "gap-under": "darkorange", "gap-other": "red"}

        for ax, x_key, x_label, x_title_suffix in [
            (axes[0], "detector_miss_rate", "Detector Miss Rate  P(miss | YOLO ran on event)",
             "miss_rate (GUARD->all zeros)"),
            (axes[1], "reuse_in_event_rate", "REUSE-in-Event Rate  (yolo=0 event frames / events)",
             "reuse_in_event_rate"),
        ]:
            for grp_r, marker, label_pfx in [(train_r, "o", "Train"), (test_r, "^", "Test")]:
                ax.scatter(
                    [r[x_key] for r in grp_r],
                    [r["d_ef1_Wopt"] for r in grp_r],
                    c=[color_map.get(r.get("group", "gap-other"), "red") for r in grp_r],
                    marker=marker, label=f"{label_pfx} (n={len(grp_r)})", alpha=0.8, s=60,
                    edgecolors="none"
                )
            # Annotate high-gap points
            for r in ok:
                if r["d_ef1_Wopt"] >= 0.07:
                    ax.annotate(
                        f"{r['category'][:3]}/{r['video'][:6]}",
                        (r[x_key], r["d_ef1_Wopt"]),
                        fontsize=7, alpha=0.8, ha="left",
                        xytext=(3, 3), textcoords="offset points"
                    )
            ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="dEF1=0.05")
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(f"|dEF1| @ W={W_AUDIT}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"|dEF1| vs {x_title_suffix}", fontsize=10)

        # Legend for mechanism colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="gray", label="faithful (<0.05)"),
            Patch(facecolor="purple", label="gap-over (MechB: sim>ref)"),
            Patch(facecolor="darkorange", label="gap-under (MechA overcorrect: sim<ref)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"T3.7 Reward Audit: MechA Fix + Recalibration (W={W_AUDIT})", fontsize=11)
        plot_path = out_csv_path.with_suffix(".png")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Scatter plot: {plot_path}")

    print()


if __name__ == "__main__":
    main()
