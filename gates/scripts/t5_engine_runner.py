"""
T5 Engine Runner -- Jetson real-hardware evaluation with INA3221 energy.

Runs 6 configurations x 26 test videos on the ASMAG engine, measuring
real energy via INA3221 current/voltage sensor on Jetson.

Configurations:
  P4: CALIB-DT3, raw (no GUARD)
  P5: CALIB-DT3, +GUARD
  P6: DQN lam=0.05 seed=2, raw
  P7: DQN lam=0.05 seed=2, +GUARD
  P6: DQN lam=0.20 seed=2, raw
  P7: DQN lam=0.20 seed=2, +GUARD

Output: t5_real/*.csv -- one file per config, format: video,ef1,activation,energy_j

CRITICAL: Gate receives canonical x_t with all 3 gating-decision features ZEROED
(reuse_success_rate=7, active_frame_rate=8, gate_closed_rate=9) before feeding to
any gate (DQN or CALIB), so x_t is policy-independent and matches the training
(feature-dump) distribution.  active_frame_rate is zeroed to remove the GT oracle
(old A3 bug fed is_active=GT).  Enforced by the shared canonicalize_feature_vector().

Usage:
  # Full run (all 6 configs x 26 videos):
  python scripts/t5_engine_runner.py

  # Single config for debugging:
  python scripts/t5_engine_runner.py --config p4_calib_raw

  # Dry-run (no engine, sim-only energy proxy):
  python scripts/t5_engine_runner.py --dry-run

  # Custom dataset root:
  python scripts/t5_engine_runner.py --dataset-root /data/cdnet2014

INA3221 integration:
  On Jetson (Linux + sysfs): reads from /sys/bus/i2c/drivers/ina3221x/
  On non-Jetson: falls back to energy proxy (warns).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from rl.rl_gate import DQNGate
from asmag_core.adaptive_gate import SharedGuardedGateBank
from asmag_core.features import (
    OnlineSceneDifficultyEstimator,
    ONLINE_MODE_FEATURE_COLUMNS,
    canonicalize_feature_vector,
    compute_is_active,
)
from asmag_core.cdnet_loader import gt_event_label
from asmag_core.controllers import ACTION_SPACE
from asmag_core.metrics import (
    compute_binary_mask_metrics,
    event_f1_from_rows,
    frame_energy,
    DEFAULT_ENERGY_PROXY_CONFIG,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SPLIT_FILE   = _REPO / "splits" / "S1_fold0.json"
_NPZ_DIR      = _REPO / "data" / "features"
_TRAIN_STATS  = _REPO / "data" / "features" / "train_stats.json"
_CALIB_PKL    = _REPO / "models" / "calib_S1_fold0_trainonly.pkl"
_T4B_OUT      = _REPO / "outputs" / "rl" / "t4b"
_OUT_DIR      = _REPO / "t5_real"

# Config registry -- tag -> (gate_type, lambda, guard)
CONFIGS = {
    "p4_calib_raw":   {"gate": "calib", "lam": None,  "guard": False},
    "p5_calib_guard": {"gate": "calib", "lam": None,  "guard": True},
    "p6_dqn05_raw":   {"gate": "dqn",   "lam": 0.05,  "guard": False},
    "p7_dqn05_guard": {"gate": "dqn",   "lam": 0.05,  "guard": True},
    "p6_dqn20_raw":   {"gate": "dqn",   "lam": 0.20,  "guard": False},
    "p7_dqn20_guard": {"gate": "dqn",   "lam": 0.20,  "guard": True},
}

# GUARD gate config -- same as production ASMAG
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
    "p4_efficient": {
        "beta": 0.7,
        "open_threshold": 0.72,
        "close_threshold": 0.45,
        "force_close_threshold": 0.65,
        "max_open_streak": 15,
        "cooldown_frames": 3,
        "sample_interval": 10,
    },
    "online_controller_guarded": {
        "resize_width_for_gating": 256,
    },
    "edge_profile": {
        "resize_width_for_gating": 256,
    },
}


# ---------------------------------------------------------------------------
# INA3221 energy measurement (Jetson sysfs)
# ---------------------------------------------------------------------------

class INA3221:
    """
    Read power from INA3221 on Jetson via sysfs.
    Channel 0 = GPU, Channel 1 = CPU, Channel 2 = SOC (varies by board).

    Falls back to None if sysfs not available (non-Jetson platforms).
    """

    # Jetson sysfs paths (varies by board and L4T version)
    _SYSFS_CANDIDATES = [
        # Orin Nano (confirmed on dat@100.72.207.8)
        "/sys/bus/i2c/drivers/ina3221x/1-0040",
        "/sys/devices/platform/bus@0/c240000.i2c/i2c-1/1-0040",
        # Xavier NX / Nano
        "/sys/bus/i2c/drivers/ina3221x/6-0040",
        "/sys/bus/i2c/drivers/ina3221/1-0040",
        "/sys/devices/c240000.i2c/i2c-1/1-0040",
    ]

    def __init__(self, channel: int = 0):
        self.channel = channel
        self.sysfs_base = None
        self.hwmon_dir = None
        self._available = False
        self._find_sysfs()

    def _find_sysfs(self):
        # Strategy: find a sysfs base that has a hwmon subdirectory with curr*_input
        for base in self._SYSFS_CANDIDATES:
            hwmon_base = os.path.join(base, "hwmon")
            if os.path.isdir(hwmon_base):
                entries = sorted(os.listdir(hwmon_base))
                for entry in entries:
                    hwmon_path = os.path.join(hwmon_base, entry)
                    if os.path.exists(os.path.join(hwmon_path, "curr1_input")):
                        self.sysfs_base = base
                        self.hwmon_dir = hwmon_path
                        self._available = True
                        return
        # Fallback: scan /sys/class/hwmon for ina3221
        hwmon_class = "/sys/class/hwmon"
        if os.path.isdir(hwmon_class):
            for entry in sorted(os.listdir(hwmon_class)):
                hwmon_path = os.path.join(hwmon_class, entry)
                name_file = os.path.join(hwmon_path, "name")
                if os.path.exists(name_file):
                    try:
                        with open(name_file) as f:
                            if "ina3221" in f.read().strip().lower():
                                if os.path.exists(os.path.join(hwmon_path, "curr1_input")):
                                    self.sysfs_base = hwmon_path
                                    self.hwmon_dir = hwmon_path
                                    self._available = True
                                    return
                    except OSError:
                        pass

    @property
    def available(self) -> bool:
        return self._available

    def read_power_mw(self) -> float:
        """Read instantaneous power in milliwatts. Uses V*I from hwmon."""
        if not self._available or not self.hwmon_dir:
            return 0.0
        ch = self.channel
        try:
            # Orin hwmon: in{N}_input = voltage (mV), curr{N}_input = current (mA)
            # Channel mapping: ch=0 -> in1/curr1, ch=1 -> in2/curr2, etc.
            volt_file = os.path.join(self.hwmon_dir, f"in{ch + 1}_input")
            curr_file = os.path.join(self.hwmon_dir, f"curr{ch + 1}_input")
            with open(volt_file) as f:
                v_mv = float(f.read().strip())
            with open(curr_file) as f:
                i_ma = float(f.read().strip())
            return v_mv * i_ma / 1000.0  # mV * mA / 1000 = mW
        except (OSError, ValueError):
            return 0.0


class EnergyMeter:
    """
    Accumulates energy over a video by sampling INA3221 between frames.
    Falls back to sim proxy on non-Jetson platforms.
    """

    def __init__(self, ina: Optional[INA3221] = None):
        self.ina = ina
        self.use_real = ina is not None and ina.available
        self._energy_j = 0.0
        self._last_time = None
        self._sample_count = 0

    def reset(self):
        self._energy_j = 0.0
        self._last_time = time.monotonic() if self.use_real else None
        self._sample_count = 0

    def sample(self):
        """Call once per frame to accumulate energy (power x dt)."""
        if not self.use_real:
            return
        now = time.monotonic()
        if self._last_time is not None:
            dt = now - self._last_time
            power_mw = self.ina.read_power_mw()
            self._energy_j += (power_mw / 1000.0) * dt  # mW -> W, then W*s = J
            self._sample_count += 1
        self._last_time = now

    def add_proxy_energy(self, frame_info: dict):
        """Add sim proxy energy for one frame (non-Jetson fallback)."""
        if self.use_real:
            return
        self._energy_j += frame_energy(frame_info, DEFAULT_ENERGY_PROXY_CONFIG)
        self._sample_count += 1

    @property
    def total_joules(self) -> float:
        return self._energy_j


# ---------------------------------------------------------------------------
# Canonical feature enforcement
# ---------------------------------------------------------------------------

def _canonicalize_features(x_t: np.ndarray) -> np.ndarray:
    """
    Zero the 3 gating-decision features so x_t is policy-independent:
      idx 7 reuse_success_rate, idx 8 active_frame_rate, idx 9 gate_closed_rate.

    The feature dump (train) zeros the same three -> train == serve. Zeroing
    active_frame_rate (idx 8) removes the GT oracle the old A3 pipeline leaked
    (is_active=GT). Delegates to the single source of truth in asmag_core.features
    so train and serve can never drift apart.
    """
    return canonicalize_feature_vector(x_t)


# ---------------------------------------------------------------------------
# Video frame reader (CDnet2014 format)
# ---------------------------------------------------------------------------

def iter_cdnet_frames(dataset_root: Path, video_id: str, temporal_roi=None):
    """
    Yield (frame_idx, bgr_frame, gt_mask) for a CDnet video.

    CDnet2014 layout:
      {dataset_root}/{category}/{video}/input/in{frame:06d}.jpg
      {dataset_root}/{category}/{video}/groundtruth/gt{frame:06d}.png
      {dataset_root}/{category}/{video}/temporalROI.txt
    """
    cat, name = video_id.split("/")
    vid_dir = dataset_root / cat / name
    input_dir = vid_dir / "input"
    gt_dir = vid_dir / "groundtruth"

    # Read temporal ROI
    roi_file = vid_dir / "temporalROI.txt"
    if temporal_roi is None and roi_file.exists():
        with open(str(roi_file)) as f:
            parts = f.read().strip().split()
            roi_start, roi_end = int(parts[0]), int(parts[1])
    elif temporal_roi is not None:
        roi_start, roi_end = temporal_roi
    else:
        roi_start, roi_end = 1, 99999

    frame_idx = 1
    while True:
        img_path = input_dir / f"in{frame_idx:06d}.jpg"
        if not img_path.exists():
            # Try .png
            img_path = input_dir / f"in{frame_idx:06d}.png"
        if not img_path.exists():
            break

        in_roi = roi_start <= frame_idx <= roi_end
        bgr = cv2.imread(str(img_path))
        gt = None
        if in_roi:
            gt_path = gt_dir / f"gt{frame_idx:06d}.png"
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None
        # Yield ALL frames with in_roi flag so caller can warm up BG model
        # on pre-ROI frames and only evaluate within ROI.
        yield frame_idx, bgr, gt, in_roi

        frame_idx += 1


# ---------------------------------------------------------------------------
# Per-video engine evaluation
# ---------------------------------------------------------------------------

def evaluate_video(
    video_id: str,
    gate_model,           # DQNGate or sklearn model (CALIB)
    gate_type: str,       # "dqn" or "calib"
    use_guard: bool,
    dataset_root: Path,
    energy_meter: EnergyMeter,
    npz_data=None,        # for dry-run (sim-only) mode
    dry_run: bool = False,
) -> dict:
    """
    Run one video through the engine and return metrics.

    Returns: {"video": str, "ef1": float, "activation": float, "energy_j": float}
    """
    # Feature estimator (computes x_t from raw gate telemetry)
    estimator_cfg = {}
    if gate_type == "calib":
        estimator_cfg["mode_policy_path"] = str(_CALIB_PKL)
    estimator = OnlineSceneDifficultyEstimator(estimator_cfg)

    # SharedGuardedGateBank for feature extraction — ALWAYS used to match NPZ
    # extraction pipeline (256px resize). GUARD decisions only applied when use_guard=True.
    guard_bank = SharedGuardedGateBank(_GUARD_CFG)

    # Reset state
    if hasattr(gate_model, "reset"):
        gate_model.reset()
    estimator.reset()
    if guard_bank:
        guard_bank.reset_runtime_state()
    energy_meter.reset()

    frame_results = []
    prev_frame = None
    yolo_calls = 0
    total_frames = 0

    if dry_run and npz_data is not None:
        # Dry-run: use NPZ features directly (no real video frames)
        roi = npz_data["temporal_roi"].astype(bool)
        feat_raw = npz_data["features"][roi].astype(np.float32)
        gt_events = npz_data["gt_event"][roi].astype(bool)

        for t in range(len(feat_raw)):
            x_t = _canonicalize_features(feat_raw[t])
            total_frames += 1

            # Mode selection
            if gate_type == "dqn":
                mode = gate_model.process_frame(x_t)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mode = str(gate_model.predict(x_t.reshape(1, -1))[0])

            # GUARD override
            if use_guard:
                # In dry-run, GUARD would check if a lighter mode should be overridden.
                # Without real frames, we can't run the gate bank. Approximate:
                # GUARD forces ACC if the ASMAG gate score is above threshold.
                # For dry-run, just use the mode as-is (GUARD effect = 0).
                pass

            is_det = mode in ("FAST", "ACC")
            if is_det:
                yolo_calls += 1

            # Energy
            energy_meter.add_proxy_energy({
                "yolo_called": int(is_det),
                "reused_prediction": 0,
                "mog2_used": int(mode == "P3_FALLBACK"),
                "framediff_used": 1,
            })

            frame_results.append({
                "gt_event": bool(gt_events[t]),
                "is_det": is_det,
                "mode": mode,
            })
    else:
        # Real engine run (with video frames)
        # Process ALL frames (including pre-ROI) for BG subtractor warm-up.
        # Only evaluate frames within temporal ROI.
        all_frames_count = 0
        for frame_idx, bgr_frame, gt_mask, in_roi in iter_cdnet_frames(dataset_root, video_id):
            all_frames_count += 1

            # SharedGuardedGateBank on EVERY frame (BG model warm-up + feature extraction)
            candidates, telemetry = guard_bank.process(
                bgr_frame, prev_frame,
                state={"is_evaluation": True, "evaluated_index": all_frames_count - 1}
            )

            # GT event flag -- canonical label (foreground == 255) via the SHARED
            # helper, identical to the NPZ gt_event used by sim. Replaces the old
            # `gt_mask > 127`, which folded value 170 ("unknown", e.g. PTZ non-ROI)
            # into the event label and scored sim vs real against different GT.
            # Used ONLY for metrics/labels below (has_event); NEVER fed to the
            # estimator (is_active=False stays -- approved feature-oracle fix).
            is_active = gt_event_label(gt_mask)
            features_dict = estimator.update(
                gate_info=telemetry,
                gate_open=True,
                reused_prediction=False,
                is_active=compute_is_active(),
                evaluated_index=all_frames_count - 1,
            )

            prev_frame = bgr_frame

            # Skip non-ROI frames for evaluation (but they warmed up BG model)
            if not in_roi:
                continue

            total_frames += 1
            energy_meter.sample()

            # Extract and canonicalize x_t
            x_t = estimator.get_feature_vector(features_dict)
            x_t = _canonicalize_features(x_t)

            # Mode selection via gate
            if gate_type == "dqn":
                mode = gate_model.process_frame(x_t)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mode = str(gate_model.predict(x_t.reshape(1, -1))[0])

            # GUARD override (only applied when use_guard=True)
            if use_guard and mode in candidates:
                cand = candidates[mode]
                if not cand["gate_open"] and mode == "FAST":
                    mode = "ACC"

            is_det = mode in ("FAST", "ACC")
            if is_det:
                yolo_calls += 1

            has_event = is_active
            frame_results.append({
                "gt_event": has_event,
                "is_det": is_det,
                "mode": mode,
            })

            guard_bank.update_after_detection(mode, is_det and has_event)

    # Final energy sample
    energy_meter.sample()

    # Compute EF1 (simplified: detection available = is_det or recently detected)
    if not frame_results:
        return {"video": video_id, "ef1": 0.0, "activation": 0.0, "energy_j": 0.0}

    # Use fast_simulate-style EF1 with W=7 staleness
    W = 7
    gt_events = np.array([r["gt_event"] for r in frame_results], dtype=bool)
    is_dets = np.array([r["is_det"] for r in frame_results], dtype=bool)
    T = len(gt_events)

    # Vectorized last-detect staleness
    fidx = np.arange(T)
    det_pos = np.where(is_dets)[0]
    if len(det_pos) == 0:
        is_fresh = np.zeros(T, dtype=bool)
    else:
        idx = np.searchsorted(det_pos, fidx, side="right") - 1
        last_det = np.where(idx >= 0, det_pos[np.maximum(idx, 0)], -W - 2)
        is_fresh = (fidx - last_det) <= W

    detect_avail = is_dets | is_fresh
    quiet = ~gt_events
    detect_avail[quiet] = is_dets[quiet] | is_fresh[quiet]

    event_tp = int(np.sum(gt_events & detect_avail))
    event_fn = int(np.sum(gt_events & ~detect_avail))
    event_fp = int(np.sum(quiet & detect_avail))

    ef1_info = event_f1_from_rows([{
        "Event_TP": event_tp, "Event_FP": event_fp,
        "Event_FN": event_fn, "Event_TN": 0,
    }])

    activation = yolo_calls / total_frames if total_frames > 0 else 0.0

    return {
        "video": video_id,
        "ef1": ef1_info["event_f1"],
        "activation": activation,
        "energy_j": energy_meter.total_joules,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_config(
    config_tag: str,
    test_videos: list[str],
    dataset_root: Path,
    out_dir: Path,
    dry_run: bool = False,
):
    """Run one configuration on all test videos and save CSV."""
    cfg = CONFIGS[config_tag]
    gate_type = cfg["gate"]
    lam = cfg["lam"]
    use_guard = cfg["guard"]

    print(f"\n{'=' * 60}")
    print(f"Config: {config_tag}  (gate={gate_type}, lam={lam}, guard={use_guard})")
    print(f"{'=' * 60}")

    # Load gate model
    if gate_type == "dqn":
        tag = f"fair_dqn_lam{lam:.2f}_seed2"
        gate_model = DQNGate.from_config(tag, str(_T4B_OUT), str(_TRAIN_STATS), lambda_cost=lam)
    else:
        with open(str(_CALIB_PKL), "rb") as f:
            obj = pickle.load(f)
        gate_model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    # INA3221 energy
    ina = INA3221(channel=0)
    if ina.available:
        print(f"  INA3221: REAL energy via {ina.sysfs_base}")
    else:
        print(f"  INA3221: NOT FOUND -- energy_j values are PROXY (not real)")
    energy_meter = EnergyMeter(ina)

    results = []
    for i, vid in enumerate(test_videos):
        npz_data = None
        if dry_run:
            cat, name = vid.split("/")
            npz_data = np.load(str(_NPZ_DIR / cat / f"{name}.npz"), allow_pickle=True)

        r = evaluate_video(
            video_id=vid,
            gate_model=gate_model,
            gate_type=gate_type,
            use_guard=use_guard,
            dataset_root=dataset_root,
            energy_meter=energy_meter,
            npz_data=npz_data,
            dry_run=dry_run,
        )
        results.append(r)
        print(f"  [{i+1:2d}/{len(test_videos)}] {vid:<45} "
              f"EF1={r['ef1']:.4f}  act={r['activation']:.4f}  E={r['energy_j']:.1f}J")

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{config_tag}.csv"
    energy_src = "ina3221" if energy_meter.use_real else "proxy"
    with open(str(csv_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "ef1", "activation", "energy_j"])
        w.writeheader()
        for r in results:
            w.writerow({
                "video": r["video"],
                "ef1": f"{r['ef1']:.4f}",
                "activation": f"{r['activation']:.4f}",
                "energy_j": f"{r['energy_j']:.1f}",
            })
    # Sidecar metadata so post-hoc analysis knows energy source
    meta_path = out_dir / f"{config_tag}.meta.json"
    with open(str(meta_path), "w") as f:
        json.dump({"energy_source": energy_src, "config": config_tag,
                    "n_videos": len(results)}, f, indent=2)
    print(f"  -> Saved: {csv_path}  [energy: {energy_src}]")

    # Summary
    ef1s = [r["ef1"] for r in results]
    acts = [r["activation"] for r in results]
    engs = [r["energy_j"] for r in results]
    print(f"  Macro avg: EF1={np.mean(ef1s):.4f}+-{np.std(ef1s):.4f}  "
          f"act={np.mean(acts):.4f}  E={np.mean(engs):.1f}J")

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=None,
                    choices=list(CONFIGS.keys()),
                    help="Run a single config (default: all 6)")
    ap.add_argument("--dataset-root", type=str, default=None,
                    help="CDnet2014 dataset root (default: from splits/S1_fold0.json)")
    ap.add_argument("--out-dir", type=str, default=str(_OUT_DIR),
                    help=f"Output directory (default: {_OUT_DIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Sim-only mode: use NPZ features instead of real video frames")
    ap.add_argument("--allow-proxy-energy", action="store_true",
                    help="Allow energy proxy fallback on real runs. "
                         "Without this flag, real runs ABORT if INA3221 is not detected. "
                         "This prevents silently collecting fake energy numbers.")
    args = ap.parse_args()

    # Load split
    with open(str(_SPLIT_FILE)) as f:
        split = json.load(f)
    test_videos = split["test"]

    # Dataset root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path(split.get("dataset_root", ""))
    if not args.dry_run and not dataset_root.is_dir():
        print(f"Dataset root not found: {dataset_root}")
        print("Use --dataset-root or --dry-run")
        sys.exit(1)

    out_dir = Path(args.out_dir)

    # Run configs
    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())

    print(f"T5 Engine Runner -- {len(configs_to_run)} config(s) x {len(test_videos)} videos")
    if args.dry_run:
        print("  MODE: dry-run (NPZ features, energy proxy)")
    else:
        print(f"  MODE: real engine (dataset: {dataset_root})")

    # INA3221 hard gate: real runs must have real energy unless explicitly opted out
    if not args.dry_run and not args.allow_proxy_energy:
        ina_check = INA3221(channel=0)
        if not ina_check.available:
            print("ABORT: INA3221 not detected and --allow-proxy-energy not set.")
            print("  Real energy is the differentiator -- proxy numbers are meaningless.")
            print("  If on Jetson: check sysfs paths, I2C bus, kernel modules.")
            print("  If testing locally: use --dry-run or --allow-proxy-energy.")
            sys.exit(1)
        print(f"  INA3221: confirmed at {ina_check.sysfs_base}")

    all_results = {}
    for tag in configs_to_run:
        all_results[tag] = run_config(tag, test_videos, dataset_root, out_dir, args.dry_run)

    # Final summary table
    print(f"\n{'=' * 72}")
    print("T5 ENGINE RUNNER -- SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Config':<20} {'EF1':>7} {'+-std':>6} {'act':>7} {'E(J)':>8} {'n':>3}")
    print("-" * 52)
    for tag, results in all_results.items():
        ef1s = [r["ef1"] for r in results]
        acts = [r["activation"] for r in results]
        engs = [r["energy_j"] for r in results]
        print(f"{tag:<20} {np.mean(ef1s):>7.4f} {np.std(ef1s):>6.4f} "
              f"{np.mean(acts):>7.4f} {np.mean(engs):>8.1f} {len(results):>3}")

    print(f"\nOutput CSVs in: {out_dir}/")
    print("Next: python scripts/t5_eval.py --real-dir", str(out_dir))


if __name__ == "__main__":
    main()
