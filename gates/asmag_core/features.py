"""
10-dimensional rolling feature vector x_t for ASMAG gate control.

Source: OnlineSceneDifficultyEstimator from ASMAG src/run_experiment.py
Extracted verbatim — do NOT reimplement this logic.

Both CALIB (deterministic) and the RL agent must receive the IDENTICAL x_t
computed by this module, so comparison is fair.
"""
import pickle
import numpy as np
import pandas as pd

ONLINE_MODE_FEATURE_COLUMNS = [
    "motion_density_mean",    # rolling mean of motion area / image area
    "motion_density_std",     # rolling std of motion density
    "component_count_mean",   # rolling mean of kept connected components
    "component_count_std",    # rolling std of component count
    "fd_mog_disagreement",    # mean |FD_area - MOG2_area| / image_area over window
    "knn_mog_disagreement",   # mean |KNN_area - MOG2_area| / image_area over window
    "illumination_variance",  # var(illumination_diff) over window
    "reuse_success_rate",     # fraction of frames where prediction was reused
    "active_frame_rate",      # fraction of frames marked is_active
    "gate_closed_rate",       # fraction of frames where gate was closed
]

# ---------------------------------------------------------------------------
# Canonical (policy-independent) feature enforcement
# ---------------------------------------------------------------------------
# Three of the ten features are aggregates of the policy's OWN runtime decisions:
#   idx 7 reuse_success_rate  -- did the policy reuse a prediction this frame?
#   idx 8 active_frame_rate   -- "is_active" for this frame
#   idx 9 gate_closed_rate    -- did the policy close the gate this frame?
# They are NOT policy-independent, so leaving them live makes x_t differ between
# CALIB and RL (each policy generates its own decision history) -> the controlled
# comparison is confounded. We hold all three to 0 so both policies see an
# IDENTICAL state. Notes:
#   * In the original ASMAG live controller, reuse_success_rate and
#     active_frame_rate were already constant 0 (reused_prediction / is_active
#     hardcoded to 0 at run_experiment.py:15104/18399).
#   * active_frame_rate has no motion/gating-free source in the original; the only
#     populated source is ground truth (Is_Active). Feeding GT (old A3 behavior)
#     turned idx 8 into a per-frame GT ORACLE. Zeroing it removes that leakage.
IDX_REUSE_SUCCESS = 7
IDX_ACTIVE_FRAME  = 8
IDX_GATE_CLOSED   = 9
CANONICAL_ZERO_INDICES = (IDX_REUSE_SUCCESS, IDX_ACTIVE_FRAME, IDX_GATE_CLOSED)


def canonicalize_feature_vector(x_t):
    """Zero the 3 gating-decision features so x_t is policy-independent.

    Single source of truth applied identically at feature dump (train) and engine
    inference (serve), so the gate sees the same distribution in both. Accepts a
    1-D [10] vector or a 2-D [T, 10] batch; returns a copy (never mutates input).
    """
    x = np.asarray(x_t).copy()
    if x.ndim == 1:
        for i in CANONICAL_ZERO_INDICES:
            x[i] = 0.0
    else:
        for i in CANONICAL_ZERO_INDICES:
            x[:, i] = 0.0
    return x


def compute_is_active(*args, **kwargs) -> bool:
    """Canonical is_active fed to OnlineSceneDifficultyEstimator — ALWAYS False.

    SINGLE SOURCE OF TRUTH guarding against GT re-contamination. The estimator must
    never receive a ground-truth signal: active_frame_rate (idx 8) is zeroed by
    canonicalize_feature_vector(), so the only consistent input is constant False.
    (In original ASMAG live, is_active was hardcoded 0; feeding GT here was the A3
    oracle bug.) Accepts/ignores any args so call sites can pass gt masks harmlessly.
    """
    return False


def normalize01(value, scale):
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(scale)))


def online_mode_from_difficulty(score, fast_threshold=0.30, p3_threshold=0.60):
    """Threshold-based fallback when no calibrated policy model is loaded."""
    if score < fast_threshold:
        return "FAST"
    if score < p3_threshold:
        return "ACC"
    return "P3_FALLBACK"


class OnlineSceneDifficultyEstimator:
    """
    Maintains a rolling window of per-frame gate statistics and computes
    the 10-dim feature vector x_t used by both CALIB and RL.

    Call update() once per frame with the gate_info dict produced by
    SharedGuardedGateBank.process().
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.window = int(cfg.get("window", 30))
        self.min_stable_frames = int(cfg.get("min_stable_frames", cfg.get("min_mode_duration", 10)))
        self.fast_threshold = float(cfg.get("fast_threshold", 0.30))
        self.p3_threshold = float(cfg.get("p3_threshold", 0.60))
        self.disagreement_high_threshold = float(cfg.get("disagreement_high_threshold", 0.35))
        self.fallback_boost_if_disagreement_high = bool(cfg.get("fallback_boost_if_disagreement_high", False))
        self.force_p3_if_fd_mog_disagreement_high = bool(cfg.get("force_p3_if_fd_mog_disagreement_high", False))
        self.force_p3_if_knn_mog_disagreement_high = bool(cfg.get("force_p3_if_knn_mog_disagreement_high", False))
        self.max_switch_per_100_frames = int(cfg.get("max_switch_per_100_frames", 6))
        self.use_hysteresis = bool(cfg.get("use_hysteresis", True))
        self.current_mode = str(cfg.get("initial_mode", "ACC"))
        self.candidate_mode = self.current_mode
        self.candidate_frames = 0
        self.mode_duration = 0
        self.mode_switch_count = 0
        self.switch_eval_indices = []
        self.history = []
        self.policy_model = None
        self.policy_name = ""
        self.policy_feature_columns = ONLINE_MODE_FEATURE_COLUMNS
        policy_path = str(cfg.get("mode_policy_path", "") or "").strip()
        if policy_path:
            with open(policy_path, "rb") as f:
                artifact = pickle.load(f)
            self.policy_model = artifact.get("model", artifact) if isinstance(artifact, dict) else artifact
            if isinstance(artifact, dict):
                self.policy_name = str(artifact.get("policy_name", "calibrated_policy"))
                self.policy_feature_columns = list(artifact.get("feature_columns", ONLINE_MODE_FEATURE_COLUMNS))
            else:
                self.policy_name = "calibrated_policy"

    def predict_calibrated_mode(self, features):
        """f(x_t) → mode.  Returns None if no model loaded (falls back to threshold rule)."""
        if self.policy_model is None:
            return None
        row = pd.DataFrame(
            [[float(features.get(col, 0.0) or 0.0) for col in self.policy_feature_columns]],
            columns=self.policy_feature_columns,
        )
        return str(self.policy_model.predict(row)[0])

    def update(self, gate_info, gate_open, reused_prediction, is_active, evaluated_index):
        """
        Ingest one frame of gate telemetry and return the feature dict.

        Args:
            gate_info: dict from SharedGuardedGateBank.process() — must contain
                       motion_density, kept_components, fd_area, mog_area,
                       knn_area, image_area (or frame_area), illumination_diff
            gate_open: bool
            reused_prediction: bool
            is_active: bool — whether this frame is within an active GT event
            evaluated_index: int — frame index within the evaluation window

        Returns:
            dict with ONLINE_MODE_FEATURE_COLUMNS keys PLUS:
                selected_mode, desired_mode, calibrated_policy_mode,
                SceneDifficulty, mode_switch_count, mode_duration, ...
        """
        frame_record = {
            "motion_density": float(gate_info.get("motion_density", 0.0) or 0.0),
            "component_count": float(gate_info.get("kept_components", 0.0) or 0.0),
            "fd_mog_disagreement": abs(float(gate_info.get("fd_area", 0.0) or 0.0) - float(gate_info.get("mog_area", 0.0) or 0.0))
            / max(1.0, float(gate_info.get("image_area", 0.0) or gate_info.get("frame_area", 0.0) or 1.0)),
            "knn_mog_disagreement": abs(float(gate_info.get("knn_area", 0.0) or 0.0) - float(gate_info.get("mog_area", 0.0) or 0.0))
            / max(1.0, float(gate_info.get("image_area", 0.0) or gate_info.get("frame_area", 0.0) or 1.0)),
            "illumination_diff": float(gate_info.get("illumination_diff", 0.0) or 0.0),
            "reuse_success": int(reused_prediction),
            "is_active": int(is_active),
            "gate_closed": int(not gate_open),
        }
        self.history.append(frame_record)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        h = pd.DataFrame(self.history)
        features = {
            "motion_density_mean": float(h["motion_density"].mean()),
            "motion_density_std": float(h["motion_density"].std(ddof=0)),
            "component_count_mean": float(h["component_count"].mean()),
            "component_count_std": float(h["component_count"].std(ddof=0)),
            "fd_mog_disagreement": float(h["fd_mog_disagreement"].mean()),
            "knn_mog_disagreement": float(h["knn_mog_disagreement"].mean()),
            "illumination_variance": float(h["illumination_diff"].var(ddof=0)),
            "reuse_success_rate": float(h["reuse_success"].mean()),
            "active_frame_rate": float(h["is_active"].mean()),
            "gate_closed_rate": float(h["gate_closed"].mean()),
        }
        features["motion_density_std_norm"] = normalize01(features["motion_density_std"], 0.05)
        features["component_count_std_norm"] = normalize01(features["component_count_std"], 12.0)
        features["fd_mog_disagreement_norm"] = normalize01(features["fd_mog_disagreement"], 0.08)
        features["knn_mog_disagreement_norm"] = normalize01(features["knn_mog_disagreement"], 0.08)
        features["illumination_variance_norm"] = normalize01(features["illumination_variance"], 400.0)
        features["reuse_failure_rate"] = 1.0 - features["reuse_success_rate"]
        features["gate_instability"] = min(features["gate_closed_rate"], 1.0 - features["gate_closed_rate"]) * 2.0

        difficulty = (
            0.20 * features["motion_density_std_norm"]
            + 0.15 * features["component_count_std_norm"]
            + 0.20 * features["fd_mog_disagreement_norm"]
            + 0.15 * features["illumination_variance_norm"]
            + 0.10 * features["reuse_failure_rate"]
            + 0.10 * features["active_frame_rate"]
            + 0.10 * features["gate_instability"]
        )

        fd_high = features["fd_mog_disagreement"] >= self.disagreement_high_threshold
        knn_high = features["knn_mog_disagreement"] >= self.disagreement_high_threshold
        if self.fallback_boost_if_disagreement_high and (fd_high or knn_high):
            difficulty = max(difficulty, self.p3_threshold)
        force_p3 = (
            (self.force_p3_if_fd_mog_disagreement_high and fd_high)
            or (self.force_p3_if_knn_mog_disagreement_high and knn_high)
        )

        calibrated_mode = self.predict_calibrated_mode(features)
        desired_mode = calibrated_mode or (
            "P3_FALLBACK" if force_p3 else online_mode_from_difficulty(
                difficulty, self.fast_threshold, self.p3_threshold
            )
        )

        if not self.use_hysteresis:
            if desired_mode != self.current_mode:
                self.mode_switch_count += 1
                self.mode_duration = 0
                self.switch_eval_indices.append(evaluated_index)
            else:
                self.mode_duration += 1
            self.current_mode = desired_mode
            self.candidate_mode = desired_mode
            self.candidate_frames = 0
            features.update({
                "SceneDifficulty": difficulty,
                "selected_mode": self.current_mode,
                "desired_mode": desired_mode,
                "calibrated_policy_mode": calibrated_mode or "",
                "calibrated_policy_name": self.policy_name,
                "emergency_p3_fallback": int(force_p3),
                "fd_mog_disagreement_high": int(fd_high),
                "knn_mog_disagreement_high": int(knn_high),
                "mode_switch_count": self.mode_switch_count,
                "mode_duration": self.mode_duration,
            })
            return features

        if desired_mode == self.current_mode:
            self.candidate_mode = desired_mode
            self.candidate_frames = 0
        elif desired_mode == self.candidate_mode:
            self.candidate_frames += 1
        else:
            self.candidate_mode = desired_mode
            self.candidate_frames = 1

        self.switch_eval_indices = [i for i in self.switch_eval_indices if evaluated_index - i < 100]
        can_switch = len(self.switch_eval_indices) < self.max_switch_per_100_frames
        if self.candidate_frames >= self.min_stable_frames and can_switch:
            self.current_mode = self.candidate_mode
            self.candidate_frames = 0
            self.mode_duration = 0
            self.mode_switch_count += 1
            self.switch_eval_indices.append(evaluated_index)
        else:
            self.mode_duration += 1

        features.update({
            "SceneDifficulty": difficulty,
            "selected_mode": self.current_mode,
            "desired_mode": desired_mode,
            "calibrated_policy_mode": calibrated_mode or "",
            "calibrated_policy_name": self.policy_name,
            "emergency_p3_fallback": int(force_p3),
            "fd_mog_disagreement_high": int(fd_high),
            "knn_mog_disagreement_high": int(knn_high),
            "mode_switch_count": self.mode_switch_count,
            "mode_duration": self.mode_duration,
        })
        return features

    def get_feature_vector(self, features_dict):
        """Extract the canonical 10-dim numpy array from a features dict."""
        return np.array(
            [float(features_dict.get(col, 0.0) or 0.0) for col in ONLINE_MODE_FEATURE_COLUMNS],
            dtype=np.float32,
        )

    def reset(self):
        self.history = []
        self.current_mode = "ACC"
        self.candidate_mode = "ACC"
        self.candidate_frames = 0
        self.mode_duration = 0
        self.mode_switch_count = 0
        self.switch_eval_indices = []
