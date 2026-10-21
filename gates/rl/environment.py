"""
Single-video RL environment for ASMAG gate control (T4).

Obs : 10-dim z-scored features (same as CALIB input — no cheating on gt_event)
Act : 0=FAST  1=ACC  2=P3_FALLBACK
Reward: per-frame, SIMPLIFIED mode (no GUARD), identical logic to reward_simulator.py
W=7 staleness window (frozen from T3.7 calibration)
"""

from __future__ import annotations
import numpy as np
from asmag_core.controllers import ACTION_SPACE, compute_cost


class VideoEnv:
    """
    Sequential MDP over one video (ROI frames only).

    In simplified mode gate_open=True for cost regardless of action, matching
    reward_simulator.py line: gate_open_for_cost = True (simplified).
    """

    obs_dim = 10
    n_actions = 3

    def __init__(
        self,
        npz_data,
        z_mean: np.ndarray,
        z_std: np.ndarray,
        lambda_cost: float,
        P_miss: float,
        R_hit: float = 1.0,
        R_maintain: float = 0.2,
        R_quiet: float = 0.1,
        W: int = 7,
    ):
        self.lam = lambda_cost
        self.P_miss = P_miss
        self.R_hit = R_hit
        self.R_maintain = R_maintain
        self.R_quiet = R_quiet
        self.W = W

        roi = npz_data["temporal_roi"].astype(bool)
        feat = npz_data["features"][roi].astype(np.float32)
        self.obs = (feat - z_mean) / z_std          # (T_roi, 10)
        self.gt_event = npz_data["gt_event"][roi].astype(bool)
        self.fidx = npz_data["frame_idx"][roi].astype(np.int64)
        self.T = int(roi.sum())

        # full temporal_roi for building T_total action sequences
        self.temporal_roi = roi
        self.T_total = len(roi)

        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.t = 0
        self.event_caught = False
        self.prev_in_event = False
        self.last_detect_fidx = (
            int(self.fidx[0]) - self.W - 1 if self.W is not None else -1
        )

    def reset(self) -> np.ndarray:
        self._reset_state()
        return self.obs[0].copy()

    def step(self, action_idx: int):
        t = self.t
        action = ACTION_SPACE[action_idx]
        in_event = bool(self.gt_event[t])
        fid = int(self.fidx[t])
        is_det = action in ("ACC", "FAST")

        if in_event and not self.prev_in_event:
            self.event_caught = False
        if is_det:
            self.last_detect_fidx = fid

        is_fresh = (
            True
            if self.W is None
            else (fid - self.last_detect_fidx) <= self.W
        )

        if in_event:
            if is_det:
                R_acc = self.R_hit if not self.event_caught else 0.0
                self.event_caught = True
            elif self.event_caught and is_fresh:
                R_acc = self.R_maintain
            else:
                R_acc = -self.P_miss
        else:
            R_acc = self.R_quiet if not is_det else 0.0

        # simplified mode: gate always "open" for cost (matches reward_simulator.py)
        C = compute_cost(action, gate_open=True)
        reward = float(R_acc - self.lam * C)

        self.prev_in_event = in_event
        self.t += 1
        done = self.t >= self.T
        next_obs = self.obs[self.t].copy() if not done else np.zeros(10, dtype=np.float32)
        return next_obs, reward, done, {}

    # ------------------------------------------------------------------
    def build_full_sequence(self, actions_roi: list) -> list:
        """Map ROI-length action list back to T_total, filling pre-ROI with P3_FALLBACK."""
        seq = ["P3_FALLBACK"] * self.T_total
        roi_pos = 0
        for i, in_roi in enumerate(self.temporal_roi):
            if in_roi:
                seq[i] = ACTION_SPACE[actions_roi[roi_pos]] if isinstance(actions_roi[roi_pos], int) else actions_roi[roi_pos]
                roi_pos += 1
        return seq
