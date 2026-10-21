"""
DQN gate wrapper for T5 reality check.

Interface mirrors CALIB's sklearn predict() API so it can be swapped in
wherever the CALIB model is called.  Critical difference:
  CALIB.predict(raw_features)   -- pipeline has internal StandardScaler
  DQNGate.predict(raw_features) -- z-scoring is INTERNAL here too

Plug-in contract
----------------
  # T4b fair-shot policies live in outputs/rl/t4b/ with "fair_" prefix:
  gate = DQNGate.from_config("fair_dqn_lam0.20_seed2",
                              repo_root / "outputs/rl/t4b",
                              repo_root / "data/features/train_stats.json",
                              lambda_cost=0.20)
  # T4 original policies (not recommended — train-selected, no regularization):
  # gate = DQNGate.from_config("dqn_lam0.20_seed2", repo_root/"outputs/rl", ..., 0.20)
  gate.reset()                    # call once at start of each video
  mode = gate.process_frame(x_t)  # per-frame in production loop
  # OR batch:
  modes = gate.predict(X)         # (T, 10) raw features -> (T,) mode strings

Fail-open: any exception in process_frame() returns "ACC" (full detection).
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


# Lazy torch import so the module is importable without GPU/torch
_torch = None
def _get_torch():
    global _torch
    if _torch is None:
        import torch as t
        _torch = t
    return _torch


class DQNGate:
    """
    Wraps a trained DQNAgent and exposes a production-safe per-frame interface.
    z-scoring is internal — callers pass RAW feature vectors.
    """

    ACTION_SPACE = ["FAST", "ACC", "P3_FALLBACK"]

    def __init__(
        self,
        policy_path: str | Path,
        train_stats_path: str | Path,
        lambda_cost: float,
        obs_dim: int = 10,
        hidden: int = 64,
        n_actions: int = 3,
    ):
        self.lambda_cost = lambda_cost
        self._policy_path = Path(policy_path)
        self._load_stats(train_stats_path)
        self._load_policy(obs_dim, hidden, n_actions)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        tag: str,
        rl_out_dir: str | Path,
        train_stats_path: str | Path,
        lambda_cost: float,
    ) -> "DQNGate":
        """Load from tag like 'dqn_lam0.20_seed2'."""
        policy_path = Path(rl_out_dir) / f"policy_{tag}.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        return cls(policy_path, train_stats_path, lambda_cost)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_stats(self, path: str | Path):
        with open(str(path)) as f:
            s = json.load(f)
        cols = s["feature_columns"]
        self.z_mean = np.array([s["mean"][k] for k in cols], dtype=np.float32)
        self.z_std  = np.array([s["std"][k]  for k in cols], dtype=np.float32)
        self.z_std  = np.where(self.z_std < 1e-7, 1.0, self.z_std)

    def _load_policy(self, obs_dim: int, hidden: int, n_actions: int):
        torch = _get_torch()
        import torch.nn as nn

        class _QNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(obs_dim, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, n_actions),
                )
            def forward(self, x):
                return self.net(x)

        q = _QNet()
        ckpt = torch.load(str(self._policy_path), map_location="cpu")
        q.load_state_dict(ckpt["q_state_dict"])
        q.eval()
        self._q = q
        self._n_actions = n_actions

    def _zscore(self, features: np.ndarray) -> np.ndarray:
        return (features.astype(np.float32) - self.z_mean) / self.z_std

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        torch = _get_torch()
        with torch.no_grad():
            q = self._q(torch.FloatTensor(obs))
        return q.argmax(dim=-1).numpy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self):
        """Call at start of each video/session. DQN is memoryless; no-op."""
        pass

    def process_frame(self, features: np.ndarray) -> str:
        """
        Single-frame prediction on RAW feature vector.
        Returns mode string: "FAST", "ACC", or "P3_FALLBACK".
        Fail-open on any exception: returns "ACC".
        """
        try:
            obs = self._zscore(np.asarray(features, dtype=np.float32).reshape(1, -1))
            idx = int(self._forward(obs)[0])
            return self.ACTION_SPACE[idx]
        except Exception:
            return "ACC"

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Batch prediction on (T, 10) RAW feature matrix.
        Returns (T,) array of mode strings. Matches CALIB sklearn interface.
        Fail-open on exception: returns all "ACC".
        """
        try:
            obs = self._zscore(np.asarray(features, dtype=np.float32))
            idxs = self._forward(obs)
            return np.array([self.ACTION_SPACE[i] for i in idxs])
        except Exception:
            return np.full(len(features), "ACC")

    def predict_indices(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction returning int action indices (0/1/2)."""
        try:
            obs = self._zscore(np.asarray(features, dtype=np.float32))
            return self._forward(obs).astype(np.int32)
        except Exception:
            return np.ones(len(features), dtype=np.int32)   # fail-open = ACC = 1
