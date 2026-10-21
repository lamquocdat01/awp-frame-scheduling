"""
RL agents for ASMAG gate control (T4).

LinearPolicy  -- REINFORCE with softmax-linear policy (sanity baseline)
DQNAgent      -- Deep Q-Network, MLP 10->64->64->3 (main RL policy)
"""

from __future__ import annotations
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Linear softmax policy (REINFORCE)
# ---------------------------------------------------------------------------

class LinearPolicy:
    """
    Softmax linear policy trained with REINFORCE.
    Parameters: W (n_actions x obs_dim) + b (n_actions).
    """

    def __init__(self, obs_dim: int = 10, n_actions: int = 3, lr: float = 1e-2, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(n_actions, obs_dim).astype(np.float64) * 0.01
        self.b = np.zeros(n_actions, dtype=np.float64)
        self.lr = lr
        self.n_actions = n_actions
        self.obs_dim = obs_dim

    def _probs(self, obs: np.ndarray) -> np.ndarray:
        lg = self.W @ obs + self.b
        lg -= lg.max()
        e = np.exp(lg)
        return e / e.sum()

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        p = self._probs(obs)
        if greedy:
            return int(np.argmax(p))
        return int(np.random.choice(self.n_actions, p=p))

    def update(self, states: list, actions: list, returns: np.ndarray, gamma: float = 0.99):
        """
        REINFORCE gradient ascent step.
        returns: per-step discounted returns (already computed by caller).
        """
        ret = np.array(returns, dtype=np.float64)
        if ret.std() > 1e-8:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        T = len(states)
        for s, a, g in zip(states, actions, ret):
            p = self._probs(s)
            ind = np.zeros(self.n_actions)
            ind[a] = 1.0
            d = ind - p                            # score function
            dW += np.outer(d, s) * g
            db += d * g
        self.W += self.lr * dW / T
        self.b += self.lr * db / T

    def run_episode(self, env, greedy: bool = False, gamma: float = 0.99):
        """Collect full episode; return (states, actions, rewards, returns)."""
        obs = env.reset()
        states, actions, rewards = [], [], []
        done = False
        while not done:
            a = self.act(obs, greedy=greedy)
            states.append(obs)
            actions.append(a)
            obs, r, done, _ = env.step(a)
            rewards.append(r)

        # discounted returns
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float64)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G
        return states, actions, rewards, returns

    def get_greedy_actions(self, env) -> list:
        obs = env.reset()
        actions = []
        done = False
        while not done:
            a = self.act(obs, greedy=True)
            actions.append(a)
            obs, _, done, _ = env.step(a)
        return actions

    def save(self, path):
        np.savez(str(path), W=self.W, b=self.b, lr=self.lr)

    @classmethod
    def load(cls, path):
        d = np.load(str(path))
        pol = cls(obs_dim=d["W"].shape[1], n_actions=d["W"].shape[0], lr=float(d["lr"]))
        pol.W = d["W"]
        pol.b = d["b"]
        return pol


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

class _QNet(nn.Module):
    def __init__(self, obs_dim: int = 10, n_actions: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, int(a), float(r), s_next, float(done)))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    Double-DQN style agent with experience replay and target network.
    MLP: obs_dim -> hidden -> hidden -> n_actions
    """

    def __init__(
        self,
        obs_dim: int = 10,
        n_actions: int = 3,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 128,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 50,   # in gradient steps
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end

        self.q = _QNet(obs_dim, n_actions, hidden)
        self.q_target = _QNet(obs_dim, n_actions, hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buf = _ReplayBuffer(buffer_capacity)
        self._steps = 0

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.q(torch.FloatTensor(obs).unsqueeze(0))
        return int(q.argmax().item())

    def push(self, s, a, r, s_next, done):
        self.buf.push(s, a, r, s_next, done)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Vectorised batch push — ~10x faster than calling push() per frame."""
        tuples = list(zip(states, actions.tolist(), rewards.tolist(),
                          next_states, dones.tolist()))
        self.buf.buf.extend(tuples)

    def update(self) -> float | None:
        if len(self.buf) < self.batch_size:
            return None
        s, a, r, s2, d = self.buf.sample(self.batch_size)

        with torch.no_grad():
            # Double-DQN: online net selects action, target net evaluates
            a_online = self.q(s2).argmax(1, keepdim=True)
            q_next = self.q_target(s2).gather(1, a_online).squeeze(1)
            target = r + self.gamma * q_next * (1.0 - d)

        pred = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._steps += 1
        if self._steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss.item()

    def decay_eps(self, amount: float):
        self.eps = max(self.eps_end, self.eps - amount)

    def run_episode(self, env, greedy: bool = False, n_updates: int = 0):
        """
        Collect one episode; optionally do n_updates gradient steps inline.
        Returns (total_reward, losses).
        """
        obs = env.reset()
        total_r = 0.0
        losses = []
        done = False
        while not done:
            a = self.act(obs, greedy=greedy)
            s_prev = obs
            obs, r, done, _ = env.step(a)
            self.push(s_prev, a, r, obs, float(done))
            total_r += r

        if n_updates > 0:
            for _ in range(n_updates):
                loss = self.update()
                if loss is not None:
                    losses.append(loss)

        return total_r, losses

    def get_greedy_actions(self, env) -> list:
        obs = env.reset()
        actions = []
        done = False
        while not done:
            a = self.act(obs, greedy=True)
            actions.append(a)
            obs, _, done, _ = env.step(a)
        return actions

    def save(self, path):
        torch.save({
            "q_state_dict": self.q.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "eps": self.eps,
            "steps": self._steps,
        }, str(path))

    def load(self, path):
        ckpt = torch.load(str(path), map_location="cpu")
        self.q.load_state_dict(ckpt["q_state_dict"])
        self.q_target.load_state_dict(ckpt["q_state_dict"])
        self.opt.load_state_dict(ckpt["opt_state_dict"])
        self.eps = ckpt.get("eps", self.eps_end)
        self._steps = ckpt.get("steps", 0)
