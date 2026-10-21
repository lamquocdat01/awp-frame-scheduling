# Gate implementations (S / RF / DQN) — `asmag-rl-vs-deterministic`

This is the source tree that produces the **measured anchors** in Figure 1 of the paper.

- **S (score-gate):** deterministic threshold on the motion-density signal
  (`asmag_core/features.py`, `asmag_core/adaptive_gate.py`).
- **RF (random-forest CALIB gate):** `scripts/train_calib.py` trains a scikit-learn
  `RandomForestClassifier` on cheap per-frame features to predict detection-availability;
  trained model in `models/`. De-oracled (no ground-truth leakage at inference).
- **DQN (RL gate):** `rl/agents.py` (DQNAgent), `rl/environment.py`, trained via
  `scripts/train_rl_fair.py`; policies under `outputs/rl/` (`t4b` = fair/de-oracled).

Evaluation entry points: `analysis/` and, in the sibling `../evaluation/analysis/`,
`compute_vii.py` / `compute_vii_frontier.py` sweep each gate's threshold to trace the
recall–energy frontier at matched activation.

`data/`, `splits/`, `outputs/` hold cached features and trained artifacts for
reproducibility; large raw video datasets are external (see top-level README).
