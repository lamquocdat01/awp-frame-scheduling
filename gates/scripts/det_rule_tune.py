"""
A3 -- EF1-TUNED deterministic threshold rule (NOT a classifier, NO pseudo-labels).

Fixed-form rule on the motion signal d = motion_density_mean (feature[0], the
rolling/EMA-like motion density already in the NPZ):

    DETECT (ACC)  if d > tau     -> is_det=True  (YOLO runs)
    SKIP  (P3_FB) otherwise       -> is_det=False

(FAST and ACC are identical for EF1/activation in the sim -- is_det = FAST|ACC --
so the only EF1-relevant threshold is the DETECT/SKIP boundary; a FAST/ACC split
would change energy only. Hence a single tau.)

Tuning: grid-search tau on the 27 TRAIN videos to MAXIMISE mean train EF1 directly
through fast_simulate (no model fit, no TR_CTRL imitation). Also pick taus matched
to the DQN activation levels for a matched-activation comparison. Apply to the 26
TEST videos; build the frontier; paired per-video vs DQN and CALIB(ExtraTrees).

Sim-only, Windows-only. Does NOT touch DQN/CALIB/features (read-only).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesClassifier

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from asmag_core.controllers import ACTION_SPACE
from rl.fast_eval import fast_simulate, dqn_actions
from rl.agents import DQNAgent
from scripts.train_calib import load_train_data

P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W = 0.5, 1.0, 0.2, 0.1, 7
NPZ_DIR = _REPO / "data" / "features"
ACC_I, P3_I = ACTION_SPACE.index("ACC"), ACTION_SPACE.index("P3_FALLBACK")
DQN_ACT = {0.05: 0.627, 0.20: 0.476}   # de-oracled DQN activation targets


def load_npz(vid):
    c, n = vid.split("/")
    return np.load(str(NPZ_DIR / c / f"{n}.npz"), allow_pickle=True)


def d_signal(npz):
    roi = npz["temporal_roi"].astype(bool)
    return npz["features"][roi][:, 0].astype(np.float64)   # motion_density_mean


def rule_ef1_act(npz, tau, lam=0.20):
    d = d_signal(npz)
    actions = np.where(d > tau, ACC_I, P3_I).astype(np.int32)
    ep = fast_simulate(npz, actions, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
    return ep["event_f1"], ep["activation"], ep["total_reward"]


def mean_over(videos, fn):
    vals = [fn(v) for v in videos]
    return np.array(vals, dtype=float)


def main():
    split = json.load(open(str(_REPO / "splits" / "S1_fold0.json")))
    train, test = split["train"], split["test"]
    train_npz = {v: load_npz(v) for v in train}
    test_npz = {v: load_npz(v) for v in test}

    # ---- grid of tau from pooled TRAIN d quantiles ----
    pooled = np.concatenate([d_signal(train_npz[v]) for v in train])
    qs = np.quantile(pooled, np.linspace(0, 1, 161))
    taus = np.unique(np.concatenate([[-1.0], qs, [pooled.max() + 1e-6]]))

    # ---- evaluate each tau on TRAIN (mean EF1, act, reward@0.05/0.20) ----
    rows = []
    for tau in taus:
        ef1 = act = r05 = r20 = 0.0
        e_l, a_l = [], []
        rw05, rw20 = [], []
        for v in train:
            npz = train_npz[v]
            d = d_signal(npz)
            a = np.where(d > tau, ACC_I, P3_I).astype(np.int32)
            ep05 = fast_simulate(npz, a, 0.05, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            ep20 = fast_simulate(npz, a, 0.20, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]
            e_l.append(ep05["event_f1"]); a_l.append(ep05["activation"])
            rw05.append(ep05["total_reward"]); rw20.append(ep20["total_reward"])
        rows.append({"tau": tau, "train_ef1": float(np.mean(e_l)), "train_act": float(np.mean(a_l)),
                     "train_rew05": float(np.mean(rw05)), "train_rew20": float(np.mean(rw20))})
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}

    # ---- selections (all on TRAIN only) ----
    sel = {}
    sel["maxEF1"]   = float(R["tau"][np.argmax(R["train_ef1"])])
    sel["maxRew05"] = float(R["tau"][np.argmax(R["train_rew05"])])
    sel["maxRew20"] = float(R["tau"][np.argmax(R["train_rew20"])])
    for lam, atgt in DQN_ACT.items():
        sel[f"matchAct{lam}"] = float(R["tau"][np.argmin(np.abs(R["train_act"] - atgt))])

    print("=== TRAIN tau selections ===")
    for k, tau in sel.items():
        i = int(np.argmin(np.abs(R["tau"] - tau)))
        print(f"  {k:<12} tau={tau:.5f}  train_ef1={R['train_ef1'][i]:.4f}  train_act={R['train_act'][i]:.4f}")

    # ---- apply selected taus to TEST ----
    def test_point(tau):
        e = mean_over(test, lambda v: rule_ef1_act(test_npz[v], tau)[0])
        a = mean_over(test, lambda v: rule_ef1_act(test_npz[v], tau)[1])
        return float(e.mean()), float(e.std()), float(a.mean())

    print("\n=== TEST points for selected taus ===")
    test_sel = {}
    for k, tau in sel.items():
        ef1, ef1s, act = test_point(tau)
        test_sel[k] = (tau, ef1, ef1s, act)
        print(f"  {k:<12} tau={tau:.5f}  TEST ef1={ef1:.4f}+-{ef1s:.3f}  act={act:.4f}")

    # ---- TEST frontier (sweep) + max-EF1 on test ----
    fe, fa = [], []
    for tau in taus:
        ef1, _, act = test_point(tau)
        fe.append(ef1); fa.append(act)
    fe, fa = np.array(fe), np.array(fa)
    imax = int(np.argmax(fe))
    print(f"\nTEST frontier: max test EF1 = {fe[imax]:.4f} at act={fa[imax]:.4f} (tau={taus[imax]:.5f})")
    front = _REPO / "outputs" / "det_rule_frontier.csv"
    front.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(str(front), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tau", "test_ef1", "test_act"])
        for tau, e, a in zip(taus, fe, fa): w.writerow([f"{tau:.6f}", f"{e:.4f}", f"{a:.4f}"])

    # ---- per-video EF1 vectors on TEST for paired tests ----
    def calib_extratrees_pervideo():
        X, y, _ = load_train_data(train, NPZ_DIR); X = np.where(np.isfinite(X), X, 0.0)
        m = ExtraTreesClassifier(n_estimators=300, max_depth=14, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(X, y)
        out = []
        for v in test:
            npz = test_npz[v]; roi = npz["temporal_roi"].astype(bool)
            feat = np.where(np.isfinite(npz["features"][roi]), npz["features"][roi], 0.0).astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = m.predict(feat)
            acts = np.array([ACTION_SPACE.index(str(p)) for p in preds], dtype=np.int32)
            out.append(fast_simulate(npz, acts, 0.20, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]["event_f1"])
        return np.array(out)

    def dqn_pervideo(lam, seed=2):
        tr = json.load(open(str(NPZ_DIR / "train_stats.json")))
        cols = tr["feature_columns"]
        zm = np.array([tr["mean"][k] for k in cols], np.float32)
        zs = np.array([tr["std"][k] for k in cols], np.float32); zs = np.where(zs < 1e-7, 1.0, zs)
        ag = DQNAgent(seed=seed); ag.load(str(_REPO / "outputs" / "rl" / "t4b" / f"policy_fair_dqn_lam{lam:.2f}_seed{seed}.pt"))
        ag.eps = 0.0
        out = []
        for v in test:
            npz = test_npz[v]; roi = npz["temporal_roi"].astype(bool)
            obs = (npz["features"][roi].astype(np.float32) - zm) / zs
            acts = dqn_actions(ag, obs)
            out.append(fast_simulate(npz, acts, lam, P_MISS, R_HIT, R_MAINTAIN, R_QUIET, W)["episode"]["event_f1"])
        return np.array(out)

    def rule_pervideo(tau):
        return mean_over(test, lambda v: rule_ef1_act(test_npz[v], tau)[0])

    print("\n=== PAIRED per-video (26 test) ===")
    calib_v = calib_extratrees_pervideo()
    dqn05_v = dqn_pervideo(0.05)
    dqn20_v = dqn_pervideo(0.20)
    rule05_v = rule_pervideo(sel["matchAct0.05"])   # rule matched to DQN act 0.627
    rule20_v = rule_pervideo(sel["matchAct0.2"])    # rule matched to DQN act 0.476
    ruleE_v  = rule_pervideo(sel["maxEF1"])

    def paired(name, a, b):
        # a vs b: positive delta = a better
        d = a - b
        wins = int(np.sum(d > 1e-9)); losses = int(np.sum(d < -1e-9)); ties = int(np.sum(np.abs(d) <= 1e-9))
        try:
            p = wilcoxon(a, b, zero_method="wilcox").pvalue
        except Exception as e:
            p = float("nan")
        print(f"  {name:<34} mean({a.mean():.4f} vs {b.mean():.4f}) dELTA={d.mean():+.4f}  "
              f"W/T/L={wins}/{ties}/{losses}  Wilcoxon p={p:.4g}")

    print(" [DQN vs CALIB(ExtraTrees)]")
    paired("DQN l=0.05  vs CALIB", dqn05_v, calib_v)
    paired("DQN l=0.20  vs CALIB", dqn20_v, calib_v)
    print(" [DQN vs tuned-rule, matched activation]")
    paired("DQN l=0.05  vs rule@act0.627", dqn05_v, rule05_v)
    paired("DQN l=0.20  vs rule@act0.476", dqn20_v, rule20_v)
    print(" [DQN vs tuned-rule @ rule's max-EF1 point]")
    paired("DQN l=0.05  vs rule@maxEF1", dqn05_v, ruleE_v)

    print(f"\nmeans: CALIB={calib_v.mean():.4f} DQN05={dqn05_v.mean():.4f} DQN20={dqn20_v.mean():.4f} "
          f"rule@maxEF1={ruleE_v.mean():.4f} rule@0.627={rule05_v.mean():.4f} rule@0.476={rule20_v.mean():.4f}")
    print(f"\nSaved frontier: {front}")


if __name__ == "__main__":
    main()
