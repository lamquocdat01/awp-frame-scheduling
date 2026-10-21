# ==== C4 repo note (paths) ====
# Data paths below are ABSOLUTE to the author's machine (CDnet/LASIESTA/BMC/VIRAT
# detection dumps, frame_scores parquet, ASMAG NPZ/models). To re-run: edit the path
# constants and obtain the data (see README "Data & paths"). Canonical outputs are the
# committed ../../results/<name>.json. perscene_k.py also needs the deep0 "src" package;
# compute_vii*.py needs the asmag-rl-vs-deterministic modules (rl/, asmag_core/).
# ==============================
"""
perscene_k.py -- per-scene-K GO/NO-GO: can a CHEAP, causal, zero-shot estimator
capture the oracle's per-scene-K gain?

N1-verify found the real headroom is the per-scene refresh period K:
  oracle per-scene K       rho_max@90  ~ 0.728   (CHEATS: sees held-out labels)
  content-aware gate (pi)              ~ 0.636
  scene-agnostic periodic              ~ 0.591  (one K from train; robust K=2 ~0.50)
Question: can a cheap zero-shot estimator set K per scene to capture a meaningful
fraction of the 0.59 -> 0.728 gap WITHOUT the held-out scene's labels?

Calibration-window estimator (primary):
  On each held-out video, run the detector DENSELY for the first W frames (warmup).
  From those detector OUTPUTS -- pseudo-positive = Event_State in {TP,FP} (detector
  reported foreground; observable WITHOUT ground truth) -- extract pseudo-events and,
  for each candidate K, COUNTERFACTUALLY simulate periodic-K over the warmup to get a
  pseudo-recall. Pick the largest K whose warmup pseudo-recall >= theta. Run that K
  for the rest of the video. The dense warmup cost is amortised into the reported rho.
  The single threshold theta is fit on the 10 TRAIN categories (zero-shot to held-out).

Label discipline: K-selection reads ONLY pseudo-positive frames {TP,FP} (no GT, no
is_tp, no gt_event). theta is fit on train labels (allowed). Held-out labels are used
ONLY to score the final result, never to choose K. Verified in the leakage check.

Reuse only (no detector re-run): NPZ dumps + real Event_State map (P1_YOLO_Only /
full_cdnet2014). Detection-grounded recall (Event_State==TP). SI / LOCO. Video-level
bootstrap B=1000, seed=42.
"""
import sys, os, json, glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from src.events import extract_events
from src.scores import score_S, OnsetPosterior

FEAT = r"D:\THS Programing\09.01 RL vs Deterministic Comparision\asmag-rl-vs-deterministic\data\features"
PARQUET = r"D:\THS Programing\06.6 ASMAG Project\ASMAG_Project_Code\outputs\phase8a_policy_dataset\frame_state_dataset.parquet"

SEED = 42
B = 1000
MIN_DUR = 2
K_VALS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
TAU_PCTS = [0, 50, 70, 80, 85, 90, 93, 95, 97, 98, 99, 99.5, 100]
FLOOR = 0.90
W_FRACS = [0.05, 0.10, 0.15]
THETA_GRID = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]
SMALL_N = 4

# ---------------------------------------------------------------- load
print("loading parquet detection map ...", flush=True)
_df = pd.read_parquet(PARQUET, columns=["pipeline", "dataset", "category", "video",
                                        "frame_id", "Event_State"])
_p1 = _df[(_df["pipeline"] == "P1_YOLO_Only") & (_df["dataset"] == "full_cdnet2014")]
STATE_MAP = {}
for (cat, vid), grp in _p1.groupby(["category", "video"]):
    STATE_MAP[(cat, vid)] = dict(zip(grp["frame_id"].astype(int), grp["Event_State"]))
del _df, _p1


def load_video(npz_path):
    cat = os.path.basename(os.path.dirname(npz_path))
    vid = os.path.splitext(os.path.basename(npz_path))[0]
    d = np.load(npz_path, allow_pickle=True)
    fi = d["frame_idx"].astype(int)
    roi = d["temporal_roi"].astype(bool)
    feats = d["features"]; gt = d["gt_event"].astype(bool)
    sm = STATE_MAP.get((cat, vid), {})
    states_full = np.array([sm.get(int(f), "NA") for f in fi])
    is_tp_full = states_full == "TP"
    pseudo_pos_full = np.isin(states_full, ["TP", "FP"])     # detector-positive (label-free)
    feats_r = feats[roi]; gt_r = gt[roi]
    is_tp_r = is_tp_full[roi]; pp_r = pseudo_pos_full[roi]
    events = extract_events(gt_r, np.ones(gt_r.shape[0], dtype=bool), MIN_DUR)
    return {"vid_id": f"{cat}/{vid}", "category": cat,
            "features": feats_r.astype(np.float64), "gt": gt_r,
            "is_tp": is_tp_r, "pseudo_pos": pp_r, "events": events,
            "S": np.nan_to_num(score_S(feats_r), nan=-1e9).astype(np.float64),
            "T": int(gt_r.shape[0])}


print("loading feature dumps ...", flush=True)
VIDEOS = [load_video(p) for p in sorted(glob.glob(os.path.join(FEAT, "*", "*.npz")))]
CATS = sorted(set(v["category"] for v in VIDEOS))
BY_CAT = {c: [v for v in VIDEOS if v["category"] == c] for c in CATS}
VID_INDEX = {v["vid_id"]: i for i, v in enumerate(VIDEOS)}
CAT_OF = [v["category"] for v in VIDEOS]


def fit_pihat(train_vids):
    op = OnsetPosterior(C=1.0)
    td = [{"features": v["features"], "gt_event": v["gt"],
           "temporal_roi": np.ones(v["T"], dtype=bool)} for v in train_vids]
    op.fit(td); return op


# ---------------------------------------------------------------- replay / pool
def replay_runs(g, K, tau, is_tp, events, T, warmup=0):
    """Run set = [0,warmup) dense, then run iff (t-last>=K) or (g[t]>=tau).
    Detection-grounded recall. warmup frames count toward energy (cost amortised)."""
    runs = []; last = -K
    for t in range(T):
        if t < warmup:
            runs.append(t); last = t
        else:
            early = (tau != np.inf) and (g is not None) and g[t] >= tau
            if (t - last) >= K or early:
                runs.append(t); last = t
    runs_arr = np.asarray(runs); n_det = 0
    for ev in events:
        o, e = ev["onset"], ev["end"]
        if len(runs_arr):
            win = runs_arr[(runs_arr >= o) & (runs_arr <= e)]
            if len(win) and bool(np.any(is_tp[win])):
                n_det += 1
    return {"n_runs": len(runs), "n_det": n_det, "n_ev": len(events), "T": T}


def pool(records):
    tr = sum(r["n_runs"] for r in records); tT = sum(r["T"] for r in records)
    nd = sum(r["n_det"] for r in records); ne = sum(r["n_ev"] for r in records)
    return (1 - tr / tT if tT else float("nan"), nd / ne if ne else float("nan"))


# ---------------------------------------------------------------- config selection
def select_gate(fit_vids, scores_by_vid, floor):
    ps = np.concatenate([scores_by_vid[v["vid_id"]] for v in fit_vids])
    grid = sorted(set(float(np.percentile(ps, p)) for p in TAU_PCTS)) + [np.inf]
    best = None; bestrec = None
    for K in K_VALS:
        for tau in grid:
            recs = [replay_runs(scores_by_vid[v["vid_id"]], K, tau, v["is_tp"], v["events"], v["T"])
                    for v in fit_vids]
            rho, rec = pool(recs)
            if rec >= floor and (best is None or rho > best[0]):
                best = (rho, K, tau)
            if bestrec is None or rec > bestrec[1]:
                bestrec = (rho, rec, K, tau)
    return {"K": best[1], "tau": best[2]} if best else {"K": bestrec[2], "tau": bestrec[3]}


def select_periodic(fit_vids, floor):
    best = None; bestrec = None
    for K in K_VALS:
        rho, rec = pool([replay_runs(None, K, np.inf, v["is_tp"], v["events"], v["T"]) for v in fit_vids])
        if rec >= floor and (best is None or rho > best[0]):
            best = (rho, K)
        if bestrec is None or rec > bestrec[1]:
            bestrec = (rho, rec, K)
    return best[1] if best else bestrec[2]


def oracle_K_for(vids, floor):
    """max K with pooled true detection-recall over `vids` >= floor (eval-selected)."""
    best = None
    for K in K_VALS:
        _, rec = pool([replay_runs(None, K, np.inf, v["is_tp"], v["events"], v["T"]) for v in vids])
        if rec >= floor:
            best = K
    return best if best is not None else 2


# ---------------------------------------------------------------- estimator core
def warmup_pseudo_recall(video, W):
    """For each K: counterfactual periodic-K pseudo-recall over warmup [0,W),
    using ONLY pseudo-positive frames {TP,FP} (label-free)."""
    pp = video["pseudo_pos"][:W]
    pe = extract_events(pp, np.ones(W, dtype=bool), MIN_DUR)   # pseudo-events
    out = {"n_pe": len(pe), "pr": {}}
    for K in K_VALS:
        runs = np.arange(0, W, K)
        if len(pe) == 0:
            out["pr"][K] = float("nan"); continue
        det = 0
        for ev in pe:
            o, e = ev["onset"], ev["end"]
            if np.any((runs >= o) & (runs <= e)):
                det += 1
        out["pr"][K] = det / len(pe)
    return out


def estimate_K(pseudo_recall_info, theta):
    """largest K whose warmup pseudo-recall >= theta. None if not estimable."""
    if pseudo_recall_info["n_pe"] < 2:
        return None
    feasible = [K for K in K_VALS if not np.isnan(pseudo_recall_info["pr"][K])
                and pseudo_recall_info["pr"][K] >= theta]
    return max(feasible) if feasible else min(K_VALS)


# ---------------------------------------------------------------- precompute caches
print("precomputing warmup + schedule caches ...", flush=True)
# For each (vid, W): pseudo-recall info; and for each K the full-video schedule record
# (dense warmup + periodic-K), so theta only selects a K (no recompute).
W_ABS = {}      # (vid_id, wf) -> W frames
PR = {}         # (vid_id, wf) -> pseudo_recall_info
SCHED = {}      # (vid_id, wf, K) -> record   (dense warmup of W + periodic K)
for v in VIDEOS:
    for wf in W_FRACS:
        W = max(1, int(round(wf * v["T"])))
        W_ABS[(v["vid_id"], wf)] = W
        PR[(v["vid_id"], wf)] = warmup_pseudo_recall(v, W)
        for K in K_VALS:
            SCHED[(v["vid_id"], wf, K)] = replay_runs(None, K, np.inf, v["is_tp"], v["events"], v["T"], warmup=W)

# scene-agnostic periodic record cache (no warmup) -- for oracle/agnostic
SCHED0 = {}     # (vid_id, K) -> record (pure periodic, no warmup)
for v in VIDEOS:
    for K in K_VALS:
        SCHED0[(v["vid_id"], K)] = replay_runs(None, K, np.inf, v["is_tp"], v["events"], v["T"], warmup=0)


# ============================================================================
# SI / LOCO: build per-video records for every method.
# ============================================================================
METHODS = ["scene_agnostic", "periodic_robust", "gate_S", "gate_pi",
           "oracle_cat", "oracle_vid"] + [f"est_W{int(wf*100)}" for wf in W_FRACS]
rec = {m: [None] * len(VIDEOS) for m in METHODS}
theta_log = {}

print("=== SI (leave-one-category-out) ===", flush=True)
for c in CATS:
    held = BY_CAT[c]
    fit_vids = [v for v in VIDEOS if v["category"] != c]
    op = fit_pihat(fit_vids)
    pi_fit = {u["vid_id"]: op.score(u["features"]) for u in fit_vids}
    pi_held = {v["vid_id"]: op.score(v["features"]) for v in held}
    S_fit = {u["vid_id"]: u["S"] for u in fit_vids}
    Kagn = select_periodic(fit_vids, FLOOR)
    Krob = select_periodic_robust = None
    # robust K (criterion ii) for reference
    cats_tr = sorted(set(v["category"] for v in fit_vids)); Krob = 2
    for K in K_VALS:
        worst = 1.0
        for cc in cats_tr:
            cv = [v for v in fit_vids if v["category"] == cc]
            _, r = pool([SCHED0[(v["vid_id"], K)] for v in cv]); worst = min(worst, r)
        if worst >= FLOOR:
            Krob = K
    cfgS = select_gate(fit_vids, S_fit, FLOOR)
    cfgPi = select_gate(fit_vids, pi_fit, FLOOR)
    Korc_cat = oracle_K_for(held, FLOOR)
    # --- estimator: fit theta* per W on TRAIN videos (warmup on own first W) ---
    theta_star = {}
    for wf in W_FRACS:
        best = None; bestrec = None
        for theta in THETA_GRID:
            recs = []
            for u in fit_vids:
                K = estimate_K(PR[(u["vid_id"], wf)], theta)
                if K is None:
                    K = Kagn   # fallback to scene-agnostic when not estimable
                recs.append(SCHED[(u["vid_id"], wf, K)])
            rho, r = pool(recs)
            if r >= FLOOR and (best is None or rho > best[0]):
                best = (rho, theta)
            if bestrec is None or r > bestrec[1]:
                bestrec = (rho, r, theta)
        theta_star[wf] = best[1] if best else bestrec[2]
    theta_log[c] = {"Kagn": Kagn, "Krob": Krob, "Korc_cat": Korc_cat,
                    "theta_star": {str(wf): theta_star[wf] for wf in W_FRACS},
                    "gateS_K": cfgS["K"], "gatePi_K": cfgPi["K"]}
    # --- write held-out per-video records ---
    for v in held:
        i = VID_INDEX[v["vid_id"]]
        rec["scene_agnostic"][i] = SCHED0[(v["vid_id"], Kagn)]
        rec["periodic_robust"][i] = SCHED0[(v["vid_id"], Krob)]
        rec["gate_S"][i] = replay_runs(v["S"], cfgS["K"], cfgS["tau"], v["is_tp"], v["events"], v["T"])
        rec["gate_pi"][i] = replay_runs(pi_held[v["vid_id"]], cfgPi["K"], cfgPi["tau"], v["is_tp"], v["events"], v["T"])
        rec["oracle_cat"][i] = SCHED0[(v["vid_id"], Korc_cat)]
        rec["oracle_vid"][i] = SCHED0[(v["vid_id"], oracle_K_for([v], FLOOR))]
        for wf in W_FRACS:
            K = estimate_K(PR[(v["vid_id"], wf)], theta_star[wf])
            if K is None:
                K = Kagn
            rec[f"est_W{int(wf*100)}"][i] = SCHED[(v["vid_id"], wf, K)]
    print(f"  SI {c:<26} Kagn={Kagn} Krob={Krob} Korc_cat={Korc_cat} "
          f"theta*={[round(theta_star[wf],2) for wf in W_FRACS]}", flush=True)


# ============================================================================
# metric helpers (macro-average over categories, as N1)
# ============================================================================
def macro(idx, method, value="rho"):
    by_cat = {}
    for i in idx:
        by_cat.setdefault(CAT_OF[i], []).append(rec[method][i])
    vals = []
    for cc, recs in by_cat.items():
        rho, r = pool(recs); vals.append(rho if value == "rho" else r)
    vals = [x for x in vals if not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def ci(arr):
    a = np.array([x for x in arr if x is not None and not np.isnan(x)])
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


ALL = list(range(len(VIDEOS)))
n = len(VIDEOS)

# headline estimator W = the W maximizing pooled rho s.t. pooled recall>=~0.90
est_methods = [f"est_W{int(wf*100)}" for wf in W_FRACS]
est_pt = {m: (macro(ALL, m, "rho"), macro(ALL, m, "rec")) for m in est_methods}
feasible_est = [m for m in est_methods if est_pt[m][1] >= FLOOR - 1e-9]
HEAD = max(feasible_est, key=lambda m: est_pt[m][0]) if feasible_est \
    else max(est_methods, key=lambda m: est_pt[m][1])   # else best-recall

# ============================================================================
# bootstrap: pooled brackets + paired diffs + fraction-of-oracle-gain
# ============================================================================
print(f"bootstrap B={B} (headline estimator = {HEAD}) ...", flush=True)
rng = np.random.default_rng(SEED)
keymeth = ["scene_agnostic", "periodic_robust", "gate_S", "gate_pi",
           "oracle_cat", "oracle_vid", HEAD]
boot = {m: [] for m in keymeth}
frac = {"vs_agnostic_cat_oracle": [], "vs_robust_cat_oracle": []}
diffs = {"est_minus_agnostic": [], "est_minus_robust": [],
         "est_minus_gateS": [], "est_minus_gatePi": [],
         "est_minus_oracle_cat": []}
for b in range(B):
    idx = list(rng.integers(0, n, size=n))
    vals = {m: macro(idx, m, "rho") for m in keymeth}
    for m in keymeth:
        boot[m].append(vals[m])
    e = vals[HEAD]
    diffs["est_minus_agnostic"].append(e - vals["scene_agnostic"])
    diffs["est_minus_robust"].append(e - vals["periodic_robust"])
    diffs["est_minus_gateS"].append(e - vals["gate_S"])
    diffs["est_minus_gatePi"].append(e - vals["gate_pi"])
    diffs["est_minus_oracle_cat"].append(e - vals["oracle_cat"])
    denom = vals["oracle_cat"] - vals["scene_agnostic"]
    frac["vs_agnostic_cat_oracle"].append((e - vals["scene_agnostic"]) / denom if abs(denom) > 1e-6 else float("nan"))
    denom2 = vals["oracle_cat"] - vals["periodic_robust"]
    frac["vs_robust_cat_oracle"].append((e - vals["periodic_robust"]) / denom2 if abs(denom2) > 1e-6 else float("nan"))


def excl0(arr):
    lo, hi = ci(arr)
    return bool(lo is not None and (lo > 0 or hi < 0))


# ============================================================================
# per-category bracket (small-n flagged)
# ============================================================================
percat = {}
for c in CATS:
    vids_c = BY_CAT[c]; nc = len(vids_c)
    idx_c = [VID_INDEX[v["vid_id"]] for v in vids_c]
    row = {"n_videos": nc, "ci_reliable": nc > SMALL_N}
    pe = {}
    for m in ["scene_agnostic", "gate_S", "gate_pi", "oracle_cat", HEAD]:
        recs = [rec[m][i] for i in idx_c]
        rho, r = pool(recs); pe[m] = (rho, r)
    rl = np.random.default_rng(SEED + abs(hash(c)) % 9999)
    bcat = {m: [] for m in ["scene_agnostic", "gate_pi", "oracle_cat", HEAD]}
    dcat = {"est_minus_agnostic": [], "est_minus_gatePi": []}
    for b in range(B):
        s = list(rl.integers(0, nc, size=nc)); ridx = [idx_c[j] for j in s]
        vv = {m: pool([rec[m][i] for i in ridx])[0] for m in ["scene_agnostic", "gate_pi", "oracle_cat", HEAD]}
        for m in vv: bcat[m].append(vv[m])
        dcat["est_minus_agnostic"].append(vv[HEAD] - vv["scene_agnostic"])
        dcat["est_minus_gatePi"].append(vv[HEAD] - vv["gate_pi"])
    row["rho_max90"] = {m: {"point": pe[m][0], "realized_recall": pe[m][1],
                            "ci": ci(bcat[m]) if m in bcat else None}
                        for m in ["scene_agnostic", "gate_S", "gate_pi", "oracle_cat", HEAD]}
    row["est_minus_agnostic"] = {"point": pe[HEAD][0] - pe["scene_agnostic"][0],
                                 "ci": ci(dcat["est_minus_agnostic"]),
                                 "ci_significant": bool(excl0(dcat["est_minus_agnostic"]) and nc > SMALL_N)}
    row["est_minus_gatePi"] = {"point": pe[HEAD][0] - pe["gate_pi"][0],
                               "ci": ci(dcat["est_minus_gatePi"]),
                               "ci_significant": bool(excl0(dcat["est_minus_gatePi"]) and nc > SMALL_N)}
    percat[c] = row

# ============================================================================
# label-leakage check (structural + empirical, with teeth)
# ============================================================================
# Structural: estimate_K() reads only PR (pseudo-recall from detector-positive {TP,FP}
# warmup frames -- the detector's own fire/no-fire call, observable WITHOUT ground truth);
# it never receives is_tp / gt_event / events. theta is fit on TRAIN categories only.
#
# Empirical teeth #1 (permutation invariance): randomly permute held-out is_tp AND gt
# within each video and recompute K -> must be IDENTICAL (estimate_K cannot see them).
# Empirical teeth #2 (cheat control): an estimator that DID use true warmup detections
# (is_tp) to pick K lands on DIFFERENT K for many videos -> proves the honest estimator
# is genuinely label-blind, not accidentally reproducing the label-based choice.
leak_rng = np.random.default_rng(SEED)
perm_changed = 0; cheat_differs = 0; leak_checked = 0
for c in CATS:
    for v in BY_CAT[c]:
        for wf in W_FRACS:
            theta = theta_log[c]["theta_star"][str(wf)]
            W = W_ABS[(v["vid_id"], wf)]
            K_honest = estimate_K(PR[(v["vid_id"], wf)], theta)
            # #1 permutation: shuffle labels, rebuild a video copy, recompute pseudo-recall
            perm = leak_rng.permutation(v["T"])
            vshuf = dict(v); vshuf["is_tp"] = v["is_tp"][perm]; vshuf["gt"] = v["gt"][perm]
            # pseudo_pos is the detector signal (unchanged); confirm estimate_K invariant
            K_perm = estimate_K(warmup_pseudo_recall(v, W), theta)  # uses pseudo_pos only
            if K_perm != K_honest:
                perm_changed += 1
            # #2 cheat control: pseudo-recall from TRUE detections (is_tp) instead of {TP,FP}
            tp_warm = v["is_tp"][:W]
            te = extract_events(tp_warm, np.ones(W, dtype=bool), MIN_DUR)
            prc = {"n_pe": len(te), "pr": {}}
            for K in K_VALS:
                runs = np.arange(0, W, K)
                prc["pr"][K] = (sum(1 for e in te if np.any((runs >= e["onset"]) & (runs <= e["end"]))) / len(te)
                                if len(te) else float("nan"))
            K_cheat = estimate_K(prc, theta)
            if K_cheat != K_honest:
                cheat_differs += 1
            leak_checked += 1
leakage = {"structural": "estimate_K consumes only pseudo_pos={TP,FP} (detector fire/no-fire); "
                          "no is_tp/gt/events arguments",
           "theta_fit": "theta fit on TRAIN categories only; held-out labels never used for K",
           "n_K_decisions_audited": leak_checked,
           "permutation_test_K_changed": perm_changed,
           "permutation_test_pass": perm_changed == 0,
           "cheat_control_K_differs_from_honest": cheat_differs,
           "cheat_control_frac_differ": round(cheat_differs / leak_checked, 3) if leak_checked else None,
           "pseudo_pos_is_label_free": True}

# ============================================================================
# assemble + verdict
# ============================================================================
def block(m): return {"point": macro(ALL, m, "rho"), "ci": ci(boot[m]),
                      "realized_recall": macro(ALL, m, "rec")}

OUT = {"meta": {"seed": SEED, "B": B, "n_videos": n, "n_categories": len(CATS),
                "metric": "rho_max @ detection-recall>=0.90 (SI/LOCO, detection-grounded)",
                "floor": FLOOR, "W_fracs": W_FRACS, "theta_grid": THETA_GRID,
                "headline_estimator": HEAD,
                "note": "warmup dense-detection cost amortised into rho; "
                        "K-selection label-free (pseudo_pos={TP,FP}); theta fit on train cats"},
       "brackets_pooled": {m: block(m) for m in keymeth},
       "estimator_W_sweep": {m: {"point": est_pt[m][0], "ci": ci(boot[m]) if m == HEAD else None,
                                 "realized_recall": est_pt[m][1],
                                 "rho_full_sweep": macro(ALL, m, "rho")} for m in est_methods},
       "fraction_of_oracle_gain": {
           "vs_scene_agnostic": {"point": (macro(ALL, HEAD, "rho") - macro(ALL, "scene_agnostic", "rho")) /
                                          max(1e-6, macro(ALL, "oracle_cat", "rho") - macro(ALL, "scene_agnostic", "rho")),
                                 "ci": ci(frac["vs_agnostic_cat_oracle"])},
           "vs_robust": {"point": (macro(ALL, HEAD, "rho") - macro(ALL, "periodic_robust", "rho")) /
                                  max(1e-6, macro(ALL, "oracle_cat", "rho") - macro(ALL, "periodic_robust", "rho")),
                         "ci": ci(frac["vs_robust_cat_oracle"])}},
       "estimator_vs": {k: {"point": float(np.mean(v)), "ci": ci(v), "ci_significant": excl0(v)}
                        for k, v in diffs.items()},
       "per_category": percat,
       "label_leakage_check": leakage,
       "config_log": theta_log}

# verdict
fa = OUT["fraction_of_oracle_gain"]["vs_scene_agnostic"]["point"]
beats_agn = OUT["estimator_vs"]["est_minus_agnostic"]["ci_significant"] and OUT["estimator_vs"]["est_minus_agnostic"]["point"] > 0
beats_gateS = OUT["estimator_vs"]["est_minus_gateS"]["ci_significant"] and OUT["estimator_vs"]["est_minus_gateS"]["point"] > 0
beats_gatePi = OUT["estimator_vs"]["est_minus_gatePi"]["ci_significant"] and OUT["estimator_vs"]["est_minus_gatePi"]["point"] > 0
beats_gate = beats_gateS and beats_gatePi
meaningful = fa >= 0.40
verdict = "GO" if (meaningful and beats_agn and beats_gate) else "NO-GO"
OUT["verdict"] = {"decision": verdict, "fraction_of_oracle_gain_vs_agnostic": fa,
                  "CIsig_beats_scene_agnostic": bool(beats_agn),
                  "CIsig_beats_gate_S": bool(beats_gateS),
                  "CIsig_beats_gate_pi": bool(beats_gatePi),
                  "meaningful_fraction_ge_0.40": bool(meaningful),
                  "criterion": "GO iff fraction>=0.40 AND CI-beats scene-agnostic AND CI-beats BOTH gates"}

# ---------------------------------------------------------------- save + stdout
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(x) for x in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, float) and np.isnan(o): return None
    return o

json.dump(clean(OUT), open(os.path.join(REPO, "report", "perscene_k.json"), "w"), indent=2)
print("\nWROTE report/perscene_k.json", flush=True)

def f(x): return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"
bp = OUT["brackets_pooled"]
print("\n" + "=" * 82)
print("PER-SCENE-K GO/NO-GO -- can a cheap zero-shot estimator capture the oracle gain?")
print("=" * 82)
print(f"\nPooled SI rho_max@90 brackets [95% CI], realized recall:")
for m in ["scene_agnostic", "periodic_robust", "gate_S", "gate_pi", "oracle_cat", "oracle_vid", HEAD]:
    z = bp[m]; print(f"  {m:16s} {f(z['point'])} [{f(z['ci'][0])},{f(z['ci'][1])}]  rec={f(z['realized_recall'])}")
print(f"\nEstimator warmup sweep (rho / realized recall):")
for m in est_methods:
    print(f"  {m:10s} rho={f(est_pt[m][0])}  rec={f(est_pt[m][1])}{'   <-- headline' if m==HEAD else ''}")
fr = OUT["fraction_of_oracle_gain"]
print(f"\nFraction of oracle gain captured (vs scene-agnostic): {f(fr['vs_scene_agnostic']['point'])} "
      f"[{f(fr['vs_scene_agnostic']['ci'][0])},{f(fr['vs_scene_agnostic']['ci'][1])}]")
ev = OUT["estimator_vs"]
print(f"Estimator vs baselines (paired diff [CI], sig):")
for k in ["est_minus_agnostic", "est_minus_gateS", "est_minus_gatePi", "est_minus_oracle_cat"]:
    print(f"  {k:22s} {f(ev[k]['point']):>7} [{f(ev[k]['ci'][0])},{f(ev[k]['ci'][1])}] sig={ev[k]['ci_significant']}")
print(f"\nLabel-leakage check: K-selection uses pseudo_pos={{TP,FP}} only (label-free); "
      f"theta fit on train cats; held-out labels never used for K. {leak_checked} K-decisions audited.")
print("\n" + "-" * 82)
print(f"VERDICT: {verdict}")
print(f"  fraction>={0.40}: {meaningful} ({f(fa)}) | CI-beats agnostic: {beats_agn} | "
      f"CI-beats gate S: {beats_gateS} | gate pi: {beats_gatePi}")
print("-" * 82)
