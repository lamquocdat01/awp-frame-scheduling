# ==== C4 repo note (paths) ====
# Data paths below are ABSOLUTE to the author's machine (CDnet/LASIESTA/BMC/VIRAT
# detection dumps, frame_scores parquet, ASMAG NPZ/models). To re-run: edit the path
# constants and obtain the data (see README "Data & paths"). Canonical outputs are the
# committed ../../results/<name>.json. perscene_k.py also needs the deep0 "src" package;
# compute_vii*.py needs the asmag-rl-vs-deterministic modules (rl/, asmag_core/).
# ==============================
"""
n3_rareevent.py — N3 rare-event regime: does content-aware gating beat the OPTIMAL
oblivious schedule at EQUAL ENERGY when events are RARE? (the deciding GO/NO-GO)

Context. Five prior studies (Deep 0, Deep 3, N1, per-scene-K, N1-crossdata) found content-
aware gating gives NO deployable advantage over a simple oblivious schedule — but all on
FOREGROUND-HEAVY footage (~76% in-event). The rare-event regime (mostly-empty timelines) is
the ONE regime where a gate could genuinely win: it can stay closed through long quiet
periods and open on events. Deep 3 explicitly left this open. This decides whether the
dissertation has a POSITIVE result.

Discipline (critical). ISO-ENERGY ONLY: compare gate vs oblivious at MATCHED activation
(energy), with detection-RECALL as the outcome. Never compare at a recall-mismatched point
(the cross-dataset BMC "+0.132 gate win" was exactly such an artifact: gate 0.83 vs periodic
0.96 recall). We verify activations match before every comparison.

Substrate. Committed `frame_scores.parquet` (per-frame gate score S, per-frame detector-
positive g_t) for CDnet2014 / LASIESTA / BMC, plus RE-PROCESSED full VIRAT (uncapped) if
its dump is ready (see _load_virat_full). Detection-grounded events = maximal g_t runs.

Regime variable. in-event fraction per video = mean(g_t) (fraction of frames inside any
event). Low = the rare-event regime.

Policies / metric.
  oblivious(a) : OPTIMAL oblivious frontier, recall = E[min(a*D, 1)] over events (Deep 3 eps).
  S-gate(a)    : OPEN iff S >= tau, tau = S-quantile(1-a); event detected iff max(S in
                 [onset,end]) >= tau. activation = mean(S>=tau). The S-gate has NO trained
                 parameter (S is the committed gate output); tau is purely the energy budget,
                 so per-video activation-matching is leak-free and iso-energy by construction.
  pi-gate(a)   : same with pi-hat = P(g_t=1 | causal features of S), calibrator FIT ON TRAIN
                 (5-fold over videos), applied to held-out; tau_pi = pi-hat-quantile(1-a).
  advantage(a) = gate_recall(a) - oblivious_recall(a'), BOTH at the gate's realized
                 activation a' (iso-energy).

Outputs: report/n3_rareevent.json, report/N3_RAREEVENT.md
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
FRAME_SCORES = Path("D:/THS Programing/09.05 Certifiable worst-case/certifiable-frame-skipping/data/processed/frame_scores.parquet")
CDNET_DET_GLOB = "D:/THS Programing/zz/09.02 Scene Comlexity Metric/scm-scene-complexity/data/detections_cpu/CDnet2014/*.json"
VIRAT_FULL_GLOB = "D:/THS Programing/zz/09.02 Scene Comlexity Metric/scm-scene-complexity/data/detections_cpu_virat_full/VIRAT/*.json"
OUT_JSON = HERE.parents[1] / "results" / "n3_rareevent.json"

SEED = 42
B_BOOT = 1000
N_FOLDS = 5
A_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
SPARSE_THR = 0.30          # in-event fraction below which a video is "sparse" (rare-event)


# ════════════════════════════════════════════════════════════════════════════
# Substrate
# ════════════════════════════════════════════════════════════════════════════
def _events(frames, gt):
    ev = []
    n = len(gt)
    i = 0
    while i < n:
        if gt[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and gt[j + 1] == 1 and frames[j + 1] == frames[j] + 1:
            j += 1
        ev.append((int(frames[i]), int(frames[j]), int(frames[j] - frames[i] + 1)))
        i = j + 1
    return ev


def load_substrate():
    df = pd.read_parquet(FRAME_SCORES).sort_values(["dataset", "video", "frame"])
    df = df[df["dataset"].isin(["CDnet2014", "LASIESTA", "BMC"])].copy()  # VIRAT capped here
    catmap = {}
    for f in glob.glob(CDNET_DET_GLOB):
        cat, vid = os.path.basename(f)[:-5].split("__", 1)
        catmap[vid] = cat
    vids = {}
    for (ds, v), vs in df.groupby(["dataset", "video"]):
        frames = vs["frame"].values.astype(np.int64)
        S = vs["score"].values.astype(np.float64)
        gt = vs["g_t"].values.astype(np.int8)
        _add_video(vids, ds, v, frames, S, gt, catmap.get(v, None))
    return vids


def _add_video(vids, ds, v, frames, S, gt, cat):
    ev = _events(frames, gt)
    durs = np.array([d for *_, d in ev], float)
    # per-event max-signal needs S aligned to frames; store frame->idx for window maxes
    fidx = {f: i for i, f in enumerate(frames)}
    ev_max_S = []
    for o, e, _ in ev:
        i0, i1 = fidx.get(o), fidx.get(e)
        ev_max_S.append(float(S[i0:i1 + 1].max()) if i0 is not None and i1 is not None else 0.0)
    vids[f"{ds}::{v}"] = dict(
        dataset=ds, video=v, frames=frames, S=S, gt=gt, cat=cat,
        events=ev, durs=durs, ev_max_S=np.array(ev_max_S),
        n_frames=len(frames), inevent=float(gt.mean()), n_ev=len(ev),
        ev_idx=[(fidx[o], fidx[e]) for o, e, _ in ev if o in fidx and e in fidx],
    )


def load_virat_full(vids):
    """Fold in re-processed (uncapped) VIRAT if its YOLO dump is ready and SANE.
    Builds g_t (boxes non-empty) + events directly from JSON; gate score S is approximated
    by a causal frame-difference motion energy on the SAME frames (the committed ASMAGPlus
    gate was not re-run for full VIRAT — documented in the report)."""
    files = sorted(glob.glob(VIRAT_FULL_GLOB))
    if not files:
        return dict(status="absent", n_videos=0)
    import cv2
    info = dict(status="present", n_videos=0, videos=[])
    # map json stem -> mp4
    mp4_root = Path("D:/THS Programing/06.6 ASMAG Project/datasets/VIRAT")
    for jf in files:
        d = json.load(open(jf))
        fr = d.get("frames", [])
        if len(fr) <= 520:            # still looks capped -> skip as not re-processed
            continue
        by = sorted(fr, key=lambda x: x["frame"])
        frames = np.array([f["frame"] for f in by], dtype=np.int64)
        gt = np.array([len(f.get("boxes", [])) > 0 for f in by], dtype=np.int8)
        stem = os.path.basename(jf)[:-5]
        vid_id = stem.split("__", 1)[1] if "__" in stem else stem
        # motion-energy proxy for S (causal |frame diff| mean), normalized to [0,1]
        S = _virat_motion_S(mp4_root / f"{vid_id}.mp4", frames, cv2)
        if S is None:
            continue
        _add_video(vids, "VIRAT", vid_id, frames, S, gt, None)
        info["n_videos"] += 1
        info["videos"].append(dict(video=vid_id, n_frames=int(len(frames)),
                                   inevent=float(gt.mean())))
    return info


def _virat_motion_S(mp4, frames, cv2):
    if not mp4.exists():
        return None
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return None
    want = set(int(f) for f in frames)
    maxf = int(frames.max())
    prev = None
    raw = {}
    idx = 0
    while idx <= maxf:
        ok, img = cap.read()
        if not ok:
            break
        if idx in want:
            g = cv2.cvtColor(cv2.resize(img, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            raw[idx] = 0.0 if prev is None else float(np.abs(g - prev).mean())
            prev = g
        idx += 1
    cap.release()
    vals = np.array([raw.get(int(f), 0.0) for f in frames], dtype=np.float64)
    if vals.max() > vals.min():
        vals = (vals - vals.min()) / (vals.max() - vals.min())
    return vals


# ════════════════════════════════════════════════════════════════════════════
# pi-hat (calibrated occupancy posterior) — train-fit
# ════════════════════════════════════════════════════════════════════════════
def causal_features(S):
    s = pd.Series(S)
    Sprev = s.shift(1).fillna(S[0]).values
    dS = S - Sprev
    rmax4 = s.rolling(4, min_periods=1).max().values
    rmean4 = s.rolling(4, min_periods=1).mean().values
    rstd4 = s.rolling(4, min_periods=1).std().fillna(0.0).values
    rmean8 = s.rolling(8, min_periods=1).mean().values
    cnt_hi = (s > 0.55).rolling(4, min_periods=1).mean().values
    hi = (S > 0.55)
    since = np.empty(len(S))
    last = -1
    for t in range(len(S)):
        if hi[t]:
            last = t
        since[t] = (t - last) if last >= 0 else float(t + 1)
    since = np.minimum(since, 30) / 30.0
    return np.column_stack([S, Sprev, dS, rmax4, rmean4, rstd4, rmean8, cnt_hi, S - rmean4, since])


def fit_pi_and_apply(train_keys, eval_keys, vids):
    """Fit logistic P(g_t=1|x) on train videos; return {key: pihat array} for eval videos."""
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for k in train_keys:
        Xs.append(causal_features(vids[k]["S"]))
        ys.append(vids[k]["gt"].astype(int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(max_iter=500, C=1.0).fit((X - mu) / sd, y)
    out = {}
    for k in eval_keys:
        Xz = (causal_features(vids[k]["S"]) - mu) / sd
        out[k] = 1.0 / (1.0 + np.exp(-(Xz @ clf.coef_[0] + clf.intercept_[0])))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Iso-energy advantage per video
# ════════════════════════════════════════════════════════════════════════════
def _ev_max_signal(vd, sig):
    """max signal within each event window."""
    return np.array([sig[i0:i1 + 1].max() for (i0, i1) in vd["ev_idx"]])


def advantage_curve(vd, sig=None, ev_max=None):
    """Per-video iso-energy advantage at each target activation in A_GRID.
       gate_recall(a) - oblivious_recall(a'), a' = gate's realized activation.
       sig = per-frame signal (defaults to S); ev_max precomputed event-max-signal."""
    if sig is None:
        sig = vd["S"]
    if ev_max is None:
        ev_max = vd["ev_max_S"]
    durs = vd["durs"]
    out = {}
    if vd["n_ev"] == 0:
        return {a: dict(act=a, gate=1.0, obliv=1.0, adv=0.0) for a in A_GRID}
    for a in A_GRID:
        tau = np.quantile(sig, 1.0 - a)
        act = float((sig >= tau).mean())              # realized energy
        gate_rec = float((ev_max >= tau).mean())       # event detected iff max-sig in window>=tau
        obliv_rec = float(np.mean(np.minimum(act * durs, 1.0)))  # OPTIMAL oblivious at SAME energy
        out[a] = dict(act=act, gate=gate_rec, obliv=obliv_rec, adv=gate_rec - obliv_rec)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Mechanism checks
# ════════════════════════════════════════════════════════════════════════════
def cue_event_auc(vd, sig=None):
    """AUC of the per-frame signal for predicting (a) event membership g_t, and
    (b) event ONSET frames. >0.5 => cue aligns with events; ~0.5 => misaligned."""
    from sklearn.metrics import roc_auc_score
    if sig is None:
        sig = vd["S"]
    gt = vd["gt"]
    res = {}
    try:
        res["auc_membership"] = float(roc_auc_score(gt, sig)) if 0 < gt.sum() < len(gt) else None
    except Exception:
        res["auc_membership"] = None
    onset = np.zeros(len(gt), dtype=int)
    for (i0, i1) in vd["ev_idx"]:
        onset[i0] = 1
    try:
        res["auc_onset"] = float(roc_auc_score(onset, sig)) if 0 < onset.sum() < len(onset) else None
    except Exception:
        res["auc_onset"] = None
    return res


# ════════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════════
def _ci(a):
    a = np.array([x for x in a if x is not None and not (isinstance(x, float) and np.isnan(x))])
    if len(a) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    print("[load] substrate ...")
    vids = load_substrate()
    print("[virat] checking re-processed full VIRAT ...")
    virat_info = load_virat_full(vids)
    print(f"        VIRAT full: {virat_info['status']} ({virat_info.get('n_videos',0)} videos)")

    keys = list(vids.keys())
    rng = np.random.default_rng(SEED)

    # ── in-event fraction inventory ──────────────────────────────────────────
    inv = [dict(key=k, dataset=vids[k]["dataset"], video=vids[k]["video"],
                n_frames=vids[k]["n_frames"], inevent=vids[k]["inevent"],
                n_ev=vids[k]["n_ev"],
                med_dur=float(np.median(vids[k]["durs"])) if vids[k]["n_ev"] else 0.0)
           for k in keys]
    inv_df = pd.DataFrame(inv)
    sparse_keys = [r["key"] for r in inv if r["inevent"] < SPARSE_THR and r["n_ev"] >= 3]
    print(f"[regime] {len(sparse_keys)} sparse videos (in-event<{SPARSE_THR}, n_ev>=3)")

    # ── pi-hat via 5-fold (for the secondary gate) ───────────────────────────
    fold_of = {k: i % N_FOLDS for i, k in enumerate(sorted(keys))}
    pihat = {}
    for f in range(N_FOLDS):
        tr = [k for k in keys if fold_of[k] != f and vids[k]["dataset"] != "VIRAT"]
        ev = [k for k in keys if fold_of[k] == f]
        pihat.update(fit_pi_and_apply(tr, ev, vids))

    # ── per-video iso-energy advantage (S-gate primary; pi-gate secondary) ───
    pervid = {}
    for k in keys:
        vd = vids[k]
        s_curve = advantage_curve(vd)                       # S-gate
        pmax = _ev_max_signal(vd, pihat[k]) if vd["n_ev"] else np.array([])
        p_curve = advantage_curve(vd, sig=pihat[k], ev_max=pmax)
        auc = cue_event_auc(vd)
        pervid[k] = dict(inevent=vd["inevent"], dataset=vd["dataset"], n_ev=vd["n_ev"],
                         med_dur=float(np.median(vd["durs"])) if vd["n_ev"] else 0.0,
                         frac_single=float(np.mean(vd["durs"] == 1)) if vd["n_ev"] else 0.0,
                         S=s_curve, pi=p_curve, auc=auc)

    # ── Step 1: regime-dependence curve (bin by in-event fraction) ───────────
    # bins: sparse(<0.3), mid(0.3-0.7), dense(>=0.7)
    def bin_of(f):
        return "sparse(<0.3)" if f < 0.30 else ("mid(0.3-0.7)" if f < 0.70 else "dense(>=0.7)")
    bins = {}
    for b in ["sparse(<0.3)", "mid(0.3-0.7)", "dense(>=0.7)"]:
        bk = [k for k in keys if bin_of(vids[k]["inevent"]) == b and vids[k]["n_ev"] >= 3]
        # per-video advantage summarised at each activation; mean + bootstrap CI over videos
        per_a = {}
        for a in A_GRID:
            advs = [pervid[k]["S"][a]["adv"] for k in bk]
            acts = [pervid[k]["S"][a]["act"] for k in bk]
            if not bk:
                per_a[a] = dict(n=0)
                continue
            boot = [np.mean([pervid[k]["S"][a]["adv"] for k in
                             [bk[i] for i in rng.integers(0, len(bk), len(bk))]]) for _ in range(B_BOOT)]
            per_a[a] = dict(n=len(bk), mean_adv=float(np.mean(advs)), ci=_ci(boot),
                            mean_act=float(np.mean(acts)))
        bins[b] = dict(n_videos=len(bk), per_activation=per_a,
                       mean_inevent=float(np.mean([vids[k]["inevent"] for k in bk])) if bk else None)

    # trend: per-video summary advantage (max over A_GRID, S-gate) vs in-event fraction
    vk = [k for k in keys if vids[k]["n_ev"] >= 3]
    summ_adv = [max(pervid[k]["S"][a]["adv"] for a in A_GRID) for k in vk]
    inev = [vids[k]["inevent"] for k in vk]
    rho_trend, p_trend = sps.spearmanr(inev, summ_adv)

    # ── Step 2: rare-event head-to-head on sparse videos (iso-energy) ────────
    head = {}
    for a in A_GRID:
        gate = [pervid[k]["S"][a]["gate"] for k in sparse_keys]
        obl = [pervid[k]["S"][a]["obliv"] for k in sparse_keys]
        act = [pervid[k]["S"][a]["act"] for k in sparse_keys]
        adv = [pervid[k]["S"][a]["adv"] for k in sparse_keys]
        padv = [pervid[k]["pi"][a]["adv"] for k in sparse_keys]
        if not sparse_keys:
            head[a] = dict(n=0); continue
        boot = [np.mean([pervid[k]["S"][a]["adv"] for k in
                         [sparse_keys[i] for i in rng.integers(0, len(sparse_keys), len(sparse_keys))]])
                for _ in range(B_BOOT)]
        bootp = [np.mean([pervid[k]["pi"][a]["adv"] for k in
                          [sparse_keys[i] for i in rng.integers(0, len(sparse_keys), len(sparse_keys))]])
                 for _ in range(B_BOOT)]
        head[a] = dict(n=len(sparse_keys),
                       gate_recall=float(np.mean(gate)), obliv_recall=float(np.mean(obl)),
                       mean_act=float(np.mean(act)),
                       S_adv=float(np.mean(adv)), S_adv_ci=_ci(boot),
                       pi_adv=float(np.mean(padv)), pi_adv_ci=_ci(bootp))

    # ── Step 3: mechanism — sparse-regime duration law + cue AUC ─────────────
    sp_durs = np.concatenate([vids[k]["durs"] for k in sparse_keys]) if sparse_keys else np.array([])
    mech = dict(
        sparse_duration_law=dict(
            n_events=int(len(sp_durs)),
            median=float(np.median(sp_durs)) if len(sp_durs) else None,
            mean=float(sp_durs.mean()) if len(sp_durs) else None,
            cv=float(sp_durs.std() / sp_durs.mean()) if len(sp_durs) else None,
            frac_single=float(np.mean(sp_durs == 1)) if len(sp_durs) else None,
            frac_ge5=float(np.mean(sp_durs >= 5)) if len(sp_durs) else None),
        cue_auc_membership=_ci([pervid[k]["auc"]["auc_membership"] for k in sparse_keys]),
        cue_auc_onset=_ci([pervid[k]["auc"]["auc_onset"] for k in sparse_keys]),
        cue_auc_membership_mean=float(np.nanmean([pervid[k]["auc"]["auc_membership"]
                                     for k in sparse_keys if pervid[k]["auc"]["auc_membership"] is not None]))
            if sparse_keys else None,
    )

    # ── verdict logic ────────────────────────────────────────────────────────
    # GO requires ALL of:
    #  (1) the gate CI-beats oblivious at equal energy at a RARE-EVENT operating point
    #      (LOW activation a<=0.20 — where you genuinely skip most frames; a win at a=0.50
    #      is dense sampling, NOT the rare-event value proposition and does not count);
    #  (2) the advantage grows as in-event fraction drops (rho_trend < 0); AND
    #  (3) the mechanism supports it: sparse events are catchable (median dur >=2, not single-
    #      frame-dominated) AND the cue aligns (mean membership AUC > 0.55).
    LOW_A = [a for a in A_GRID if a <= 0.20]
    sparse_win_lowa = any(head[a]["n"] > 0 and head[a]["S_adv_ci"][0] > 0 for a in LOW_A)
    sparse_win_anya = any(head[a]["n"] > 0 and head[a]["S_adv_ci"][0] > 0 for a in A_GRID)
    trend_ok = (rho_trend is not None) and (rho_trend < 0)
    dl = mech["sparse_duration_law"]
    mech_ok = (dl["median"] is not None and dl["median"] >= 2
               and dl["frac_single"] is not None and dl["frac_single"] < 0.5
               and mech["cue_auc_membership_mean"] is not None
               and mech["cue_auc_membership_mean"] > 0.55)
    verdict = "GO" if (sparse_win_lowa and trend_ok and mech_ok) else "NO-GO"
    sparse_win = sparse_win_lowa

    results = dict(
        meta=dict(seed=SEED, B=B_BOOT, a_grid=A_GRID, sparse_thr=SPARSE_THR,
                  iso_energy="oblivious recall E[min(a*D,1)] compared at gate's realized activation",
                  virat=virat_info),
        inventory=dict(
            per_dataset_inevent={ds: inv_df[inv_df.dataset == ds]["inevent"].describe().to_dict()
                                 for ds in inv_df.dataset.unique()},
            sparse_videos=[dict(dataset=vids[k]["dataset"], video=vids[k]["video"],
                                inevent=round(vids[k]["inevent"], 4), n_ev=vids[k]["n_ev"],
                                med_dur=float(np.median(vids[k]["durs"]))) for k in sparse_keys],
            n_sparse=len(sparse_keys),
            inevent_quantiles={str(q): float(inv_df["inevent"].quantile(q))
                               for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]}),
        regime_bins=bins,
        trend=dict(spearman_inevent_vs_advantage=float(rho_trend) if rho_trend is not None else None,
                   p=float(p_trend) if p_trend is not None else None,
                   note="negative spearman => advantage grows as in-event fraction drops"),
        head_to_head_sparse=head,
        mechanism=mech,
        verdict=dict(verdict=verdict, sparse_win_lowa=bool(sparse_win_lowa),
                     sparse_win_anya=bool(sparse_win_anya),
                     trend_grows_as_sparse=bool(trend_ok), mechanism_supports=bool(mech_ok)),
    )

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT_JSON}")
    _summary(results)
    return results


def _summary(R):
    print("\n" + "=" * 74)
    print("N3 RARE-EVENT SUMMARY")
    print("=" * 74)
    iq = R["inventory"]["inevent_quantiles"]
    print(f"in-event fraction quantiles: median={iq['0.5']:.2f} q25={iq['0.25']:.2f} "
          f"min={iq['0']:.3f} | n_sparse(<{R['meta']['sparse_thr']})={R['inventory']['n_sparse']}")
    print(f"VIRAT full re-process: {R['meta']['virat']['status']} "
          f"({R['meta']['virat'].get('n_videos',0)} videos)")
    print("\nRegime-dependence (S-gate advantage at equal energy, by in-event bin):")
    for b, d in R["regime_bins"].items():
        if d["n_videos"] == 0:
            continue
        a = 0.10
        pa = d["per_activation"].get(a, {})
        print(f"  {b:14s} n={d['n_videos']:2d} inev~{d['mean_inevent']:.2f}  "
              f"adv@a=0.10 = {pa.get('mean_adv',float('nan')):+.3f} {pa.get('ci')}")
    print(f"\nTrend (Spearman in-event vs advantage) = "
          f"{R['trend']['spearman_inevent_vs_advantage']:.3f}  "
          f"(neg => grows as sparse)")
    print("\nRare-event head-to-head on sparse videos (iso-energy; gate-oblivious):")
    for a in R["meta"]["a_grid"]:
        h = R["head_to_head_sparse"][a]
        if h.get("n", 0) == 0:
            continue
        print(f"  a={a:.2f} (act {h['mean_act']:.3f}): gate {h['gate_recall']:.3f} vs "
              f"obliv {h['obliv_recall']:.3f} | S-adv {h['S_adv']:+.3f}{h['S_adv_ci']} "
              f"pi-adv {h['pi_adv']:+.3f}{h['pi_adv_ci']}")
    m = R["mechanism"]
    dl = m["sparse_duration_law"]
    print(f"\nMechanism: sparse durations median={dl['median']} cv={dl['cv']:.2f} "
          f"frac_single={dl['frac_single']:.2f} frac>=5={dl['frac_ge5']:.2f}")
    print(f"           cue-event membership AUC mean={m['cue_auc_membership_mean']:.3f} "
          f"CI{m['cue_auc_membership']}  onset AUC CI{m['cue_auc_onset']}")
    v = R["verdict"]
    print("\n" + "-" * 74)
    print(f"VERDICT: {v['verdict']}  (sparse_win@low-a={v['sparse_win_lowa']}, "
          f"win@any-a={v['sparse_win_anya']}, trend_grows={v['trend_grows_as_sparse']}, "
          f"mechanism={v['mechanism_supports']})")
    print("-" * 74)


if __name__ == "__main__":
    main()
