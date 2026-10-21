# -*- coding: utf-8 -*-
"""
awp_experiments.py
==================================================================
Thí nghiệm số (numerical experiments) cho Đề tài 03 — AwP Frame Scheduling
"Learning-Augmented Frame Scheduling: Consistency–Robustness Bounds
 for Gating with Untrusted Motion Predictors".

NỘI DUNG (bám theo results/experiment_plan.md):
  - Hiện thực HEDGE-GATE(λ): chia ngân sách kích hoạt giữa "backbone periodic"
    (đảm bảo sàn recall) và "advice gate" (bám theo bộ dự đoán).
  - Chạy 2 chế độ nhiễu hoá bộ dự đoán (predictor-corruption, T10):
      * NOISE       : nhiễu hoán vị (uninformative) -> gate tụt về mức NGẪU NHIÊN.
      * ADVERSARIAL : dự đoán bị "lái" sang vùng nhiễu động (clutter) lệch khỏi sự
                      kiện (đúng hiện tượng cross-scene/LODO của P-A) -> gate sụp
                      XUỐNG DƯỚI mức oblivious (hiện thực worst-case của Định lý 1).
  - Kiểm chứng 3 tính chất lý thuyết:
      (1) Periodic certificate:  R(periodic) >= min(1, (W+1)/K),  K=ceil(1/a).
      (2) Consistency–Robustness frontier (Định lý 2): đánh đổi giữa recall khi
          dự đoán hoàn hảo (η→0) và recall khi dự đoán đối kháng (η→1).
      (3) Smoothness (Proposition 4): recall suy giảm trơn theo η ở chế độ NOISE.
      + HEDGE-GATE luôn giữ "sàn backbone" min(1, λ(W+1)/K) dù dự đoán đối kháng.

LƯU Ý DỮ LIỆU
--------------------------------------------------------------------
Bộ video gốc (121 video) KHÔNG có trong workspace, nên corruption sweep chạy trên
một EVENT PROCESS TỔNG HỢP được HIỆU CHỈNH theo thống kê thật của P-A
(sparse_duration_law: median=1, mean≈6.13, CV≈4.67, frac_single≈0.596). Các điểm
NEO (anchors) là SỐ ĐO THẬT đọc từ content-aware-frameskip-evaluation/results/*.json.
Đây là kiểm chứng số chuẩn mực cho bài lý thuyết; thí nghiệm trên video gốc (T10)
sẽ thay phần synthetic khi có dataset.
==================================================================
"""

import os
import json
import csv
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. THAM SỐ TOÀN CỤC
# ------------------------------------------------------------------
SEED = 42
W = 7                      # cửa sổ staleness: recall nếu detector bắn trong [onset, onset+W]
B = 1000                   # số lần bootstrap (cấp video)
N_VIDEOS = 40
T_FRAMES = 1500
TARGET_INEVENT = 0.15      # tỉ lệ frame trong sự kiện (regime sparse < 0.3)

A_GRID = [0.10, 0.20, 0.333]
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
CORRUPT_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
REGIMES = ["noise", "adversarial"]

# Thống kê thật của P-A để hiệu chỉnh
SPARSE_LAW = dict(median=1.0, mean=6.1286, cv=4.6711, frac_single=0.5957, frac_ge5=0.1440)
CUE_AUC_ONSET = (0.4934, 0.7913)     # dải onset-AUC đo được

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC_DIR = os.path.abspath(os.path.join(HERE, ".."))
RESULTS_DIR = os.path.join(TOPIC_DIR, "results")
PA_RESULTS = os.path.abspath(os.path.join(
    TOPIC_DIR, "..", "..", "content-aware-frameskip-evaluation", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==================================================================
# 1. ĐỌC NEO THẬT TỪ BỘ P-A
# ==================================================================
def load_pa_anchors():
    out = {}
    try:
        a3 = json.load(open(os.path.join(PA_RESULTS, "a3_learned_result.json")))
        grid = a3["frontier_low_a"]["grid"]            # grid[a][pol] = [recall, activation]
        out["frontier"] = {float(a): {p: grid[a][p][0] for p in ("periodic", "S", "RF")}
                           for a in grid}
        out["oracle_alwayson"] = a3["oracle_alwayson_recall_mean"]
        out["det_gate_gap"] = a3["det_gate_RF"]["gap_detgate_minus_periodic"]
    except Exception as e:
        print("  [warn] a3_learned_result.json:", e)
    try:
        pk = json.load(open(os.path.join(PA_RESULTS, "perscene_k.json")))
        out["perscene"] = {k: pk["brackets_pooled"][k]["point"] for k in pk["brackets_pooled"]}
        out["frac_oracle_gain"] = pk["fraction_of_oracle_gain"]["vs_scene_agnostic"]["point"]
    except Exception as e:
        print("  [warn] perscene_k.json:", e)
    try:
        n1 = json.load(open(os.path.join(PA_RESULTS, "n1_crossdata.json")))
        out["pooled_SS"] = n1["pooled"]["SS"]
        out["pooled_SI"] = n1["pooled"]["SI"]
    except Exception as e:
        print("  [warn] n1_crossdata.json:", e)
    return out if out else None


# ==================================================================
# 2. SINH EVENT PROCESS + VÙNG NHIỄU ĐỘNG (clutter) — hiệu chỉnh theo P-A
# ==================================================================
def sample_durations(n, rng):
    """Sinh n thời lượng khớp luật P-A: median=1, mean≈6.1, CV≈4.7, P(d=1)≈0.596.
    Hỗn hợp: điểm khối tại d=1 + đuôi log-normal nặng cho d>=2."""
    d = np.ones(n, dtype=int)
    u = rng.random(n)
    heavy = u >= SPARSE_LAW["frac_single"]            # ~40.4% sự kiện dài hơn 1 frame
    k = int(heavy.sum())
    if k:
        # log-normal lệch phải (mu=0.85, σ=1.85): khớp mean≈6.1, CV≈4.7, đuôi nặng (max lớn)
        vals = rng.lognormal(mean=0.85, sigma=1.85, size=k)
        d[heavy] = np.clip(np.round(vals).astype(int) + 1, 2, 6000)
    return d


def make_video(T, target_inevent, rng):
    """Sinh 1 video. Trả về dict: gt[T] bool, events list, clutter[T] bool.

    clutter = các "burst nhiễu động" (motion clutter: rung camera, nền động...) nằm
    NGOÀI cửa sổ sự kiện. Đây là thứ mà bộ dự đoán cross-scene hay bắn nhầm vào."""
    gt = np.zeros(T, dtype=bool)
    events = []
    approx_n = max(4, int(target_inevent * T / SPARSE_LAW["mean"]))
    durs = sample_durations(approx_n * 4, rng)
    # khoảng nghỉ trung bình để đạt tỉ lệ in-event mong muốn
    gap_mean = max(3.0, (1 - target_inevent) / target_inevent * SPARSE_LAW["mean"])
    t = int(rng.integers(0, 30))
    di = 0
    while t < T and di < len(durs):
        d = int(min(durs[di], T - t))
        if d < 1:
            break
        gt[t:t + d] = True
        events.append({"onset": t, "end": t + d - 1, "duration": d})
        di += 1
        gap = 2 + int(rng.geometric(1.0 / gap_mean))
        t = t + d + gap

    # cửa sổ sự kiện [onset, onset+W] = vùng "positive" mà gate cần bắn trúng
    win = np.zeros(T, dtype=bool)
    for ev in events:
        win[ev["onset"]:min(T, ev["onset"] + W + 1)] = True

    # clutter: ~50% số frame, dạng burst, đặt ở vùng KHÔNG phải cửa sổ sự kiện
    clutter = np.zeros(T, dtype=bool)
    n_clutter_target = int(0.5 * T)
    placed = 0
    guard = 0
    while placed < n_clutter_target and guard < 10 * T:
        guard += 1
        start = int(rng.integers(0, T))
        blen = int(2 + rng.geometric(0.15))           # độ dài burst clutter
        seg = np.arange(start, min(T, start + blen))
        seg = seg[~win[seg]]                          # bỏ phần trùng cửa sổ sự kiện
        clutter[seg] = True
        placed = int(clutter.sum())
    return {"gt": gt, "events": events, "win": win, "clutter": clutter}


# ==================================================================
# 3. BỘ DỰ ĐOÁN + 2 CHẾ ĐỘ NHIỄU HOÁ + ĐO η
# ==================================================================
def predictor_score(vid, T, d_prime, p, regime, rng):
    """Sinh điểm số dự đoán s[t] theo mức nhiễu p ∈ [0,1] và chế độ.

      base = nhiễu Gaussian; tín hiệu mạnh d_prime đặt ở cửa sổ sự kiện (win).
      - regime='noise'       : p phần frame bị HOÁN VỊ điểm số (mất thông tin -> ngẫu nhiên).
      - regime='adversarial' : tín hiệu CHUYỂN DẦN từ cửa sổ sự kiện sang clutter:
                               trọng số (1-p) lên win, p lên clutter -> bắn lệch khỏi sự kiện."""
    win = vid["win"]; clutter = vid["clutter"]
    score = rng.normal(0.0, 1.0, size=T)
    if regime == "noise":
        score[win] += d_prime
        if p > 0:                                     # hoán vị p phần frame
            k = int(round(p * T))
            if k > 1:
                idx = rng.choice(T, size=k, replace=False)
                score[idx] = score[idx][rng.permutation(k)]
    else:  # adversarial
        score[win] += (1.0 - p) * d_prime             # còn (1-p) tín hiệu đúng
        score[clutter] += p * (d_prime + 1.0)         # p tín hiệu LÁI sang clutter (mạnh hơn chút)
    return score


def onset_auc(score, win):
    """AUC (Mann–Whitney) phân biệt điểm số ở cửa sổ sự kiện vs phần còn lại."""
    a = score[win]; b = score[~win]
    if len(a) == 0 or len(b) == 0:
        return 0.5
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # hạng trung bình cho giá trị trùng
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    cum = np.concatenate([[0], np.cumsum(counts)])
    avg = (cum[:-1] + 1 + cum[1:]) / 2.0
    ranks = avg[inv]
    R1 = ranks[:len(a)].sum()
    return float((R1 - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))


def eta_from_auc(auc):
    """η = 1 - AUC ∈ [0,1]. AUC=1 -> η=0 (hoàn hảo); 0.5 -> 0.5 (vô dụng); <0.5 -> η>0.5 (đối kháng)."""
    return float(min(1.0, max(0.0, 1.0 - auc)))


# ==================================================================
# 4. CHÍNH SÁCH HEDGE-GATE(λ)  (iso-energy chính xác)
# ==================================================================
def hedge_gate_fire(score, T, a, lam):
    """Chia ngân sách: backbone periodic tiêu λ·a (bắn đều), advice gate tiêu (1-λ)·a
    (bắn vào frame điểm cao nhất chưa bị backbone chiếm). Tổng bắn = round(a*T) (iso-energy)."""
    budget = int(round(a * T))
    fired = np.zeros(T, dtype=bool)
    n_bb = int(round(lam * budget))
    if n_bb >= 1:
        fired[np.unique(np.linspace(0, T - 1, n_bb, dtype=int))] = True
    remaining = budget - int(fired.sum())
    if remaining > 0:
        cand = np.where(~fired)[0]
        top = cand[np.argsort(-score[cand], kind="mergesort")[:remaining]]
        fired[top] = True
    return fired


def sat_gate_fire(score, T, a, q):
    """SAT-GATE(q): chính sách 'backbone bão hòa' — Pareto-optimal ở MỌI regime ngân sách
    (Định lý 2-3, bản general). Dial q∈(0,1] = rate bắn các frame advised (điểm-cue cao).
    Frame advised được bắn ở rate q (thưa ra thay vì dồn rate 1 như HEDGE), phần ngân sách
    tiết kiệm được đổ vào 'sàn' oblivious phủ ĐỀU toàn bộ frame -> nâng recall worst-case ở
    regime ngân sách lớn (β>1/2). q=1 trùng điểm cuối của HEDGE; q<1 tách lên trên frontier.
    Iso-energy: tổng bắn = round(a*T)."""
    budget = int(round(a * T))
    fired = np.zeros(T, dtype=bool)
    if budget <= 0:
        return fired
    q = min(1.0, max(1e-9, q))
    order = np.argsort(-score, kind="mergesort")
    n_adv = min(T, budget)                     # pool advised = top-cue
    adv_pool = order[:n_adv]
    n_adv_fire = int(round(q * n_adv))
    if n_adv_fire > 0:                          # bắn advised ở rate q (thưa đều, tất định)
        idx = np.unique(np.linspace(0, n_adv - 1, n_adv_fire, dtype=int))
        fired[adv_pool[idx]] = True
    remaining = budget - int(fired.sum())      # phần còn lại -> sàn oblivious phủ đều
    if remaining > 0:
        floor_idx = np.unique(np.linspace(0, T - 1, remaining, dtype=int))
        fired[floor_idx[~fired[floor_idx]]] = True
        deficit = budget - int(fired.sum())    # bù nếu va chạm làm hụt ngân sách
        if deficit > 0:
            cand = np.where(~fired)[0][:deficit]
            fired[cand] = True
    return fired


def recall_windowed(fired, events, W):
    """Recall: sự kiện được tính nếu có lần bắn trong [onset, onset+W]."""
    if not events:
        return float("nan")
    fidx = np.where(fired)[0]
    if len(fidx) == 0:
        return 0.0
    det = 0
    for ev in events:
        lo = ev["onset"]; hi = lo + W
        j = np.searchsorted(fidx, lo)
        if j < len(fidx) and fidx[j] <= hi:
            det += 1
    return det / len(events)


# ==================================================================
# 5. CẬN LÝ THUYẾT
# ==================================================================
def periodic_certificate(a, W):
    """R(periodic) >= min(1,(W+1)/K), K=ceil(1/a)."""
    K = math.ceil(1.0 / a)
    return min(1.0, (W + 1.0) / K), K


def thm1_upper(a, W, eta):
    """Cận trên Định lý 1 (δ=0): R(gate) <= 1 - η·(1 - min(1,(W+1)/K))."""
    cert, _ = periodic_certificate(a, W)
    return 1.0 - eta * (1.0 - cert)


def oblivious_random(a, W):
    """Recall kỳ vọng nếu bắn NGẪU NHIÊN tỉ lệ a: 1-(1-a)^(W+1) (mức 'noise' η=1)."""
    return 1.0 - (1.0 - a) ** (W + 1)


# ==================================================================
# 6. BOOTSTRAP + VÒNG QUÉT CHÍNH
# ==================================================================
def bootstrap_ci(vals, B, seed, ci=0.95):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    means = vals[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print("=" * 70)
    print("AwP Frame Scheduling — numerical experiments (Topic 03)")
    print("=" * 70)
    anchors = load_pa_anchors()
    print("[1] Neo P-A:", "OK" if anchors else "KHÔNG có")

    # 6.1 corpus video
    rng = np.random.default_rng(SEED)
    videos = [make_video(T_FRAMES, TARGET_INEVENT, rng) for _ in range(N_VIDEOS)]
    all_d = np.array([e["duration"] for v in videos for e in v["events"]])
    calib = dict(n_events=int(all_d.size), median=float(np.median(all_d)),
                 mean=float(all_d.mean()), cv=float(all_d.std() / all_d.mean()),
                 frac_single=float((all_d == 1).mean()), frac_ge5=float((all_d >= 5).mean()),
                 mean_inevent=float(np.mean([v["gt"].mean() for v in videos])))
    print("[2] corpus calib:", {k: round(v, 3) for k, v in calib.items()})

    # 6.2 d_prime để AUC(p=0) ~ 0.95
    try:
        from scipy.stats import norm
        d_prime = math.sqrt(2) * norm.ppf(0.95)
    except Exception:
        d_prime = math.sqrt(2) * 1.645
    print("[3] d_prime=%.3f" % d_prime)

    # 6.3 vòng quét: regime x a x λ x p  (+ đo η theo regime,p)
    print("[4] Sweep regimes=%s ..." % REGIMES)
    sweep_rows = []
    eta_map = {}                          # eta_map[(regime,p)] = η trung bình
    for regime in REGIMES:
        # đo η (không phụ thuộc a,λ)
        for p in CORRUPT_GRID:
            rngp = np.random.default_rng(SEED + 7 + int(p * 100))
            aucs = [onset_auc(predictor_score(v, T_FRAMES, d_prime, p, regime, rngp), v["win"])
                    for v in videos]
            eta_map[(regime, p)] = eta_from_auc(float(np.mean(aucs)))
        for a in A_GRID:
            for lam in LAMBDA_GRID:
                for p in CORRUPT_GRID:
                    rngp = np.random.default_rng(SEED + 7 + int(p * 100))
                    per = []
                    for v in videos:
                        s = predictor_score(v, T_FRAMES, d_prime, p, regime, rngp)
                        per.append(recall_windowed(hedge_gate_fire(s, T_FRAMES, a, lam),
                                                   v["events"], W))
                    m, lo, hi = bootstrap_ci(per, B, SEED + int(a * 1000) + int(lam * 10))
                    sweep_rows.append(dict(regime=regime, a=a, lam=lam, corruption_p=p,
                                           eta=round(eta_map[(regime, p)], 4),
                                           recall_mean=round(m, 4), ci_lo=round(lo, 4),
                                           ci_hi=round(hi, 4)))
    print("    η(noise):", {p: round(eta_map[("noise", p)], 3) for p in CORRUPT_GRID})
    print("    η(adv):  ", {p: round(eta_map[("adversarial", p)], 3) for p in CORRUPT_GRID})

    _write_csv(os.path.join(RESULTS_DIR, "corruption_sweep.csv"),
               ["regime", "a", "lam", "corruption_p", "eta", "recall_mean", "ci_lo", "ci_hi"],
               sweep_rows)

    # 6.4 consistency–robustness: consistency=recall(p=0); robustness=recall(p=1, adversarial)
    p_lo, p_hi = CORRUPT_GRID[0], CORRUPT_GRID[-1]
    cr_rows = []
    for a in A_GRID:
        cert, K = periodic_certificate(a, W)
        for lam in LAMBDA_GRID:
            c = _lk(sweep_rows, "adversarial", a, lam, p_lo)["recall_mean"]
            r = _lk(sweep_rows, "adversarial", a, lam, p_hi)["recall_mean"]
            cr_rows.append(dict(a=a, lam=lam, consistency_recall=c, robustness_recall=r,
                                backbone_floor=round(min(1.0, lam * (W + 1) / K), 4),
                                periodic_certificate=round(cert, 4),
                                oblivious_random=round(oblivious_random(a, W), 4)))
    _write_csv(os.path.join(RESULTS_DIR, "consistency_robustness.csv"),
               ["a", "lam", "consistency_recall", "robustness_recall", "backbone_floor",
                "periodic_certificate", "oblivious_random"], cr_rows)

    # 6.5 theory bounds (gate thuần λ=0)
    tb_rows = []
    for regime in REGIMES:
        for a in A_GRID:
            cert, K = periodic_certificate(a, W)
            for p in CORRUPT_GRID:
                eta = eta_map[(regime, p)]
                tb_rows.append(dict(regime=regime, a=a, K=K, eta=round(eta, 4),
                                    periodic_certificate=round(cert, 4),
                                    oblivious_random=round(oblivious_random(a, W), 4),
                                    thm1_upper_bound=round(thm1_upper(a, W, eta), 4),
                                    measured_pure_gate=_lk(sweep_rows, regime, a, 0.0, p)["recall_mean"]))
    _write_csv(os.path.join(RESULTS_DIR, "theory_bounds.csv"),
               ["regime", "a", "K", "eta", "periodic_certificate", "oblivious_random",
                "thm1_upper_bound", "measured_pure_gate"], tb_rows)

    # 6.6 anchors_pa.csv
    if anchors and "frontier" in anchors:
        rows = []
        for a in sorted(anchors["frontier"]):
            row = {"a": a}; row.update({k: round(v, 4) for k, v in anchors["frontier"][a].items()})
            rows.append(row)
        _write_csv(os.path.join(RESULTS_DIR, "anchors_pa.csv"), ["a", "periodic", "S", "RF"], rows)

    # 6.7 summary + figures
    summary = build_summary(sweep_rows, cr_rows, tb_rows, eta_map, calib, anchors, d_prime)
    json.dump(summary, open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w"),
              indent=2, ensure_ascii=False, allow_nan=False)
    make_figures(sweep_rows, cr_rows, tb_rows, eta_map, anchors)

    print("[5] Xong -> %s" % RESULTS_DIR)
    print("    sanity periodic (λ=1, noise):")
    for a in A_GRID:
        cert, K = periodic_certificate(a, W)
        v = _lk(sweep_rows, "noise", a, 1.0, 0.0)["recall_mean"]
        print("      a=%.3f recall=%.3f cert=%.3f (K=%d)" % (a, v, cert, K))
    print("    adversarial collapse (λ=0, p=1):")
    for a in A_GRID:
        print("      a=%.3f gate=%.3f  obliv_random=%.3f" %
              (a, _lk(sweep_rows, "adversarial", a, 0.0, 1.0)["recall_mean"], oblivious_random(a, W)))
    return summary


# ------------------------------------------------------------------ utils
def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def _lk(rows, regime, a, lam, p):
    for r in rows:
        if (r["regime"] == regime and abs(r["a"] - a) < 1e-9
                and abs(r["lam"] - lam) < 1e-9 and abs(r["corruption_p"] - p) < 1e-9):
            return r
    return None


def build_summary(sweep, cr, tb, eta_map, calib, anchors, d_prime):
    out = {
        "config": dict(seed=SEED, W=W, B=B, n_videos=N_VIDEOS, T=T_FRAMES, a_grid=A_GRID,
                       lambda_grid=LAMBDA_GRID, corrupt_grid=CORRUPT_GRID, regimes=REGIMES,
                       d_prime=round(d_prime, 4)),
        "note": ("Synthetic event-process hiệu chỉnh theo P-A sparse_duration_law; anchors là "
                 "số đo thật. Thí nghiệm video gốc (T10) vẫn cần dataset 121 video."),
        "calibration_check": {"target_pa": SPARSE_LAW,
                              "synthetic": {k: round(v, 4) for k, v in calib.items()}},
        "eta_by_regime_p": {rg: {str(p): round(eta_map[(rg, p)], 4) for p in CORRUPT_GRID}
                            for rg in REGIMES},
        "deployment_eta_band_cue_auc": [round(eta_from_auc(CUE_AUC_ONSET[1]), 4),
                                        round(eta_from_auc(CUE_AUC_ONSET[0]), 4)],
    }
    # (1) smoothness ở chế độ noise: độ dốc & tương quan recall~η
    smooth = {}
    for a in A_GRID:
        for lam in LAMBDA_GRID:
            etas = np.array([eta_map[("noise", p)] for p in CORRUPT_GRID])
            recs = np.array([_lk(sweep, "noise", a, lam, p)["recall_mean"] for p in CORRUPT_GRID])
            if etas.std() > 1e-9 and recs.std() > 1e-9:
                slope = float(np.polyfit(etas, recs, 1)[0]); r = float(np.corrcoef(etas, recs)[0, 1])
            else:
                slope, r = 0.0, None
            smooth["a%.3f_lam%.2f" % (a, lam)] = dict(slope=round(slope, 4),
                pearson_r=(round(r, 4) if r is not None else None),
                recall_eta_min=round(float(recs[0]), 4), recall_eta_max=round(float(recs[-1]), 4))
    out["smoothness_noise"] = smooth
    # (2) HEDGE giữ sàn? (adversarial robustness >= backbone_floor - tol)
    floor_ok = all(r["robustness_recall"] + 0.03 >= r["backbone_floor"] for r in cr)
    out["consistency_robustness"] = cr
    out["hedge_floor_respected"] = bool(floor_ok)
    # (3) Định lý 1 là CẬN TRÊN cho instance ĐỐI KHÁNG (worst-case), không phải cho mọi
    #     quá trình. Vì vậy chỉ kiểm tại cực đối kháng (p=1): gate thuần phải <= trần.
    eta_max_adv = eta_map[("adversarial", CORRUPT_GRID[-1])]
    ceiling_ok, looseness = True, {}
    for a in A_GRID:
        bound_p1 = thm1_upper(a, W, eta_max_adv)
        meas_p1 = _lk(sweep, "adversarial", a, 0.0, CORRUPT_GRID[-1])["recall_mean"]
        if meas_p1 > bound_p1 + 1e-6:
            ceiling_ok = False
        looseness["a%.3f_p1.0" % a] = round(bound_p1 - meas_p1, 4)
    out["theorem1_ceiling_holds_at_adversarial_extreme"] = bool(ceiling_ok)
    # Độ "lỏng" của cận: bound - measured. Lớn => cận yếu. Khi cert=1 (a>=1/(W+1)≈0.125) bound=1
    # bất kể η, trong khi gate sụp ~0 => cận informal HIỆN TẠI VÔ HIỆU ở vùng ngân sách cao.
    out["theorem1_bound_looseness_adversarial"] = looseness
    out["finding_for_T05"] = (
        "Cận informal R<=1-η(1-(W+1)/K) trở nên vô hiệu khi (W+1)/K>=1 (tức a>=1/(W+1)≈0.125): "
        "lúc đó bound=1 với mọi η nhưng gate đối kháng sụp về ~0. T05 cần một dạng cận dưới mạnh "
        "hơn cho vùng ngân sách cao (vd. dựa trên 'budget đặt sai chỗ' thay vì coverage cửa sổ).")
    # (4) adversarial: gate thuần có sụp XUỐNG DƯỚI oblivious_random không (worst-case Thm1)?
    adv_below = {("a%.3f" % a): bool(_lk(sweep, "adversarial", a, 0.0, 1.0)["recall_mean"]
                                     < oblivious_random(a, W))
                 for a in A_GRID}
    out["adversarial_gate_below_oblivious"] = adv_below
    # (5) sanity vs neo thật
    sanity = {}
    for a in A_GRID:
        cert, K = periodic_certificate(a, W)
        e = dict(simulated_periodic=_lk(sweep, "noise", a, 1.0, 0.0)["recall_mean"],
                 certificate=round(cert, 4),
                 adversarial_pure_gate=_lk(sweep, "adversarial", a, 0.0, 1.0)["recall_mean"])
        if anchors and "frontier" in anchors:
            keys = sorted(anchors["frontier"], key=lambda x: abs(x - a))
            e["pa_periodic_nearest"] = round(anchors["frontier"][keys[0]]["periodic"], 4)
            e["pa_gateS_nearest"] = round(anchors["frontier"][keys[0]]["S"], 4)
            e["pa_gateRF_nearest"] = round(anchors["frontier"][keys[0]]["RF"], 4)
            e["pa_nearest_a"] = keys[0]
        sanity["a%.3f" % a] = e
    out["sanity_vs_pa"] = sanity
    return out


# ------------------------------------------------------------------ figures
def make_figures(sweep, cr, tb, eta_map, anchors):
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(LAMBDA_GRID)))

    # Fig 2 — smoothness (NOISE): recall vs η
    fig, axes = plt.subplots(1, len(A_GRID), figsize=(5 * len(A_GRID), 4.2), sharey=True)
    for ax, a in zip(axes, A_GRID):
        for ci, lam in enumerate(LAMBDA_GRID):
            etas = [eta_map[("noise", p)] for p in CORRUPT_GRID]
            recs = [_lk(sweep, "noise", a, lam, p)["recall_mean"] for p in CORRUPT_GRID]
            ax.plot(etas, recs, "-o", color=colors[ci], lw=2, ms=4, label="λ=%.2f" % lam)
        cert, K = periodic_certificate(a, W)
        ax.axhline(cert, ls="--", color="gray", lw=1.1, label="periodic cert")
        ax.axhline(oblivious_random(a, W), ls=":", color="brown", lw=1.1, label="oblivious random")
        e_lo, e_hi = eta_from_auc(CUE_AUC_ONSET[1]), eta_from_auc(CUE_AUC_ONSET[0])
        ax.axvspan(e_lo, e_hi, color="orange", alpha=0.12)
        ax.set_title("a=%.3f (K=%d)" % (a, K)); ax.set_xlabel("prediction error η"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("windowed recall R (W=7)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("NOISE regime — HEDGE-GATE recall degrades smoothly in eta (Prop. 4). "
                 "Uninformative noise slides recall toward the oblivious-random level without breaking the floor.", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "fig2_noise_smoothness.png"), dpi=300)
    plt.close(fig)

    # Fig 3 — consistency–robustness frontier
    fig, ax = plt.subplots(figsize=(6.2, 5))
    mk = ["o", "s", "^"]
    for mi, a in enumerate(A_GRID):
        xs = [r["consistency_recall"] for r in cr if abs(r["a"] - a) < 1e-9]
        ys = [r["robustness_recall"] for r in cr if abs(r["a"] - a) < 1e-9]
        lams = [r["lam"] for r in cr if abs(r["a"] - a) < 1e-9]
        ax.plot(xs, ys, "-", color="gray", lw=1, alpha=0.5)
        sc = ax.scatter(xs, ys, c=lams, cmap="viridis", s=95, marker=mk[mi],
                        edgecolor="k", zorder=3, label="a=%.3f" % a)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("trust λ")
    ax.set_xlabel("Consistency: recall at eta=0 (perfect advice)")
    ax.set_ylabel("Robustness: recall at eta=1 (adversarial advice)")
    ax.set_title("Consistency-Robustness trade-off (Theorem 2)\n"
                 "lambda=0: high consistency / low robustness; lambda=1: reverse")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "fig3_consistency_robustness.png"), dpi=300)
    plt.close(fig)

    # Fig 4 — ADVERSARIAL: recall vs p, gate sụp dưới oblivious, HEDGE giữ sàn
    fig, axes = plt.subplots(1, len(A_GRID), figsize=(5 * len(A_GRID), 4.2), sharey=True)
    for ax, a in zip(axes, A_GRID):
        for ci, lam in enumerate(LAMBDA_GRID):
            recs = [_lk(sweep, "adversarial", a, lam, p)["recall_mean"] for p in CORRUPT_GRID]
            ax.plot(CORRUPT_GRID, recs, "-o", color=colors[ci], lw=2, ms=4, label="λ=%.2f" % lam)
        cert, K = periodic_certificate(a, W)
        ax.axhline(cert, ls="--", color="gray", lw=1.1, label="periodic cert")
        ax.axhline(oblivious_random(a, W), ls=":", color="brown", lw=1.1, label="oblivious random")
        # dải neo P-A (S & RF) tại a gần nhất
        if anchors and "frontier" in anchors:
            keys = sorted(anchors["frontier"], key=lambda x: abs(x - a))
            s_v = anchors["frontier"][keys[0]]["S"]; rf_v = anchors["frontier"][keys[0]]["RF"]
     