# -*- coding: utf-8 -*-
"""
awp_real_data.py  —  T10 trên DỮ LIỆU THẬT (FULL 68 video, siết CI cho §VI)
==================================================================
Đọc dataset gốc D:\\THS Programing\\06.6 ASMAG Project\\datasets\\
  - BMC synth1/synth2 (20 video): frames/*.jpg + gt/*.png  (fg = pixel sáng >200)
  - LASIESTA (48 video): I_xx/*.bmp + I_xx-GT/*.png         (fg = kênh sáng >200)

Cue (π̂/S-gate) tính TRUNG THỰC từ frame (không dùng GT): cue[t] = mean|Δgray|.
gt_event[t] = (pixel fg trong GT > ngưỡng). Sự kiện = run liên tiếp, min_duration=2.
Chạy HEDGE-GATE(λ) + corruption sweep (nhiễu hoán vị), W=7, bootstrap B=1000, seed=42.

TỐI ƯU cho FULL run: đọc ảnh bằng OpenCV (cv2) + đa tiến trình (multiprocessing) +
cache .npz theo tham số. Mặc định CHẠY TOÀN BỘ (MAX_*=None). Có thể chạy nhiều lần;
mỗi lần cache thêm video đã trích (an toàn khi bị giới hạn thời gian).
==================================================================
"""
import os
import re
import glob
import csv
import json
import numpy as np
import cv2
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awp_experiments import (hedge_gate_fire, recall_windowed, periodic_certificate,
                             oblivious_random, bootstrap_ci, onset_auc, eta_from_auc,
                             A_GRID, LAMBDA_GRID, CORRUPT_GRID, W, B, SEED, RESULTS_DIR)

DATASETS_ROOT = r"D:\THS Programing\06.6 ASMAG Project\datasets"
DOWNSCALE = (64, 48)          # (W,H) ảnh xám nhỏ -> nhanh; motion density vẫn ổn định
MAX_FRAMES = None            # UNCAPPED: đọc TOÀN BỘ frame mỗi video (~175k tổng)
MAX_VIDEOS_PER_DS = None      # FULL: tất cả video
FG_THR = 200                  # pixel foreground (object) trong GT
FG_FRAC_THR = 0.0008          # tỉ lệ pixel fg tối thiểu để coi là "có sự kiện"
MIN_DURATION = 2
N_WORKERS = 8                 # số core khả dụng
TAG = "64x48_fALL"         # cache riêng cho bản uncapped, không đè cache cũ f300
CACHE_DIR = os.path.join(RESULTS_DIR, "awp_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
NAT = re.compile(r"(\d+)")


def natkey(s):
    return [int(x) if x.isdigit() else x for x in NAT.split(os.path.basename(s))]


def _cache_path(vid_id):
    return os.path.join(CACHE_DIR, "%s__%s.npz" % (vid_id, TAG))


def extract_one(job):
    """Worker đa tiến trình: trích (cue, gt_event) cho 1 video, cache .npz. Trả (vid_id, n_frames)."""
    vid_id, frames, gts = job
    cache = _cache_path(vid_id)
    if os.path.exists(cache):
        return (vid_id, -1)                       # đã có
    frames = sorted(frames, key=natkey)
    gts = sorted(gts, key=natkey)
    if MAX_FRAMES:
        frames, gts = frames[:MAX_FRAMES], gts[:MAX_FRAMES]
    n = min(len(frames), len(gts))
    if n < 5:
        return (vid_id, 0)
    cue = np.zeros(n, dtype=np.float32)
    gt = np.zeros(n, dtype=bool)
    prev = None
    tot_px = DOWNSCALE[0] * DOWNSCALE[1]
    for t in range(n):
        g = cv2.imread(frames[t], cv2.IMREAD_GRAYSCALE)
        if g is None:
            cue[t] = cue[t - 1] if t else 0.0
        else:
            g = cv2.resize(g, DOWNSCALE).astype(np.float32)
            cue[t] = float(np.mean(np.abs(g - prev))) if prev is not None else 0.0
            prev = g
        m = cv2.imread(gts[t], cv2.IMREAD_COLOR)
        if m is not None:
            m = cv2.resize(m, DOWNSCALE)
            fg = int((m.max(axis=2) > FG_THR).sum())
            gt[t] = (fg > FG_FRAC_THR * tot_px)
    if n > 1:
        cue[0] = cue[1]
    np.savez_compressed(cache, cue=cue, gt=gt)
    return (vid_id, n)


def extract_events(gt, min_duration=MIN_DURATION):
    events = []; T = len(gt); t = 0
    while t < T:
        if gt[t]:
            s = t
            while t < T and gt[t]:
                t += 1
            if (t - s) >= min_duration:
                events.append({"onset": s, "end": t - 1, "duration": t - s})
        else:
            t += 1
    return events


def enumerate_videos():
    vids = []
    for sub in ["synth1", "synth2"]:
        base = os.path.join(DATASETS_ROOT, "BMC", sub)
        if os.path.isdir(base):
            for v in sorted(os.listdir(base)):
                fdir = os.path.join(base, v, "frames"); gdir = os.path.join(base, v, "gt")
                if os.path.isdir(fdir) and os.path.isdir(gdir):
                    vids.append(("BMC", "BMC_%s_%s" % (sub, v),
                                 glob.glob(fdir + "/*"), glob.glob(gdir + "/*")))
    las = os.path.join(DATASETS_ROOT, "LASIESTA")
    if os.path.isdir(las):
        for gtd in sorted(glob.glob(las + "/*-GT")):
            base = gtd[:-3]
            if os.path.isdir(base):
                vids.append(("LASIESTA", "LAS_" + os.path.basename(base),
                             glob.glob(base + "/*.bmp") + glob.glob(base + "/*.jpg"),
                             glob.glob(gtd + "/*.png")))
    # CDnet2014: <cat>/<video>/input + groundtruth + temporalROI.txt (fg pixel == 255)
    cd = os.path.normpath(os.path.join(DATASETS_ROOT, "..", "dataset cdnet2014", "archive", "dataset"))
    if os.path.isdir(cd):
        for cat in sorted(os.listdir(cd)):
            catd = os.path.join(cd, cat)
            if not os.path.isdir(catd):
                continue
            for v in sorted(os.listdir(catd)):
                vd = os.path.join(catd, v)
                ind = os.path.join(vd, "input"); gtd = os.path.join(vd, "groundtruth")
                if not (os.path.isdir(ind) and os.path.isdir(gtd)):
                    continue
                fr = sorted(glob.glob(ind + "/*.jpg"), key=natkey)
                gg = sorted(glob.glob(gtd + "/*.png"), key=natkey)
                s, e = 1, len(fr)
                roi = os.path.join(vd, "temporalROI.txt")
                if os.path.exists(roi):
                    try:
                        s, e = [int(x) for x in open(roi).read().split()[:2]]
                    except Exception:
                        pass
                fr = fr[s - 1:e]; gg = gg[s - 1:e]   # cat theo temporalROI (vung danh gia)
                if fr and gg:
                    vids.append(("CDnet", "CDN_%s_%s" % (cat, v), fr, gg))
    return vids


def main():
    print("=" * 70); print("T10 REAL-DATA — FULL 68-video (cv2 + multiprocessing)"); print("=" * 70)
    allv = enumerate_videos()
    sel, cnt = [], {}
    for ds, vid, f, g in allv:
        if not f or not g:
            continue
        if MAX_VIDEOS_PER_DS and cnt.get(ds, 0) >= MAX_VIDEOS_PER_DS:
            continue
        sel.append((ds, vid, f, g)); cnt[ds] = cnt.get(ds, 0) + 1
    print("[1] Chọn %d video. Theo ds: %s" % (len(sel), cnt))

    # --- trích song song (skip video đã cache) ---
    jobs = [(vid, f, g) for ds, vid, f, g in sel if not os.path.exists(_cache_path(vid))]
    print("[2] Cần trích %d video (đã cache %d)." % (len(jobs), len(sel) - len(jobs)))
    if jobs:
        with Pool(N_WORKERS) as pool:
            done = 0
            for vid_id, n in pool.imap_unordered(extract_one, jobs):
                done += 1
                if n and n > 0:
                    print("    [trích %d/%d] %s n=%d" % (done, len(jobs), vid_id, n), flush=True)

    # --- nạp corpus từ cache ---
    corpus = []
    for ds, vid, f, g in sel:
        cp = _cache_path(vid)
        if not os.path.exists(cp):
            continue
        z = np.load(cp); cue = z["cue"]; gt = z["gt"].astype(bool)
        ev = extract_events(gt)
        if len(ev) >= 2:
            win = np.zeros(len(gt), dtype=bool)
            for e in ev:
                win[e["onset"]:min(len(gt), e["onset"] + W + 1)] = True
            corpus.append(dict(ds=ds, vid=vid, cue=cue, events=ev, win=win, T=len(gt)))
    n_extracted = sum(1 for _, vid, _, _ in [(d, v, f, g) for d, v, f, g in sel] if os.path.exists(_cache_path(vid)))
    print("[3] Đã cache %d/%d video; %d video có >=2 sự kiện. Sự kiện: %d, frame: %d" %
          (n_extracted, len(sel), len(corpus), sum(len(c["events"]) for c in corpus),
           sum(c["T"] for c in corpus)))
    if n_extracted < len(sel):
        print("    [CHƯA ĐỦ — chạy lại để trích nốt %d video còn lại rồi mới xuất kết quả]"
              % (len(sel) - n_extracted))
        return None

    # --- onset-AUC cue thật ---
    aucs = [onset_auc(c["cue"], c["win"]) for c in corpus]
    eta_dep = eta_from_auc(float(np.mean(aucs)))
    by_ds = {}
    for c in corpus:
        by_ds.setdefault(c["ds"], []).append(onset_auc(c["cue"], c["win"]))
    print("[4] onset-AUC cue: mean=%.3f [%.3f,%.3f] -> η_dep=%.3f (P-A [0.49,0.79]); theo ds: %s"
          % (float(np.mean(aucs)), float(np.min(aucs)), float(np.max(aucs)), eta_dep,
             {k: round(float(np.mean(v)), 3) for k, v in by_ds.items()}))

    def corrupt(cue, p, rng):
        out = cue.copy(); k = int(round(p * len(cue)))
        if k > 1:
            idx = rng.choice(len(cue), size=k, replace=False)
            out[idx] = cue[idx][rng.permutation(k)]
        return out

    # --- corruption sweep + bootstrap (video-level, B=1000) ---
    rows = []
    for a in A_GRID:
        for lam in LAMBDA_GRID:
            for p in CORRUPT_GRID:
                rng = np.random.default_rng(SEED + 7 + int(p * 100))
                per = [recall_windowed(hedge_gate_fire(corrupt(c["cue"], p, rng), c["T"], a, lam),
                                       c["events"], W) for c in corpus]
                m, lo, hi = bootstrap_ci(per, B, SEED + int(a * 1000) + int(lam * 10))
                rng2 = np.random.default_rng(SEED + int(p * 100))
                etas = [onset_auc(corrupt(c["cue"], p, rng2), c["win"]) for c in corpus]
                rows.append(dict(a=a, lam=lam, corruption_p=p,
                                 eta=round(eta_from_auc(float(np.mean(etas))), 4),
                                 recall_mean=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

    with open(os.path.join(RESULTS_DIR, "corruption_sweep_REAL.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["a", "lam", "corruption_p", "eta", "recall_mean", "ci_lo", "ci_hi"])
        w.writeheader(); [w.writerow(r) for r in rows]

    deployed = {}
    for a in A_GRID:
        cert, K = periodic_certificate(a, W)
        g0 = _get(rows, a, 0.0, 0.0); per = _get(rows, a, 1.0, 0.0)
        deployed["a%.3f" % a] = dict(real_cue_gate=g0["recall_mean"], gate_ci=[g0["ci_lo"], g0["ci_hi"]],
                                     periodic=per["recall_mean"], periodic_ci=[per["ci_lo"], per["ci_hi"]],
                                     certificate=round(cert, 4),
                                     gate_minus_periodic=round(g0["recall_mean"] - per["recall_mean"], 4))

    summary = dict(
        source="REAL video FULL (BMC synth1/2 + LASIESTA) @ THS Programing/06.6 ASMAG Project/datasets",
        config=dict(seed=SEED, W=W, B=B, downscale=DOWNSCALE, max_frames=MAX_FRAMES,
                    n_videos_selected=len(sel), n_videos_with_events=len(corpus),
                    n_events=sum(len(c["events"]) for c in corpus),
                    n_frames=sum(c["T"] for c in corpus)),
        cue_onset_auc=dict(mean=round(float(np.mean(aucs)), 4), min=round(float(np.min(aucs)), 4),
                           max=round(float(np.max(aucs)), 4),
                           by_dataset={k: round(float(np.mean(v)), 4) for k, v in by_ds.items()},
                           pa_reference=[0.49, 0.79]),
        eta_deployed=round(eta_dep, 4),
        deployed_gate_vs_periodic=deployed,
        note="FULL 68-video real T10. CI hẹp hơn bản subset. cue=motion-diff, events từ GT.")
    json.dump(summary, open(os.path.join(RESULTS_DIR, "metrics_summary_REAL.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, allow_nan=False)

    # --- hình recall vs η ---
    fig, axes = plt.subplots(1, len(A_GRID), figsize=(5 * len(A_GRID), 4.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.88, len(LAMBDA_GRID)))
    for ax, a in zip(axes, A_GRID):
        for ci, lam in enumerate(LAMBDA_GRID):
            etas = [_get(rows, a, lam, p)["eta"] for p in CORRUPT_GRID]
            recs = [_get(rows, a, lam, p)["recall_mean"] for p in CORRUPT_GRID]
            los = [_get(rows, a, lam, p)["ci_lo"] for p in CORRUPT_GRID]
            his = [_get(rows, a, lam, p)["ci_hi"] for p in CORRUPT_GRID]
            ax.plot(etas, recs, "-o", color=colors[ci], lw=2, ms=4, label="λ=%.2f" % lam)
            ax.fill_between(etas, los, his, color=colors[ci], alpha=0.12)
        cert, K = periodic_certificate(a, W)
        ax.axhline(cert, ls="--", color="gray", lw=1.1, label="periodic cert")
        ax.axhline(oblivious_random(a, W), ls=":", color="brown", lw=1.1, label="oblivious random")
        ax.set_title("REAL a=%.3f (K=%d)" % (a, K)); ax.set_xlabel("prediction error eta (real cue)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("windowed recall R (W=7)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("T10 real-data UNCAPPED (%d-video corpus BMC+LASIESTA+CDnet, B=1000): "
                 "HEDGE-GATE recall vs η (dải = CI95)." % len(corpus), fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "fig5_real_corruption.png"), dpi=300)
    plt.close(fig)

    print("[5] Deployed gate vs periodic:")
    for k, v in deployed.items():
        print("    %s: gate=%.3f%s periodic=%.3f%s gap=%+.3f" %
              (k, v["real_cue_gate"], tuple(v["gate_ci"]), v["periodic"], tuple(v["periodic_ci"]),
               v["gate_minus_periodic"]))
    print("[6] Xong FULL -> corruption_sweep_REAL.csv, metrics_summary_REAL.json, fig5_real_corruption.png")
    return summary


def _get(rows, a, lam, p):
    for r in rows:
        if abs(r["a"] - a) < 1e-9 and abs(r["lam"] - lam) < 1e-9 and abs(r["corruption_p"] - p) < 1e-9:
            return r
    return None


if __name__ == "__main__":    main()
