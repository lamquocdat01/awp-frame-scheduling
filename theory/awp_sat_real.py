# -*- coding: utf-8 -*-
"""
awp_sat_real.py — SAT-GATE trên CORPUS THẬT (uncapped cache 64x48_fALL).
==================================================================
Nạp cache như awp_percategory.py. Với mỗi a in {0.1,0.2,0.333} quét dial
q in {1.0,0.95,0.9,0.8,0.7}: tính windowed recall (video-level bootstrap CI,
B=1000, seed=42) của SAT-GATE trên cue thật, so với HEDGE(λ=0)=pure gate và
periodic(λ=1). Xuất results/sat_real.csv.

Kỳ vọng lý thuyết: corpus thật là sparse-event + cue near-uninformative
(η≈0.5) => mọi chính sách bị giới hạn bởi SÀN OBLIVIOUS. SAT-GATE KHÔNG vượt
HEDGE đáng kể (SAT>HEDGE là hiện tượng dense/high-budget đã chứng minh ở
synthetic, Bước 1). Nếu SAT vượt HEDGE đáng kể trên real => điều tra.
==================================================================
"""
import os, csv, numpy as np
import awp_real_data as R
from awp_experiments import (sat_gate_fire, hedge_gate_fire, recall_windowed,
                             periodic_certificate, oblivious_random,
                             onset_auc, eta_from_auc, bootstrap_ci,
                             W, B, SEED, RESULTS_DIR)

A_GRID = [0.10, 0.20, 0.333]
Q_GRID = [1.0, 0.95, 0.9, 0.8, 0.7]


def load_corpus():
    corpus = []
    for ds, vid, f, g in R.enumerate_videos():
        cp = R._cache_path(vid)
        if not os.path.exists(cp):
            continue
        z = np.load(cp); cue = z["cue"]; gt = z["gt"].astype(bool)
        ev = R.extract_events(gt)
        if len(ev) >= 2:
            win = np.zeros(len(gt), bool)
            for e in ev:
                win[e["onset"]:min(len(gt), e["onset"] + W + 1)] = True
            corpus.append(dict(ds=ds, vid=vid, cue=cue, events=ev, win=win, T=len(gt)))
    return corpus


def main():
    corpus = load_corpus()
    aucs = [onset_auc(c["cue"], c["win"]) for c in corpus]
    print("[corpus] %d videos >=2ev; events=%d frames=%d; onset-AUC=%.3f (eta=%.3f)"
          % (len(corpus), sum(len(c["events"]) for c in corpus),
             sum(c["T"] for c in corpus), float(np.mean(aucs)),
             eta_from_auc(float(np.mean(aucs)))))

    rows = []
    for a in A_GRID:
        cert, K = periodic_certificate(a, W)
        obliv = oblivious_random(a, W)
        # baselines (không phụ thuộc q)
        per_periodic = [recall_windowed(hedge_gate_fire(c["cue"], c["T"], a, 1.0), c["events"], W)
                        for c in corpus]
        per_hedge0 = [recall_windowed(hedge_gate_fire(c["cue"], c["T"], a, 0.0), c["events"], W)
                      for c in corpus]
        pm, plo, phi = bootstrap_ci(per_periodic, B, SEED + int(a * 1000) + 1)
        hm, hlo, hhi = bootstrap_ci(per_hedge0, B, SEED + int(a * 1000) + 2)
        for q in Q_GRID:
            per_sat = [recall_windowed(sat_gate_fire(c["cue"], c["T"], a, q), c["events"], W)
                       for c in corpus]
            sm, slo, shi = bootstrap_ci(per_sat, B, SEED + int(a * 1000) + int(q * 100))
            # MATCHED oblivious fraction: SAT(q) spends (1-q)B obliviously; the HEDGE with the
            # same oblivious content is lambda = 1-q. This is the fair mechanism comparison
            # (vs. the raw HEDGE(lam=0) endpoint which has ZERO oblivious content).
            lam_m = round(1.0 - q, 4)
            per_hedge_m = [recall_windowed(hedge_gate_fire(c["cue"], c["T"], a, lam_m), c["events"], W)
                           for c in corpus]
            hmm, hmlo, hmhi = bootstrap_ci(per_hedge_m, B, SEED + int(a * 1000) + int(lam_m * 100) + 50)
            rows.append(dict(
                a=a, q=q, K=K,
                sat_recall=round(sm, 4), sat_ci_lo=round(slo, 4), sat_ci_hi=round(shi, 4),
                hedge_lam0_recall=round(hm, 4), hedge_ci_lo=round(hlo, 4), hedge_ci_hi=round(hhi, 4),
                hedge_matched_lam=lam_m, hedge_matched_recall=round(hmm, 4),
                hedge_matched_ci_lo=round(hmlo, 4), hedge_matched_ci_hi=round(hmhi, 4),
                periodic_recall=round(pm, 4), periodic_ci_lo=round(plo, 4), periodic_ci_hi=round(phi, 4),
                periodic_cert=round(cert, 4), oblivious_random=round(obliv, 4),
                sat_minus_hedge0=round(sm - hm, 4),
                sat_minus_hedge_matched=round(sm - hmm, 4),
                sat_minus_periodic=round(sm - pm, 4)))

    fields = ["a", "q", "K", "sat_recall", "sat_ci_lo", "sat_ci_hi",
              "hedge_lam0_recall", "hedge_ci_lo", "hedge_ci_hi",
              "hedge_matched_lam", "hedge_matched_recall", "hedge_matched_ci_lo", "hedge_matched_ci_hi",
              "periodic_recall", "periodic_ci_lo", "periodic_ci_hi",
              "periodic_cert", "oblivious_random",
              "sat_minus_hedge0", "sat_minus_hedge_matched", "sat_minus_periodic"]
    with open(os.path.join(RESULTS_DIR, "sat_real.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); [w.writerow(r) for r in rows]

    print("== SAT-GATE vs HEDGE(λ=0 endpoint), HEDGE(λ=1-q matched-oblivious), periodic(λ=1) ==")
    print("  a     q    SAT    HEDGE0 HEDGEm(λ)   per   |  SAT-HEDGE0  SAT-HEDGEm  SAT-per")
    max_adv0, max_advm = 0.0, 0.0
    for r in rows:
        max_adv0 = max(max_adv0, r["sat_minus_hedge0"])
        max_advm = max(max_advm, abs(r["sat_minus_hedge_matched"]))
        print("  %.3f %.2f %.3f  %.3f  %.3f(%.2f) %.3f |  %+0.3f      %+0.3f      %+0.3f"
              % (r["a"], r["q"], r["sat_recall"], r["hedge_lam0_recall"],
                 r["hedge_matched_recall"], r["hedge_matched_lam"], r["periodic_recall"],
                 r["sat_minus_hedge0"], r["sat_minus_hedge_matched"], r["sat_minus_periodic"]))
    max_satper = max(r["sat_minus_periodic"] for r in rows)
    print("[dial-mismatch] max(SAT - HEDGE_λ0) = %+0.4f  (SAT(q<1) has oblivious content, HEDGE(λ=0) has none;"
          " NOT a frontier advantage — also appears at a=0.2 where β=1)" % max_adv0)
    print("[matched test]  max|SAT - HEDGE_matched(λ=1-q)| = %0.4f  -> %s"
          % (max_advm, "SAT ≈ HEDGE at matched oblivious budget (same mechanism on real data)"
             if max_advm < 0.05 else "SAT differs from matched HEDGE — INVESTIGATE further"))
    print("[deployment]    max(SAT - periodic) = %+0.4f  -> periodic %s"
          % (max_satper, "still wins (SAT never exceeds oblivious floor on real cue)"
             if max_satper < 0 else "beaten by SAT — INVESTIGATE"))
    print("Saved results/sat_real.csv")
    return rows


if __name__ == "__main__":
    main()
