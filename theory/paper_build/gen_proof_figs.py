# -*- coding: utf-8 -*-
"""
gen_proof_figs.py — Biểu đồ GIẢNG GIẢI (didactic) cho file proofs song ngữ.
Tạo 5 hình PNG vào proof_figs/: model+Lemma1, Theorem 1', frontier (Thm2), smoothness (Prop4),
converse (p_b). Chú thích trong hình bằng tiếng Anh; phần song ngữ để ở văn bản Word.
"""
import os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

OUT = "/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/proof_figs"
os.makedirs(OUT, exist_ok=True)
BLUE, GREEN, RED, GRAY = "#2c6fbb", "#1e7a33", "#c0392b", "#888888"


def _timeline(ax, T, onsets, W, fires, title, fire_color=BLUE, fire_label="detector fire"):
    """Vẽ 1 timeline: ô sự kiện [onset,onset+W] (xanh lá), các lần bắn (vạch dọc)."""
    ax.set_xlim(-0.5, T + 0.5); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_title(title, fontsize=10, loc="left")
    for o in onsets:                                   # cửa sổ recall [o, o+W]
        ax.add_patch(Rectangle((o, 0.30), W, 0.40, color=GREEN, alpha=0.22))
        ax.plot([o], [0.5], marker="v", color=GREEN, ms=9)
    for f in fires:
        ax.axvline(f, 0.12, 0.88, color=fire_color, lw=1.8)
    # đánh dấu caught/missed
    for o in onsets:
        caught = any(o <= f <= o + W for f in fires)
        ax.text(o + W / 2.0, 0.80, "✓" if caught else "✗",
                ha="center", color=(GREEN if caught else RED), fontsize=13, fontweight="bold")
    ax.set_xlabel("frame t")


# ---------- Fig 1: model + Lemma 1 ----------
def fig_model():
    T, W = 24, 2
    onsets = [2, 9, 13, 20]
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    K1 = 3; fires1 = list(range(0, T + 1, K1))            # K=3 ≤ W+1=3  -> R=1
    _timeline(axes[0], T, onsets, W, fires1,
              "K = 3 ≤ W+1 = 3 : every window contains a fire → recall R = 1")
    K2 = 6; fires2 = list(range(0, T + 1, K2))            # K=6 > W+1   -> coverage (W+1)/K
    _timeline(axes[1], T, onsets, W, fires2,
              "K = 6 > W+1 = 3 : a fire lands in a window only (W+1)/K = 1/2 of the time")
    fig.suptitle("Model & Lemma 1 — windowed recall. Green band = staleness window [o, o+W]; "
                 "▼ = event onset; vertical line = detector fire.", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pfig1_model.png"), dpi=200); plt.close(fig)


# ---------- Fig 2: Theorem 1' ----------
def fig_thm1p():
    T, W = 30, 2
    onsets = [4, 12, 23]                                   # 3 events
    clutter = [1, 2, 7, 8, 9, 16, 17, 18, 27, 28]          # high-cue "clutter", away from events
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4))
    # (a) pure-advice gate: fires all budget on clutter -> R=0
    _timeline(axes[0], T, onsets, W, clutter, fire_color=RED,
              title="(a) Pure-advice gate fires the top-cue (clutter) frames → 0 events caught (R = 0)")
    # (b) HEDGE: backbone periodic (blue) catches some; clutter wasted (faint red)
    Kbb = 7; backbone = list(range(0, T + 1, Kbb))
    for f in clutter:
        axes[1].axvline(f, 0.12, 0.88, color=RED, lw=1.0, alpha=0.25)
    _timeline(axes[1], T, onsets, W, backbone, fire_color=BLUE,
              title="(b) HEDGE-GATE: oblivious backbone (blue) guarantees floor λβ regardless of advice")
    fig.suptitle("Theorem 1′ — advice can be adversarial (points away from events). "
                 "Trusting it fully → R=0; a backbone rhythm guarantees a floor.", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pfig2_thm1prime.png"), dpi=200); plt.close(fig)


# ---------- Fig 3: frontier (Theorem 2) ----------
def fig_frontier():
    beta = 0.8
    c = np.linspace(1 - (1 - beta), 1, 200)               # r = beta(1-c)/(1-beta) >=0 region
    fig, ax = plt.subplots(figsize=(6.4, 5))
    cc = np.linspace(beta, 1, 200); rr = beta * (1 - cc) / (1 - beta)
    ax.plot(cc, rr, color=BLUE, lw=2.5, label="frontier  r = β(1−c)/(1−β)")
    ax.fill_between(cc, 0, rr, color=BLUE, alpha=0.10)
    lams = [0, 0.25, 0.5, 0.75, 1.0]
    cs = [(1 - L) + L * beta for L in lams]; rs = [L * beta for L in lams]
    ax.scatter(cs, rs, c=lams, cmap="viridis", s=110, edgecolor="k", zorder=3)
    for L, x, y in zip(lams, cs, rs):
        ax.annotate("λ=%.2f" % L, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.annotate("trust advice\n(consistent, fragile)", (1.0, 0.0), xytext=(0.86, 0.06),
                fontsize=8, color=RED, ha="center")
    ax.annotate("periodic backbone\n(robust, less consistent)", (beta, beta), xytext=(0.83, 0.66),
                fontsize=8, color=GREEN, ha="center")
    ax.set_xlabel("consistency  c   (recall when advice is perfect, η=0)")
    ax.set_ylabel("robustness  r   (recall when advice is adversarial, η=1)")
    ax.set_title("Theorem 2 — tight trade-off (β = 0.8 example)\nevery non-adaptive gate lives on/under the line; HEDGE-GATE is ON it")
    ax.grid(alpha=0.3); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pfig3_frontier.png"), dpi=200); plt.close(fig)


# ---------- Fig 4: smoothness (Prop 4) ----------
def fig_smooth():
    beta = 0.8; eta = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for L, kappa, col in [(0.0, 1.0, RED), (0.5, 0.85, BLUE), (1.0, 0.0, GREEN)]:
        floor = L * beta
        R = floor + (1 - eta) * kappa * (1 - floor)
        ax.plot(eta, R, color=col, lw=2.2, label="λ=%.2f (floor λβ=%.2f)" % (L, floor))
        ax.axhline(floor, ls=":", color=col, lw=1)
    ax.set_xlabel("prediction error  η"); ax.set_ylabel("expected recall  E[R]")
    ax.set_title("Proposition 4 — recall is a STRAIGHT line in η (no cliff)\nE[R] = λβ + (1−η)·κ(λ)·(1−λβ)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pfig4_smoothness.png"), dpi=200); plt.close(fig)


# ---------- Fig 5: converse (p_b bars) ----------
def fig_converse():
    rng = np.random.default_rng(1)
    N = 16; advised = [2, 5, 9, 12]                        # set A (cue marks these)
    p = np.clip(rng.uniform(0.05, 0.5, N), 0, 1)
    for b in advised:
        p[b] = rng.uniform(0.7, 0.95)                      # gate fires advised blocks w.h.p. (consistency)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [GREEN if b in advised else GRAY for b in range(N)]
    ax.bar(range(N), p, color=colors, edgecolor="k")
    # adversary picks the smallest non-advised bars for events (B=4)
    nonadv = [b for b in range(N) if b not in advised]
    chosen = sorted(nonadv, key=lambda b: p[b])[:4]
    for b in chosen:
        ax.bar(b, p[b], color=RED, edgecolor="k")
        ax.text(b, p[b] + 0.02, "event", ha="center", fontsize=7, color=RED, rotation=90)
    ax.set_xticks(range(N)); ax.set_xlabel("block b")
    ax.set_ylabel("p_b = P(gate fires in block b)")
    ax.set_title("Theorem 2 converse — fixed firing probabilities p_b (same in both instances).\n"
                 "Green = advised set A (high p_b → consistency). Adversary hides events on the "
                 "SMALLEST non-advised bars (red) → low robustness.")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="advised A"), Patch(color=GRAY, label="non-advised"),
                       Patch(color=RED, label="events at η=1 (smallest p_b)")], fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pfig5_converse.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_model(); fig_thm1p(); fig_frontier(); fig_smooth(); fig_converse()
    print("Saved 5 didactic figures ->", OUT)
    print(os.listdir(OUT))
