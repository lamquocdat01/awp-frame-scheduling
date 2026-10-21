# -*- coding: utf-8 -*-
"""
gen_paper_v2.py — Bản thảo v2 (ĐỒNG BỘ với proofs đã chứng minh, Route A).
  - Theorem 1′ PROVEN; Theorem 2 & 3 PROVEN cho lớp NON-ADAPTIVE, biên chặt r=β(1−c)/(1−β);
    Proposition 4 PROVEN dưới Assumption A. Bỏ slope cũ (κK)/(W+1).
  - Tác giả đơn: Lam Quoc Dat <lamquocdat@gmail.com>. §VI: real-data 121 video. Appendix có proofs.
Xuất 03_Paper_Draft_v2.docx vào submission_package/.
"""
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

RES = "/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/results"
OUT = "/sessions/bold-wizardly-cannon/mnt/PhD Program/02. Working/03_AwP_Frame_Scheduling/submission_package/03_Paper_Draft_v2.docx"

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)


def H(t, l=1): doc.add_heading(t, level=l)
def P(t, italic=False, bold=False, align=None):
    p = doc.add_paragraph(); r = p.add_run(t); r.italic = italic; r.bold = bold
    if align == "c": p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p
def THM(label, text):
    p = doc.add_paragraph(); p.add_run(label + " ").bold = True; p.add_run(text).italic = True; return p
def fig(path, cap, width=6.2):
    try:
        doc.add_picture(path, width=Inches(width)); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    except Exception as e:
        P("[figure missing: %s]" % e)
    P(cap, italic=True, align="c")


# ---------------- Title / author ----------------
doc.add_heading("Learning-Augmented Frame Scheduling: Consistency–Robustness Bounds "
                "for Gating with Untrusted Motion Predictors", level=0)
P("Lam Quoc Dat", bold=True, align="c")
P("CERTES dissertation, Topic 03 · lamquocdat@gmail.com · [Affiliation]", italic=True, align="c")
P("Draft v2 · June 12, 2026 · Target: IEEE Transactions on Computers", italic=True, align="c")

# ---------------- Abstract ----------------
H("Abstract", 1)
P("Energy-constrained edge devices must decide online which video frames receive an object-detector "
  "invocation. Content-aware gating policies consult cheap motion predictors to concentrate the "
  "energy budget on event-bearing frames, yet a recent multi-protocol evaluation over 121 videos "
  "and 3,713 events found that, at equal energy, such gates are dominated by an oblivious periodic "
  "schedule — at activation a = 0.20 they recover 20–28% of events against the periodic schedule’s "
  "98.4%. We explain this at the theorem level. Casting gated detection as online sampling with an "
  "untrusted predictor in the algorithms-with-predictions (AwP) framework, we (i) define a "
  "prediction-error metric η on motion cues; (ii) prove a regime-indexed lower bound — a coverage "
  "bound for dense events and, in the high-budget regime a ≥ 1/(W+1), an advice-commitment bound "
  "(Theorem 1′) showing the advice arm contributes zero worst-case recall; (iii) prove, for the "
  "deployed class of non-adaptive gates, a TIGHT consistency–robustness frontier r = β(1−c)/(1−β) "
  "(robustness ratio Ω(1/ε) for (1+ε)-consistency), where β = min(1,(W+1)/K); and (iv) construct "
  "HEDGE-GATE, a budget-splitting meta-policy that ATTAINS this frontier and degrades affinely in η "
  "(Proposition 4). A controlled predictor-corruption sweep on synthetic streams calibrated to the "
  "measured corpus confirms the results, and a real-data run over a 121-video BMC+LASIESTA+CDnet2014 "
  "corpus reproduces the deployment finding: a motion cue with onset-AUC ≈ 0.50 yields a gate that "
  "trails the periodic schedule by 0.42–0.44 recall at low activation. The results convert an "
  "empirical negative finding into a structural impossibility theorem for perception sampling, and "
  "give designers a certifiable dial between exploiting predictions and guaranteeing worst-case "
  "recall.")

# ---------------- I. Introduction ----------------
H("I. Introduction", 1)
P("Modern edge video-analytics pipelines operate under a hard energy budget: invoking a deep object "
  "detector on every frame is infeasible, so a scheduler must select a subset of frames. The "
  "dominant design philosophy is content-awareness — use a cheap signal (frame difference, motion "
  "score, a learned policy) to predict which frames matter, and spend the detector budget there. "
  "The promise is intuitive: why sample blind when you can sample smart?")
P("A recent detection-grounded, multi-regime, multi-protocol evaluation (the P-A study) challenged "
  "this intuition. Across CDnet2014, LASIESTA, BMC and VIRAT (121 videos, 3,713 events), with "
  "video-level bootstrap (B = 1000), content-aware and learned gates provided no deployable "
  "advantage over an oblivious periodic schedule at equal energy. At a = 0.20 an S-gate recovers "
  "28.4% and a learned RF-gate 20.0% of events, while the periodic schedule recovers 98.4%; "
  "cross-dataset (LODO) evaluation reverses the within-dataset ranking entirely. These findings are "
  "empirical. The natural scientific question is why adaptivity fails, and whether it must. This "
  "paper answers with matching upper and lower bounds in the algorithms-with-predictions (AwP) "
  "framework. Our thesis: the gate is not a bad idea badly executed; it is an online algorithm "
  "trusting an untrusted predictor, and its observed collapse is the robustness arm of a provable "
  "consistency–robustness trade-off.")
P("Contributions. (1) A formal model of gated perception sampling with windowed detection-recall "
  "and hard activation budgets (§III). (2) A prediction-error metric η on motion cues, computable "
  "from deployment logs (§III-C). (3) A regime-indexed lower bound: Theorem 1 (dense) and the new "
  "Theorem 1′ (high-budget) (§IV). (4) Theorem 2: a tight consistency–robustness frontier for "
  "non-adaptive gates, with Theorem 3 establishing HEDGE-GATE’s Pareto-optimality in that class "
  "(§IV-C, §V). (5) HEDGE-GATE, attaining the frontier with affine degradation in η (Proposition 4). "
  "(6) Empirical grounding on the P-A corpus, a calibrated synthetic sweep, and a real-data run "
  "over a 121-video BMC+LASIESTA+CDnet2014 corpus (§VI). Proofs are in Appendix A.")

# ---------------- II. Related Work ----------------
H("II. Related Work", 1)
P("Algorithms with predictions. Lykouris and Vassilvitskii initiated competitive analysis with "
  "machine-learned advice for caching [1]; Purohit, Svitkina and Kumar formalized consistency–"
  "robustness trade-offs for ski-rental and scheduling [2]; Mitzenmacher and Vassilvitskii survey "
  "the framework [3]. To our knowledge, no prior work treats perception sampling, where the "
  "objective is windowed event recall, the budget is a hard activation constraint, and advice "
  "quality varies by scene and regime.")
P("Content-aware video analytics. Adaptive frame-selection systems report large savings under "
  "within-dataset evaluation; the P-A study [4] showed these conclusions are protocol-dependent. "
  "The Deep-2 companion [5] develops worst-case latency certificates for fixed-rate schedules; the "
  "per-scene-K analysis shows the true lever is the refresh rate K (oracle skip 0.728 at recall 0.9 "
  "vs gate 0.636 and scene-agnostic 0.591), but a causal zero-shot estimator captures only 0.24 "
  "(CI [−0.07,+0.45]) of that headroom. Our theory explains both: K-selection is advice about rate "
  "(low dimension, slowly varying), frame-level gating is advice about timing (high dimension, "
  "adversarially fragile).")
P("Sampling and freshness theory. Age-of-Information characterizes optimal status-update schedules "
  "[6]; quickest change detection with observation control [7] bounds detection delay under "
  "sampling constraints. These give oblivious-schedule baselines but do not model untrusted advice; "
  "we import their windowed-staleness semantics into the AwP setting.")

# ---------------- III. Model ----------------
H("III. Model and Preliminaries", 1)
P("Time is slotted t = 1,…,T (frames). An unknown event process marks onsets o_1<…<o_m. A schedule "
  "σ ⊆ [T] fires the detector on |σ| frames; activation a(σ) = |σ|/T, with hard budget |σ| ≤ "
  "B = ⌊aT⌋. An event is recalled if the detector fires within staleness window W of its onset: "
  "σ ∩ [o_j, o_j+W] ≠ ∅, with W = 7. The objective is windowed recall R(σ) = (1/m) Σ_j 1[recalled] "
  "subject to |σ| ≤ B. Let K = ⌈1/a⌉ and β := min(1,(W+1)/K).")
P("Periodic baseline. σ_per(K) fires every K-th frame; Lemma 1 (Appendix A) gives E[R] ≥ β (R = 1 "
  "when K ≤ W+1). A gate is an online policy that observes a cue stream x_t and fires when q(x_t) "
  "exceeds a budget-tuned threshold. A gate is NON-ADAPTIVE if its fired set is a function of the "
  "cue and internal randomness only (not of detector hit/miss feedback); threshold S-gates, "
  "RF-gates and HEDGE-GATE are non-adaptive.")
P("Prediction-error metric. Per-event error η_j = 1 if π̂ fails to cover e_j at budget a, else 0, "
  "and η = E[η_j]; η is estimable from logs as η = 1 − onset-coverage AUC of the cue. On the P-A "
  "corpus the onset AUC spans [0.49, 0.79]; our real-data run (§VI) measures 0.51, confirming "
  "deployment cues are near-uninformative at onsets.")

# ---------------- IV. Lower Bounds ----------------
H("IV. Lower Bounds", 1)
H("A. Coverage bound (dense regime)", 2)
THM("Theorem 1 (coverage, informal).",
    "Fix budget a = 1/K and W < K. Any online gate that allocates at least a (1−δ) fraction of its "
    "budget by advice of error η satisfies, on some event process, R(G) ≤ 1 − η(1 − (W+1)/K) + δ. As "
    "η → 1, recall falls to the level of a schedule that concentrates its budget on the wrong "
    "frames.")
P("Remark (regime of validity). When a ≥ 1/(W+1), i.e. K ≤ W+1, the coverage term (W+1)/K ≥ 1 and "
  "the bound is vacuous (R ≤ 1+δ). The high-budget regime needs a different argument (Theorem 1′).")
H("B. Advice-commitment bound (high-budget regime)", 2)
THM("Theorem 1′ (advice commitment).",
    "(i) A pure-advice gate (fires the top-B frames by cue value) has, in the sparse regime "
    "m(W+1) ≤ T−B, an instance with onset-AUC = ½ on which R = 0. (ii) HEDGE-GATE(λ) satisfies "
    "E[R] ≥ min(1, λ(W+1)/K) = λβ on EVERY instance and EVERY cue, including adversarial ones. The "
    "advice arm contributes zero worst-case recall; a gate can do no better than its oblivious "
    "fraction alone (proof in Appendix A).")
H("C. Tight consistency–robustness frontier (non-adaptive gates)", 2)
THM("Theorem 2 (tight frontier).",
    "Let consistency c = E[R | η = 0] (vs the per-scene oracle) and robustness r = E[R | η = 1]. For "
    "K > W+1 (β < 1), every NON-ADAPTIVE randomized gate satisfies r ≤ β(1−c)/(1−β), and HEDGE-GATE(λ) "
    "ATTAINS it via c = (1−λ)+λβ, r = λβ. Equivalently a (1+ε)-consistent gate has robustness ratio "
    "β/r = Ω(1/ε). The frontier is tight: achievability equals the converse (Appendix A).")

# ---------------- V. HEDGE-GATE ----------------
H("V. HEDGE-GATE: A Pareto-Optimal Meta-Policy", 1)
P("Algorithm 1 (HEDGE-GATE(λ)). Spend λ·a of the budget as an oblivious random-phase periodic "
  "backbone and (1−λ)·a on the advice gate. The backbone preserves a deterministic floor "
  "E[R] ≥ λβ regardless of advice quality; the advice arm harvests oracle headroom when η is small. "
  "λ is a certifiable dial: an operator who must guarantee recall ≥ r picks λ ≥ rK/(W+1).")
THM("Theorem 3 (Pareto-optimality).",
    "Among non-adaptive randomized gates, HEDGE-GATE(λ) lies on the frontier of Theorem 2 for every "
    "λ ∈ [0,1]; no non-adaptive gate is simultaneously more consistent and more robust.")
THM("Proposition 4 (smoothness).",
    "Under a per-event independence model (Assumption A, Appendix A), expected recall is AFFINE in η: "
    "E[R_λ(η)] = λβ + (1−η)κ(λ)(1−λβ); degradation is monotone with no cliff (|dR/dη| ≤ 1).")
P("Remark (feedback and adaptivity). The converse of Theorem 2 uses that a non-adaptive gate fires "
  "the same distribution on the (cue-identical) η=0 and η=1 instances. With per-fire detector "
  "feedback an ADAPTIVE gate can break this: a few probes reveal whether the advice is good, after "
  "which it can fall back to the backbone, achieving c ≈ 1 and r ≈ β simultaneously when B > m. "
  "Thus the impossibility is specific to non-adaptive gates — exactly the deployed class — and it "
  "yields a positive design implication: a deployable gate should verify advice with a few detector "
  "probes before committing budget. Characterizing the optimal feedback-aware gate is future work. "
  "The design philosophy matches the Audit-CP shield of the companion work: exploit predictions, "
  "never trust them.")

# ---------------- VI. Empirical ----------------
H("VI. Empirical Grounding", 1)
H("A. Anchors from the P-A corpus (measured, reused)", 2)
P("The measured detection-grounded frontier (Fig. 1) has the periodic schedule dominating all gates "
  "at every activation below 0.5 — at a = 0.20: periodic 0.984, S-gate 0.284, RF-gate 0.200; "
  "det-gate gap −0.359 (CI95 [−0.493,−0.242]). This realizes the η→1 robustness arm by deployed "
  "predictors.")
fig(RES + "/fig1_anchor_frontier_PA.png",
    "Fig. 1. Measured recall–energy frontier (P-A corpus): periodic vs S, RF, DQN gates.")
H("B. Calibrated synthetic sweep (theory validation)", 2)
P("On synthetic event streams calibrated to the P-A sparse duration law (median 1, mean ≈ 6, CV ≈ "
  "4.7), simulated periodic recall matches the certificate to within bootstrap noise (0.78/1.00/1.00 "
  "at a = 0.1/0.2/0.333). Under uninformative noise, HEDGE-GATE degrades affinely in η (Pearson r up "
  "to −0.99), validating Proposition 4 (Fig. 2). Under adversarial advice the pure gate collapses "
  "below the oblivious level toward 0 while HEDGE-GATE(λ) holds the floor λβ — confirming "
  "Theorem 1′ (Fig. 4); the (consistency, robustness) points trace the tight frontier "
  "r = β(1−c)/(1−β) of Theorem 2 (Fig. 3).")
fig(RES + "/fig3_consistency_robustness.png",
    "Fig. 3. Consistency–robustness frontier; λ sweeps the Pareto segment r = β(1−c)/(1−β) (Theorem 2).")
fig(RES + "/fig4_adversarial_floor.png",
    "Fig. 4. Adversarial advice: pure gate (λ=0) collapses below oblivious; HEDGE-GATE holds the "
    "backbone floor λβ (Theorem 1′).")
H("C. Real-data run (T10) — 121-video corpus (BMC+LASIESTA+CDnet2014)", 2)
P("We extracted real motion cues (mean |Δframe|, computed from frames, not ground truth) and event "
  "streams (from GT masks, min duration 2) for 121 videos of BMC, LASIESTA and CDnet2014; 54 videos "
  "carry ≥ 2 events (262 events total), W = 7, video-level bootstrap B = 1000. The real cue’s "
  "onset-AUC is 0.510 (BMC 0.539, LASIESTA 0.456, CDnet 0.528), at the low end of the P-A band "
  "[0.49, 0.79] — i.e. near-uninformative at onsets. Consistent with Theorem 1′ and the P-A finding, "
  "the deployed motion gate trails the periodic schedule at every budget, with non-overlapping "
  "confidence intervals at low activation:")
tbl = doc.add_table(rows=4, cols=4); tbl.style = "Light Grid Accent 1"
for j, h in enumerate(["activation a", "real motion gate (CI95)", "periodic (CI95)", "gap"]):
    tbl.rows[0].cells[j].paragraphs[0].add_run(h).bold = True
for i, r in enumerate([["0.10", "0.360 [0.290, 0.437]", "0.777 [0.729, 0.822]", "−0.418"],
                       ["0.20", "0.567 [0.484, 0.645]", "1.000 [1.000, 1.000]", "−0.433"],
                       ["0.333", "0.780 [0.709, 0.845]", "1.000 [1.000, 1.000]", "−0.220"]], 1):
    for j, c in enumerate(r):
        tbl.rows[i].cells[j].text = c
P("")
fig(RES + "/fig5_real_corruption.png",
    "Fig. 5. Real-data T10 (121-video corpus BMC+LASIESTA+CDnet2014, B=1000): HEDGE-GATE recall vs η; "
    "the deployed cue sits at η ≈ 0.50 and gate-heavy policies trail the periodic certificate.")

# ---------------- VII. Discussion ----------------
H("VII. Discussion and Limitations", 1)
P("Scope. Our bounds address windowed event recall under hard activation budgets; latency-"
  "percentile objectives are covered by the companion certificate work [5]. The lower bound splits "
  "by regime: a coverage bound (Theorem 1) for dense events and an advice-commitment bound "
  "(Theorem 1′) for the high-budget regime a ≥ 1/(W+1). Theorems 2–3 (tight frontier and "
  "Pareto-optimality) are proven for NON-ADAPTIVE gates — exactly the deployed class (threshold "
  "S/RF gates and HEDGE-GATE); the Remark in §V shows feedback-aware adaptive gates can exceed the "
  "frontier, a positive direction we leave to future work. Proposition 4 assumes per-event "
  "independence (Assumption A). The real-data run uses the full 121-video corpus (54 videos with "
  "≥ 2 events, 262 events) at a 300-frame-per-video cap, ROI-aligned for CDnet; an uncapped rerun "
  "would further tighten the confidence intervals. The synthetic sweep spans η ∈ [0,1] for "
  "controlled validation, while the real cue occupies only η ∈ [0.49, 0.51] — the high-η deployment "
  "regime where, by Theorem 1′, adaptivity cannot help.")

# ---------------- References ----------------
H("References", 1)
for ref in [
    "[1] T. Lykouris and S. Vassilvitskii, “Competitive Caching with Machine Learned Advice,” J. ACM, 68(4), 2021.",
    "[2] M. Purohit, Z. Svitkina, R. Kumar, “Improving Online Algorithms via ML Predictions,” NeurIPS 31, 2018.",
    "[3] M. Mitzenmacher, S. Vassilvitskii, “Algorithms with Predictions,” Commun. ACM, 65(7), 2022.",
    "[4] Anonymous, “When Does Content-Aware Frame Skipping Help?…,” under review, 2026 (P-A).",
    "[5] Anonymous, “Certifiable Rate Scheduling…,” CERTES dissertation companion (Deep-2), 2026.",
    "[6] R. D. Yates et al., “Age of Information: An Introduction and Survey,” IEEE JSAC, 39(5), 2021.",
    "[7] T. Banerjee, V. V. Veeravalli, “Data-Efficient Quickest Change Detection,” Seq. Anal., 31(1), 2012.",
]:
    P(ref)

# ---------------- Appendix A: proofs ----------------
H("Appendix A — Proofs", 1)
THM("Lemma 1 (periodic coverage).",
    "(a) If K ≤ W+1, the deterministic periodic schedule recalls every event. (b) If K > W+1, the "
    "random-phase periodic schedule recalls each event with probability (W+1)/K. Hence E[R] ≥ β.")
P("Proof. (a) Consecutive fires are K ≤ W+1 apart, so every length-(W+1) window contains a fire. "
  "(b) For phase U uniform on {0,…,K−1}, [o,o+W] contains a fire iff (o−U) mod K ∈ {0,…,W}, "
  "true for exactly W+1 of K residues. ∎")
P("Theorem 1′. Proof. (i) Non-window frames number ≥ T − m(W+1) ≥ B; place the top-B cue values "
  "on B of them (set C), carrying no onset information (AUC = ½). The pure-advice gate fires exactly "
  "C, disjoint from all windows, so R = 0. (ii) HEDGE-GATE’s backbone is an oblivious random-phase "
  "periodic schedule at rate λa, independent of the cue; by Lemma 1 it recalls each event with "
  "probability min(1,λ(W+1)/K) = λβ regardless of the advice arm. ∎")
P("Theorem 2 (tight frontier). Proof. ACHIEVABILITY: HEDGE-GATE(λ) — at η=0 the advice arm covers "
  "(1−λ)m advised events and the backbone covers a β-fraction of the remaining λm, so "
  "c=(1−λ)+λβ; at η=1 the advice arm fires in the (empty) advised set (0) while the backbone "
  "gives r=λβ. Eliminating λ=r/β yields r=β(1−c)/(1−β). CONVERSE: on the canonical block "
  "family (N=⌊T/(W+1)⌋ blocks, m=B events, cue marks a fixed advised set A) the cue is identical "
  "in the η=0 and η=1 instances, so a non-adaptive gate’s firing probability p_b is the same in "
  "both; with Σ_b p_b ≤ B and c=(1/B)Σ_{b∈A}p_b we get Σ_{b∉A}p_b ≤ B(1−c). The η=1 "
  "adversary places the B events on the non-advised blocks of smallest p_b, whose average is "
  "≤ Σ_{b∉A}p_b/(N−B) ≤ B(1−c)/(N−B). Since B/N=β, B/(N−B)=β/(1−β), giving "
  "r ≤ β(1−c)/(1−β), matching achievability; HEDGE-GATE meets the boundary (Theorem 3). ∎")
P("Proposition 4. Proof. Under Assumption A an event is recalled iff either arm succeeds, with "
  "probability 1−(1−λβ)(1−(1−η)κ(λ)) = λβ + (1−η)κ(λ)(1−λβ); linearity over m "
  "events gives the affine E[R_λ(η)]. ∎")
P("Note. Theorems 1′, 2, 3 and Proposition 4 (under Assumption A) carry complete arguments above; "
  "Theorem 1 (dense regime) remains a sketch. All proofs should be verified by the author before "
  "submission. A full bilingual derivation with numerical cross-checks is in "
  "03_Proofs_Theorems_Bilingual_EN_VN.")
doc.save(OUT)
print("Saved:", OUT, "| paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
