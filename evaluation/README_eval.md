# Content-Aware Frame Skipping: A Multi-Regime, Multi-Protocol Evaluation

Analysis code, result artifacts, and manuscript for the methodology paper
**"When Does Content-Aware Frame Skipping Help? A Multi-Regime, Multi-Protocol Evaluation for
Energy-Efficient Edge Object Detection."**

> **Status:** manuscript under review; **figures and §II (Related Work) pending**; author/affiliation and
> repo URL are placeholders in the `.tex`. This repository is currently **private** (see *Before going
> public* below).

## Thesis
Under honest evaluation, **content-aware and learned frame-skipping gates provide no deployable advantage**
over a simple periodic (fixed-rate, oblivious) schedule at equal energy. The apparent advantages reported
same-scene are evaluation artifacts or mechanical (K-quantization recovery). The **real lever is the
per-scene refresh *rate* (K), not content-awareness** — but that headroom is *oracle-only* and not
capturable by any cheap, causal, zero-shot estimator. Recommended design: a **certifiable fixed-rate
oblivious schedule** (worst-case latency certificate in the companion paper).

## What's here

| § | claim | script | result | finding |
|---|---|---|---|---|
| IV | Evaluation-protocol dependence | `analysis/protocol_dependence/n1_crossdata.py` | `results/n1_crossdata.json` | Within-dataset (same-scene / LOCO) endorses the gate; true cross-dataset **LODO reverses the ranking** — the oblivious schedule CI-beats the gate on every held-out dataset. |
| V | The real lever is per-scene *K* | `analysis/per_scene_k/perscene_k.py` | `results/perscene_k.json` | Oracle per-scene-K skip ρmax@90 **0.728** (per-video 0.832) ≫ gate **0.636** ≫ scene-agnostic global-K **0.591**; gate recovers only ≈⅓ of the oracle headroom. A causal zero-shot estimator captures **0.24 [−0.07,+0.45]** of the gain (CI includes 0; does not beat scene-agnostic); label-leakage control: honest 0%, cheat 45–85%. **Verdict: NO-GO.** |
| VI | Event-regime dependence | `analysis/rare_event/n3_rareevent.py` | `results/n3_rareevent.json` | Gate's equal-energy disadvantage shrinks as videos get sparser (Spearman −0.358) but never becomes a low-energy win; under detection-grounding VIRAT in-event fraction = **1.0** ("rare-event" is largely a misnomer). |
| VII | Learned policies too | `analysis/learned_gating/compute_vii.py`, `compute_vii_frontier.py` | `results/a3_learned_result.json` | An RL (DQN) gate and a deterministic gate both **CI-lose to periodic** at matched activation. Iso-90%-recall: periodic needs **0.125** activation vs gates' **0.81–0.89** (periodic ~7× more efficient). At a=0.20 gates cover **0.20–0.28** of events vs periodic's **0.984**. |

Convergent debunk across protocols (IV) **and** regimes (VI) **and** method classes (VII).

## Honest limitations
- **Low-activation craters:** the gates collapse where deployment lives — at a=0.20, recall 0.20–0.28 vs
  periodic 0.98 (detection-grounded, staleness W=7).
- **Scene-dependence:** the *magnitude* of periodic's advantage is amplified by CDnet's long events; the
  *direction* (periodic ≥ content-aware/learned at equal energy) is what's robust.
- **DQN λ-floor:** the RL gate's reachable activation bottomed at **0.476 within the trained penalty range
  (λ≤0.80)** — this is **not** a hard architectural floor; larger λ would lower activation only by dropping
  events (collapsing recall toward the gate frontier, not toward periodic).
- **Per-scene-K is real but uncapturable:** the oracle headroom exists, but a safe online estimator faces a
  **sample-complexity wall** (≈ ln δ / ln(1−ε) events to certify a miss floor ε; see the Deep-2 companion
  on certifiable rate scheduling). Not overclaimed: K-selection is the lever, but not cheaply realizable.

## Reproduce
Each script writes its canonical JSON to `results/`. Statistics use **video-level bootstrap, B=1000,
seed=42**. The committed `results/*.json` are the canonical numbers the paper cites.

```
pip install -r requirements.txt
python analysis/protocol_dependence/n1_crossdata.py     # -> results/n1_crossdata.json
python analysis/rare_event/n3_rareevent.py              # -> results/n3_rareevent.json
python analysis/per_scene_k/perscene_k.py               # needs the deep0 'src' package (see below)
python analysis/learned_gating/compute_vii.py           # needs the asmag-rl modules (see below)
python analysis/figures/make_figures.py                 # -> paper/figures/fig1-4.pdf (reads results/*.json)
```

**Figures (Fig. 1–4): generated** by `analysis/figures/make_figures.py` from `results/*.json` →
`paper/figures/{fig1_protocol_flip, fig2_perscene_k, fig3_regime_trend, fig4_duration_law}.pdf` (vector,
bootstrap-CI error bars). The paper has not been compiled here (no TeX toolchain on the build machine);
`pdflatex ×2` should be run where TeX is available.

## Data & paths (not committed — large)
Scripts contain **absolute paths to the author's machine** (flagged in each file header). To re-run, set
those constants and obtain the source data:
- **CDnet2014, LASIESTA, BMC, VIRAT** detection dumps (per-frame detector outputs / `frame_scores.parquet`)
  and CDnet NPZ feature dumps. CDnet2014: <http://changedetection.net>; LASIESTA, BMC, VIRAT from their
  respective sources.
- Detector outputs and NPZ/`Event_State` maps are produced by the **ASMAG-TRC pipeline** (companion
  detection/gating codebase). `analysis/learned_gating/compute_vii*.py` import that pipeline's modules
  (`rl/`, `asmag_core/`) — i.e. the de-oracled A3 fork of ASMAG-TRC — and load its NPZ + trained
  RF-CALIB / DQN-T4b models. `analysis/per_scene_k/perscene_k.py` imports a local `src/` package
  (`events`, `scores`) from the Deep-0 working tree. These dependencies are **not vendored here**; the
  committed `results/*.json` are the canonical outputs.

## Repository layout
```
paper/      methodology_paper.tex   figures/ (pending)
analysis/   protocol_dependence/  per_scene_k/  rare_event/  learned_gating/
results/    n1_crossdata.json  perscene_k.json  n3_rareevent.json  a3_learned_result.json
```

## Before going public
§II Related Work **citations** (currently unverified `\cite` placeholders) · author block/email confirm in
the `.tex` · LICENSE choice · `pdflatex` compile + visual verification (no TeX on the build machine) · a
clean reproduce-run. (Figures 1–4 and author/repo-URL meta are now in.) See `c4_complete_report.md`.

## License
**Placeholder — license not yet chosen.** Suggested: **MIT** for code (`analysis/`) and **CC-BY-4.0** for
the paper/text (`paper/`). See `LICENSE`.

## Companion
Deep-3 (certifiable worst-case detection latency for frame-skipping) and Deep-2 (certifiable rate
scheduling / sample-complexity of per-scene-K) are companion works.
