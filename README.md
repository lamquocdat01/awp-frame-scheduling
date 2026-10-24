# Learning-augmented scheduling under hard activation budgets — code, derived data, reproduction

Code and derived data for *"Learning-augmented scheduling under hard activation budgets: tight
consistency--robustness bounds, the limits of feedback, and optimal budget allocation"* (Lam Quoc Dat,
School of Business and Technology, FPT University), **version 3, submitted to the European Journal of
Operational Research**. Release tag of the submitted version: **`v3.0.1-submission`**.

The paper proves the exact consistency--robustness frontier of online activation under a hard budget with
untrusted predictions (non-adaptive policies), the limits of feedback for adaptive policies (ceiling, price of
adaptivity, sprinkle-and-verify), and the optimal allocation of activation budgets across a tier of nodes, and
instantiates the model with a measured energy law (Jetson Orin Nano) and a 121-video real-data study.

## Reproduce every number of the v3 paper (one command)
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-v3.txt   # .venv/bin on Linux/macOS
.venv/Scripts/python theory/v3_reproduce.py          # numbers, sweeps, allocation, TPDS artefacts (~15 min)
.venv/Scripts/python theory/v3_reproduce.py --full   # + adaptive-gate Monte Carlo (~40 min)
```
Exit code 0 iff every regenerated output equals the committed one (details below). No video is needed:
`results/awp_cache/*.npz` holds the derived per-video cue and event mask; the raw corpora (BMC, LASIESTA,
CDnet2014) are **not** redistributed here and are available from their maintainers.

**Earlier version.** The TPDS-era snapshot lives at `github.com/lamquocdat01/AwP-frame-scheduling-public`
(commit 66e48c2). It is superseded: its Theorem 3 statement is corrected in v3 and three of its result files are
truncated (see the erratum below). Its history is also contained in this repository.

**License.** MIT — see `LICENSE`.

## Version 3 — what is new and what the reproduction checks
Paper v3 (target: European Journal of Operational Research) adds the limits of feedback for adaptive gates
(Theorems 4-6, Corollary 7, Conjecture 8), optimal tier budget allocation (Theorem 7), and corrects the
HEDGE-GATE clause of Theorem 3 (the block-aware split of Algorithm 1 is Pareto-optimal in every regime; the
previously reported 0.19 gap belongs to a naive overlapping split). Notes with full proofs: `theory/v3_notes/`.
v3.0.1 (P3C): Algorithm 1 states the two uniform draws the proof of Theorem 3 needs (uniformly random backbone
set on the canonical family; uniform choice among the free advised blocks), and the real-data trust-dial sweep
(fig5, supplement S2.5) is re-run with that split (`theory/v3_sweep_blockaware.py`; the lambda in {0,1} rows are
identical to the TPDS sweep, checked).
v3.0.1-submission (P4): EJOR format (30-page limit; complete proofs in supplement S1), APA references, and the
synthetic sweep (supplement S2.4, fig24) re-run with the block-aware split (`theory/v3_sweep_blockaware.py --synthetic`).

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-v3.txt   # (.venv/bin on Linux/macOS)
.venv/Scripts/python theory/v3_reproduce.py          # numbers + allocation + TPDS artefacts (~10 min)
.venv/Scripts/python theory/v3_reproduce.py --full   # + adaptive-gate Monte Carlo (~40 min)
```
`v3_reproduce.py` first snapshots the committed outputs, then runs every generator to completion (phase 1), and
only then compares (phase 2): `numbers.json`, `a8a/a8c_summary.json` key by key, every regenerated CSV cell by
cell, `numbers.tex` byte for byte, and the TPDS real-data/synthetic artefacts in `results/` regenerated from source
by `theory/v3_regen_tpds_results.py` (sha256). Exit code 0 iff 0 differences. All v3 outputs are written
atomically (`theory/v3_io.py`: `*.tmp` then `os.replace`).

*Erratum (v3).* The TPDS snapshot of this repository (66e48c2) contained truncated copies of
`results/metrics_summary.json`, `results/metrics_summary_REAL.json` and `results/corruption_sweep_REAL.csv`
(the TPDS scripts wrote them non-atomically in the Windows code page). The complete files, restored in v3, are
byte-identical to the ones regenerated from source (`theory/v3_regen_tpds_results.py`); no number changes.
| v3 file | role |
|---|---|
| `theory/v3_numbers.py` | every number of the v3 paper from `results/awp_cache/*.npz` (per-video cue + event mask, derived; no video) and `energy/fit.json`; re-verifies the TPDS Table III and `results/percategory.csv` row by row |
| `theory/v3_a8a_mc.py` | adaptive gates: ceiling check, sprinkle-and-verify against the adversary family (seeds 42-46) |
| `theory/v3_a8c_alloc.py` | tier allocation: uniform vs exact DP vs bang-bang vs envelope bound (n = 8/32/128) |
| `theory/v3_check_thm3_hedge.py` | audit of the Theorem 3 correction (Monte Carlo + exact enumeration of the two uniform draws) |
| `theory/v3_sweep_blockaware.py` | real-data trust-dial sweep with the block-aware split of Algorithm 1 (fig5, S2.5); checks the lambda in {0,1} invariants |
| `theory/v3_fig5.py` | fig5: block-aware (solid) vs the TPDS frame-level split (dashed) |
| `theory/v3_path_check.py` | every file path printed in the paper/supplements exists in this repository |
| `theory/v3_regen_tpds_results.py` | regenerates the TPDS result files from source and compares them (sha256 / key by key) |
| `theory/v3_io.py` | atomic writes used by every v3 script |
| `theory/v3_figures.py`, `v3_fig24.py` | 300-dpi redraws at printed size |
| `results/v3/`, `Revision_v3/` | outputs (CSV/JSON, numbers.json/tex, figures, verified refs.bib) |

The manuscript source is not part of this repository. `gates/` and `evaluation/` are kept for the history of the
TPDS version; v3 does not use them.

## Earlier version (TPDS, 2026-08) — original layout and scripts

### Repository layout
| Folder | Contents |
|---|---|
| `theory/` | The paper's own code: theory + synthetic sweep, real-data pipeline, the general-frontier check, and figure generators. Reproduces every result/figure. |
| `theory/paper_build/` | Author scripts that build the manuscript/report artifacts (env-specific paths; not needed to reproduce results). |
| `energy/` | **Measured energy anchor** (Jetson Orin Nano 8 GB, 15 W mode, YOLO26s@640, 30 fps): the 18-run activation sweep (`raw/`), the thermal-control run (`ctl/`, `control_thermal.json`), the orchestration and analysis scripts, and `ENERGY_REPORT.md`. Reproduces the fit `W(a) = 4.9571 + 6.9089·a` (R² = 0.99899) and the thermal-confound bound. |
| `gates/` | The **S / RF / DQN** gate implementations (`asmag-rl-vs-deterministic`): score-gate, random-forest CALIB gate, and DQN reinforcement-learning gate, with trained models. Produces the measured anchors. See `gates/README.md`. |
| `evaluation/` | Analysis scripts + pre-computed result JSONs for the detection-grounded anchors consumed by `theory/awp_experiments.py`. See `evaluation/README.md`. |
| `results/` | Output CSVs and metrics JSON (synthetic sweep, real-data run, per-category, general frontier). |
| `figures/` | Figures of the submitted paper: `fig1`, `fig3`, `fig5`, `fig24` (merged noise/adversarial panel) and the schematics `sch2`, `sch5`. `fig2`, `fig4`, `fig6` are the pre-merge originals, still produced by the scripts in `theory/`; `fig6` ships as supplementary material with the paper. |
| `paper/` | Manuscript source (`paper.tex`, `IEEEtran.cls`) and typeset `paper.pdf` — the TPDS submission version (12 pp, single-column redrawn figures), identical to the submitted package. |

### Quick start (TPDS scripts)
```bash
pip install -r requirements.txt

# 1) Theory only (no data): verify the general tight frontier + SAT-GATE
python theory/validate_frontier.py            # -> results/frontier_general.csv, figures fig3

# 2) Synthetic sweep + theory bounds (no data)
python theory/awp_experiments.py              # -> results/*.csv, metrics_summary.json, fig2/fig4

# 3) Real-data pipeline (needs the video datasets; see below)
export ASMAG_DATASETS=/path/to/datasets       # BMC + LASIESTA (+ ../dataset cdnet2014/...)
python theory/awp_real_data.py                # uncapped extraction -> corruption_sweep_REAL.csv, fig5
python theory/awp_percategory.py              # -> percategory.csv, fig6
python theory/awp_sat_real.py                 # SAT-GATE on the real corpus

# 4) Regenerate the English figures
python theory/replot_english.py               # fig2, fig4, fig5
python theory/replot_fig1.py                  # fig1 (measured anchors)

# 5) Energy anchor: re-derive the W(a) fit from the raw 18-run sweep
python energy/analyze_awp_energy.py           # -> energy/E_of_a.csv, energy/fit.json
```

### What are S, RF, DQN? (the measured-anchor gates)
- **S — score-gate:** fires frames whose cheap motion-density signal exceeds a swept threshold (deterministic). Signal = `asmag_core/features.py` (motion-density mean).
- **RF — random-forest CALIB gate:** a `RandomForestClassifier` predicting per-frame detection-availability, trained by `gates/scripts/train_calib.py` (de-oracled: no ground-truth leakage). Model in `gates/models/`.
- **DQN — reinforcement-learning gate:** a deep-Q agent (`gates/rl/agents.py`), trained by `gates/scripts/train_rl_fair.py`; policies in `gates/outputs/rl/`.
All three are evaluated with detection-grounded windowed event-recall (W=7) at matched
activation against the oblivious periodic schedule; the resulting anchors live in
`evaluation/results/a3_learned_result.json` and feed Figure 1.

### Datasets (not bundled)
Public benchmarks — download separately and point `ASMAG_DATASETS` at them:
BMC (Background Models Challenge), LASIESTA, and CDnet2014. The video files are large and
are **not** included; the pipeline caches per-video cues/events on first run.

### Citation (TPDS version)
Please cite the paper *"Learning-Augmented Scheduling under Hard Activation Budgets:
Tight Consistency–Robustness Bounds with Untrusted Predictors"* (IEEE Transactions on
Parallel and Distributed Systems, under review). See `paper/paper.pdf`.
