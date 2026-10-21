# Learning-Augmented Scheduling under Hard Activation Budgets — code & reproducibility

Public code and data for the paper *"Learning-Augmented Scheduling under Hard Activation
Budgets: Tight Consistency–Robustness Bounds with Untrusted Predictors"*
(Lam Quoc Dat, FSB / FPT University).

The paper studies online scheduling under a hard activation budget with an untrusted
predictor, proves a **general tight consistency–robustness frontier** for non-adaptive
policies (attained by **SAT-GATE**, and by the simpler **HEDGE-GATE** in the scarce-budget
regime), extends the frontier to a device–edge tier where **skewed budget allocation
strictly beats uniform provisioning**, and instantiates the model on energy-budgeted
detector scheduling at the edge: a **measured energy anchor** on a Jetson Orin Nano plus a
121-video BMC + LASIESTA + CDnet2014 real-data study.

## Repository layout
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

## Quick start
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

## What are S, RF, DQN? (the measured-anchor gates)
- **S — score-gate:** fires frames whose cheap motion-density signal exceeds a swept threshold (deterministic). Signal = `asmag_core/features.py` (motion-density mean).
- **RF — random-forest CALIB gate:** a `RandomForestClassifier` predicting per-frame detection-availability, trained by `gates/scripts/train_calib.py` (de-oracled: no ground-truth leakage). Model in `gates/models/`.
- **DQN — reinforcement-learning gate:** a deep-Q agent (`gates/rl/agents.py`), trained by `gates/scripts/train_rl_fair.py`; policies in `gates/outputs/rl/`.
All three are evaluated with detection-grounded windowed event-recall (W=7) at matched
activation against the oblivious periodic schedule; the resulting anchors live in
`evaluation/results/a3_learned_result.json` and feed Figure 1.

## Datasets (not bundled)
Public benchmarks — download separately and point `ASMAG_DATASETS` at them:
BMC (Background Models Challenge), LASIESTA, and CDnet2014. The video files are large and
are **not** included; the pipeline caches per-video cues/events on first run.

## Citation
Please cite the paper *"Learning-Augmented Scheduling under Hard Activation Budgets:
Tight Consistency–Robustness Bounds with Untrusted Predictors"* (IEEE Transactions on
Parallel and Distributed Systems, under review). See `paper/paper.pdf`.

## License
MIT — see `LICENSE`.

## Version 3 (2026-10, staging) — what is new and how to reproduce
Paper v3 (target: European Journal of Operational Research) adds the limits of feedback for adaptive gates
(Theorems 4-6, Corollary 7, Conjecture 8), optimal tier budget allocation (Theorem 7), and corrects the
HEDGE-GATE clause of Theorem 3 (the block-aware split of Algorithm 1 is Pareto-optimal in every regime; the
previously reported 0.19 gap belongs to a naive overlapping split). Notes with full proofs: `theory/v3_notes/`.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-v3.txt   # (.venv/bin on Linux/macOS)
.venv/Scripts/python theory/v3_reproduce.py          # numbers + allocation, compares Revision_v3/numbers.json key by key
.venv/Scripts/python theory/v3_reproduce.py --full   # + adaptive-gate Monte Carlo (~40 min)
```
| v3 file | role |
|---|---|
| `theory/v3_numbers.py` | every number of the v3 paper from `results/awp_cache/*.npz` (per-video cue + event mask, derived; no video) and `energy/fit.json`; re-verifies the TPDS Table III and `results/percategory.csv` row by row |
| `theory/v3_a8a_mc.py` | adaptive gates: ceiling check, sprinkle-and-verify against the adversary family (seeds 42-46) |
| `theory/v3_a8c_alloc.py` | tier allocation: uniform vs exact DP vs bang-bang vs envelope bound (n = 8/32/128) |
| `theory/v3_check_thm3_hedge.py` | audit of the Theorem 3 correction |
| `theory/v3_figures.py`, `v3_fig24.py` | 300-dpi redraws at printed size |
| `results/v3/`, `Revision_v3/` | outputs (CSV/JSON, numbers.json/tex, figures, verified refs.bib) |

The manuscript source is not part of this repository. `gates/` and `evaluation/` are kept for the history of the
TPDS version; v3 does not use them.
