# Detection-grounded anchor evaluation

Analysis scripts (`analysis/`) and pre-computed results (`results/*.json`) for the
equal-energy, cross-dataset evaluation whose numbers appear as the "measured anchors"
in the paper (Figure 1). `theory/awp_experiments.py::load_pa_anchors()` reads
`results/a3_learned_result.json` (periodic / S / RF frontier grid, DQN points) to build
`results/anchors_pa.csv`.

- `a3_learned_result.json` — periodic vs S vs RF frontier over activation, DQN points,
  iso-90 activations, matched-activation gaps (metric: detection-grounded event-recall, W=7).
- `n1_crossdata.json` — leave-one-dataset-out protocol dependence (ranking flip).
- `n3_rareevent.json`, `perscene_k.json` — rare-event and per-scene rate analyses.

The upstream gate training/inference modules imported by `analysis/learned_gating/*`
(`rl/`, `asmag_core/`) are provided under `../gates/`.
