# Exact Phase2D changed-file list

Base `eafcea7`; unrelated root research notes remain untouched.

## Source changes

- `backend/network_layer/cell.py` — Three-line optional experimental allocator hook; original P/F functions and default arguments preserved.
- `backend/network_layer/feasibility_gated_fairness.py` — Causal bounded service windows, optimistic attainable FIFO targets, gated credits, optional unchanged old-credit comparator.
- `backend/tools/phase2d_study.py` — Frozen validation-only checkpoint replays, nonlearning paired policies, read-only decisions/windows/shadows and strict replay/reserve checks.
- `backend/tools/phase2d_report.py` — Streaming measured analyses, per-UE/stratum tables and six editable SVG/PDF figures; no new simulations.
- `backend/tests/test_phase2d.py` — Ten nonlearning unit/integration tests: credit gating, partial/equality FIFO targets, integer/caps, old F identity, bounded state, encoder aliasing and saved-forward replay.

## Research artifacts

All files below are new; no C2/C3 data/checkpoint was modified.

- `backend/evaluations/phase2d/CHANGED_FILES.md`
- `backend/evaluations/phase2d/DESIGN.md`
- `backend/evaluations/phase2d/README.md`
- `backend/evaluations/phase2d/REPORT.md`
- `backend/evaluations/phase2d/a4_decision_summary.csv`
- `backend/evaluations/phase2d/a4_recorded_candidates.csv`
- `backend/evaluations/phase2d/analysis.json`
- `backend/evaluations/phase2d/decision_diagnostics.csv`
- `backend/evaluations/phase2d/decisions.jsonl.gz`
- `backend/evaluations/phase2d/deprivation_counts.csv`
- `backend/evaluations/phase2d/environment.json`
- `backend/evaluations/phase2d/figures/01_a4_recorded_behavior.pdf`
- `backend/evaluations/phase2d/figures/01_a4_recorded_behavior.svg`
- `backend/evaluations/phase2d/figures/02_switch_conditions.pdf`
- `backend/evaluations/phase2d/figures/02_switch_conditions.svg`
- `backend/evaluations/phase2d/figures/03_exact_harm_case.pdf`
- `backend/evaluations/phase2d/figures/03_exact_harm_case.svg`
- `backend/evaluations/phase2d/figures/04_risk_saturation.pdf`
- `backend/evaluations/phase2d/figures/04_risk_saturation.svg`
- `backend/evaluations/phase2d/figures/05_fixed_parameter_tradeoff.pdf`
- `backend/evaluations/phase2d/figures/05_fixed_parameter_tradeoff.svg`
- `backend/evaluations/phase2d/figures/06_scheduler_cost.pdf`
- `backend/evaluations/phase2d/figures/06_scheduler_cost.svg`
- `backend/evaluations/phase2d/figures/CAPTIONS.md`
- `backend/evaluations/phase2d/focused_resources.json`
- `backend/evaluations/phase2d/focused_tests.log`
- `backend/evaluations/phase2d/harm_ticks.csv.gz`
- `backend/evaluations/phase2d/initial_test_resources.json`
- `backend/evaluations/phase2d/initial_tests.log`
- `backend/evaluations/phase2d/interval_outcomes.csv.gz`
- `backend/evaluations/phase2d/per_ue.csv`
- `backend/evaluations/phase2d/protocol.json`
- `backend/evaluations/phase2d/prototype_paired.csv`
- `backend/evaluations/phase2d/prototype_scenarios.csv`
- `backend/evaluations/phase2d/prototype_summary.csv`
- `backend/evaluations/phase2d/provenance.json`
- `backend/evaluations/phase2d/regression.log`
- `backend/evaluations/phase2d/regression_resources.json`
- `backend/evaluations/phase2d/report.log`
- `backend/evaluations/phase2d/report_resources.json`
- `backend/evaluations/phase2d/runs.jsonl.gz`
- `backend/evaluations/phase2d/runtime.json`
- `backend/evaluations/phase2d/sampled_deprivation.csv`
- `backend/evaluations/phase2d/service_segments.csv`
- `backend/evaluations/phase2d/shadow_completions.csv`
- `backend/evaluations/phase2d/shadow_summary.csv`
- `backend/evaluations/phase2d/shadows.jsonl.gz`
- `backend/evaluations/phase2d/state_discrimination.csv`
- `backend/evaluations/phase2d/study.log`
- `backend/evaluations/phase2d/study_resources.json`
- `backend/evaluations/phase2d/switch_correlations.csv`
- `backend/evaluations/phase2d/switch_strata.csv`
- `backend/evaluations/phase2d/ue_windows.csv.gz`
- `backend/evaluations/phase2d/validation_checks.json`
