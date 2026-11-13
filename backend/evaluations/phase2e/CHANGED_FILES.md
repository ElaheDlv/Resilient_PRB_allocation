# Exact Phase2E changed-file list

Base `fbafd09`. All changes are new files. Original G1/P/F, simulator, DQN,
state/reward/action sources and all earlier experiment artifacts are unchanged.
Three unrelated root research notes were left untouched.

## Source additions

- `backend/network_layer/equivalent_g1.py` — Separate fixed-lambda G1 execution; computes the same credits/kernel, omits only provably unused telemetry and zero-credit work.
- `backend/tools/phase2e_study.py` — Prospective disjoint exogenous discovery, protected-source checks, profiling, complete-tick timing, exact replay comparisons and frozen independent P/F/G1 evaluation.
- `backend/tools/phase2e_report.py` — All paired regime/severity/UE outcomes, tail/runtime scopes, profile components, optimization comparisons and six SVG/PDF figures without new simulation.
- `backend/tests/test_phase2e.py` — Five nonlearning tests, including500 randomized direct-allocation comparisons, bounded seed separation and actual per-tick equivalence/timing.

## Research artifacts

Every delivered artifact path is listed below. No new neural checkpoints exist.

- `backend/evaluations/phase2e/CHANGED_FILES.md`
- `backend/evaluations/phase2e/PROTOCOL.md`
- `backend/evaluations/phase2e/README.md`
- `backend/evaluations/phase2e/REPORT.md`
- `backend/evaluations/phase2e/analysis.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61448192_j0.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61448192_j10.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61448192_j20.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61460480_j0.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61460480_j10.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61460480_j20.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61476864_j0.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61476864_j10.json`
- `backend/evaluations/phase2e/configs/independent_contested_seed61476864_j20.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61444096_j0.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61444096_j10.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61444096_j20.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61489152_j0.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61489152_j10.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61489152_j20.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61505536_j0.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61505536_j10.json`
- `backend/evaluations/phase2e/configs/independent_feasible_seed61505536_j20.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61452288_j0.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61452288_j10.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61452288_j20.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61456384_j0.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61456384_j10.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61456384_j20.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61464576_j0.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61464576_j10.json`
- `backend/evaluations/phase2e/configs/independent_overloaded_seed61464576_j20.json`
- `backend/evaluations/phase2e/consistency.csv`
- `backend/evaluations/phase2e/discover.log`
- `backend/evaluations/phase2e/discover_resources.json`
- `backend/evaluations/phase2e/discover_runtime.json`
- `backend/evaluations/phase2e/environment.json`
- `backend/evaluations/phase2e/equivalence.json`
- `backend/evaluations/phase2e/equivalence.log`
- `backend/evaluations/phase2e/equivalence_resources.json`
- `backend/evaluations/phase2e/equivalence_runs.jsonl.gz`
- `backend/evaluations/phase2e/equivalence_runtime.json`
- `backend/evaluations/phase2e/evaluate.log`
- `backend/evaluations/phase2e/evaluate_resources.json`
- `backend/evaluations/phase2e/evaluate_runtime.json`
- `backend/evaluations/phase2e/exogenous_probes.json`
- `backend/evaluations/phase2e/figures/01_profile_cost.pdf`
- `backend/evaluations/phase2e/figures/01_profile_cost.svg`
- `backend/evaluations/phase2e/figures/02_equivalent_optimization.pdf`
- `backend/evaluations/phase2e/figures/02_equivalent_optimization.svg`
- `backend/evaluations/phase2e/figures/03_independent_placements.pdf`
- `backend/evaluations/phase2e/figures/03_independent_placements.svg`
- `backend/evaluations/phase2e/figures/04_independent_tradeoffs.pdf`
- `backend/evaluations/phase2e/figures/04_independent_tradeoffs.svg`
- `backend/evaluations/phase2e/figures/05_individual_ue_harms.pdf`
- `backend/evaluations/phase2e/figures/05_individual_ue_harms.svg`
- `backend/evaluations/phase2e/figures/06_complete_tick_cost.pdf`
- `backend/evaluations/phase2e/figures/06_complete_tick_cost.svg`
- `backend/evaluations/phase2e/figures/CAPTIONS.md`
- `backend/evaluations/phase2e/figures/placement_indices.csv`
- `backend/evaluations/phase2e/final_tests.log`
- `backend/evaluations/phase2e/final_tests_resources.json`
- `backend/evaluations/phase2e/independent_manifest.json`
- `backend/evaluations/phase2e/independent_original_checks.jsonl.gz`
- `backend/evaluations/phase2e/independent_runs.jsonl.gz`
- `backend/evaluations/phase2e/initial_tests.log`
- `backend/evaluations/phase2e/initial_tests_resources.json`
- `backend/evaluations/phase2e/manifest_freeze.json`
- `backend/evaluations/phase2e/optimization_timings.csv`
- `backend/evaluations/phase2e/paired_differences.csv`
- `backend/evaluations/phase2e/paired_per_ue.csv`
- `backend/evaluations/phase2e/per_ue.csv`
- `backend/evaluations/phase2e/profile_components.csv`
- `backend/evaluations/phase2e/profile_optimized.csv`
- `backend/evaluations/phase2e/profile_optimized.log`
- `backend/evaluations/phase2e/profile_optimized_contested.prof`
- `backend/evaluations/phase2e/profile_optimized_feasible.prof`
- `backend/evaluations/phase2e/profile_optimized_overloaded.prof`
- `backend/evaluations/phase2e/profile_optimized_resources.json`
- `backend/evaluations/phase2e/profile_optimized_runtime.json`
- `backend/evaluations/phase2e/profile_original.csv`
- `backend/evaluations/phase2e/profile_original.log`
- `backend/evaluations/phase2e/profile_original_contested.prof`
- `backend/evaluations/phase2e/profile_original_feasible.prof`
- `backend/evaluations/phase2e/profile_original_overloaded.prof`
- `backend/evaluations/phase2e/profile_original_resources.json`
- `backend/evaluations/phase2e/profile_original_runtime.json`
- `backend/evaluations/phase2e/protocol.json`
- `backend/evaluations/phase2e/provenance.json`
- `backend/evaluations/phase2e/recovery_windows.csv`
- `backend/evaluations/phase2e/regime_means.csv`
- `backend/evaluations/phase2e/regression.log`
- `backend/evaluations/phase2e/regression_resources.json`
- `backend/evaluations/phase2e/report.log`
- `backend/evaluations/phase2e/report_resources.json`
- `backend/evaluations/phase2e/scenarios.csv`
- `backend/evaluations/phase2e/service_windows.csv.gz`
- `backend/evaluations/phase2e/summary.csv`
- `backend/evaluations/phase2e/tick_times.csv.gz`
- `backend/evaluations/phase2e/traces/urllc_seed61452288_overloaded.csv`
- `backend/evaluations/phase2e/traces/urllc_seed61456384_overloaded.csv`
- `backend/evaluations/phase2e/traces/urllc_seed61464576_overloaded.csv`
- `backend/evaluations/phase2e/validation_checks.json`
