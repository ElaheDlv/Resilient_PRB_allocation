# Exact Phase 2F changed-file inventory

All listed files are **new** relative to `2c31c96`. No previously tracked file
was changed or removed. The original reward, DQN, state/action interface, schedulers,
physical models and prior experiment artifacts remain byte-for-byte unchanged.
The three preexisting untracked root research notes remain uncommitted.

Total: **65 new files** (4 source/test files and 61 audit artifacts).

| File | Change/purpose |
| --- | --- |
| `backend/network_layer/reward_audit.py` | New read-only saved-total identification and honest unlogged-component bounds; original reward untouched. |
| `backend/tools/phase2f_audit.py` | New frozen-replay harness, pure component verification, service/ledger instrumentation and input checks. |
| `backend/tools/phase2f_report.py` | New saved/replay analysis, terminal checkpoint aggregation and seven SVG/PDF exports. |
| `backend/tests/test_phase2f.py` | Six audit unit/integration tests, including real frozen-instrumented/uninstrumented replay equivalence. |
| `backend/evaluations/phase2f/ABLATION_PLAN.md` | Unexecuted matched three-arm hypothesis-driven reward study. |
| `backend/evaluations/phase2f/ANALYSIS_PLAN.md` | Prerun diagnostic criteria and explicit later-interval label clarification. |
| `backend/evaluations/phase2f/MATHEMATICAL_AUDIT.md` | Exact original reward, scales, ledger/timing contracts, identifiable totals and limits. |
| `backend/evaluations/phase2f/README.md` | Data dictionary, guarded reproduction commands, scope and excluded tests. |
| `backend/evaluations/phase2f/REPORT.md` | Measured findings, concrete trajectories, objective agreement, terminal analysis and decision. |
| `backend/evaluations/phase2f/actual_terminal_intervals.csv` | Actual final transitions containing administrative charges and component shares. |
| `backend/evaluations/phase2f/analysis.json` | Machine-readable component, temporal and concrete-example summaries. |
| `backend/evaluations/phase2f/candidate_ordering.csv` | 24 candidate means and original reward/aggregate-first ranks. |
| `backend/evaluations/phase2f/environment.json` | Software versions, CPU/memory/GPU safeguards and global sampled peaks. |
| `backend/evaluations/phase2f/failure_credit_timing.csv` | Measured failure lags, later-decision-interval fractions and mode differences. |
| `backend/evaluations/phase2f/failure_timing_histogram.csv.gz` | First-known failure counts by budget, lag, mode and decision-issue gap. |
| `backend/evaluations/phase2f/figures/01_saved_episode_accounting.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/01_saved_episode_accounting.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/02_interval_components.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/02_interval_components.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/03_service_recovery_and_history.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/03_service_recovery_and_history.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/04_switch_magnitudes.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/04_switch_magnitudes.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/05_reserve_and_observed_service.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/05_reserve_and_observed_service.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/06_reward_objective_order.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/06_reward_objective_order.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/07_terminal_accounting.pdf` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/07_terminal_accounting.svg` | Reward-component/diagnostic figure; scope and source tables in figures/CAPTIONS.md. |
| `backend/evaluations/phase2f/figures/CAPTIONS.md` | Exact scope and source-table descriptions for all seven figures. |
| `backend/evaluations/phase2f/final_tests.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/final_tests_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/initial_tests.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/initial_tests_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/interval_component_distributions.csv` | Six-component interval distributions, stratified by declared replay role. |
| `backend/evaluations/phase2f/observed_prefix_accounting.csv.gz` | Read-only hypothetical closure count at observed prefixes; no alternate rollouts. |
| `backend/evaluations/phase2f/ordering_comparisons.csv` | 276 pairwise mean objective and identified cost deltas, A minus B. |
| `backend/evaluations/phase2f/protocol.json` | Frozen18-replay selection, original reward parameters and input SHA-256 pins. |
| `backend/evaluations/phase2f/provenance.json` | New artifact hashes and evidence of unchanged original tracked code/data. |
| `backend/evaluations/phase2f/regression.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/regression_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/replay.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/replay_checks.json` | 18 exact replay and reward reconciliation checks; zero updates. |
| `backend/evaluations/phase2f/replay_component_totals.csv` | Six exact original cost totals and shares for18 replay cases. |
| `backend/evaluations/phase2f/replay_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/replay_runtime.json` | Measured application elapsed time. |
| `backend/evaluations/phase2f/replay_summaries.jsonl.gz` | Original physical/control summaries plus measured component totals. |
| `backend/evaluations/phase2f/report.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/report_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/reward_intervals.csv.gz` | 5,382 interval component sums, service and historical cohort diagnostics. |
| `backend/evaluations/phase2f/reward_ticks.csv.gz` | 53,820 paid-tick original components, controls and observed service. |
| `backend/evaluations/phase2f/saved.log` | Sequential command/test output and resource-guard completion report. |
| `backend/evaluations/phase2f/saved_checks.json` | Saved data counts, ordering checks and missing-training-history flags. |
| `backend/evaluations/phase2f/saved_component_distributions.csv` | Identifiable component distributions by split and workload regime. |
| `backend/evaluations/phase2f/saved_episode_components.csv` | 504 saved episode decompositions with explicit missing splits and bounds. |
| `backend/evaluations/phase2f/saved_resources.json` | Archived affinity, command, elapsed time, sampled CPU/RSS peaks and watchdog status. |
| `backend/evaluations/phase2f/saved_runtime.json` | Measured application elapsed time. |
| `backend/evaluations/phase2f/service_improvement_with_history_examples.csv` | Adjacent-interval on-time-completion improvements with positive history cost. |
| `backend/evaluations/phase2f/strong_history_carry_examples.csv` | Positive historical cost with no new failures and timely completions >= arrivals. |
| `backend/evaluations/phase2f/successful_service_historical_runs.csv` | Longest consecutive strong-current-service/history-positive run per replay. |
| `backend/evaluations/phase2f/switch_cost_associations.csv` | Descriptive switched/stayed interval costs; no causal effect estimates. |
| `backend/evaluations/phase2f/temporal_and_resource_diagnostics.csv` | Validity, saturation runs and headroom/unused-activation accounting. |
| `backend/evaluations/phase2f/terminal_checkpoint_summary.csv` | Terminal means/maxima/shares by regime, split and checkpoint/training episode. |
| `backend/evaluations/phase2f/ue_intervals.csv.gz` | 43,056 per-UE interval service, deadline-risk and queue snapshots. |
| `backend/evaluations/phase2f/CHANGED_FILES.md` | This exact file-by-file inventory. |
