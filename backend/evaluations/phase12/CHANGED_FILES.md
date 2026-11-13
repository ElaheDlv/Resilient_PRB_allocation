# Exact Phase 1.2 changed files

43 files relative to `6bce0d8`. Root research notes are excluded.

| Path | Purpose |
|---|---|
| `backend/evaluations/phase12/CHANGED_FILES.md` | This exact changed-file inventory. |
| `backend/evaluations/phase12/README.md` | Policy mathematics, operating regions, protocol amendments, timing/accounting definitions and exact commands. |
| `backend/evaluations/phase12/REPORT.md` | Measured comparisons, original and new UE harm attribution, uncertainty and decision. |
| `backend/evaluations/phase12/analysis.json` | All held-out groups, seed intervals, paired harm lists and operating-region certificates. |
| `backend/evaluations/phase12/audit.json` | Per-UE outcome agreement, exclusive partitions and useful-demand/idle-capacity checks over 240 episodes. |
| `backend/evaluations/phase12/boundary_plan.json` | Supplementary zero-result rho0.995 validation-screening plan. |
| `backend/evaluations/phase12/boundary_validation_calibration.json` | Exogenous radio/quota calibration and assignments for boundary_validation. |
| `backend/evaluations/phase12/boundary_validation_summary.json` | Complete per-run measurements, hashes and attribution for boundary_validation. |
| `backend/evaluations/phase12/compatibility.json` | Phase 0 smoke checks and all six saved hash matches. |
| `backend/evaluations/phase12/existing_candidate_harms.json` | Original harmed UEs, actual service deficits, resource consumption and finite-vs-long-run feasibility. |
| `backend/evaluations/phase12/heldout_calibration.json` | Exogenous radio/quota calibration and assignments for heldout. |
| `backend/evaluations/phase12/heldout_summary.json` | Complete per-run measurements, hashes and attribution for heldout. |
| `backend/evaluations/phase12/original_case_regression.json` | Two Phase 1.1 severe-seed4008 trajectory hash matches. |
| `backend/evaluations/phase12/per_class.csv` | Every held-out class and case: failures, unresolved traffic, served-byte/HOL metrics and counts. |
| `backend/evaluations/phase12/per_ue.csv` | Every held-out UE and phase: offered/completed/failed/unresolved counts, channels, service, PRBs, HOL and starvation. |
| `backend/evaluations/phase12/per_ue_recovery.json` | Individual first compliant holds; retained separately from simultaneous all-UE recovery. |
| `backend/evaluations/phase12/provenance.json` | Base revision, source/input hashes, environment and included/excluded cohort counts. |
| `backend/evaluations/phase12/recovery_heldout_calibration.json` | Exogenous radio/quota calibration and assignments for recovery_heldout. |
| `backend/evaluations/phase12/recovery_heldout_summary.json` | Complete per-run measurements, hashes and attribution for recovery_heldout. |
| `backend/evaluations/phase12/recovery_plan.json` | Recoverable ON-overload/OFF-feasible control and independent 10000-series cohort. |
| `backend/evaluations/phase12/recovery_trace_evidence.json` | Recoverable-overload measured queue/grant/service samples and explicitly reconstructed credit states. |
| `backend/evaluations/phase12/recovery_validation_calibration.json` | Exogenous radio/quota calibration and assignments for recovery_validation. |
| `backend/evaluations/phase12/recovery_validation_summary.json` | Complete per-run measurements, hashes and attribution for recovery_validation. |
| `backend/evaluations/phase12/seed_audit.json` | Disclosed aborted seed-selection launch, excluded seeds and corrected independent cohorts. |
| `backend/evaluations/phase12/selection.json` | Frozen validation-only alpha selection, scores, criterion and primary held-out seeds. |
| `backend/evaluations/phase12/test_results.txt` | Final full-suite pytest output. |
| `backend/evaluations/phase12/threshold_heldout_calibration.json` | Exogenous radio/quota calibration and assignments for threshold_heldout. |
| `backend/evaluations/phase12/threshold_heldout_summary.json` | Complete per-run measurements, hashes and attribution for threshold_heldout. |
| `backend/evaluations/phase12/threshold_plan.json` | Validation-refined finite-budget threshold and independent 9000-series cohort. |
| `backend/evaluations/phase12/threshold_validation_calibration.json` | Exogenous radio/quota calibration and assignments for threshold_validation. |
| `backend/evaluations/phase12/threshold_validation_summary.json` | Complete per-run measurements, hashes and attribution for threshold_validation. |
| `backend/evaluations/phase12/validation_calibration.json` | Exogenous radio/quota calibration and assignments for validation. |
| `backend/evaluations/phase12/validation_summary.json` | Complete per-run measurements, hashes and attribution for validation. |
| `backend/network_layer/cell.py` | Select optional fairness mode; prepare service credits, debit actual service, and clear per-UE credit state on deregistration. |
| `backend/network_layer/deadline_scheduler.py` | Add bounded credit-priority grant stage before existing feasibility rank; preserve integer/useful-demand caps and guard. |
| `backend/network_layer/fair_service_credit.py` | New causal admitted-load/actual-byte service-credit state with bounded memory. |
| `backend/network_layer/phase1_metrics.py` | Add allocation-time median and p99 while retaining original timing metrics. |
| `backend/settings/phase0_config.py` | Register deadline_fair and validate optional fairness fraction; legacy remains default. |
| `backend/tests/test_phase1.py` | Extend real episode-reset test to check credit state cleanup. |
| `backend/tests/test_phase12.py` | Credit/service, conservation, replay, jammer, finite-bound and cohort-separation regression tests. |
| `backend/tools/phase0_evaluate.py` | Expose optional fairness scheduler in evaluation CLI; fixed controller unchanged. |
| `backend/tools/phase12_analyze.py` | Finite-window feasibility bounds, paired intervals, UE/class exports, individual recovery and measured/reconstructed trace evidence. |
| `backend/tools/phase12_study.py` | Paired capacity-normalized workload matrix, validation-only fraction selection, independent supplementary cohorts and calibration. |
