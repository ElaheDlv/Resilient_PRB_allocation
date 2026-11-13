# Exact Phase 1.1 changed files

26 files relative to `cb35202`. Existing root research notes are excluded.

| Path | Purpose |
|---|---|
| `backend/evaluations/phase11/CHANGED_FILES.md` | This exact changed-file inventory. |
| `backend/evaluations/phase11/README.md` | Mathematical definitions, measurement/censoring semantics, assumptions and exact commands. |
| `backend/evaluations/phase11/REPORT.md` | Measured failure diagnosis, controlled pilot tables, individual harms and journal recommendation. |
| `backend/evaluations/phase11/compatibility_summary.json` | All Phase 0 smoke checks and six saved trajectory matches. |
| `backend/evaluations/phase11/heldout_analysis.json` | Aggregate/class seed intervals, paired differences, resource checks and individual harmed UEs. |
| `backend/evaluations/phase11/heldout_summary.json` | All 72 once-evaluated paired held-out runs, per-UE measurements and recovery windows. |
| `backend/evaluations/phase11/per_class_attribution.csv` | Deadline-class and phase outcomes, channel/rate, service and violation summaries. |
| `backend/evaluations/phase11/per_ue_attribution.csv` | Each UE, policy, seed and phase: disjoint outcomes, channel/rate, PRBs/bytes, HOL and starvation. |
| `backend/evaluations/phase11/provenance.json` | Base revision, source/input hashes and environment versions. |
| `backend/evaluations/phase11/reproduction_j10_seed3456_deadline_slack_s5.jsonl.gz` | Complete gzip event/allocation stream for the matched moderate-jamming seed3456 case. |
| `backend/evaluations/phase11/reproduction_j10_seed3456_proportional_wc_s5.jsonl.gz` | Complete gzip event/allocation stream for the matched moderate-jamming seed3456 case. |
| `backend/evaluations/phase11/reproduction_summary.json` | All 18 original fixed-quota reproduced runs with per-UE/class attribution. |
| `backend/evaluations/phase11/selection.json` | Separate validation scores, frozen candidate threshold and predeclared held-out seeds. |
| `backend/evaluations/phase11/test_results.txt` | Final complete pytest results. |
| `backend/evaluations/phase11/trace_evidence.json` | Concrete matched old-policy batch/tick examples and held-out recovery/harm snapshots. |
| `backend/evaluations/phase11/validation_summary.json` | All 18 candidate validation runs, kept separate from held-out evaluation. |
| `backend/network_layer/cell.py` | Select the optional feasibility policy and invoke read-only pre-service allocation diagnostics outside scheduler timing. |
| `backend/network_layer/deadline_scheduler.py` | Add optimistic feasibility/EDF/remaining-work ranking using the existing integer, FIFO, cap and starvation framework; original slack path preserved. |
| `backend/network_layer/queue_measurements.py` | Optional event sink for independent complete batch attribution; original counters and queues unchanged. |
| `backend/network_layer/scheduler_diagnostics.py` | New per-UE/phase service, rate/channel, starvation, HOL, batch partition and gzip event/allocation diagnostics. |
| `backend/settings/phase0_config.py` | Accept deadline_feasible as an optional mode; preserve legacy defaults. |
| `backend/tests/test_phase11.py` | Nine new tests including three parametrized real-simulator pairing/noninterference/jammer-transition cases. |
| `backend/tools/phase0_evaluate.py` | Optional diagnostics and new scheduler CLI; suppress verbose diagnostics from console summary. |
| `backend/tools/phase11_analyze.py` | Seed-level paired confidence intervals, per-UE/class CSV exports and matched trace evidence. |
| `backend/tools/phase11_study.py` | Predeclared reproduction, separate validation selection and held-out paired fixed-quota experiments. |
| `backend/evaluations/phase11/audit_results.json` | Automated audit of all 108 episodes: independent outcome, phase cohort, byte/PRB sums, useful-demand and eligible-idle checks. |
