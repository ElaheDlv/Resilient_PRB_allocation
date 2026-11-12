# Exact Phase 1 changed-file inventory

30 files; prior root research notes are excluded.

| Path | Purpose |
|---|---|
| `backend/evaluations/phase1/CHANGED_FILES.md` | This exact file-by-file inventory. |
| `backend/evaluations/phase1/PILOT_REPORT.md` | Measured comparison tables, validation, interpretation and journal recommendation. |
| `backend/evaluations/phase1/README.md` | Mathematical rule, assumptions, configuration, measurement definitions and reproduction commands. |
| `backend/evaluations/phase1/compatibility_summary.json` | Phase 0 smoke rerun and comparison against all six saved trajectories. |
| `backend/evaluations/phase1/main_scenarios/scenario_j0.json` | Generated paired eight-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/main_scenarios/scenario_j10.json` | Generated paired eight-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/main_scenarios/scenario_j20.json` | Generated paired eight-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/pilot_runs.json` | All 42 per-run records including per-UE/class counters, recovery and environment metadata. |
| `backend/evaluations/phase1/pilot_summary.json` | Main three-seed fixed and frozen-DQN grouped results and pairing checks. |
| `backend/evaluations/phase1/pilot_table.csv` | Grouped pilot metrics for direct analysis. |
| `backend/evaluations/phase1/provenance.json` | Base commit, changed-source hashes and original trace hashes. |
| `backend/evaluations/phase1/stress_scenarios/scenario_j0.json` | Generated paired twelve-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/stress_scenarios/scenario_j10.json` | Generated paired twelve-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/stress_scenarios/scenario_j20.json` | Generated paired twelve-URLLC-UE scenario; original traces referenced unchanged. |
| `backend/evaluations/phase1/stress_summary.json` | Separately labeled twelve-URLLC-UE, single-seed results. |
| `backend/evaluations/phase1/test_results.txt` | Final full-suite pytest output. |
| `backend/network_layer/base_station.py` | Pass sample duration to the measurement-only HOL tracker. |
| `backend/network_layer/cell.py` | Optional URLLC slack allocator, starvation state/cleanup and allocation timing/resource diagnostics. |
| `backend/network_layer/deadline_scheduler.py` | New pure integer slack scheduler with virtual FIFO and starvation guard. |
| `backend/network_layer/phase1_metrics.py` | New per-UE/class, recovery, resource, timing and exogenous-pair measurements. |
| `backend/network_layer/queue_measurements.py` | Budget classes, offered-batch partitions and HOL violation runs; preserve existing byte/batch counters. |
| `backend/network_layer/simulation_engine.py` | Optional after-service observer before the next RIC action. |
| `backend/network_layer/ue.py` | Add fixed queue-budget class metadata. |
| `backend/network_layer/ue_budgets.py` | New balanced private-RNG budget assignment utility. |
| `backend/network_layer/xApps/xapp_resilient_dqn_allocator.py` | Apply deterministic fixed budget assignments during episode UE registration; DQN formulation unchanged. |
| `backend/network_layer/xApps/xapp_resilient_policy_validator.py` | Add fixed reserve-allocation comparison policy and reset initialization. |
| `backend/settings/phase0_config.py` | Allow optional deadline_slack mode and configurable starvation threshold. |
| `backend/tests/test_phase1.py` | New unit and actual-simulator integration coverage. |
| `backend/tools/phase0_evaluate.py` | Expose fixed allocation, per-UE cap, deadline mode and optional Phase 1 metrics. |
| `backend/tools/phase1_pilot.py` | New paired pilot driver with frozen-DQN and higher-load options. |
