# Phase 1 pilot and validation report

Date: November 12, 2026. Base commit: `4f6e0e6`. Branch: `phase1/deadline-aware-ue-scheduler`.

## Validation scope

`python -m pytest -q`: **27 passed** in 16.39 seconds in the recorded full-suite run. The original 15 Phase 0 tests remain present. New tests cover private/balanced budget assignment, per-episode budget and starvation reset, slack ordering and deterministic ties, integer/cap/useful-demand conservation, zero-rate and sub-byte service, continuous overload/starvation, partial service, deadline equality, offered-batch failure/censoring partitions, HOL durations, recovery confirmation/censoring, and paired actual-simulator jammer transitions.

All 18 main scheduler pairs match their exogenous hashes and budget assignments. All 9 fixed-controller pairs additionally match service-quota histories exactly. The 3 higher-load fixed pairs also match. Requested URLLC cohort sizes and offered-batch partitions were checked for all 42 runs.

All four Phase 0 smoke checks pass again. All six full-episode trajectory hashes—including legacy, measured legacy, causal paced/unpaced and causal work-conserving—match the saved Phase 0 hashes. This verifies the specific Phase 0 smoke scenarios, not every historical paper result. See `compatibility_summary.json`.

## Main experiment

Each run lasts 5 seconds, with a 1 ms simulation step, 8 eMBB and 8 URLLC UEs, original per-UE traffic traces and radio model, a fixed physical reserve pool of 18 PRBs, and the existing periodic jammer (2.5-second period, 50% duty, severity 0/10/20 lost PRBs). Seeds are 1234, 2345 and 3456.

Two illustrative queue-budget classes are balanced four/four: tight = 2 ms and relaxed = 10 ms, assignment seed 2026. They are not validated application-specific industrial deadlines. Fixed-control comparisons activate exactly 3 reserved PRBs for URLLC and 0 for eMBB, including before initial service. Jammed baseline quotas change as in Phase 0; each A/B pair uses identical quotas at every service tick.

All following main tables show arithmetic means of the three per-seed results. Percentages are percentage points on a 0–100 scale, not relative improvements. This is a small pilot without statistical significance claims.

### A/B: fixed reserve, scheduler isolation

“Known failed” includes overflow, completed late fully admitted batches, and overdue unfinished batches. The upper bound also treats unfinished/not-yet-due batches as failures. Completion-only and served-byte statistics use different denominators.

| Lost PRBs | Scheduler | Offered-batch failure lower–upper % | Completed batch late % | Served-byte late % | Sampled HOL violation % | Worst-UE known failure % | Recovered / OFF windows |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Proportional WC | 0.000–0.003 | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| 0 | Deadline slack | 0.000–0.002 | 0.000 | 0.000 | 0.000 | 0.000 | N/A |
| 10 | Proportional WC | 0.000–0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 6/6 |
| 10 | Deadline slack | 0.637–0.641 | 0.638 | 0.019 | 0.000 | 0.987 | 6/6 |
| 20 | Proportional WC | 31.393–31.427 | 31.424 | 31.255 | 31.060 | 33.080 | 4/6 |
| 20 | Deadline slack | 32.584–32.632 | 32.630 | 32.288 | 32.190 | 32.600 | 5/6 |

### Per-class outcomes and violation durations

Longest duration is the mean across seeds of each run’s longest per-UE observed HOL-violation run. Mean run duration pools observed runs within each seed, then averages seeds, retaining right-censor flags in per-UE data. These are sampled HOL durations, not durations of plant failure.

| Lost PRBs | Scheduler | Tight known failure % | Relaxed known failure % | Mean observed run ms | Mean longest run ms |
|---:|---|---:|---:|---:|---:|
| 0 | Proportional WC | 0.000 | 0.000 | 0.000 | 0.000 |
| 0 | Deadline slack | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | Proportional WC | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | Deadline slack | 0.683 | 0.592 | 0.000 | 0.000 |
| 20 | Proportional WC | 33.065 | 29.722 | 887.429 | 1648.000 |
| 20 | Deadline slack | 32.590 | 32.578 | 138.452 | 821.333 |

### Resource use and execution time

All fixed-control rows activate 3 reserve PRBs throughout. Allocated/unused values below are mean URLLC slice PRBs; inactive reserve is not counted as unused active quota. Timing uses CPU inference with one PyTorch thread. Scheduler times are per URLLC allocation call; wall time covers the entire 5-second loop including telemetry and hashing, excluding initialization and final report serialization.

| Lost PRBs | Scheduler | eMBB Mbps | URLLC allocated / unused PRBs | Scheduler mean / p95 µs | Loop wall s |
|---:|---|---:|---:|---:|---:|
| 0 | Proportional WC | 94.708 | 57.667 / 5.333 | 5.67 / 8.70 | 2.029 |
| 0 | Deadline slack | 94.708 | 57.667 / 5.333 | 167.85 / 206.86 | 2.870 |
| 10 | Proportional WC | 93.939 | 57.167 / 4.333 | 5.60 / 8.26 | 2.063 |
| 10 | Deadline slack | 93.939 | 57.167 / 4.333 | 166.91 / 202.32 | 2.903 |
| 20 | Proportional WC | 93.176 | 56.500 / 3.500 | 6.62 / 9.87 | 2.195 |
| 20 | Deadline slack | 93.176 | 56.500 / 3.500 | 165.02 / 205.98 | 3.009 |

### Interpretation of the isolated scheduler result

- No jamming: neither method has known deadline failures; slack scheduling adds computation without a measured deadline benefit. Small upper bounds reflect not-yet-due traffic at truncation.
- Moderate jamming: proportional WC has zero known failures, while slack scheduling introduces about 0.64%. Completed late batches coexist with zero sampled post-service HOL violations: some late batches finish on a tick whose remaining head is compliant/empty. The separate metrics reveal this difference rather than hiding it.
- Severe jamming: aggregate known failure increases from 31.39% to 32.58%. Tight-class failure decreases slightly (33.07% to 32.59%), while relaxed-class failure increases (29.72% to 32.58%). Worst-UE failure and longest observed violation improve modestly/substantially, respectively, and confirmed recovery coverage rises from 4/6 to 5/6 OFF windows. Those continuity/fairness benefits do not outweigh aggregate failures automatically.
- These severe fixed-control averages are dominated by seed 3456: its aggregate failure is 94.18% with proportional WC versus 97.7525% with slack; seeds 1234 and 2345 have no known severe-case failures. Three seeds do not establish a robust population-level gain.
- eMBB throughput and mean URLLC allocated/unused PRBs are identical in each fixed-control comparison. The differences arise from allocation across UEs rather than extra slice resources. The least-slack rule keeps serving overdue backlog; this can favor equalizing overdue heads instead of maximizing timely completions. This mechanism is a plausible explanation, not a proved causal decomposition.
- The greedy implementation uses roughly 165–168 µs per URLLC allocation call versus 6 µs for the proportional allocator on this host. Both pilot loops run faster than simulated time, but slack scheduling has a material CPU cost and should not be presented as free.

## Optional frozen causal-DQN transfer comparison

The checkpoint is `backend/checkpoints/res_dqn/xAppResilientDQNAllocator_20260123_215931_final.pt`, SHA-256 `57968f350effc428ad0cdd6914f6a281fd410ba276d9674f6b509e8c2f5484a3`. Its 15 inputs, network architecture, reward and weights are unchanged; there is no Phase 1 training. These are not independently optimized or matched-cost controllers.

| Lost PRBs | Scheduler | Known batch failure % | Mean active reserve PRBs | eMBB Mbps |
|---:|---|---:|---:|---:|
| 0 | Proportional WC | 0.000 | 1.860 | 94.955 |
| 0 | Deadline slack | 0.000 | 1.916 | 94.976 |
| 10 | Proportional WC | 3.580 | 3.523 | 94.616 |
| 10 | Deadline slack | 0.288 | 4.135 | 94.774 |
| 20 | Proportional WC | 19.636 | 6.731 | 95.398 |
| 20 | Deadline slack | 7.108 | 5.825 | 94.258 |

Frozen-controller deadline outcomes improve in attacked cases, but scheduler-dependent queues change the controller’s resource decisions. At severe jamming the slack combination has lower mean activated reserve and lower eMBB throughput. The fixed-quota experiment does not establish a general aggregate scheduler advantage, so the frozen result is a coupled transfer observation, not evidence of a superior newly trained controller.

## Separately labeled overload check

One additional placement seed (1234), 12 URLLC UEs, the same 8 eMBB UEs, original per-UE traces/radio, class budgets, reserve split and five-second horizon. This changes population only and is not pooled with the main experiment.

| Lost PRBs | Scheduler | Known batch failure % | Sampled HOL violation % | Worst-UE failure % |
|---:|---|---:|---:|---:|
| 0 | Proportional WC | 99.237 | 99.132 | 99.740 |
| 0 | Deadline slack | 99.377 | 99.275 | 99.420 |
| 10 | Proportional WC | 99.397 | 99.313 | 99.800 |
| 10 | Deadline slack | 99.485 | 99.410 | 99.520 |
| 20 | Proportional WC | 99.490 | 99.423 | 99.840 |
| 20 | Deadline slack | 99.567 | 99.495 | 99.600 |

Both methods fail almost all offered budgets even without jamming in this load check. Slack slightly improves tight-class and worst-UE outcomes while worsening the relaxed class and aggregate rate. This is a capacity/load-feasibility warning: reprioritization does not create service capacity. It is not an operating regime supporting credible industrial service guarantees.

## Recommendation for the T-ICPS direction

Keep the scheduler as an optional, mechanically validated research comparator. Do **not** replace the Phase 0 proportional baseline or claim an overall deadline improvement from this pilot. Repository defaults remain legacy. The continuity-versus-completion tradeoff may be worth studying if a real industrial application justifies the chosen outcome, but this evidence is not sufficient as the main journal contribution.

Before extending control, select application-supported queue budgets, establish per-UE load feasibility, expand paired placements/load regimes, and investigate why already-overdue work consumes resources under this rule. Independently evaluate starvation-guard sensitivity and matched-cost controller behavior. Prediction will not repair a fundamentally infeasible service regime. No predictive DQN development was begun.

## Reproducibility artifacts

- `pilot_summary.json`: main grouped results, settings and 18 paired checks.
- `stress_summary.json`: separately labeled load check and 3 paired checks.
- `pilot_runs.json`: all 42 per-run outputs, including per-UE/per-class metrics, censoring, recovery, resource costs, hashes and environment versions.
- `pilot_table.csv`: grouped comparison data in machine-readable form.
- `main_scenarios/` and `stress_scenarios/`: the generated catalogs; original source traces/catalogs were not edited.
- `compatibility_summary.json`: new Phase 0 smoke results and all six saved-trajectory comparisons.
- `provenance.json`: base branch and source hashes. Full transient logs/reports can be regenerated with the commands in README.

## File-by-file implementation summary

| File | Change |
|---|---|
| `backend/network_layer/ue_budgets.py` | Private deterministic balanced class assignment; Phase 0 override precedence. |
| `backend/network_layer/deadline_scheduler.py` | Pure integer least-slack allocator, virtual FIFO and starvation guard. |
| `backend/network_layer/phase1_metrics.py` | Offered-batch partitions, paired scenario/quota hashing, recovery, classes, resource and timing summaries. |
| `backend/network_layer/cell.py` | Optional intra-URLLC scheduler, shared eMBB proportional baseline, timing/resource diagnostics and starvation cleanup. |
| `backend/network_layer/ue.py` | Fixed class-label metadata. |
| `backend/network_layer/xApps/xapp_resilient_dqn_allocator.py` | Episode budget-assignment hook only; no changes to state, reward, action list or network architecture. |
| `backend/network_layer/xApps/xapp_resilient_policy_validator.py` | Fixed-reserve evaluation policy, initialized before service and after reset. |
| `backend/network_layer/queue_measurements.py` | Class metadata, complete offered-batch outcome counters and sampled HOL runs; original counters retained. |
| `backend/network_layer/base_station.py` | Pass actual sample interval to measurement-only HOL statistics. |
| `backend/network_layer/simulation_engine.py` | Optional read-only observer after BS service and before RIC actions. |
| `backend/settings/phase0_config.py` | Optional scheduler mode and configurable starvation threshold. |
| `backend/tools/phase0_evaluate.py` | New optional modes, fixed reserve/caps and Phase 1 telemetry; Phase 0 defaults preserved. |
| `backend/tools/phase1_pilot.py` | Controlled scenario generation, paired runs, frozen-transfer option, cohort checks and summaries. |
| `backend/tests/test_phase1.py` | New unit and actual-simulator integration tests; Phase 0 tests unchanged. |
| `backend/evaluations/phase1/` | Guide, this report, generated catalogs, per-run/grouped results, compatibility and provenance artifacts. |
