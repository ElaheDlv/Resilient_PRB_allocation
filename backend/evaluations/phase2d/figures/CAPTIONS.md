**01_a4_recorded_behavior.** All 108 recorded A4 candidate/scenario runs; learner seeds12026/22026, checkpoints3/6/9. Associations are not causal; workloads and reserve actions differ. Data: ../a4_recorded_candidates.csv.

**02_switch_conditions.** Aggregate-selected A4, both learner seeds. Feature change excludes mode, previous actions, quotas and exposure flags; includes workload/service/queue and deadline-risk inputs. This does not identify measurement noise. Data: ../decision_diagnostics.csv.

**03_exact_harm_case.** Placement53260288 j20, learner12026, aggregate-selected A1/A4 ep6. Original 94.667-pp degradation case. Curves sampled every10ms from stored 1-ms observations; dashed lines mark OFF/ON. Different learned quotas are visible, so scheduler-only causality is not inferred. Data: ../harm_ticks.csv.gz.

**04_risk_saturation.** Overloaded aggregate-selected A4 replays, both learner seeds. Every fifth decision plotted; full conditional ranges/stds for all30 inputs and service deficits are in ../state_discrimination.csv. Saturation is a valid failure fraction reaching1, not an arbitrary encoder clip.

**05_fixed_parameter_tradeoff.** Exploratory contested C2 validation: two placements x three severities, same reactive reserve trajectory. All seven policies are reported, with no lambda optimization or confidence interval. G0 equals F, so points overlap. Data: ../prototype_scenarios.csv and ../prototype_summary.csv.

**06_scheduler_cost.** Wall-clock allocation cost includes the experimental gate/telemetry query but excludes post-service credit/window updates, diagnostics and I/O. CPU affinity6,7,11,14, one numerical thread; no hard real-time guarantee. Feasible/overloaded use j20 only; contested uses j0/10/20. Data: ../prototype_summary.csv.
