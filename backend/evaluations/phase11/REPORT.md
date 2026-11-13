# Phase 1.1 measured diagnosis and recommendation

Base `cb35202`; feature branch `phase1.1/scheduler-diagnosis`. All 108 study episodes completed: 18 Phase 1 fixed-quota reproductions, 18 candidate validation episodes, and 72 held-out episodes (8 independent paired seeds × 3 severities × 3 policies). All paired exogenous, budget-assignment, and service-quota checks passed. All 18 reproduction trajectories exactly match the saved Phase 1 hashes. No DQN, prediction, jammer, traffic, or reserve-partition change.

## A. What the original discrepancy means

The failed-offered denominator is **all offered batches**, including dropped and unfinished traffic. Each offered batch is assigned exactly once to overflow failure, on-time full completion, late-only full completion, full-admission unfinished-overdue failure, or not-yet-due unresolved. Overflow takes precedence over late completion and censoring. Lower failure=(overflow+late-only+unfinished-overdue)/offered; upper adds not-yet-due unresolved. Raw completed/late-completed/unfinished counts are descriptive overlapping counters; the disjoint partition supplies the failure numerator. Event-derived partitions agree with sidecar offered, overflow, late-only and served-byte counters; byte conservation is also asserted. See README for exact boundaries and CSV semantics.

Reproduction totals below pool the three original seeds; each class has 60,000 offered batches. No buffer overflow occurred. The extra moderate-jamming failures are all late completions, not overflow or terminal censoring. Severe-jamming failures also contain small, explicitly counted overdue unfinished cohorts.

| Jamming | Policy | Class | Completed | Late-only | Overflow | Unfinished overdue | Unresolved not due |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | proportional_wc | tight | 59998 | 0 | 0 | 0 | 2 |
| 0 | proportional_wc | relaxed | 59998 | 0 | 0 | 0 | 2 |
| 0 | deadline_slack | tight | 60000 | 0 | 0 | 0 | 0 |
| 0 | deadline_slack | relaxed | 59998 | 0 | 0 | 0 | 2 |
| 10 | proportional_wc | tight | 59997 | 0 | 0 | 0 | 3 |
| 10 | proportional_wc | relaxed | 59997 | 0 | 0 | 0 | 3 |
| 10 | deadline_slack | tight | 60000 | 410 | 0 | 0 | 0 |
| 10 | deadline_slack | relaxed | 59996 | 355 | 0 | 0 | 4 |
| 20 | proportional_wc | tight | 59971 | 19823 | 0 | 16 | 13 |
| 20 | proportional_wc | relaxed | 59972 | 17833 | 0 | 0 | 28 |
| 20 | deadline_slack | tight | 59980 | 19546 | 0 | 8 | 12 |
| 20 | deadline_slack | relaxed | 59946 | 19538 | 0 | 9 | 45 |

Per-UE and per-class counts, PRBs, bytes, channel/rate statistics, starvation, and before/during/after results are in `per_ue_attribution.csv` and `per_class_attribution.csv`. “During/after” service statistics refer to sample time, whereas batch outcomes refer to arrival cohorts followed to the horizon. Jammed episodes start ON, so their before segment is empty; the no-jamming condition supplies a separate reference. Phase 1 HOL metrics retain horizon censor flags; segment durations are observed, potentially truncated runs.

## B. Mechanisms supported by concrete traces

1. **Residual-byte neglect at the equality boundary.** Moderate jamming, seed3456, URLLC_4 (relaxed, CQI15/MCS27, 155.593487 bytes/PRB): at 0.962 s the head arrived at 0.952 s, has exactly its 10-ms budget age, and has 160 bytes left. Slack grants one PRB, serving 155 bytes and leaving 5. At 0.963 s that offered batch (ID 15228, 650 bytes originally) completes with 11-ms residence. Proportional completes the identical batch with 2-ms residence. This is an allocation difference at a still-attainable deadline, not an overflow or pre-existing infeasibility.

2. **Why the old rank can leave that tail.** After the first PRB, the remaining service penalty is only 5/155.593487≈0.032 PRB-intervals. Other virtual heads with larger remaining service penalties then rank ahead. The old rule continually reduces a head’s urgency as it approaches completion; it optimizes a local slack ranking rather than the count of timely completed batches. A synthetic unit test with three due heads verifies an instance where old slack completes one batch and the candidate completes two under the same quota. This unit example explains the rule; the paired simulator trace establishes that the residual-tail mechanism occurs in the actual pilot.

3. **Late departure is invisible in post-service HOL compliance.** At 0.963 s URLLC_4 receives 8 PRBs, serves 1,244 bytes, and removes that expired head; the new HOL wait is 9 ms and therefore compliant. A sampled compliant head cannot erase the 11-ms completed-batch miss. Across the moderate-jamming reproduction, old slack has 765 late-only failures while proportional has zero, even though both have zero sampled post-service HOL violation frequency. The measurements describe different events, not contradictory reliability claims.

4. **Expired traffic continues to consume service.** Severe jamming, seed3456, old slack at 0.072 s: relaxed UEs 2 and 4 have 11-ms-old heads against 10-ms budgets and receive 10 and 4 PRBs. Proportional also serves expired heads (e.g. tight UE7 at 0.039 s, age 3 ms, six PRBs). Such service cannot make the expired batch timely, although it drains FIFO blockage and may serve later batches too. The pre-head PRB counter must not be called precisely wasted capacity. In the moderate example, the first expired service is already a consequence of the tail miss; it cannot explain the initial miss by itself.

5. **Poor-rate preference is built into the rank, but its separate causal contribution is unproven.** At equal remaining bytes and budget age, the negative h/b term favors lower b. The seed3456 rates include UE0 at 57.363 versus UE4 at 155.593 bytes/PRB, so a fresh 650-byte head has single-PRB service penalties 11.331 versus 4.178 intervals. The traces also show demand-proportional scheduling compensates slower UEs; allocation alone does not establish this compensation is the source of the extra failures. This experiment does not isolate radio-efficiency preference from the rank’s completion effect.

6. **No evidence of wasteful excess grants, eligible idle capacity, or prolonged relaxed zero-service starvation.** The study checks grant useful-demand bounds and eligible-unused intervals. In the original problematic seed, old slack’s longest zero-byte-service run is 1 ms (proportional 0 ms), including relaxed UEs. Relaxed queues nevertheless hover near their budget: at 0.951 s UE4 is 10 ms old with 6 bytes at the head, compared with proportional’s 2-ms head. The issue is insufficient timely service/completion, not long exclusion from service.

For severe seed3456, old slack reduces the longest observed HOL run from 4.944 to 2.464 s and recovers in one OFF window where proportional recovers in neither, yet raises known failures from 37,672 to 39,101 of 40,000 offered batches. This is the continuity/completion tradeoff being diagnosed. `trace_evidence.json` contains matched ticks and completion events; the retained moderate seed3456 gzip streams include every service/arrival/allocation event.

## C. Candidate and validation

The candidate shares integer useful-demand limits, virtual FIFO, and protected service with old slack, but ranks potentially feasible heads first, then earliest deadline and shortest remaining PRB requirement. Infeasible/expired FIFO heads remain eligible as protected/fallback work and are never discarded; partially served heads keep their original arrival time. Its feasibility estimate is optimistic and local, assuming current rate/quota can be attained at future service timestamps. It is not joint admission control or a deadline guarantee. Exact equations, zero-rate handling, tie ordering and conditional starvation bound are in README.

Validation used separate seeds 3001/3002 and budget assignment 2027; guards 3/5/10 all scored 1.99625% mean failure across validation conditions, with 15.97% mean worst-UE failure. The predeclared tie rule selected guard 3. This gives no evidence of a tuned-guard benefit. Held-out seeds 4001–4008, budget assignment 2026, were evaluated once after selection; old slack retained guard 5.

## D. Held-out paired comparison

Percentages are mean known-failed offered-batch fractions across eight independent seeds. Tight/relaxed columns are deadline-class percentages. Unresolved traffic is included in the separately reported upper bounds.

| Jamming | Policy | Failure % | Tight % | Relaxed % | Upper failure % | HOL violation % | Worst-UE failure % |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | proportional_wc | 28.080 | 33.134 | 23.026 | 28.131 | 26.861 | 33.432 |
| 0 | deadline_slack | 33.050 | 33.031 | 33.069 | 33.124 | 30.915 | 33.205 |
| 0 | deadline_feasible | 4.212 | 3.060 | 5.364 | 4.240 | 4.212 | 33.695 |
| 10 | proportional_wc | 52.245 | 56.034 | 48.456 | 52.319 | 51.270 | 56.093 |
| 10 | deadline_slack | 55.474 | 55.484 | 55.464 | 55.556 | 54.618 | 55.527 |
| 10 | deadline_feasible | 7.631 | 12.182 | 3.080 | 7.639 | 7.631 | 61.047 |
| 20 | proportional_wc | 71.038 | 72.430 | 69.646 | 71.129 | 70.702 | 72.453 |
| 20 | deadline_slack | 71.653 | 71.653 | 71.652 | 71.742 | 71.316 | 71.707 |
| 20 | deadline_feasible | 9.261 | 18.521 | 0.000 | 9.267 | 9.261 | 74.085 |

Paired candidate-minus-proportional failure differences in percentage points; negative favors the candidate. Intervals are descriptive 95% Student t intervals over eight paired seed differences (df 7), not over the many dependent batches. Eight seeds and rare bad placements limit generalization; no multiplicity-adjusted significance claim.

| Jamming | Aggregate difference [95% CI] | Tight difference [95% CI] | Relaxed difference [95% CI] |
|---|---:|---:|---:|
| 0 | -23.868 [-55.249, 7.513] | -30.074 [-65.476, 5.327] | -17.662 [-50.862, 15.538] |
| 10 | -44.614 [-80.764, -8.464] | -43.852 [-78.367, -9.336] | -45.376 [-84.554, -6.198] |
| 20 | -61.777 [-94.352, -29.203] | -53.909 [-82.181, -25.637] | -69.646 [-106.542, -32.749] |

| Jamming | Policy | Recovered OFF windows / observed | Max zero-service run (ms) | Mean scheduler time (µs) | Mean diagnostic-enabled loop (s) | eMBB (Mbps) |
|---|---|---:|---:|---:|---:|---:|
| 0 | proportional_wc | 0/0 | 0.0 | 8.31 | 3.509 | 96.000 |
| 0 | deadline_slack | 0/0 | 1.0 | 198.02 | 4.538 | 96.000 |
| 0 | deadline_feasible | 0/0 | 3.0 | 256.16 | 4.588 | 96.000 |
| 10 | proportional_wc | 8/16 | 0.0 | 9.01 | 3.595 | 95.961 |
| 10 | deadline_slack | 8/16 | 1.0 | 190.01 | 4.539 | 95.961 |
| 10 | deadline_feasible | 6/16 | 3.0 | 238.48 | 4.548 | 95.961 |
| 20 | proportional_wc | 6/16 | 0.0 | 8.61 | 3.760 | 95.560 |
| 20 | deadline_slack | 6/16 | 1.0 | 189.85 | 4.756 | 95.560 |
| 20 | deadline_feasible | 4/16 | 3.0 | 233.16 | 4.677 | 95.560 |

All policies activate 3 reserve PRBs and retain identical service quotas. All held-out over-useful-demand counters and eligible-unused UE-interval counters are checked in `heldout_analysis.json`; actual allocations/unused PRBs and per-UE bytes are retained in stage summaries/CSV. Scheduler timing excludes the diagnostic hook; loop wall time includes gzip tracing and measurements and is host-dependent. All-UE recovery requires 10 ms of sustained compliant HOL after OFF; non-recovered windows remain censored, not removed from denominators. Windows within a seed are not independent replicates.

Per-UE harms are enumerated in `heldout_analysis.json` (candidate-minus-baseline), not hidden by the overall gain:

- J 0: 2/64 paired UE outcomes have a larger known-failed fraction; maximum increase 75.100 percentage points.
- J 10: 2/64 paired UE outcomes have a larger known-failed fraction; maximum increase 43.360 percentage points.
- J 20: 1/64 paired UE outcomes have a larger known-failed fraction; maximum increase 16.260 percentage points.

## E. Research recommendation

1. **Why did old slack increase failures?** Its per-quantum service-time subtraction spreads grants and can deprioritize nearly finished heads at the deadline boundary. The matched seed3456 batch demonstrates a 5-byte tail completing late under slack and on time under proportional. Once late, draining it can restore sampled HOL compliance, shortening observed violation runs without undoing the completion miss. Overflow, censoring mistakes and prolonged zero-service starvation do not explain the reproduced discrepancy. Expired service under severe overload is an additional persistence mechanism, shared with proportional, rather than the initial cause of every miss.

2. **Does the candidate reduce misses versus proportional?** Yes for the measured aggregate held-out population: severe failure falls 71.038%→9.261% (paired difference−61.777 percentage points, descriptive 95% CI[−94.352,−29.203]); moderate 52.245%→7.631%. The no-jamming mean also improves 28.080%→4.212%, but its paired interval spans zero. These are failed **offered-batch** outcomes including overdue unfinished traffic, not packet reliability. Both class means improve, but individual UEs can worsen. This held-out cohort includes overloaded placements even without jamming, unlike the original three-seed pilot; those load regimes must not be pooled or mistaken for a changed jammer model.

3. **Does it improve severe recovery without harming UEs?** No. Severe all-UE recovered windows fall 6/16→4/16; mean worst-UE failure rises 72.453%→74.085%. Seed 4008 tight UE3 worsens 81.80%→98.06% failures (+16.26 points). The worse individual increases are 75.10 points without jamming and 43.36 points with moderate jamming. The candidate achieves most of its gain by keeping seven UEs timely while concentrating persistent failures in another, rather than reducing every UE's delay. A three-tick no-service guard does not guarantee a two-ms budget or an adequate long-term service share.

   Concrete severe seed4008 post-OFF evidence at 1.300 s: all 63 PRBs are allocated. Seven UEs each finish 650 bytes; UE3 receives 9 PRBs at 69.7955 bytes/PRB, serves 628 bytes, and retains an 804-ms-old HOL. Its new load is 650 bytes/ms, so this allocation cannot clear its backlog. Fluid demand is 59.433 PRBs per interval, but independently clearing every new batch requires 64 integer PRBs. This is per-batch rounding overhead and persistent fallback disadvantage, even though no grant exceeds useful-demand ceilings. At 3.800 s its HOL is 1.695 s old; at 5 s it has 1,128,395 unfinished bytes. Proportional's corresponding HOL ages are 48 ms and 0 ms at those last two timestamps. Candidate UE3 has 3,170 late completions plus 1,733 overdue unfinished batches; they are disjoint and sum to its 4,903 known failures. Full paired snapshots are in trace_evidence.json. Thus work conservation and zero-service protection alone do not establish completion fairness or recovery.

4. **Which mechanism is best supported?** For aggregate timely completion, the candidate's completion-oriented rank bundle is supported better than the original residual least-slack rule in this study. Proportional is better supported as the general baseline when all-UE recovery and service fairness matter. The experiment changes feasible/expired-head preference, deadline ordering and remaining-work tie-breaking together; it does not isolate a causal benefit from the feasibility screen alone. The current head sizes generally fit a full interval quota, so the screen mainly separates expired heads here. Neither scheduler is uniformly best, and a new predictor is not justified by these findings.

5. **Include this in T-ICPS?** Include the diagnosis and optional candidate as a controlled scheduling tradeoff if the industrial application justifies aggregate completion priorities versus per-device guarantees. Do not present it as a general resilient-industrial improvement or a standalone novel journal contribution. A fairness/service-share mechanism, feasibility/load characterization including integer overhead, and application-supported queue budgets would be needed before recommending deployment or making industrial deadline claims. Separate ablations would be required to credit any particular component. Keep proportional as the reference; defer DQN retraining and prediction.

## Tests, compatibility and artifacts

The automated audit of all 108 episodes passes (`audit_results.json`), including independent failure partitions, phase cohort/resource sums and absence of excess grants or eligible idle capacity. All 36 unit/integration tests pass (see `test_results.txt`), covering original Phase 0/1 behavior plus candidate integer/cap/useful-demand conservation, subbyte/zero-rate service, expired/infeasible fallback, partial FIFO and equality, deterministic rank/guard behavior, residual-tail completion, partial-overflow/censor accounting, read-only diagnostic noninterference, and real jammer ON/OFF transitions under all three schedulers. All six Phase 0 saved trajectory hashes and the legacy/paced-unpaced checks pass again. These are scoped compatibility checks, not a reproduction of every published experiment.

`README.md` gives exact commands and accounting; `CHANGED_FILES.md` lists every changed path; `provenance.json` gives source/trace hashes and environment versions. Stage summaries, CSV exports, `selection.json`, `heldout_analysis.json`, and concrete trace evidence provide the measured results. Full regenerable traces for other episodes remain in the experiment output directory and are not all duplicated here. No predictive control or DQN retraining has begun.
