# Phase 1.2: measured fairness and feasibility decision

Base `6bce0d8`; feature branch `phase1.2/fairness-feasibility`. No DQN, prediction, radio/jammer model, physical reserve sizing, or source-trace changes. The optional fairness allocator and measurements were evaluated with fixed reserve activation of three URLLC PRBs from the unchanged pool of 18.

Included study: 60 tuning-validation episodes, 16 zero-result boundary-screening episodes, 16 threshold-validation episodes, 16 recoverable-overload validation episodes, and 240 paired held-out episodes. Primary held-out seeds are 7001–7006; threshold seeds 9001–9006; recoverable-overload seeds 10001–10006. Every cell has six independent seeds. Supplementary scenarios were added to resolve specific operating-region/recovery limitations; they are labeled separately and did not retune fairness. An aborted seed-selection launch is excluded, with all 8001–8006 seeds excluded from held-out analysis.

## 1. Why individual UEs were harmed

The original harmed UEs all have enough capacity to serve their own 650-byte/ms demand if given the available quota. They suffer **service-rate deprivation**, not prolonged total exclusion: receiving positive service every few milliseconds does not prevent a FIFO backlog when mean served bytes are below arrivals. The feasibility rank can keep serving fresh feasible heads at other UEs while a particular expired head stays in fallback. Tight budgets can expire before the three-tick zero-service guard intervenes. Per-UE offered load, budget, CQI/MCS, PRB/byte service, failed/unresolved batches, HOL runs, and ON/OFF segments are retained in `existing_candidate_harms.json`; all original UE counters remain in the Phase 1.1 artifacts.

Original five-second harmed cases (P=proportional, F=feasibility; failures are known-failed offered batches):

| Jamming | Seed / UE index | Budget ms | Bytes/PRB | P failure % | F failure % | P served/offered % | F served/offered % |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 4004 / 6 | 10 | 69.795 | 0.00 | 75.10 | 99.87 | 72.56 |
| 0 | 4006 / 6 | 10 | 57.363 | 90.48 | 96.54 | 97.91 | 54.00 |
| 10 | 4006 / 6 | 10 | 57.363 | 97.02 | 98.56 | 95.53 | 40.29 |
| 10 | 4007 / 5 | 2 | 57.363 | 51.80 | 95.16 | 100.00 | 76.28 |
| 20 | 4008 / 3 | 2 | 69.795 | 81.80 | 98.06 | 100.00 | 65.29 |

Concrete example: severe seed4008 UE3 needs about 9.31 PRBs/ms at 69.7955 bytes/PRB. Feasibility gives mean 6.0912 PRBs/ms and serves 424.36 bytes/ms, versus proportional 9.3692 PRBs/ms and 650.00 bytes/ms. Its failure rate rises 81.80%→98.06%, including 3,170 late completions and 1,733 overdue unfinished batches;3 not-yet-due batches remain censored. At 1.300 s after OFF it gets 9 PRBs (628 bytes) against 650 new bytes; seven other queues complete their new batches. Clearing all fresh batches independently requires 64 PRBs while the quota is 63, but fluid load is 59.433. Proportional amortizes rounding across intervals; the feasibility fallback remains deprived. This is avoidable allocation-induced harm on top of unavoidable ON-period misses. The committed Phase 1.1 trace evidence contains those snapshots; both corresponding original trajectories reproduce exactly in the current code.

Global infeasibility is a separate statement. The necessary deadline-window bound proves some failures unavoidable in the other four jammed/strongly overloaded harmed cases. For no-jamming seed 4004, fluid load 63.038 exceeds quota 63 in the long run, yet the five-second deadline-window bound does **not** prove failure unavoidable; do not replace that distinction with a blanket overload claim. These bounds cannot determine which UE must fail or establish that the observed failure percentage is optimal.

## 2. Fairness mechanism and validation

The new `deadline_fair` policy maintains bounded service credits from causally observed admitted queue increments and actual serviced bytes. It first retains the old zero-service guard, spends at most floor(alpha*quota) PRBs repaying normalized positive service deficits, then uses the existing feasibility/EDF/remaining-work rule. Integer useful-demand caps, virtual FIFO, deterministic ties, partial-batch timestamps, and work conservation are retained. Credit clipping bounds memory, **not real backlog or deadline unfairness**. Exact equations and limitations are in README.

Only alpha was tuned, on seeds 6001/6002 with separate budget assignment 2028. Among 0.15/0.35/0.65, the predefined criterion selected 0.35: minimize mean worst-UE failure while retaining at least half the baseline-relative aggregate gain. Validation mean worst-UE failure 32.5500% versus 32.5556% at 0.15 is a tiny distinction; aggregate failures 14.1524% versus 8.1038% reveal substantial selection cost for that small difference. The parameter is not claimed optimal and was not changed after held-out evaluation.

## 3. Operating regions

The source trace has synchronous constant 650-byte one-ms batches. Experimental traces scale its byte lengths only, with a common size per seed/regime; timestamps, radio conditions and budget assignment remain paired. Target fluid load is normalized to the 57-PRB ON quota. No-jamming/OFF quota is 63. Source files are unchanged. These are controlled synthetic operating regions, not validation of realistic industrial traffic or end-to-end 2/10-ms deadlines.

| Regime | Target ON load | Actual fluid PRBs/interval range | Interpretation |
|---|---:|---:|---|
| nominal | 0.6000 | 34.119–34.163 | One-interval integer-drain certificate; generous feasible margin. |
| near | 0.9600 | 54.644–54.688 | Integer contention possible; observed deadline outcomes may still saturate at zero. |
| threshold | 1.0005 | 56.942–57.027 | Supplementary finite-budget boundary selected using validation only; no impossibility certificate is a feasibility proof. |
| recovery | 1.0500 | 59.795–59.847 | ON overload with OFF capacity margin; tests recovery without persistent OFF overload. |
| overloaded | 1.1500 | 65.492–65.522 | Demand exceeds even OFF/no-jamming capacity; some misses are physically unavoidable. |

Certificate details in `analysis.json`: sum ceil(batch_bytes/b_i)<=quota is a constructive one-interval certificate. A positive excess in sum(due_batch_count_i*batch_bytes/b_i)−window_capacity is a necessary-bound violation proving zero-miss scheduling impossible in that window. The first ON segment contains 1,249 service ticks; OFF begins at 1.250 s. Fluid load below quota alone is not a deadline proof. The finite-budget threshold follows zero-result validation at 0.96 and 0.995; all screening results are retained.

## 4. Held-out outcomes

Each table cell below is aggregate / worst-UE known-failed offered-batch percentage, averaged over six paired seeds. Worst UE is selected separately within each seed/policy; individual identity changes are also enumerated in `analysis.json`. Lower failure includes overflow, fully admitted late completion, and fully admitted overdue unfinished traffic exactly once. Not-yet-due unfinished traffic is censored and included only in separately reported upper bounds.

| Regime / jamming | Proportional | Slack | Feasible | Fair |
|---|---:|---:|---:|---:|
| nominal / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| nominal / 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| near / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| near / 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| threshold / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| threshold / 20 | 0.000 / 0.000 | 0.276 / 0.411 | 0.869 / 6.956 | 0.020 / 0.161 |
| recovery / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| recovery / 20 | 84.237 / 92.594 | 88.301 / 88.428 | 12.132 / 97.056 | 35.805 / 97.022 |
| overloaded / 0 | 94.840 / 98.311 | 96.630 / 96.744 | 12.075 / 96.600 | 37.767 / 96.578 |
| overloaded / 20 | 98.445 / 99.494 | 98.743 / 98.806 | 18.493 / 98.689 | 43.021 / 98.678 |

Deadline-class table cells are tight / relaxed known-failure percentages:

| Regime / jamming | Proportional | Slack | Feasible | Fair |
|---|---:|---:|---:|---:|
| nominal / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| nominal / 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| near / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| near / 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| threshold / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| threshold / 20 | 0.000 / 0.000 | 0.257 / 0.294 | 0.000 / 1.739 | 0.000 / 0.040 |
| recovery / 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| recovery / 20 | 92.460 / 76.014 | 88.281 / 88.321 | 24.264 / 0.000 | 44.258 / 27.351 |
| overloaded / 0 | 98.261 / 91.419 | 96.657 / 96.603 | 20.160 / 3.990 | 59.800 / 15.735 |
| overloaded / 20 | 99.471 / 97.419 | 98.744 / 98.742 | 36.986 / 0.000 | 86.042 / 0.000 |

Paired **Fair minus Feasible** differences, severe jamming, percentage points with descriptive 95% Student t intervals over six independent seed differences (df 5). Negative favors fairness; these exploratory intervals are not multiplicity-adjusted.

| Regime | Aggregate difference [CI] | Worst-UE difference [CI] |
|---|---:|---:|
| nominal | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| near | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| threshold | -0.849 [-2.556, 0.857] | -6.794 [-20.445, 6.856] |
| recovery | 23.673 [23.133, 24.213] | -0.033 [-0.055, -0.011] |
| overloaded | 24.528 [24.503, 24.552] | -0.011 [-0.029, 0.007] |

Individual harms and confidence intervals versus proportional are included in `analysis.json`. Group upper failure bounds, unresolved traffic, served-byte lateness, sampled HOL violations and all per-seed/per-UE/class counters are preserved. No UE, batch or millisecond is treated as an independent seed.

## 5. Recovery, resource use and computation

Recovered **simultaneous all-UE** OFF windows / observed windows, severe jamming. There is one window per seed; failures to recover remain censored. No-jamming recovery is undefined.

| Regime | Proportional | Slack | Feasible | Fair |
|---|---:|---:|---:|---:|
| nominal | 6/6 | 6/6 | 6/6 | 6/6 |
| near | 6/6 | 6/6 | 6/6 | 6/6 |
| threshold | 6/6 | 6/6 | 6/6 | 6/6 |
| recovery | 6/6 | 6/6 | 0/6 | 0/6 |
| overloaded | 0/6 | 0/6 | 0/6 | 0/6 |

`per_ue_recovery.json` reports individual first compliant holds; individual recoveries at disjoint times do not prove simultaneous recovery. In sustained overload, even OFF capacity is insufficient, so that recovery comparison alone cannot diagnose scheduler quality; the separate recoverable-overload condition addresses this limitation.

Allocated PRB utilization percentages (allocated/quota); all policies activate the same 3 reserve PRBs. Mean allocated/unused PRBs and their seed intervals are also in `analysis.json`.

| Regime / jamming | Proportional | Slack | Feasible | Fair |
|---|---:|---:|---:|---:|
| nominal / 0 | 61.38 | 61.38 | 61.38 | 61.38 |
| nominal / 20 | 64.99 | 64.99 | 64.99 | 64.99 |
| near / 0 | 92.59 | 92.59 | 92.59 | 92.59 |
| near / 20 | 96.73 | 96.75 | 96.76 | 96.76 |
| threshold / 0 | 96.03 | 96.03 | 96.03 | 96.03 |
| threshold / 20 | 98.26 | 98.32 | 98.36 | 98.34 |
| recovery / 0 | 100.00 | 100.00 | 100.00 | 100.00 |
| recovery / 20 | 100.00 | 100.00 | 100.00 | 100.00 |
| overloaded / 0 | 100.00 | 100.00 | 100.00 | 100.00 |
| overloaded / 20 | 100.00 | 100.00 | 100.00 | 100.00 |

Computation table cells are median / p 95 / p 99 microseconds, averaged across per-seed allocation-time quantiles. Allocation timing includes credit preparation, excludes post-service credit debit, network service, diagnostic callbacks and I/O; it is not a total-policy overhead or hard real-time guarantee. Loop runtime and individual quantiles are retained per run.

| Regime / jamming | Proportional | Slack | Feasible | Fair |
|---|---:|---:|---:|---:|
| nominal / 0 | 5.2 / 9.4 / 16.7 | 115.5 / 164.9 / 213.4 | 154.8 / 214.6 / 257.2 | 179.3 / 245.1 / 289.0 |
| nominal / 20 | 6.0 / 10.7 / 19.8 | 116.5 / 161.5 / 212.9 | 152.6 / 213.6 / 263.1 | 180.6 / 242.1 / 291.1 |
| near / 0 | 5.2 / 8.8 / 15.3 | 167.8 / 231.6 / 275.0 | 221.1 / 287.6 / 330.0 | 247.2 / 315.5 / 364.8 |
| near / 20 | 7.4 / 11.9 / 22.4 | 167.3 / 223.1 / 270.8 | 227.7 / 302.8 / 351.1 | 251.4 / 321.8 / 367.7 |
| threshold / 0 | 4.7 / 7.9 / 13.7 | 174.4 / 237.0 / 283.3 | 222.6 / 291.8 / 333.3 | 250.2 / 322.8 / 371.1 |
| threshold / 20 | 7.5 / 11.4 / 20.9 | 177.8 / 238.8 / 288.0 | 258.7 / 333.5 / 383.7 | 272.4 / 345.8 / 395.7 |
| recovery / 0 | 5.4 / 9.9 / 17.4 | 181.0 / 238.7 / 282.2 | 237.8 / 308.8 / 355.2 | 269.4 / 337.5 / 376.2 |
| recovery / 20 | 8.0 / 13.7 / 25.3 | 186.7 / 250.8 / 300.4 | 220.6 / 294.9 / 339.5 | 253.4 / 328.7 / 379.6 |
| overloaded / 0 | 7.3 / 13.3 / 24.0 | 196.3 / 260.1 / 302.5 | 237.7 / 312.1 / 355.4 | 273.8 / 348.0 / 394.7 |
| overloaded / 20 | 7.8 / 12.6 / 22.4 | 189.0 / 253.9 / 322.0 | 236.4 / 312.0 / 373.8 | 268.1 / 344.8 / 404.8 |

## 6. Decision

1. **What causes worst-UE degradation?** The feasibility rank protects heads that are still timely and demotes expired FIFO heads. Partial service does not reset their age. A UE can receive service every interval yet get less than its arrivals; integer rounding while repeatedly clearing other queues makes this persist even after OFF. Radio capacity limits mean some ON-period failures are unavoidable, but they do not force that particular UE to absorb nearly all failures. The bounded service-credit intervention changes byte-service distribution but does not guarantee adequate backlog repayment or deadline fairness.

   New recoverable-overload seed10001 provides direct evidence. At 2.000 s, after OFF, fairness leaves UE5 with 282,804 queued bytes and 319-ms HOL age. It grants 10 PRBs at 81.9 bytes/PRB, serving 819 bytes against 884 newly arriving bytes. The reconstructed pre-service credit is 13,666 bytes, capped at 17,680, and the fairness allowance is 22 PRBs shared with other deprived UEs. That credit is much smaller than the accumulated backlog and does not prescribe a catch-up service plan. Proportional's same UE has only 20,037 queued bytes and 22-ms HOL at this timestamp, and all UEs subsequently recover. Queue/grant/service snapshots and explicitly labeled causal credit reconstruction are in `recovery_trace_evidence.json`.

2. **Can fairness improve without losing most aggregate gains?** Byte-service fairness improves, and about 69% of the feasibility scheduler's aggregate advantage over proportional is retained in sustained overload (both no-jamming and severe conditions). But this is not the requested solution to worst-UE deadline harm: severe overloaded mean worst-UE failure changes only 98.689%→98.678%, with the paired interval spanning zero, while aggregate failure rises 18.493%→43.021%. The tight class worsens 36.986%→86.042%; relaxed failures remain 0%. Service-Jain fairness rises 0.917→0.976, showing why a throughput-fairness metric cannot stand in for deadline fairness. Multiple evaluated individual UEs worsen substantially; their identities and deltas are enumerated. At the finite-budget threshold, fairness reduces the feasibility scheduler's aggregate 0.869%→0.020% and worst-UE 6.956%→0.161%, but six-seed intervals span zero. Proportional has 0% known failures there; fairness harms one observed UE by 0.967 percentage points relative to it.

3. **Are the large no-jamming improvements robust across realistic load regimes?** No such broad claim is supported. All policies have zero known no-jamming failures in the nominal, near, threshold and recoverable-overload controls. The dramatic no-jamming advantage occurs only in sustained overload, where offered fluid work exceeds even the OFF/no-jamming quota. These are synthetic constant-batch capacity-conditioned traces; application realism, burstiness, heterogeneous offered rates and calibrated industrial budgets remain unvalidated. Terminal unresolved traffic is retained in upper bounds rather than silently called successful.

4. **Does fairness improve all-UE recovery?** No. Nominal/near/threshold policies all recover 6/6; sustained-overload policies all recover 0/6, where insufficient OFF capacity is a confound. In the added ON-overloaded/OFF-feasible control, proportional and original slack recover 6/6, while both feasibility variants recover 0/6. Thus the recovery failure is not explained away by insufficient OFF fluid capacity. The fairness policy's aggregate 35.805% versus proportional 84.237% comes with worse mean worst-UE failure 97.022% versus 92.594%. Individual recovery holds do not imply simultaneous recovery.

5. **Enough evidence to integrate deadline-risk-aware slice control?** Not yet as a new learned controller or as a journal performance claim. Keep proportional as the general baseline; retain feasibility/fairness policies as controlled tradeoff comparators. The present fairness candidate does not resolve individual deadline harm or recovery and should not replace that baseline. A future small activation-quota sweep within the existing 18-PRB pool could test whether extra service capacity actually restores per-UE feasibility and recovery before modifying controller state/reward. Establish an application-supported delay endpoint and traffic model, and separate unavoidable load failures from allocation-induced failures. The unchanged RIC decision period is 10 ms while one synthetic queue budget is 2 ms, which also limits causal reaction claims. None of that new control work, retraining or prediction was implemented here.

**No Pareto improvement is claimed.** Deadline-class and individual harm, uncertainty, recovery failures, unresolved traffic and computation costs remain explicit. These results do not prove that every possible fairness policy must fail; they reject this bounded recent-service-credit mechanism as a sufficient fix.

## Validation and reproduction

All 45 tests pass, including actual-byte credit debiting, bounded state, zero/partial service, integer/cap/useful-demand conservation, alpha-zero replay, episode cleanup, four-policy jammer transitions and pairing, finite-vs-long-run capacity bounds, and validation/held-out seed-separation regression tests. All six saved Phase 0 smoke hashes match, including paced/unpaced equivalence. Two original Phase 1.1 severe-seed4008 trajectories (proportional and feasibility) also match exactly. These are scoped compatibility checks, not a revalidation of every historical publication result.

All included paired runs match exogenous trajectories, budgets and quotas; all observed UE capacities match their calibration. Accounting audits check exclusive failure partitions, useful-demand bounds and absence of eligible unused resources. Source/input hashes and cohort exclusions are in provenance. Root research notes were left untouched.

Exact commands and mathematical details: `README.md`. Exact changed paths: `CHANGED_FILES.md`. Measured exports: `analysis.json`, `per_ue.csv`, `per_class.csv`, `per_ue_recovery.json`, stage summaries and calibration/selection files. Full event/allocation streams and generated CSV/config inputs remain regenerable from the documented scripts. No predictive or deadline-risk DQN development was begun.

Zero-variance t intervals in all-zero cells describe these six observed seeds only; they are not reliability guarantees or bounds on rare unobserved failures.
