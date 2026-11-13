# Phase 2D measured decision report

**Recommendation: retain the binary learned controller for now. Keep the gated
lambda=1 heuristic as an isolated nonlearning third-scheduler candidate for
independent validation, not as a demonstrated need for parameterized RIC control.**

Persistent under-service is reproducible, the old fairness mechanism can move
rather than resolve the disadvantaged UE, and the 30-feature state loses per-UE
service-history information. Gated protection offers a useful development
trade-off in one informative contested placement, but harms other UEs, has
substantial computation cost, and cannot cure physical overload. No DQN was
trained, no action/state/reward was changed, and no frozen future holdout was used.

## Scope, audit and implementation

Base `eafcea7`; branch `phase2d/scheduling-diagnostics-fairness`.
`DESIGN.md` provides the source audit, exact mathematical specification, units,
normalization, causal windows and limitations. `protocol.json` was frozen before
experiments. It pins the saved C3 checkpoints and validation input hashes.
`CHANGED_FILES.md` lists every changed path and its purpose.

| Path / function | Audited behavior |
| --- | --- |
| `backend/network_layer/proportional_scheduler.py:6`, `proportional_allocation` | Capped proportional water filling, integer remainder redistribution; no deadlines or miss history. |
| `backend/network_layer/deadline_scheduler.py:20`, `deadline_allocation` | Zero-service guard, then slack or optimistic feasibility/EDF/remaining-work order. Expired FIFO heads remain fallback-eligible. |
| `backend/network_layer/fair_service_credit.py:21`, `prepare`; `:57`, `service` | Existing bounded EWMA admitted-load credits, actual-byte debit; no recent deadline-failure weighting or reachable-target gate. |
| `backend/network_layer/cell.py:346` deadline dispatch/service path | Current P is the Phase0 work-conserving proportional scheduler; F uses feasibility=True. Positive service resets zero-service exclusion even if it is below arrivals. |
| `backend/network_layer/ue_telemetry.py:133`, `_window`; `:190`, `collect` | Read-only queue/HOL/capacity and per-UE arrival-cohort failures, with exposure/history validity. |
| `backend/network_layer/deadline_risk.py:27`, `encode_deadline_risk`; `joint_dqn_schema.encode_joint_state` | Aggregate risk reduction and fixed30 encoder; per-UE identity and byte-service histories are omitted. |

P, F, the Phase1.2 credit implementation and the deadline-allocation kernel were
not rewritten. The sole shared simulator edit is an optional allocator callback
in Cell's existing deadline branch. With no callback, its original function and
arguments are used. Experimental state belongs to a fresh app/cell. The unchanged
F kernel receives alternative gated credits only in the isolated G experiments.
The old alpha=.35 mechanism is an explicit comparator, not presented as new.

42 saved-checkpoint replays include aggregate-selected A4 on all18 validation
entries for both learner seeds; worst-first A4 on the four contested severe
cases; and A1/A2 comparators for the two worst recorded harm examples. All42
match C3 control/parameter/budget/exogenous hashes and complete physical outcome
partitions. All70 nonlearning runs (10 scenarios x7 policies) share identical
reactive-reserve trajectory hashes and exogenous conditions. G0 matches F's
whole-episode per-UE measurements in all10 scenarios. Read-only diagnostics and
shadow computations did not change saved policy behavior.

## A4 switching: direct observations and limits

Aggregate-selected A4,12 scenario/seed runs per regime,299 decisions each:

| Regime | F choices / decisions | Mode switches | Switches with identical measured UE-condition vector | Worst-risk=1 / valid decisions |
| --- | ---: | ---: | ---: | ---: |
| feasible | 2446/3588 | 1205 | 552 | 0/3480 |
| contested | 2695/3588 | 1093 | 0 | 853/3480 |
| overloaded | 3142/3588 | 828 | 0 | 3468/3480 |

The condition vector contains the existing workload/service/queue and five
health-risk values, excluding policy mode, previous increments, quotas and
exposure flags. Identical condition vectors do not imply identical full states.
Example: feasible placement53276672, no jamming, learner22026 ep9, at120ms switches
to F with zero measured condition-vector change, zero 100-ms service deficit,
zero worst failure risk and80 on-time completions in the following10ms. There
are552 such switches in feasible cases, where both policies have zero failures.
This is concrete unnecessary churn in these cases, not proof of sensor noise.
The model is deterministic and no observation-noise intervention was performed;
its own mode/previous-action/quota inputs and endogenous queues can create
feedback. A causal claim about which input caused a switch would require a
separate intervention. Training instability remains supported by C3's
nonmonotonic/seed-variable validation, but cannot be disentangled from action
representation with fixed checkpoints and no retraining.

Next-10-ms associations change sign with context. Aggregate-selected contested
runs have mean1.75 newly observed known-failure events after switches versus3.76
after stays; overloaded runs have56.41 versus33.41. For worst-first contested
replays, the association reverses strongly (22.93 versus0.38). These are event
counts per10ms, not arrival-cohort failure probabilities. Overlapping100-ms
cohort windows, prior modes, reserve actions, severity and learner state confound
attribution. `switch_strata.csv` further separates severity and learner seed.
Do not conclude from these associations that switching caused improvement/harm.

Pure shadows on the first serviced tick after each RIC command use identical
current FIFO queues, quota, current radio rates and prior starvation ages:

| Regime | Sampled service ticks | P strictly more timely completions | F strictly more | Chosen mode strictly worse | Switch to a worse immediate choice |
| --- | ---: | ---: | ---: | ---: | ---: |
| feasible | 3588 | 0 | 0 | 0 | 0 |
| contested | 3588 | 11 | 2036 | 97 | 67 |
| overloaded | 3588 | 0 | 3562 | 432 | 408 |

F is often immediately preferable under overload, yet A4 sometimes selects P,
including408 switches to an immediately worse choice. These shadows count
fully admitted timely batch completions at one actual service timestamp. They
are not long-horizon policy rollouts: a mode with fewer immediate completions
can repay a deprived UE's older FIFO prefix. Thus the table identifies local
completion opportunities, not a proof that F is universally optimal. Full
per-decision grants are retained in `shadows.jsonl.gz`.

## Sustained deprivation and the94.667-pp harm case

For contested placement53260288, j20, learner12026 ep6, tight-budget UE1 has
3000 offered batches at650bytes/ms and57.3628bytes/PRB/tick. It requires about
11.331 fluid PRBs/tick. A4 gives mean9.186 PRBs and serves1,577,143bytes; A1 gives
11.950 PRBs and serves all1,950,000bytes. A4 has2840 known failures (2269 late
completions plus571 overdue unfinished),3 not-yet-due unresolved batches and
372,857 unfinished bytes; A1 has zero failures. Overflow is zero in this case.
The offered partition is exact and unfinished traffic is not counted twice.

A4's initial ON segment serves61.28% of UE1's offered bytes, versus100% for A1.
In the OFF segment A4 serves109.98% of new offered bytes, so it is then **draining**
backlog rather than still suffering a recent service-rate deficit. Nevertheless,
it cannot repay the earlier debt before the next ON period, which serves57.15%
of new offered traffic. This distinguishes persistent batch lateness from
persistent *current* under-service. The maximum zero-service gap in initial ON
is only3ticks; OFF has no zero-service tick. Positive but insufficient service,
not prolonged total exclusion, explains much of the growth.

Concrete frames from `harm_ticks.csv.gz`: at1.250s UE1 has314,053queued bytes,
483-ms HOL age,17PRBs (975served bytes) and URL quota69. At1.300s it has313,313
queued bytes and482-ms HOL despite positive8PRB/458byte service at quota60.
At2.000s the 100-ms deficit is zero while266,763queued bytes and410-ms HOL
remain. The G gate would correctly refuse to treat this enormous prefix as
currently reachable within its2-ms budget. A single mode switch cannot undo
previous expiry. Both learned mode and reserve trajectories differ between A1
and A4, visible in figure03, so this example alone does not isolate the scheduler.

Scheduler-only evidence comes from fixed reactive reserve comparisons on the
same scenario. P gives relaxed UE6 mean11.338PRBs and zero failures; F gives
8.333PRBs against11.331 required and76.867% failures, while every other F UE has
zero failures. The old fair rule reduces UE6 to7.900% but moves the victim to
tight UE1 (75.767%). G1 spreads loss: worst UE6 is21.433%, and other UEs have
7.200–15.300% failures. This establishes allocation-associated persistent
under-service and a redistribution of harm with **identical quotas and channels**.
`per_ue.csv`, `service_segments.csv`, `deprivation_counts.csv` and sampled window
streams preserve the evidence. Deprivation sample counts are not independent
subjects; tiny positive deficits can also be benign rounding transients, as in
some A1 ticks, and must be interpreted with backlog growth, HOL and failures.

The exact harm-case quota distributions also show reserve-control differences:

| Controller | ON mean URL quota | ON ticks below59.087fluid PRBs | OFF mean URL quota |
| --- | ---: | ---: | ---: |
| A4 | 58.469 | 50.34% | 64.471 |
| A1 | 67.333 | 0.57% | 73.510 |

A4 never reaches the64-PRB empty-queue integer-drain certificate in ON; A1
reaches it in73.71% of ON ticks. These comparisons describe different learned
reserve trajectories, not a controlled scheduler-only intervention. The stronger
scheduler-only evidence is the paired reactive-reserve experiment above.

Resource sufficiency is time/state dependent. On this placement fluid offered
load is59.087PRBs/tick and the empty-queue one-tick integer-drain requirement is64.
The reactive controller supplies59URLLC PRBs in settled severe ON and60OFF;
fluid margin in OFF does not prove that a large existing FIFO debt can meet
current deadlines. Some A4/A1 learned quotas exceed64, but an old hundreds-kB
queue cannot be cleared within2ms just because fresh arrivals would fit. We do
not prove zero misses impossible in every contested finite window, nor an
optimal miss rate. In overloaded C2 scenarios, the existing necessary bounds
prove zero misses impossible even with optimistic full-pool OFF quota78. That
capacity limit does not prove that every UE must have99% failures.

## Assessment of the30-feature state and underlying telemetry

The worst risk is the **maximum actual matured cohort failure fraction**, not a
heuristic clipping operation. A fully failing UE sets it to1 regardless of its
identity or the severity/number of other deprived UEs. In selected overloaded
A4 replays it is exactly1 on3468/3480 valid observations (99.66%). Warmup-invalid
placeholders are excluded from this calculation.

Other inputs retain discrimination even conditional on worst=1:

| Input | Observed range under worst=1 |
| --- | ---: |
| Aggregate URLLC normalized queue | 0.03476–1.00000 |
| Active overdue-UE fraction | 0.50000–1.00000 |
| Tight matured failure fraction | 0.50000–0.95619 |
| Relaxed matured failure fraction | 0.00000–0.87640 |
| Transformed maximum HOL pressure | 0.95122–0.99930 |
| Additional measured max per-UE100-ms deficit | 0.36758–0.96885 |

The state is therefore not entirely saturated. Aggregate queue can clip,
HOL transformation compresses large ages, and exposure flags become constant
after warmup. More fundamentally, the encoder discards per-UE identities and
recent byte-service histories. A unit-test collision swaps the failed UE within
a class and obtains exactly the same30-vector. The underlying read-only report
can identify that UE through per-UE cohorts, queue, budget and capacity, but
it does not expose recent offered/served **byte** counts or queue slope directly.
Those require causal counter differences/retained snapshots. HOL alone cannot
say whether a queue is growing or draining; at2.000s above, rates are adequate
but old deadline damage remains.

A bounded complementary feature **max_i max(0,a_i-b_i)/max(1,a_i)** over a full
100-ms post-service window is justified as a diagnostic hypothesis. It varies
while worst failure is constant and explicitly measures recent under-service.
It still loses UE identity, can saturate, and becomes zero during backlog drain;
it must be interpreted with HOL/queue/prefix feasibility and a history-validity
indicator. Neither this feature nor the current snapshot proves intrinsic
radio/global infeasibility or optimal scheduling. No feature is added to the
DQN here, and no evidence that adding it improves learning is claimed.

## Fixed-parameter development results

G reuses F's integer allocator and zero-service guard. It protects up to
floor(lambda*quota) PRBs using a bounded recent offered-minus-served deficit,
weighted by a recent matured-miss fraction, **only** when some fully admitted
unexpired batch is reachable through the current FIFO prefix under an optimistic
local capacity bound. Misses alone earn no credit. Credits are capped by that
prefix, recomputed each round and debited virtually as PRBs are assigned.
There is no new expiry, dropping or batch-reordering rule. `DESIGN.md` gives the
exact equations, including partial service, zero rates, ties and uncertainty.
Lambda is a service-protection **budget share**, not a per-UE minimum guarantee
or a linear mix of schedulers. G0 is F, not P. The 1+m multiplier is a bounded
experimental design choice, not a theoretically optimal or tuned coefficient.

All figures below are means across the prespecified paired validation scenarios;
only two placements exist per regime. Feasible/overloaded use j20 only, contested
uses j0/j10/j20. No lambda was optimized and no confidence intervals are inferred.

| Regime | Policy | Aggregate failures (%) | Worst UE (%) | Median allocation (µs) | p99 allocation (µs) |
| --- | --- | ---: | ---: | ---: | ---: |
| feasible | P | 0.000 | 0.000 | 7.4 | 13.8 |
| feasible | F | 0.000 | 0.000 | 196.1 | 261.1 |
| feasible | OLD035 | 0.000 | 0.000 | 220.9 | 317.3 |
| feasible | G0 | 0.000 | 0.000 | 198.1 | 272.5 |
| feasible | G025 | 0.000 | 0.000 | 483.3 | 680.5 |
| feasible | G05 | 0.000 | 0.000 | 483.6 | 620.8 |
| feasible | G1 | 0.000 | 0.000 | 483.4 | 642.5 |
| contested | P | 4.419 | 9.228 | 9.6 | 17.8 |
| contested | F | 3.189 | 25.511 | 260.1 | 371.5 |
| contested | OLD035 | 3.307 | 25.100 | 284.7 | 380.1 |
| contested | G0 | 3.189 | 25.511 | 262.6 | 358.5 |
| contested | G025 | 3.288 | 24.794 | 586.4 | 840.6 |
| contested | G05 | 7.244 | 24.261 | 586.8 | 877.9 |
| contested | G1 | 3.749 | 5.861 | 595.9 | 890.8 |
| overloaded | P | 99.400 | 99.800 | 10.8 | 18.1 |
| overloaded | F | 37.279 | 99.433 | 225.4 | 312.0 |
| overloaded | OLD035 | 62.033 | 99.433 | 259.0 | 415.9 |
| overloaded | G0 | 37.279 | 99.433 | 227.2 | 300.8 |
| overloaded | G025 | 37.279 | 99.433 | 835.4 | 1207.8 |
| overloaded | G05 | 37.279 | 99.433 | 837.8 | 1224.7 |
| overloaded | G1 | 37.279 | 99.433 | 834.3 | 1226.9 |

Computation columns average per-run quantiles, not pooled-event quantiles. They
include credit/gate preparation and the experimental per-ms telemetry query,
but exclude post-service window/credit updates, measurement observers and I/O.
They are not total execution cost or hard real-time guarantees. In overloaded
runs G's mean per-run p99 exceeds the simulated1-ms step. It is **not yet a
lightweight implementation suitable for a real1-ms scheduler**. Caching existing
10-ms telemetry, gating before expensive cohort scans and profiling could help,
but were not implemented or measured after observing these results.

The gated lambda values are behaviorally distinct in contested cases, and their
trade-off is nonmonotonic: G.5 is substantially worse in aggregate than G.25 or
G1. Under overload, every G value has F's outcomes: the gate rejects extra
protection for unreachable timely targets, while F's guard/fallback still serves
expired queues. The old ungated rule raises overload aggregate failure from
37.279% to62.033% without improving the worst UE. This is useful evidence for
the gate, but not evidence that physical overload has been solved.

The apparent G1 gain is concentrated in **one informative placement**. At
placement53260288 j10, aggregate/worst results are P8.558/18.367, F9.525/76.200,
G1 9.537/13.733. At j20 they are P17.958/37.000, F9.608/76.867,
G1 12.958/21.433. All policies have zero failures at j0 and at every evaluated
severity for the second contested placement53284864. Thus averaging six runs
does not imply six informative independent replications.

Across the48 paired UE/scenario cases in contested development, G1 harms14
and benefits2 relative to F; relative to P it harms8 and benefits8. **No Pareto
improvement is claimed.** Full class, UE, upper-bound and unresolved counts are
retained. eMBB throughput and reserve consumption are identical across prototype
policies because their trajectories are paired: contested means83.989Mbps and
5.419activated PRBs; severe feasible92.334Mbps/10.413PRBs; severe overload
96.000Mbps/10.413PRBs. G1 recovers4/4 contested jammer-OFF windows, P4/4 and F2/4.
Recovery is a consecutive10-ms all-UE HOL-compliant hold, not proof that all
batches met budgets. It fails under physical overload for every policy.

## Seven research answers and next-step decision

1. **Reproducible deprivation?** Yes. Exact C3 replays and fixed-quota P/F
   comparisons show positive service below arrivals, growing FIFO debt and
   concentrated per-UE failure. Persistent lateness can also continue while
   recent service exceeds arrivals because old debt has not been repaid.
2. **Scheduling or capacity?** Both. Shared-quota P/F differences establish
   scheduling-associated harm. Full-pool overload bounds independently establish
   unavoidable zero-miss infeasibility; they do not fix the optimal distribution
   or count of failures. A4's learned reserve choices additionally confound its
   comparison with A1/A2.
3. **Useful fairness trade-off?** G1 provides promising worst-UE/recovery trade-offs
   in one contested placement, at aggregate/individual harm and computation cost.
   The old mechanism often moves the victim. No general superiority is established.
4. **Tunable parameter necessary?** Distinct behavior is demonstrated, necessity
   is not. The nonmonotonic grid does not justify interpolating or automatically
   adapting lambda, and a fixed G1 already supplies the observed fairness benefit.
5. **Meaningful new RIC decision?** Potentially, if independently validated
   operating-region trade-offs persist. These10 development scenarios do not
   show adaptive selection outperforming a simple fixed setting or rule.
6. **Simple heuristic enough?** A fixed G1 delivers the observed fairness/recovery
   benefit in this pilot; a simple rule might avoid needless computation when
   deficits/targets are absent, but no rule was evaluated and none is claimed superior.
7. **Expand98 to196 actions?** No. There is no adaptive-control advantage,
   independent final test, runtime readiness or all-UE improvement evidence.
   196=49x4 choices. Retaining P/F plus G.25/.5/1 requires5 choices (245), since
   G0 duplicates F; choosing a smaller menu would need prospective justification.

Retain the current binary learned interface for compatibility; this does not
endorse A4 over the stronger A2 aggregate baseline from C3. Preserve P/F as essential
comparators, and keep G1 isolated for possible independent third-scheduler
validation. Do not retrain or expand the controller on the strength of this
exploratory pilot. Industrial end-to-end reliability is not established by
these synthetic2/10-ms **queueing** budgets, periodic traces and static channels.

## Tests, runtime and resource safeguards

105 regression tests passed, with7 tests that perform DQN training deliberately
deselected. Final focused nonlearning checks passed10 tests, covering actual
saved-checkpoint observer noninterference, F/G0 pairing, partial FIFO prefix,
equality boundaries, integer/useful/cap conservation, zero-service protection,
bounded histories, failures-alone gating and state collisions. Regression also
covers existing queue/PRB/jammer/legacy paths. No old paper experiments are
claimed unchanged merely from these checks. All42 full replay outcomes and70
paired nonlearning runs passed their independent runtime assertions.

The sequential study took **537.37s (8.96min)**. CPUs **6,7,11,14** were inherited through sched_setaffinity; numerical threads1, workers1, DataLoader workers0, CUDA disabled and checked uninitialized. Sampled study peak aggregate RSS was **726.36MiB**, peak process-tree CPU **138.60%** (100%=one logical CPU). The regression peak was1525.11MiB. No memory warning/stop occurred and no test was reduced for resource reasons.

Memory is monitored, not hard-capped: no writable delegated cgroup/user systemd
memory controller is available. The50-ms RSS watchdog warns at2GiB, stops its
own process tree at2.5GiB and stops below4GiB MemAvailable; it can miss
instantaneous peaks or short-lived workers. No unrelated process or GPU setting
was changed. Exact per-command resource reports, logs, software versions and
reproduction commands are included. Six editable SVG/PDF diagnostic pairs have
captions and underlying numerical data. Large raw CSV streams are compressed
losslessly; the analysis script reads either form. The study stops here.
