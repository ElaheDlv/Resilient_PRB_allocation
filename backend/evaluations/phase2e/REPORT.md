# Phase2E independent G1 validation and efficiency decision

**Do not integrate G1 as a third scheduler. Retain P/F and the98-action interface.**

The Phase2D worst-UE gain does not reproduce at a useful magnitude independently.
Across the nine new contested scenarios, F has5.192% aggregate and41.537% mean
worst-UE failures; frozen G1 has16.510% and41.493%. Thus G1 adds11.318percentage
points of aggregate failures for a0.044pp difference in mean worst-UE failure.
Its severe contested worst-UE result is slightly worse than F. It systematically
harms previously successful UEs, adds no recovery benefit, and costs more.
The optimization is decision-equivalent, but it does not rescue this utility
trade-off or establish a complete1ms execution budget.

## Frozen algorithm and independent design

Branch `phase2e/independent-g1-efficiency`, base `fbafd09`. Exact file inventory
is in `CHANGED_FILES.md`. `PROTOCOL.md` and `protocol.json` specify the design,
proof/limits of the optimization, timing scopes and resource budget.

The original Phase2D `ExperimentalFairAllocator`, lambda=1, credit equations,
100ms service window, matured exposure20, FIFO-prefix reachability, zero-service
guard and F kernel are unchanged. P/F source functions, DQN, reward,30-feature
state and98-action representation are unchanged. The implementation remains an
optional isolated tool allocator, not a newly registered production/RIC mode.

Predeclared candidate indices15001–15024 map to actual seeds=index*4096. All384
new per-UE registration streams are disjoint from all1152 streams in the prior
C2 candidate blocks, including the future-model block bounds. No future-model
holdout manifest/scenario/outcome was evaluated. Twenty-four20ms P/zero-reserve
probes retain **only exogenous** rates, positions, CQI, periodic arrivals and
budgets. The first three original650-byte candidates satisfying the C2 feasible
certificate and first three unused contested candidates are selected in seed
order. Three remaining placements use the preexisting C2 normalized-overload
length-scaling definition at85.8fluid PRBs/tick. Only generated length-column
copies change; original traffic files, timestamps, budget assignment, radio,
jammer, reserve18 and3s horizon remain fixed.

Membership/config/trace hashes were frozen before any **independent** G1 run;
there was no G1-based selection, replacement, tuning or later extension.
`exogenous_probes.json` and `manifest_freeze.json` retain this audit.

| Regime | Placement seed | Batch bytes | Fluid PRBs/tick | One-tick integer demand |
| --- | ---: | ---: | ---: | ---: |
| feasible | 61444096 | 650 | 50.600 | 54 |
| feasible | 61489152 | 650 | 40.138 | 45 |
| feasible | 61505536 | 650 | 50.097 | 54 |
| contested | 61448192 | 650 | 56.831 | 59 |
| contested | 61460480 | 650 | 59.782 | 63 |
| contested | 61476864 | 650 | 59.888 | 65 |
| overloaded | 61452288 | 1089 | 85.769 | 90 |
| overloaded | 61456384 | 876 | 85.741 | 90 |
| overloaded | 61464576 | 816 | 85.757 | 89 |

81 paired independent episodes compare P/F/G1 on all27 entries (nine placements
x three severities), with the same frozen ReactiveReserve policy and complete
quota/exogenous/budget trajectory equality. Reference P/F receive only read-only
deficit measurement; G1 uses its original service window. Separate profiling and
equivalence work uses development inputs and is not counted as independent evidence.
Three placements per stratum remain a small model-specific study. Repeated UEs,
ticks and severities are not independent placement replicates; no confidence
interval, rare-event reliability guarantee or population superiority is claimed.

## Profiling and the exact optimization

Original G1 profiles make6588 full telemetry collections per3s episode:3000 G1
pre-allocation queries plus3588 queries in the existing reference RIC/monitoring
path. Collections rebuild16UE reports, including repeated100ms cohort processing
and pending-FIFO consistency/overdue scans. These dominate cost. In the overloaded
original profile, collection takes14.927 cumulative profiled seconds, compared
with0.100s for all24000 reachability calls and0.076s for all3000 post-service
updates. These are **instrumented cProfile** figures, not real-time measurements;
nested cumulative costs are not additive.

The sole optimization is lazy collection based on the frozen zero-credit
identity. A nonzero credit requires a full window, positive offered-minus-served
bytes and a reachable target. If no UE satisfies those prerequisites, original
credits are all0 for every possible matured-miss fraction. Skip the unused full
query and no-op fair loop, retaining the original zero-service guard and F
kernel. Otherwise call the same current-time collector and use exactly the old
arithmetic and allocation logic. No stale cache, approximate timestamps, new
coefficient, altered target or heuristic is introduced. The original algorithm
file is untouched; optimized execution is in `equivalent_g1.py`.

| Profile regime | Original collections | Optimized collections | Reduction |
| --- | ---: | ---: | ---: |
| feasible | 6588 | 3588 | 45.54% |
| contested | 6588 | 6365 | 3.38% |
| overloaded | 6588 | 3589 | 45.52% |

Repeated100ms service-window updates and FIFO target scans were investigated,
but were much smaller than telemetry cost and are preserved. The substantial
fast-path benefit occurs mostly where protection cannot act. In the contested
profile credits are active frequently, so collection falls by only3.39%.

Unprofiled original/optimized pairs on all10 Phase2D scenarios compare every
16UE downlink grant and offered/served/overflow/queue/HOL trajectory at every
service tick through SHA256 digests. They also require exact control digests,
per-UE/class/overall physical outcomes, unresolved accounting, recovery, rewards
(including the common joint monitor), causal deficit summaries and exogenous/
quota hashes. All10 pairs match for30000 ticks per implementation. Three more
original G1 replays on the new severe contested placements also match optimized
independent execution: **13 exact full-horizon pairs /39000ticks per implementation**.
500 randomized differential cases test direct integer-grant equality with partial
batches, expiry/equality, missing/warmup exposure, zero rates/quotas and caps.

Private telemetry query counts and pruning watermarks naturally differ. The
unused debug `last_protection.miss_fraction` is null when not queried; this is
explicit, not a fabricated zero failure. Scheduling, reported numerical outcomes,
rewards and physical trajectories match exactly. No claim of private-cache or
wall-time identity is made, and no behaviorally different candidate replaces G1.

Unprofiled development replay timings below average per-run p99 values within
regime, not pooled-event p99:

| Development regime | Original full-tick p99 (µs) | Optimized full-tick p99 (µs) | Original scheduler transaction p99 (µs) | Optimized transaction p99 (µs) |
| --- | ---: | ---: | ---: | ---: |
| feasible | 2613.3 | 2272.6 | 832.2 | 365.7 |
| contested | 2846.0 | 2868.1 | 964.3 | 958.7 |
| overloaded | 5814.3 | 4999.8 | 1350.3 | 474.5 |

The contested timing improvement is negligible despite unchanged outcomes;
feasible/overload fast paths improve substantially. Original-first fixed replay
order and ordinary host-load variation limit small timing comparisons. The
profiled runs are excluded from these measurements and from independent timing.

## Independent deadline results by regime and severity

Cells are **aggregate / mean worst-UE known failed offered-batch percentages**,
averaging the three placements at that regime/severity. Worst UE is selected
separately per run; individual identities are not hidden in the per-UE files.

| Regime | Jammer PRBs | P | F | G1 |
| --- | ---: | ---: | ---: | ---: |
| feasible | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| feasible | 10 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| feasible | 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| contested | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| contested | 10 | 57.665 / 63.989 | 7.756 / 62.044 | 24.208 / 61.778 |
| contested | 20 | 59.449 / 65.744 | 7.821 / 62.567 | 25.321 / 62.700 |
| overloaded | 0 | 99.317 / 99.767 | 33.101 / 99.333 | 33.101 / 99.333 |
| overloaded | 10 | 99.358 / 99.800 | 33.112 / 99.367 | 33.112 / 99.367 |
| overloaded | 20 | 99.403 / 99.800 | 37.276 / 99.433 | 37.276 / 99.433 |

Corresponding tight / relaxed deadline-class failure percentages:

| Regime | Jammer PRBs | P tight / relaxed | F tight / relaxed | G1 tight / relaxed |
| --- | ---: | ---: | ---: | ---: |
| feasible | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| feasible | 10 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| feasible | 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| contested | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| contested | 10 | 63.831 / 51.500 | 0.000 / 15.511 | 28.850 / 19.567 |
| contested | 20 | 65.631 / 53.267 | 7.878 / 7.764 | 30.214 / 20.428 |
| overloaded | 0 | 99.761 / 98.872 | 66.203 / 0.000 | 66.203 / 0.000 |
| overloaded | 10 | 99.792 / 98.925 | 66.225 / 0.000 | 66.225 / 0.000 |
| overloaded | 20 | 99.800 / 99.006 | 74.553 / 0.000 | 74.553 / 0.000 |

`scenarios.csv` reports known failures, offered exposure, upper bounds, not-yet-due
unfinished traffic and administrative terminal charges separately for every run.
`per_ue.csv` includes every UE's budget, capacity, PRBs, service/overflow/unfinished
bytes, on-time/late/overdue counts and measured100ms deficits. Raw service windows
are in `service_windows.csv.gz`. Known failures are the existing disjoint union
of overflow-affected batches, fully admitted late completions and fully admitted
overdue unfinished batches. The upper bound adds not-yet-due admitted unfinished
batches. Administrative reward closure charges are **not physical failures** and
are never added to this score. These are batch/byte queueing metrics, not packet
reliability or industrial end-to-end control deadlines.

## Individual placements, consistency and harm

All three contested placements have zero failures at j0. Placement61448192 also
has zero failures for P/F/G1 at j10/j20. It is a valid nominal/easier member of
the frozen contested region and is not replaced because its performance is zero.
The other two placements are informative and reject the development conclusion:

| Contested seed | Jammer | P aggregate / worst (%) | F aggregate / worst (%) | G1 aggregate / worst (%) |
| --- | ---: | ---: | ---: | ---: |
| 61448192 | 10 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 61448192 | 20 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 61460480 | 10 | 85.737 / 95.833 | 11.567 / 92.533 | 35.404 / 91.700 |
| 61460480 | 20 | 88.517 / 98.533 | 11.646 / 93.167 | 37.846 / 93.233 |
| 61476864 | 10 | 87.258 / 96.133 | 11.700 / 93.600 | 37.221 / 93.633 |
| 61476864 | 20 | 89.829 / 98.700 | 11.817 / 94.533 | 38.117 / 94.867 |

Against F at j10, G1 improves worst UE in1/3 placements by0.833pp, worsens one by
0.033pp and ties the zero-failure placement. At j20 it improves **none**, worsens
both informative placements by0.067/0.333pp and ties the easier placement.
Maximum paired aggregate harm is26.300pp. G1 improves on P in the two hard
placements, but F already delivers much lower aggregate failure at similar
worst-UE failure, so that does not justify a third scheduler.

Compared with F, G1 harms29/72 contested UE-scenario cases, benefits3 and ties40
(29 harms among48 jammed cases). Compared with P, it benefits32 and harms0 of72
on the deadline metric, but costs much more; this is not an all-objective Pareto
improvement. Tables count repeated UE/scenario observations, not independent
subjects. No global Pareto claim is made.

Concrete severe placement61460480 illustrates **victim migration**. F leaves
relaxed UE0 at93.167% failure (mean deficit0.3004, served/offered70.14%); other
UEs have zero failures. G1 reduces UE0 to35.267% and serves all its offered bytes,
but tight UE1 moves from zero to93.233% failure (mean deficit0.2175,
served/offered78.38%). Six other previously successful UEs acquire27.87–30.43%
failures even though all their offered bytes are served by the horizon. Matching
long-window service rates therefore does not ensure timely batch completion.
This is direct paired-policy evidence with identical quotas/radio/traffic; it
does not isolate which individual credit term causes the redistribution.

Feasible scenarios have zero failures for all policies, with G1/F identical
allocation hashes in all9 cases. Under physical overload, G1 and F have the same
deadline failure outcomes in all9 cases;8/9 allocation hashes also match (equal
failure rates do not imply identical grants). The C2 full-pool bound still proves
zero misses impossible for these overload workloads. In contested cases the
full physical reserve could in principle support the empty-queue constructive
quota, but the fixed reactive split can give insufficient URLLC service; scheduler
comparisons keep that split identical. No bound is used to claim the observed
miss rate or victim identity is unavoidable/optimal.

No recovery benefit reproduces: P/F/G1 each recover6/6 feasible jammer-OFF
windows,2/6 contested windows and0/6 overloaded windows. No-jamming recovery is
undefined (no OFF event). Recovery remains a consecutive10ms all-UE HOL-compliant
hold, not a guarantee of timely completion for every offered batch.
`recovery_windows.csv` preserves times and censoring. eMBB throughput and reserve
usage are exactly paired; separate regime/severity values are in `summary.csv`.
There is no resource or eMBB gain compensating the F-relative deadline harm.

## Actual per-tick computation, not just allocator timing

The wall timer encloses the entire `engine.step(.001)`: traffic/radio/queue
simulation, both slices' scheduling/service, ordinary metrics/telemetry,
reference RIC/reward accounting and mandatory G1 post-service updates. The raw
CPU timer encloses the same scope. Setup/reset and external audit hashing/CSV
export are excluded. Baseline P/F have an extra read-only service window for
comparable deficit measurements. This is the measured complete simulator path,
not a production radio implementation; baseline full-tick cost is not P/F's
standalone scheduling cost. All-tick values include episode-finalization work;
nonterminal, RIC-boundary and ordinary-tick summaries are separately retained.

Table entries average per-run quantiles over9 scenarios/regime; p99 values from
different scopes must **not** be added. Full per-placement/severity p95/p99/max
and >1ms counts are in `scenarios.csv` and `tick_times.csv.gz`.

| Regime | Policy | Allocator p95 / p99 (µs) | Allocator+G update p99 (µs) | Complete tick p95 / p99 (µs) | Complete ticks >1ms (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| feasible | P | 7.4 / 10.1 | 10.1 | 1799.1 / 1882.7 | 10.84 |
| feasible | F | 201.4 / 249.3 | 249.3 | 1987.2 / 2072.6 | 12.23 |
| feasible | G1 | 213.4 / 258.5 | 265.6 | 2011.8 / 2112.1 | 12.43 |
| contested | P | 9.5 / 15.4 | 15.4 | 1847.9 / 1937.3 | 11.09 |
| contested | F | 269.6 / 314.0 | 314.0 | 2179.9 / 2327.4 | 53.99 |
| contested | G1 | 467.6 / 567.7 | 574.7 | 2271.1 / 2393.9 | 64.49 |
| overloaded | P | 10.7 / 17.1 | 17.1 | 3652.0 / 4969.5 | 90.07 |
| overloaded | F | 267.4 / 369.7 | 369.7 | 3819.5 / 5170.3 | 97.54 |
| overloaded | G1 | 291.7 / 534.8 | 542.8 | 3848.5 / 5217.5 | 97.80 |

Optimized G1 transaction p99 ranges234.8–1207.5µs across independent scenarios;
one of27 cases exceeds1ms even for this narrower scheduler-plus-update scope.
In overloaded seed61464576 j10 it exceeds1ms on79/3000 ticks, with p99=1207.5µs
and max1618.2µs. Its complete-tick p99 is5997.4µs; nonterminal p99 is the same.
The issue is not merely a terminal reporting spike. Process-CPU p99 is also
high (e.g. mean2322µs in contested G1), so OS preemption alone does not explain
full-loop budget failures. The existing reference monitor still makes thousands
of full100ms cohort queries after optimization, especially costly with growing
pending queues. Changing or approximating that reward/observation path was outside
scope and was not silently done to improve timing.

G1 is **not demonstrated feasible for a complete1ms loop in this implementation**.
Most local transaction p99 values improve below1ms, but that is not a hard
real-time guarantee; full-tick p99 fails in all27 independent G1 cases. P/F's
instrumented full path also misses the benchmark. Affinity is not exclusive CPU
isolation, wall times depend on host load, and three placements are not a
worst-case execution-time proof. A production implementation would need its own
complete-path budget/profile; this report does not certify any scheduler for
industrial deployment. Retaining P/F is a research/interface recommendation.

## Decision against the five criteria

1. **Reproducible worst-UE benefit?** Not relative to F at a useful magnitude.
   Moderate improvements are tiny/inconsistent and severe cases are worse.
2. **Acceptable aggregate loss?** No defensible justification is supplied for
   an11.318pp mean contested loss or up to26.300pp case loss for that negligible
   worst-UE benefit. No application-supported universal SLA threshold was invented;
   these observed losses are not endorsed as acceptable.
3. **Systematic individual harm?** Yes,29/72 contested F-relative UE cases,
   including migration of a near-total-failure victim to a previously successful
   tight-budget UE and widespread new lateness.
4. **Computational practicality?** The exact optimization removes wasted work,
   especially when credits cannot act. Active contested fairness remains costly,
   and the measured complete path does not meet1ms. Standalone allocator timing
   would have hidden that fact.
5. **Third-mode integration?** No. Keep G1 as a documented negative/conditional
   experimental comparator, retain P/F and the98-action interface, and do not
   proceed to retraining, parameter tuning or action-space expansion from these
   data. This is a bounded independent study, not proof that every fairness
   algorithm must fail or that the existing F has adequate per-UE reliability.

## Validation, resources and reproduction

110 regression tests pass;7 training tests are deliberately deselected. Five new
nonlearning tests cover frozen sources/disjoint streams, quantile semantics,
500 differential allocation cases, the exact telemetry-skip condition and real
saved-physics allocation/service equivalence with complete tick timing. All13
full-horizon original/optimized comparisons pass, all81 independent runs share
paired conditions/quota trajectories, and measured radio rates match calibration.
No neural model was constructed in the study stages; regression includes existing
nontraining schema/checkpoint-forward checks. No DQN update occurred and no old
paper results are claimed revalidated. Six SVG/PDF figure pairs retain captions/numeric data.

The independent command (81 evaluations +3 original checks) took **365.95s (6.10min)**. The five guarded study stages together took **592.81s (9.88min)**, excluding tests/reporting. All110 predeclared full episodes and24 short probes completed; the budget was not extended. Independent sampled peak process-tree RSS was **823.18MiB**, peak CPU **138.81%** (100%=one logical CPU). Regression peak was **1423.55MiB**.

CPUs6,7,11,14 are enforced by inherited sched_setaffinity; CPU-only execution,
one numerical-library thread, one sequential worker, no DataLoader workers and
CUDA disabled/uninitialized. No memory warning/stop triggered and no test was
reduced for resource limits. Memory is monitored, **not hard-capped**: no writable
delegated cgroup/user systemd memory controller is available. The50ms process-tree
RSS watchdog warns at2GiB, stops its own group at2.5GiB and below4GiB system
MemAvailable; instantaneous peaks/short-lived workers can be missed. No unrelated
training process or system/GPU setting was changed. Exact resource reports,
software versions, hashes, file inventory and reproduction commands are included.
The study is complete and stops here.
