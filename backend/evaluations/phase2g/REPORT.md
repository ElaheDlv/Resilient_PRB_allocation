# Phase 2G: controlled joint-DQN reward ablation

**Decision: retain R0 as the reference reward. R2 is a promising development
candidate, but this bounded experiment does not justify replacing R0.**
R2 improves both final checkpoints and reduces the selected two-seed mean failure
rate, yet its selected aggregate benefit is almost zero in one seed, overloaded
performance worsens in that seed, and one UE suffers a95.13-point increase.
R2 also retains a constant fairness signal in persistent overload. R1 is less
consistent and introduces even larger individual harms. This is not evidence
that historical fairness is universally necessary, or that R0 itself satisfies
industrial requirements; it is insufficient evidence for a safe reward replacement.

Branch `phase2g/controlled-reward-ablation`, base `2c8c438`.
See [exact equations/implementation](METHODS.md), [frozen protocol](protocol.json),
[prospective analysis criteria](ANALYSIS_PLAN.md), [figures](figures/CAPTIONS.md),
[reproduction/data dictionary](README.md), and [file inventory](CHANGED_FILES.md).
The Phase2F prospective plan was followed without coefficient tuning or new arms.

## Implementation and pretraining validation

Only new optional modules were added. Original reward, DQN updates/architecture,
state encoding,98 actions/masks, P/F schedulers, physical reserve18, traffic,
channel and jammer implementations remain unchanged. The arms are:

- R0: original `joint-reward-v1`, including ten tick-level historical worst-risk costs.
- R1: remove only that historical term; version `joint-reward-no-history-v1`.
- R2: replace it with `.25*max_u(interval_new_physical_failures_u/10)`;
  version `joint-reward-interval-events-v1`. Charge at the tenth observed service
  tick. Do not clip, and exclude terminal administrative charges.

The original exactly-once ledger supplies per-UE deltas through a subclass;
scalar count conservation is asserted at every consume. Counter resets preserve
unpaid initialization, interval attribution and fresh episode state. Checkpoint
metadata records arm, equation, version, schema/settings and protocol hash;
cross-arm loads are rejected. No new observation input is introduced.

**10 new correctness tests passed; the pretraining nontraining regression passed
126 tests,7 deselected in88.26s.** Tests cover overflow/expiry/late-completion union,
per-UE conservation, partial service, exact deadline equality, unresolved terminal
traffic, reset/bootstrap isolation, causal boundaries, no future service access,
R0 numerical identity and unclipped R2 normalization. A synthetic20-failure UE
costs.5, while alternating two UEs correctly uses the maximum of interval sums.
Real frozen32-interval comparisons produce identical controls and physical UE
outcomes across arms, with R0 reward exactly matching the original app.

Initial test-harness failures (float equality and fresh-subprocess settings/import
setup) were corrected before training; attempts are archived. The final protocol
hash and passing gate were verified before any learning. Seven existing tests
that perform learning were excluded to avoid training outside the six prescribed
learners; no resource failure caused that exclusion.

## Exact bounded experiment and reproducibility

Six sequential learners: three arms ×NN seeds12026/22026. Within each seed,
policy, target, empty Adam state and action/replay RNG are copied from one saved
initialization. All have identical original architecture, optimizer, epsilon,
replay and target-update settings; the exact C3 nine-episode scenario order is
preserved. Every learner completes2691 transitions and2660 optimizer updates.
Checkpoints at897/1794/2691 interactions each receive all18 C2 validation entries.
Totals:54 training episodes,18 checkpoints,324 greedy validation episodes,
15,960 optimizer updates and113,022 paid10ms interval records.

All validation runs preserve parameters, RNG, replay and learning counters. All
paired exogenous/deadline hashes match. All intervals and physical per-UE outcome
partitions reconcile; terminal charges equal unresolved-not-due traffic and never
enter R2 event counters. Exact primary and secondary selections were recomputed
from original Fraction-based physical keys. See `result_integrity.json`.

**R0 matches all108 historical C3 A4 validation comparisons** in policy parameters,
controls, original reward and physical per-UE outcomes. This verifies historical
reproducibility for those quantities; measured CPU timing is not asserted identical.
All18 checkpoint schemas/update counts and324 evaluated fingerprints pass audit.
No future-model holdout was loaded or evaluated, no budget extended, and no
additional reward arm or hyperparameter search was run.

## Primary selected-policy results

Aggregate-first selection is the unchanged C2 hierarchy in every arm. Percentages
use offered FIFO batches; these are not packet-level reliability. Known failures
include the exactly-once overflow/late/overdue union. Unresolved-not-due batches
are separate; on-time completion is100% minus the upper failure bound. Mean worst
UE is the mean of each scenario's worst-UE fraction, not the worst pooled UE.
Each seed averages the same18 cases; the two-seed mean is descriptive.

| NN seed | Arm | Selected episode | Known failures % | Mean worst UE % | On-time completions % | Reserve PRBs | eMBB Mbit/s | Switches/3s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12026 | R0 | 6 | 13.215 | 44.261 | 86.749 | 11.025 | 92.248 | 1.000 |
| 12026 | R1 | 6 | 14.155 | 32.831 | 85.810 | 11.897 | 90.122 | 110.611 |
| 12026 | R2 | 9 | 13.200 | 33.263 | 86.788 | 9.746 | 90.006 | 44.889 |
| 22026 | R0 | 9 | 21.596 | 38.765 | 78.364 | 9.427 | 90.219 | 172.667 |
| 22026 | R1 | 9 | 15.322 | 33.759 | 84.646 | 11.157 | 90.314 | 125.889 |
| 22026 | R2 | 9 | 13.405 | 33.128 | 86.560 | 13.919 | 90.330 | 101.833 |
| mean | R0 | — | 17.406 | 41.513 | 82.557 | 10.226 | 91.233 | 86.833 |
| mean | R1 | — | 14.738 | 33.295 | 85.228 | 11.527 | 90.218 | 118.250 |
| mean | R2 | — | 13.302 | 33.195 | 86.674 | 11.832 | 90.168 | 73.361 |

R1 changes aggregate failures by **+0.940 /−6.275 points** (seeds12026/22026).
It does not consistently improve the primary objective. R2 changes them by
**−0.0155 /−8.1912 points**. Its two-seed average improves4.1034 points, but almost
all of that selected-policy gain comes from the second initialization. R2's
on-time completion changes are+0.0389/+8.1961 points, so pending traffic does not
explain the reported aggregate gains. All unresolved/upper rates are retained
in `validation_metrics.csv` and `selected_summary.csv`.

R2 lowers mean worst-UE failure by10.998/5.637 points, largely through contested
cases. This average does not protect every UE or remedy nearly100% worst-UE
failure in overloaded workloads.

## Load and jamming strata

The primary choices above are fixed, then stratified without scenario replacement:

| Seed | Regime | R0 failures % | R1 failures % | R2 failures % | R0 worst % | R1 worst % | R2 worst % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12026 | feasible | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12026 | contested | 4.125 | 0.000 | 0.001 | 33.000 | 0.000 | 0.006 |
| 12026 | overloaded | 35.520 | 42.464 | 39.598 | 99.783 | 98.494 | 99.783 |
| 22026 | feasible | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 22026 | contested | 3.722 | 0.869 | 0.074 | 16.511 | 2.456 | 0.422 |
| 22026 | overloaded | 61.067 | 45.096 | 40.141 | 99.783 | 98.822 | 98.961 |
| mean | feasible | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mean | contested | 3.924 | 0.435 | 0.037 | 24.756 | 1.228 | 0.214 |
| mean | overloaded | 48.293 | 43.780 | 39.869 | 99.783 | 98.658 | 99.372 |

R2 reduces contested failures in both seeds. In overload, it **worsens seed12026
by4.078 points**, while improving seed22026 by20.926 points. Every overloaded
severity is worse for the first seed. The almost-zero overall gain there is
cancellation between contested improvement and overload loss, not robust benefit
across operating regions. Feasible cases already have zero failures for every arm.

Two-seed descriptive means by jamming severity:

| Regime | Jam PRBs | R0 failures % | R1 failures % | R2 failures % |
| --- | ---: | ---: | ---: | ---: |
| feasible | 0 | 0.000 | 0.000 | 0.000 |
| feasible | 10 | 0.000 | 0.000 | 0.000 |
| feasible | 20 | 0.000 | 0.000 | 0.000 |
| contested | 0 | 0.000 | 0.000 | 0.000 |
| contested | 10 | 2.782 | 0.045 | 0.000 |
| contested | 20 | 8.989 | 1.259 | 0.112 |
| overloaded | 0 | 46.219 | 41.375 | 38.489 |
| overloaded | 10 | 49.393 | 43.603 | 38.808 |
| overloaded | 20 | 49.269 | 46.361 | 42.311 |

Each stratum has two placements per NN seed. Both seeds' individual stratum
values are in `selected_summary.csv` and Figure09; the pooled means must not hide
the opposite overload directions. No significance or population generalization
is inferred from these dependent development scenarios.

## Learning progress at matched interactions

Figures01–04 use the same fixed validation18 scenarios at every checkpoint.
Exploratory training returns across changing loads are not convergence evidence.
R2's aggregate failures at the **final matched2691-interaction checkpoint** are
13.200%/13.405%, versus R0's20.882%/21.596%: improvements in both seeds. That is
real positive evidence within this pilot. However, R0's best earlier checkpoint
for seed12026 already achieved13.215%, making the prespecified selected-policy
comparison almost tied. Training maxima/budgets remain identical despite
selection of different earlier checkpoints.

Learning is nonmonotonic. R2 seed12026 failures move26.161→28.720→13.200%;
seed22026 moves35.564→22.835→13.405%. Its seed22026 middle checkpoint has mean
worst-UE failure53.739%, above R0's45.072%. These short curves do not establish
convergence or uniformly improved sample efficiency.

## Individual UEs and deadline classes

Across144 matched UE/scenario pairs per seed in the primary selection:

| Arm | Seed | UE/scenario failure increases | Decreases | Largest increase (points) |
| --- | ---: | ---: | ---: | ---: |
| R1 |12026|29|13|98.367|
| R1 |22026|8|43|4.767|
| R2 |12026|8|3|95.133|
| R2 |22026|0|46|0.000|

The other pairs tie in known failure rate. These counts are dependent observations,
not independent trials or statements about all other service metrics. All signs,
including gains, unresolved changes, served bytes and HOL changes are archived
in `per_ue_paired_differences.csv`; no harmful rows were removed.

Concrete R2 harm: seed12026, overloaded placement53252096, **no jamming**, UE5
(tight budget) rises from **2.900% to98.033% known failures** (87→2941 of3000
batches), loses684,253 served bytes, and has3 extra unresolved-not-due batches.
Its HOL violation frequency rises95.167 points. This compares the prespecified
R0 episode6 and R2 episode9 choices. Overloaded worst-UE metrics barely change
because another UE already fails almost completely: a maximum statistic can hide
which UE is harmed. This does not prove either that the harm was unavoidable
or that it was caused by one particular scheduler action. The harm also persists
at **matched final2691-interaction checkpoints**: this same UE rises from.833%
under final R0 to98.033% under final R2 (**+97.200 points**). Five first-seed
UE/scenario pairs worsen at that matched endpoint; none worsen in seed22026.
Thus the individual-harm concern is not only a result of comparing different
selected checkpoint episodes. The full checkpoint-level UE records are in
`per_ue_validation.csv`.

Concrete R1 harm: seed12026, overloaded placement53252096, severe j20, relaxed
UE0 rises0→98.367% known failures (0→2951/3000), with11 extra pending batches.
R0 and R1 both use episode6 in this comparison. A lower mean worst-UE statistic
therefore does not establish individual fairness or Pareto improvement.

| Seed | Arm | Tight-class failures % | Relaxed-class failures % |
| --- | --- | ---: | ---: |
| 12026 | R0 | 24.995 | 1.435 |
| 12026 | R1 | 12.051 | 16.258 |
| 12026 | R2 | 26.399 | 0.000 |
| 22026 | R0 | 26.273 | 16.920 |
| 22026 | R1 | 17.723 | 12.920 |
| 22026 | R2 | 10.208 | 16.602 |
| mean | R0 | 25.634 | 9.178 |
| mean | R1 | 14.887 | 14.589 |
| mean | R2 | 18.304 | 8.301 |

R2's mean tight-class improvement comes from the second seed; tight-class failure
is actually worse in the first seed. R1 shifts substantial loss toward relaxed
traffic there. No application-specific acceptable harm threshold was supplied;
these large changes cannot be declared acceptable by aggregate optimization alone.

## Resources, eMBB, switching and recovery trade-offs

For R2 relative to R0, mean reserve changes **−1.279/+4.492 PRBs** by seed;
eMBB throughput changes **−2.241/+0.111 Mbit/s**. R1 uses+.871/+1.730 PRBs,
with eMBB−2.126/+.095 Mbit/s. The strong second-seed R2 improvement is accompanied
by substantially more activated reserve; the first-seed near-tie loses eMBB.
These are observed trade-offs, not isolated causal mediation estimates. Activated
reserve is not a counterfactual estimate of PRBs needed for successful service.

R2 mode switches change1→44.889 and172.667→101.833 per3s episode. It is **not
consistently more stable**: switching improves in one seed and worsens in the
other. R1 increases the first seed's switching even more. Figure05 and the CSVs
retain occupancy and switching separately from reserve-command adjustments.

Under the existing10ms consecutive all-UE HOL-compliance recovery definition,
primary R0 recovers13/24 jammer-OFF windows across both seeds, versus16/24 for
R1 and R2. Feasible recovery is8/8 for all; contested recovery5/8→8/8; overloaded
recovery remains **0/8 for every arm**. Recovered windows have zero onset delay
(the required compliance hold is still10ms); unrecovered windows are censored,
not assigned zero recovery time. This does not assert timely completion of every
batch following jammer-OFF.

## Fairness feedback and numerical stability

R2 does remove historical carry from its fairness charge: zero new failures gives
zero event fairness, regardless of old risk. Original-reference accounting on the
selected R2 contested trajectories has10/9 intervals with historical cost but no
new events, which R2 does not penalize through its fairness term.

However, **every selected R2 overloaded interval costs exactly.25 in fairness
shaping**, in both seeds: maximum10 newly known failures/UE, interval standard
deviation0, and zero fraction0. The historical worst risk on those trajectories
is also saturated for99.879% of valid ticks. R2 has no explicit cap; the workload
and maximum statistic make its signal constant. Its overloaded fairness cost is
74.75 over299 intervals, versus approximately72.52 historical cost in R0, partly
due to R0's100ms validity warmup. Removing history does not repair the maximum
statistic's lack of discrimination while one UE keeps failing ten batches.
The direct aggregate failure component still changes and remains informative.
See `interval_signal_diagnostics.csv` and Figure06.

These observations establish event alignment and a structural limitation, not a
claim that the constant term caused the individual harm. Physical events also
remain delayed by budget expiry (3/11ms first-known samples here), and R2 changes
warmup and target timing as well as historical persistence. The experiment cannot
separate all those mechanisms or infer unseen alternative service outcomes.

All15,960 optimizer diagnostic records are finite. Maximum replay-minibatch |Q|
values are R0:10.353/11.341; R1:6.966/7.878; R2:11.161/11.503 by seed. Maximum
Huber losses are R0:.335/.439; R1:.178/.279; R2:.434/.368. No numerical divergence
was observed within the budget. Figure07 nevertheless shows growing TD/Q scales
near the end; no convergence or long-run stability claim follows. R1's lower
loss/Q magnitude is expected under a smaller reward scale and is not evidence
of better physical learning.

## Secondary worst-first selection

The unchanged secondary hierarchy is applied to every arm independently:

| Seed | Arm | Worst-first episode | Known failures % | Mean worst UE % |
| --- | --- | ---: | ---: | ---: |
| 12026 | R0 | 9 | 20.882 | 35.728 |
| 12026 | R1 | 6 | 14.155 | 32.831 |
| 12026 | R2 | 3 | 26.161 | 33.261 |
| 22026 | R0 | 3 | 24.589 | 36.911 |
| 22026 | R1 | 9 | 15.322 | 33.759 |
| 22026 | R2 | 9 | 13.405 | 33.128 |

R2 seed12026 chooses episode3 because worst-UE failure33.26111% is slightly below
episode9's33.26296%, despite aggregate failures26.161% versus13.200%. Relative
to the worst-first R0 choice, aggregate failures are worse in that seed. This
exposes the sensitivity of a strict lexicographic selector; the criterion was
not changed after outcomes. Primary and secondary results must not be mixed to
create a favorable headline.

## Execution resources and stopping point

The six learners plus validations took **1803.74s (30.06min)** application time,
1805.68s including guard/startup. Mean simulation-loop times are5.09s per training
episode and4.54s per validation episode; setup/serialization overhead is included
in the experiment total, not those loop-only measurements.

All workloads were sequential, affinity-bound to logical CPUs6,7,11,14; numerical
threads1, DataLoader workers0, CPU-selected PyTorch, CUDA disabled/uninitialized.
Experiment sampled peak process-tree CPU was138.80% (100%=one logical core),
aggregate RSS **0.716GiB**. The nontraining regression peak was **1.500GiB**.
No2GiB warning or2.5GiB memory stop occurred. The guard monitored aggregate RSS
and system available RAM; **memory was not hard-capped** because no writable
cgroup was available. CPU affinity is enforced; sampled CPU/RSS peaks can miss
short transients. No unrelated job or system configuration was modified.
See `environment.json` and all archived resource reports.

Retain R0 as the reproducible reference, without treating it as an industrial
QoS solution under overload. R2 merits further *consideration*, supported by its
matched final-checkpoint gains and contested-load results, but not integration
on this evidence: selected improvement is concentrated in one seed, one overloaded
UE suffers95 points of new loss, reserve/eMBB trade-offs remain, switching is mixed,
and the fairness signal is still constant under overload. R1 does not justify
removing fairness. No coefficients were tuned, no extra arms/models trained, and
no future holdout evaluated. **This bounded experiment is complete; no subsequent
training or control development was started.**
