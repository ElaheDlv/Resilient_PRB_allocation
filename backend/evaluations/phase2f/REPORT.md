# Phase 2F: joint reward alignment audit

**Recommendation: retain the original reward as the reference; investigate its
historical worst-UE term with a small matched ablation before revising it.**
The arithmetic and action/reward accounting pass. Historical persistence and
terminal concentration are real limitations, but this audit does not establish
that either caused the joint policy's inferior learning. No training, alternative
reward evaluation, weight changes, DQN edits or action-space changes were performed.

Branch `phase2f/joint-reward-alignment-audit`, base `2c31c96`.
See [mathematics](MATHEMATICAL_AUDIT.md), [frozen protocol](protocol.json),
[analysis plan](ANALYSIS_PLAN.md), and [prospective study](ABLATION_PLAN.md).

## Coverage and missing information

All **72 training and 432 validation episode records** were audited. Four episode
component totals are identifiable: newly known failures, administrative terminal
charges, reserve activation and scheduler switching. Their subtraction from the
recorded cost identifies the **sum** of worst-UE and eMBB costs. Those two cannot
be separated in failing episodes without interval service/risk histories. They
are left null with explicit bounds, not estimated from whole-episode throughput.
No-failure episodes identify both exactly because historical failure risk is zero.

Evolving training actions, exploration/replay RNG history and continuation states
were not saved. Therefore complete training interval decompositions are unavailable
without prohibited retraining. This is a substantive limit on the requested audit.

Only **18 frozen-checkpoint validation replays** were run, selected in the frozen
protocol before replay. They provide **53,820 paid ticks / 5,382 transitions**.
All original controls, returns, per-UE/class physical outcomes and resources match
C3; all six components sum to the original rewards. Parameters, RNG fingerprints
and update counters remain unchanged; replay insertion and learning are forbidden.
The nine primary grid cases use A4, NN seed12026, selected episode6, one existing
placement per regime, jamming0/10/20. Additional seed, method and checkpoint
comparisons are reported separately. This is a diagnostic sample, not an independent
performance experiment or a reconstruction of all training trajectories. No
future-model holdout was evaluated.

## Mathematical scale

For each1ms tick, the original cost is

`new_failures/80 + terminal_charges/80 + .025*valid_worst_risk`
`+ .01*max(0,1-eMBB_bytes/12000) + .002*reserve_PRBs/18 + .01*mode_switch`.

Reward is its negative; ten ticks form one10ms transition. A mode switch is charged
once after action application. Reserve-command changes alone do not incur that
switch cost. The first10 initialization ticks are outside learned reward, while
physical episode measurements include them. This scope difference is explicit,
not evidence of a transition-timing error.

One new failure costs.0125; one mode switch costs.01, equivalent to.8 new failures
or9 PRBs held for10ms. Full reserve costs.02 per10ms, requiring1.6 fewer expected
new failures to repay **if all other components were equal**. These are algebraic
exchange rates, not measured counterfactual service benefits. Saturated worst
risk costs.25 per interval: twice the direct cost of one UE failing10 batches,
but one quarter of all eight UEs failing80 batches. Fairness pressure is therefore
stronger when failures are concentrated. The failure denominator is a fixed
reference exposure, not a contemporaneous failure-probability estimate.

## Measured component contributions

Original positive costs summed over each3s primary grid replay:

| Regime | Jamming | New failures | Worst risk | eMBB | Reserve | Switch | Terminal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feasible | 0 | 0.0000 | 0.0000 | 0.3852 | 4.4289 | 0.0100 | 0.0000 |
| feasible | 10 | 0.0000 | 0.0000 | 1.4129 | 3.6400 | 0.0100 | 0.0000 |
| feasible | 20 | 0.0000 | 0.0000 | 1.8175 | 4.1833 | 0.0100 | 0.0000 |
| contested | 0 | 0.0000 | 0.0000 | 1.7369 | 3.9856 | 0.0100 | 0.4250 |
| contested | 10 | 33.3875 | 65.6750 | 3.2263 | 3.4111 | 0.0100 | 0.1375 |
| contested | 20 | 35.5000 | 69.8000 | 4.2783 | 2.9800 | 0.0100 | 0.0375 |
| overloaded | 0 | 75.9250 | 72.5224 | 0.0000 | 3.0000 | 0.0100 | 0.0750 |
| overloaded | 10 | 112.4125 | 72.5235 | 0.0000 | 3.0000 | 0.0100 | 0.1125 |
| overloaded | 20 | 112.6875 | 72.5235 | 0.0000 | 3.0000 | 0.0100 | 0.1125 |

All training/validation saved-total distributions, including the unresolved
worst+eMBB residual, are in `saved_component_distributions.csv`. Full six-component
interval distributions for each declared replay role are in
`interval_component_distributions.csv`. Primary grid10ms distributions:

| Regime | Component | Median | p95 | Maximum | Cost share |
| --- | --- | ---: | ---: | ---: | ---: |
| feasible | failure | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| feasible | worst | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| feasible | embb | 0.0026 | 0.0118 | 0.0166 | 22.74% |
| feasible | reserve | 0.0144 | 0.0200 | 0.0200 | 77.07% |
| feasible | switch | 0.0000 | 0.0000 | 0.0100 | 0.19% |
| feasible | terminal | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| contested | failure | 0.1250 | 0.1250 | 0.1250 | 30.67% |
| contested | worst | 0.2500 | 0.2500 | 0.2500 | 60.32% |
| contested | embb | 0.0099 | 0.0210 | 0.0211 | 4.11% |
| contested | reserve | 0.0167 | 0.0200 | 0.0200 | 4.62% |
| contested | switch | 0.0000 | 0.0000 | 0.0100 | 0.01% |
| contested | terminal | 0.0000 | 0.0000 | 0.4250 | 0.27% |
| overloaded | failure | 0.3750 | 0.3750 | 0.5000 | 57.02% |
| overloaded | worst | 0.2500 | 0.2500 | 0.2500 | 41.21% |
| overloaded | embb | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| overloaded | reserve | 0.0200 | 0.0200 | 0.0200 | 1.70% |
| overloaded | switch | 0.0000 | 0.0000 | 0.0100 | 0.01% |
| overloaded | terminal | 0.0000 | 0.0000 | 0.1125 | 0.06% |

These quantiles pool three severities within a regime and are descriptive; the
897 intervals per regime are correlated, not897 independent experimental units.
Feasible-grid cost is mostly reserve activation; contested-grid cost is mostly
historical worst risk; overloaded-grid direct events dominate but historical
risk remains substantial. eMBB deficit is zero in these overloaded grid cases,
not necessarily in every alternative workload. A zero clipped deficit means the
per-tick target was met; throughput above it earns no extra credit.

## Historical saturation, persistence and credit ambiguity

In contested primary cases, worst risk is saturated for **89.04% (j10)** and
**94.59% (j20)** of valid ticks; uninterrupted saturated runs last2583/2744ms.
Overloaded primary cases are saturated for **99.86–99.90%** of valid ticks,
with runs2897–2898ms. Risk is valid on2901 of3000 physical ticks. This is a
maximum matured-cohort failure fraction reaching1, not a numerical clipping bug.
Once one UE's cohorts all fail, improvements in other UEs need not move this term.
The direct new-failure term remains informative under that saturation.

Across the18 replays, **21 intervals** have historical cost but no new failures;
**14** also complete at least as many on-time batches as newly offered. A
continuous favorable-service/history-positive run lasts **100ms**. The concrete
example in `strong_history_carry_examples.csv` is A4, NN22026, checkpoint3,
contested placement53260288, j20, interval1610–1620ms:

- No newly known failures;95 on-time completions against80 new batches.
- URLLC serves58,601bytes against52,000 arrivingbytes.
- Selected worst UE `URLLC_1` ends with zero queue, zero HOL wait and zero100ms
  service deficit, but historical failed/matured risk is.85567.
- Historical cost.225515; eMBB.006402; reserve.020000; switch.010000;
  total.261917. History supplies86.1% of that interval's cost.
- Its83 failed historical cohorts include43 arriving under the other scheduler
  mode; the oldest failed arrival is99ms old. Arrival mode does not identify
  which action caused failure. Worst-UE identity ties are possible.

Figure03 and adjacent interval/per-UE tables show the actual trajectory. Current
service cannot retroactively repair a late batch. The historical offset can be
largely determined before a current action; a negative absolute reward does not
by itself show that the better current action receives a worse action advantage.
This is demonstrated lag and limited immediate discrimination, not proof that
removing the fairness objective would improve learning.

First-known failure lag is3ms for2ms budgets and11ms for10ms budgets in these
replays, consistent with strict `delay > budget` on1ms samples. In the primary
grid,30.02% of tight-budget and100% of relaxed-budget failure events have a
later decision interval at charge than at arrival (even if the action value repeats). In severe second-seed cases,
13.08%/45.22% have a different scheduler mode. See `failure_credit_timing.csv`.
A new decision interval does not necessarily mean a different action value.
These are measured event lags, not incorrect transition storage or causal blame.

## Switching costs

Switch cost is small relative to saturated overload penalties, but **not globally
negligible**. Across all saved validation trajectories, it supplies10.84% of
feasible cost,0.334% of contested cost and0.183% of overloaded cost. During
training these shares are14.61%,6.89% and0.204%. These are ratios of summed costs,
not means of episode percentages; zero-cost episodes are handled explicitly.

Among A4 validation episodes, mean switch costs are.968/.786/1.799 for
feasible/contested/overloaded, with maxima2.52/2.35/2.99. The primary NN12026 grid
switches only once, so its.01 episode cost must not be generalized. Figure04 uses
all saved A3/A4 validation switch totals to show this heterogeneity.
`switch_cost_associations.csv` separately describes switched/stayed intervals.
Their service/failure differences are observational and confounded by state,
checkpoint and workload; no switch-weight causal effect is inferred. The fixed
exchange rate supports material influence near a tie, not proof of suppressed
beneficial switches.

## Reserve and observed service

Two concrete findings distinguish commands from outcomes:

1. Contested j20 primary replay at171ms: action1, deltas(-9,-9), zero reserve,
   URLLC quota54, one newly known failure,4,770bytes served versus5,200 arriving.
   Across that replay,1420 paid ticks combine new failures with unused pool
   headroom;2219 combine failures with URLLC quota below the existing certificate
   for draining fresh demand even though full URLLC reserve would fit that
   certificate. This identifies low activation alongside losses. It does **not**
   establish that those old failed batches were savable at that instant, nor that
   reallocating eMBB reserve would improve the complete objective.
2. Feasible j0 primary interval120–130ms: action85, deltas(+9,-9),160 reserve
   PRB-ticks activated,10 unused by the actual allocations;80 timely completions,
   no failures, and eMBB deficit zero. The interval pays.017778 reserve cost.
   Over this replay23.76% of activated PRB-time is unused by actual grants;
  1440 individual ticks have unused activation, no fresh failures and no eMBB
   target deficit. The second-seed feasible severe replay has84.04% unused
   activation. This is measured idle activated capacity, not inferred service
   under a zero-reserve action.

The primary overloaded cases waste50% of activated PRB-time in the observed
allocations while still failing traffic: extra service in one slice cannot
repair an infeasible batch in another. Figure05 and per-tick records retain both
facts. Calling every such activation 'without benefit' would require an unobserved
paired intervention; only the unused portion is demonstrated here.

## Alignment with physical objectives versus inadequate learning

All24 checkpoints were scored on the identical18 existing validation scenarios.
There are **6/276** disagreements between mean reward ordering and the C2
aggregate-first lexicographic hierarchy; **zero** reversals of six-criterion
mean dominance. The **best five candidates match** under both rankings. These
are descriptive finite-sample comparisons, not confidence claims or proof of
population Pareto alignment. The six mean criteria omit individual-UE constraints.

A concrete disagreement: A3-12026-9 has1.895 percentage points fewer aggregate
known failures than A4-12026-9, yet mean return is.8423 lower. It also has13.406
points worse mean worst-UE failure and different resources/switching. This is a
scalar fairness/resource trade-off conflicting with aggregate-first priority,
not an isolated weight defect. Another pair A2-12026-3 versus A2-22026-3 differs
by only.052 aggregate-failure points while using different reserve and eMBB.
Exact deltas and identified/residual cost differences are retained in
`ordering_comparisons.csv`; unavailable worst/eMBB splits are not invented.

The best A2 checkpoint has mean return **-68.3621**, versus best A4 **-76.8142**.
Both use the **same original reward**. A4's action set includes A2's fixed-F
choices; consequently the reward does not inherently prefer the observed poorer
A4 policy. Limited exploration/optimization, function approximation and the short
2691-transition budget remain explanations; this audit does not isolate any one.
Do not confuse a poorer learned policy with a demonstrated objective mismatch.
Whole-episode worst-UE fraction, integrated rolling maxima and clipped per-tick
eMBB deficits also differ mathematically from final physical summary metrics.

## Terminal charges and horizons

Closure charges exactly the still-live batches not already failed; overdue or
overflow failures are not counted again. They are **administrative unresolved
charges**, not actual misses. The charge is.0125 per batch. Across saved data
maximum charge is.7 (56 batches). Feasible cases have zero. Maximum episode-cost
share is2.25% in contested training and11.95% in contested validation; overloaded
maxima are.234%/.263%. These are heterogeneous, not universally negligible.

In the primary contested j0 replay,34 unresolved-not-due batches cost.425,
**96.41% of the final10ms transition's .440804 cost**, despite zero new failures
and zero historical worst risk there. It is6.90% of episode cost. This creates a
boundary incentive on unfinished work, distinct from realized deadline failures.
All saved physical horizons are3s; checkpoints3/6/9 change training duration,
not physical duration. `terminal_checkpoint_summary.csv` separates these budgets.

`observed_prefix_accounting.csv.gz` and Figure07 apply only the existing
administrative counting equation at already-observed100/500/1000/2000/3000ms
prefixes, without closing the ledger. Their varying relative shares demonstrate
accounting sensitivity; they are **not alternate-horizon policy outcomes**.
At actual termination, the count matches the original ledger. No valid empirical
claim about learned performance at other horizons is possible from these logs.
The30-state representation has no explicit time-to-go; this may make the
administrative boundary partially observed. Discounting also reduces distant
charges' initial-state influence (gamma=.99 per10ms), without removing their full
local influence at the last transition. Neither state nor terminal rule changed.

## Validation, resources and decision

- **116 passed,7 deselected** in84.62s; six audit-specific tests also passed in an
  earlier focused run; a final focused rerun passed6 tests in2.68s. The excluded tests execute DQN learning, including their
  episode-reset/update variants; excluding them respects the no-training scope.
  Existing nontraining accounting, masks, causality, replay and scheduler tests ran.
- 18 replays:93.58s application elapsed,95.35s including guard/startup; mean5.20s
  per representative3s diagnostic episode, including CSV instrumentation. This
  is audit throughput, not production scheduling-loop latency.
- Affinity CPUs6,7,11,14 inherited by children; one workload at a time, one numerical
  thread per process, zero DataLoader workers. CUDA disabled and CPU devices selected.
- Maximum sampled process-tree RSS **1.419GiB** (regression); replay.708GiB.
  Maximum sampled CPU **138.71%**, where100% is one logical core. These are sampled
  peaks, not guarantees about short-lived sub-sample peaks.
- 2GiB warning /2.5GiB monitored stop; neither triggered. No writable delegated
  cgroup was available, so memory was **monitored, not hard-capped**. The guard
  additionally stops below4GiB system available RAM. Initial available RAM was
  about23GiB. No unrelated process/settings were modified.

Retain R0 for reproducibility. Its direct event accounting is sound and its broad
checkpoint ordering supports intended aggregate performance. Do not revise
weights based only on scales or correlations. The strongest prospective hypothesis
is whether absolute rolling worst-risk shaping obscures credit compared with
new per-UE failure events. [The bounded matched ablation plan](ABLATION_PLAN.md)
specifies that test, preserving state/action/initialization/budget and physical
validation criteria. It was **not run**. No DQN retraining or reward revision is
part of Phase2F.
