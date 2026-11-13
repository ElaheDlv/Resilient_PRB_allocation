# Phase 2C-3 measured decision report

**The joint DQN has not demonstrated a robust overall advantage at this bounded budget.**
Under aggregate-first selection, A2 has the lowest measured aggregate failure,
and A4 is worse than A2 in aggregate and mean worst-UE failure while consuming
more reserve. A4 offers a different aggregate/fairness operating point under
worst-first selection, with better HOL recovery, but substantial individual-UE
harms remain. These are validation-selected exploratory results, not held-out
estimates or evidence of convergence.

## Implementation and frozen experiment

Branch: `phase2c3/matched-objective-ablation-pilot`; base `64a861e`.
The exact file inventory and change explanations are in `CHANGED_FILES.md`.
`PROTOCOL.md` and `protocol.json` document the prospective design and input hashes.

Only action masks differ between A1/A2/A3/A4; every learner uses the unchanged
Phase 2B 30-feature state, 98-action network, reward, Huber loss, clipping, Adam,
replay and target update setup. A1 fixes proportional scheduling; A2 fixes
feasibility-aware scheduling; A3 retains only the current reactive reserve
pair and learns its mode; A4 learns the full joint action. A3 applies one atomic
bridge command and recomputes its restriction in the replay next-state mask.
No predictor, architectural/reward change, jamming change or physical reserve
resize was introduced. The original reserve remains 18 PRBs.

Two independent learner seeds, 12026/22026, use the same nine preselected training
entries, mixed in three regime/severity-balanced blocks. Nine of 18 frozen
training entries are used per learner; both placements appear within each
regime. Every learner has 2,691 interactions and 2,660 optimizer updates.
There are 72 training episodes, 432 greedy validation episodes, and 24 saved
checkpoints (episodes 3/6/9; 897/1,794/2,691 transitions). All 18 frozen validation
entries are used at each checkpoint. Pairing checks verify identical arrival,
position, CQI/MCS, effective-capacity and budget-assignment hashes across runs.
The frozen future-model holdout was neither evaluated nor used for selection.

## Both prespecified selections

Aggregate-first uses the exact C2 lexicographic `checkpoint_key`. Worst-first
swaps its aggregate and worst-UE terms, retaining every other tie break. No
hard fairness SLA is invented. Both are reported; no criterion is chosen after
seeing results. Each learner seed selects its own checkpoint.

| Method | NN seed | Aggregate-first episode | Worst-first episode |
| --- | ---: | ---: | ---: |
| A1 | 12026 | 6 | 6 |
| A1 | 22026 | 9 | 9 |
| A2 | 12026 | 6 | 6 |
| A2 | 22026 | 6 | 6 |
| A3 | 12026 | 3 | 3 |
| A3 | 22026 | 9 | 9 |
| A4 | 12026 | 6 | 9 |
| A4 | 22026 | 9 | 3 |

Results below average the same 18 scenarios and two learner seeds, balancing
all nine regime/severity strata. “Worst UE” is the mean of each run's maximum
per-UE failure percentage, not the maximum over the whole experiment. NN range
is the range of the two seed means; it is **not a confidence interval**. Reusing
a placement at three severities and two initializations does not create six
independent geometry draws.

| Selection | Method | Known failed (%) | Worst UE (%) | Upper bound (%) | eMBB (Mbps) | Reserve (PRBs) | OFF recovered | NN failure range (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aggregate | A1 | 35.389 | 35.786 | 35.481 | 91.246 | 11.980 | 15/24 | 4.642 |
| aggregate | A2 | 11.899 | 39.697 | 11.935 | 89.986 | 6.677 | 14/24 | 1.125 |
| aggregate | A3 | 16.624 | 45.853 | 16.672 | 90.840 | 5.419 | 12/24 | 2.616 |
| aggregate | A4 | 17.406 | 41.513 | 17.443 | 91.233 | 10.226 | 13/24 | 8.381 |
| fairness | A1 | 35.389 | 35.786 | 35.481 | 91.246 | 11.980 | 15/24 | 4.642 |
| fairness | A2 | 11.899 | 39.697 | 11.935 | 89.986 | 6.677 | 14/24 | 1.125 |
| fairness | A3 | 16.624 | 45.853 | 16.672 | 90.840 | 5.419 | 12/24 | 2.616 |
| fairness | A4 | 22.736 | 36.319 | 22.788 | 90.952 | 10.493 | 16/24 | 3.707 |

`summary.csv` additionally reports tight/relaxed classes, served/offered eMBB,
unfinished bytes, not-yet-due unfinished batches and every regime/severity.
`selected_scenarios.csv` and `selected_per_ue.csv` contain both selections;
`validation_scenarios.csv` and `per_ue.csv` retain **all** candidates.

## Operating regions and allocation-associated failures

Aggregate-first results, averaging both learner seeds and all three severities:

| Regime | Method | Aggregate failure (%) | Worst UE (%) | Tight (%) | Relaxed (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| feasible | A1 | 0.000 | 0.000 | 0.000 | 0.000 |
| feasible | A2 | 0.000 | 0.000 | 0.000 | 0.000 |
| feasible | A3 | 0.000 | 0.000 | 0.000 | 0.000 |
| feasible | A4 | 0.000 | 0.000 | 0.000 | 0.000 |
| contested | A1 | 6.969 | 7.569 | 7.537 | 6.401 |
| contested | A2 | 2.470 | 19.758 | 3.062 | 1.878 |
| contested | A3 | 9.022 | 37.775 | 13.572 | 4.472 |
| contested | A4 | 3.924 | 24.756 | 5.694 | 2.153 |
| overloaded | A1 | 99.199 | 99.789 | 99.781 | 98.617 |
| overloaded | A2 | 33.228 | 99.333 | 64.460 | 1.997 |
| overloaded | A3 | 40.850 | 99.783 | 68.273 | 13.428 |
| overloaded | A4 | 48.293 | 99.783 | 71.207 | 25.380 |

All selected policies have zero known failures in feasible nominal scenarios;
these cases cannot establish a deadline-control advantage. In contested load,
A4 improves on A1 and A3 in aggregate, but A2 remains better in aggregate.
A2's good aggregate number hides worse worst-UE performance: under contested
severe jamming its aggregate/worst percentages are 7.409/59.275, versus A4's
8.989/52.008 and A1's 20.907/22.708. These are different service trade-offs.
A4 also introduces moderate-jamming contested failures (2.782%) where A1/A2
have zero, so a uniform improvement is not supported.

The frozen C2 overload certificates establish that zero misses are impossible
even with optimistic full-pool OFF capacity under this static periodic model.
They do **not** establish that every observed miss, or 99% failure for a particular
UE, is unavoidable. A2's overload aggregate is 33.228%, compared with A4's
48.293% and A1's 99.199%; allocation choices still matter greatly under overload.
The worst-UE risk feature saturates at one on approximately 99.55% of valid
sampled overloaded validation decisions across candidates. This limits its
observed discrimination; it is not proof of the cause of a particular action.
No reward/state change is made in response to that observation.

## Paired trade-offs and individual harms

Differences below subtract the comparator from A4 within identical scenario and
learner seeds. Positive failure differences are worse; there are 36 paired
run observations per comparison, clustered on six placements and two learner
seeds. No significance or confidence interval is claimed.

| Selection | Comparator | Aggregate delta (pp) | Worst-UE delta (pp) | eMBB delta (Mbps) | Reserve delta (PRBs) | UE cases worse / better |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| aggregate | A1 | -17.984 | +5.727 | -0.013 | -1.754 | 16 / 66 of 288 |
| aggregate | A2 | +5.506 | +1.816 | +1.247 | +3.549 | 76 / 5 of 288 |
| aggregate | A3 | +0.782 | -4.340 | +0.394 | +4.807 | 42 / 64 of 288 |
| fairness | A1 | -12.654 | +0.533 | -0.294 | -1.487 | 8 / 80 of 288 |
| fairness | A2 | +10.836 | -3.378 | +0.966 | +3.816 | 91 / 8 of 288 |
| fairness | A3 | +6.111 | -9.533 | +0.112 | +5.074 | 52 / 49 of 288 |

These UE cases are repeated paired measurements, not independent subjects.
As a concrete harm case, for validation placement 53260288, j20, learner seed
12026, aggregate-selected A4 raises URLLC_1 failure by **94.667 percentage points**
relative to A1 on the same exogenous scenario. `paired_per_ue.csv` identifies
this exact run/UE and all other harms; `selected_per_ue.csv` includes its offered
exposure, budget class, bytes/PRB, allocations, served/unfinished traffic and
known failures. A lower average worst-UE score therefore cannot justify a
claim that no UE is harmed. Neither selection demonstrates a Pareto improvement
across UEs and service/resource metrics.

## Switching, recovery and learning progress

| Selection | Method | Feasibility-mode time (%) | Mode switches / decision |
| --- | --- | ---: | ---: |
| aggregate | A1 | 0.000 | 0.0000 |
| aggregate | A2 | 100.000 | 0.0000 |
| aggregate | A3 | 89.704 | 0.0354 |
| aggregate | A4 | 76.694 | 0.2904 |
| fairness | A1 | 0.000 | 0.0000 |
| fairness | A2 | 100.000 | 0.0000 |
| fairness | A3 | 89.704 | 0.0354 |
| fairness | A4 | 69.019 | 0.5356 |

A4 does not simply remain in one mode. It switches much more than A3. However,
more switching does not translate into lower aggregate failures than A2 or A3.
A4-versus-A2 comparisons include different learned reserve actions, so they do
not isolate the causal benefit of switching with reserve decisions held fixed.
That benefit remains unproven; no extra counterfactual experiment is added here.

OFF recovery means **all URLLC HOL waits comply for a consecutive 10 ms** after
an observed jammer-OFF transition. It is not a guarantee that every offered
batch completed on time. Unrecovered windows are censored at the next ON event
or horizon; their times are never treated as successful recoveries. A4 recovers
13/24 windows under aggregate-first and 16/24 under worst-first, compared with
A1 15/24, A2 14/24 and A3 12/24. The worst-first recovery improvement coexists
with batch failures and individual harms. Per-window times/censoring are in
`recovery_windows.csv` and the raw episode records.

Fixed-validation curves show nonmonotonic learning. A4 seed12026 moves from
22.970% aggregate failure at episode3 to 13.215% at episode6, then back to
20.882% at episode9; its worst-UE score moves 36.611→44.261→35.728%.
Seed22026 is different: 24.589→26.008→21.596% aggregate and
36.911→45.072→38.765% worst-UE. Neither return curves, Q magnitudes nor TD loss
establish convergence or stable superiority. All methods share the same reward;
training episode difficulty still changes, so raw return oscillations are not
interpreted as learning progress. The seven vector figure pairs and their
underlying data/captions are in `figures/`; `updates.csv` records every one of
21,280 optimizer updates. Bands denote two-seed ranges, not confidence intervals.

## Accounting, validity and compatibility

Known failed/offered batches count the disjoint union of overflow-affected
batches, fully admitted late completions and fully admitted overdue unfinished
batches. The upper bound adds fully admitted unfinished traffic not yet due.
Unfinished bytes can include already-known failures and are not added a second
time to failed-batch counts. Administrative terminal reward charges remain
separate from physical known failures. The largest upper-minus-known gap across
all validation candidates is 0.233 percentage points, too small to explain the
main method differences. Raw records retain terminal/live-ledger accounting.
These are batch/served-byte queueing measurements, not packet reliability.

The full pre-pilot regression suite passed **100 tests in 97.28 s**. Final C3
focused checks passed **9 tests in 6.28 s**, including the two added selection/
shared-learner checks (the other seven overlap the full regression run).
Tests cover physical-mask intersection, fixed modes, atomic/current-state A3
commands and next-mask recomputation, matched reward, real updates, causal
interval replay, terminal transitions, repeatable resets, pairing and frozen
evaluation. Existing queue, PRB, jammer and legacy tests passed as part of the
suite. Historical paper experiments were **not** rerun; their numerical results
are not claimed unchanged. The sole shared-runner edit is an optional app-class
hook whose default preserves its old call path.

All 24 saved checkpoints were reloaded and their experiment metadata/update
counts verified. All 432 recorded validation parameter hashes match the saved
weights. Evaluation preserves the agent's parameter fingerprint, RNG state,
interaction/update counters and replay behavior. Snapshots are not exact training
resume files because replay and RNG continuation state are not serialized.
See `checkpoint_audit.json`, test logs and `README.md` reproduction commands.

## Runtime and resource limits

The entire sequential training/validation command took **2337.76 s (38.96 min)**. Mean service-loop runtime per 3-s simulated episode was 4.886 s for training and 4.418 s for validation; validation median 3.852 s, range 2.690–7.357 s. Full-command time also includes setup/reset, garbage collection, checkpointing, logging and guard overhead.

All training, evaluation, test and report workloads use inherited CPU affinity
**6,7,11,14**, one numerical-library thread, one worker, no DataLoader workers
and CUDA disabled. Sampled training-command peak aggregate RSS was
**732.64 MiB** and peak process-tree CPU
**138.81%** (100% is one logical CPU).
The full regression peak was **1,553.35 MiB** and 138.39% CPU. No 2-GiB warning
or 2.5-GiB stop triggered, and no test was skipped/reduced for memory.

Affinity is enforced; memory is **monitored, not hard-capped**. A writable
delegated cgroup/user systemd memory controller is unavailable. The 50-ms
process-tree RSS watchdog and <4-GiB MemAvailable stop cannot guarantee capture
of instantaneous peaks or short-lived processes. Numerical thread safeguards
alone are not described as isolation. CUDA initialization is checked after each
episode. No unrelated process was terminated or modified. Exact sampled peaks,
CPU/RSS measurement limitations, thresholds and times for all commands are in
`*_resources.json`; software versions are in `environment.json`.

## Research decision

Retain A2 as the strongest aggregate deadline baseline in this pilot. Retain A1
as an essential fairness comparator and report A4's worst-first trade-off as
exploratory, not as an across-the-board joint-control improvement. A4 has lower
aggregate failures than A1, but does not beat A2 on aggregate deadlines, does
not consistently improve worst-UE outcomes, and uses more reserve than A2/A3.
Its greater switching and learner-seed variability undermine a robust-superiority
claim at the tested budget. Some eMBB/recovery trade-offs are favorable, but
individual UE harm and overloaded performance remain material.

This is a useful negative/inconclusive ablation, not yet sufficient evidence
for a journal claim of superior joint DQN control. Only two learner seeds and
two placements per regime, nine training entries per learner, periodic traffic,
static channels, fixed queueing budgets and 3-s horizons were tested. No
industrial end-to-end control guarantee follows from them. Any later study
would need prospective uncertainty/generalization validation and a clearer
fairness/completion objective; those are recommendations, not changes made here.
The authorized bounded pilot is complete. No frozen future-model holdout,
full-scale sweep, predictor or further training was started.
