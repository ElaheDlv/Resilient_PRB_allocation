# Phase 2G equations, implementation and validity boundaries

The authoritative prospective specification remains the unchanged Phase2F
`ABLATION_PLAN.md`. This implementation introduces **three arms only**, with the
original30-state,98-action/mask interfaces, two256-unit hidden layers and DQN
update unchanged. Full joint A4 is used in all arms. No additional heuristic,
prediction, channel/traffic/jammer change or reserve resizing is introduced.

## Exact rewards

Let i denote one causal action interval containing service ticks t=1,...,10.
Let f_u,t count newly known, exactly-once **physical** batch failures for UE u,
f_t=sum_u f_u,t, c_t administrative terminal unresolved charges, h_t the original
valid/masked100ms worst-UE matured-cohort fraction, E_t actual eMBB bytes, R_t
activated reserve (0..18), and z_t the first-service-tick mode-switch flag.
The unchanged components at each1ms tick are:

```
D_t = f_t /80
T_t = c_t /80
B_t = .01 * max(0,1-E_t/12000)
U_t = .002 * R_t/18
S_t = .01 * z_t
```

The learned interval reward for arm a is

`r_i(a) = -sum_t(D_t + T_t + B_t + U_t + S_t) - H_i(a)`

with exactly the predeclared fairness terms:

```
H_i(R0) = sum_t(.025 * h_t)
H_i(R1) = 0
H_i(R2) = .25 * max_u(sum_t f_u,t /10)
```

R2 uses **maximum of interval sums**, never sum of per-tick maxima. The denominator
10 is the fixed per-UE reference arrival exposure (1000/s ×.010s), independent of
observed arrivals. It is deliberately **unclipped**:20 first-known failures for
one UE in an interval cost.5. Terminal charges remain solely in T_t and never
increment f_u,t. A physically failed batch may remain queued, but later expiry or
completion cannot count it again.

R0 returns the exact original tick reward, preserving its floating-point summation
order. R1/R2 replace only the original historical contribution. R2's fairness
charge is emitted on the tenth observed service tick after action application;
no future tick is inspected and no recursive simulator stepping occurs. All arms
have the same ten-tick reward collection boundaries. R2 is defined here for the
prescribed complete10ms intervals; the app asserts against a truncated interval.

Versions: R0=`joint-reward-v1`; R1=`joint-reward-no-history-v1`;
R2=`joint-reward-interval-events-v1`. Each checkpoint includes its arm, equation,
version, state/action metadata, training settings and protocol hash. Cross-arm
checkpoint loads are rejected. Metadata does not add any network input.

## Exactly-once counters and isolation

`PerUEFailureLedger` subclasses the original ledger. Its `_fail` executes the
original implementation once, and increments a per-UE pending counter only on
its first false→true failure transition. The original `consume()` contract stays
unchanged. Per-UE consumed deltas are exposed separately and their sum is asserted
equal to the original scalar count on every consume. Initialization consumes and
discards the first10ms in the original way; those deltas cannot leak into the
first paid R2 interval. Pending counters reset at consume, interval counters at
action application, and the entire ledger at every fresh episode.

The existing matched app runs full A4 with its original state encoder, bridge,
mask and P/F schedulers. The optional ablation app replaces only the ledger and
reward adapter before any arrivals. The base reward, agent, state/action catalog,
queue, physical simulator and prior experiment files are not edited.

## Matched bounded training

The protocol creates one stored initialization per NN seed12026/22026. Each arm
copies that seed's exact policy, target, empty Adam state and Python action/replay
RNG state; fingerprints and target equality are checked before every learner.
All learners use the original C3 training order, epsilon1→.05 over.8 of the
2691-transition budget, gamma.99, Adam1e-4, Huber loss, clip10, replay2048,
batch32 and target updates every100 optimizer steps.

Six learners run sequentially. Each has9 fresh3s episodes,299 paid transitions
per episode,2691 interactions and2660 optimizer updates. Checkpoints after
3/6/9 episodes yield897/1794/2691 interactions. Every checkpoint receives all18
existing validation scenarios:324 frozen evaluations total. Evaluation is greedy
and does not update parameters, interactions, updates, RNG or replay; all are
checked. Scenario setup resets exogenous seeds independently of learner RNG.
Endogenous actions/masks and thus later RNG consumption may diverge after learning;
matched initialization does not force different policies to take identical actions.

## Selection and interpretation

Primary checkpoint selection is the exact original C2 aggregate-first balanced
physical hierarchy, including final earlier-checkpoint tie break. Worst-first
selection swaps only its first two criteria and is reported separately for every
arm. Neither uses the arm's own return to compare objectives. Candidate points
are fixed validation evaluations, not exploratory-training convergence claims.

Known physical failures, unresolved-not-due traffic, and their upper bound remain
separate. Completed-batch, byte-delay and HOL metrics retain their original labels.
These are FIFO **batch/byte** measurements, not packet-level reliability.
Per-UE comparisons match NN seed, scenario identity and UE identity. Some placements
assign different channel quality to a UE; comparisons never pool identities to
hide individual harms.

Two NN seeds and two placements/regime are a small development pilot. Repeated
checkpoints, ticks, UE observations and jamming severities are dependent; no
statistical significance or population Pareto guarantee is inferred. R2 can remove
historical persistence while remaining insensitive when a UE loses ten batches
in every interval. Reward changes also alter target scale. Physical outcomes,
resource use and per-UE trade-offs—not lower TD loss—decide usefulness. No causal
mediation claim (e.g., 'switching caused the improvement') follows from correlated
outcome changes. Future-model holdout data are neither loaded nor evaluated.
