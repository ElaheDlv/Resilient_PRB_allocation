# Phase2F mathematical and instrumentation audit

Base2c31c96. Phase2B `joint-reward-v1`, all weights and all DQN/state/action
sources remain unchanged. This is not a reward experiment or retraining run.

## Exact implemented equation and units

There are U=8 URLLC UEs, reference arrival rate nu0=1000batches/UE/s,
Delta=10ms RIC/learning intervals, dt=1ms service ticks, Nref=U*nu0*Delta=80.
For service tick t, actual served eMBB bytes are B_E,t, target rate96e6bits/s,
so target service is12000bytes/tick. Let f_t be **newly known** failed batches,
c_t terminal administrative unresolved charges, h_t valid maximum per-UE
100ms matured-cohort failure fraction (or a neutral masked0 when invalid),
R_t activated reserve PRBs (0..18), and z_t indicate a mode switch's **first
subsequent serviced tick**. The exact negative reward is:

    r_t = -(f_t/80 + c_t/80
            + .025*h_t
            + .01*max(0,1-B_E,t/12000)
            + (.002/18)*R_t
            + .01*z_t).

Ten ticks are summed into one DQN transition, without internal recursive stepping.
`.025=.25*dt/Delta`, `.01=.10*dt/Delta`, and `.002=.02*dt/Delta`.
Each switch is charged once, not ten times. A first action issued after tick10
is applied to ticks11..20. Initialization ticks1..10 contribute physical metrics
and bootstrap counts, but no learned reward. C3 has299 transitions and2990 paid
ticks per3s episode. Logged return is the **undiscounted** sum; DQN targets use
gamma=.99 per10ms transition, so the effective learning objective also discounts
delays. Discounting ten intervals gives .99^10≈.904, not zero feedback.

| Component | Per10ms reference scale | Important interpretation |
| --- | ---: | --- |
| Newly known failure | .0125 per batch;1 if80 become known | Exactly once per batch; fixed reference exposure, not current arrivals or a probability estimator. |
| Historical worst UE | 0.. .25 per fully valid interval | Absolute rolling failure rate, not new failures or progress. |
| eMBB deficit | 0.. .10 | Clipped at each1ms service tick; later catch-up above target is not a negative cost. |
| Reserve |0.. .02; .001111 per activated PRB held10ms | Activation, not actual extra PRBs needed by the observed grants. |
| Switching | .01 per mode change | .8 newly known batches,9PRB-intervals,4% of saturated worst cost. |
| Terminal closure | .0125 per not-yet-failed live batch | Administrative finite-horizon charge, not a physical deadline miss. |

At steady one-UE complete failure, f≈10/interval yields event cost.125 and worst
cost.25. With all8 UEs failing, event cost1 outweighs worst.25. Thus the relative
fairness pressure depends on how concentrated failures are, not just severity.
The reference-rate normalization is appropriate to the fixed one-batch/ms workload
used here; changing arrival rates would require another normalization audit.

## Source-level contracts

- `network_layer/joint_reward.py`: `RewardConfig`, `FailureLedger._fail`,
  `on_event`, `sample`, `terminal`, `consume`, and `tick_reward`.
- Overflow first marks an arrival failed; a fully dropped arrival need not remain
  live. Late completion or expiry can mark a still-unfailed batch once. Subsequent
  overdue samples and completion do not charge that batch again.
- `JointCausalAdapter.prepare_tick` samples the ledger, closes only at actual
  terminal time, consumes current tick failures and computes actual eMBB served
  byte deltas and activated occupancy **for the interval just served**.
- `JointCausalAdapter._compute_reward` consumes that service outcome and resets
  the first-tick switching flag. The unmodified `CausalInterval.observe` sums
  one-tick rewards and stores state/action/next state with the next boundary mask.
- `PilotAdapter._compute_reward` calls the joint reward, but discards its six
  parts and returns `{'joint_reward': monitor}` for C3's internal P dispatch.
  `xAppMatchedPilot` deliberately uses that dispatch for allA1..A4 so objectives
  match despite different masks. C3 logs only the resulting episode return.
- `deadline_risk.encode_deadline_risk` uses per-UE matured offered/failed counts
  in an arrival-cohort window(t-100ms,t], after full history and20 matured
  exposures per UE. All8 must be valid for worst risk. The maximum reaches1
  when a UE's entire matured cohort failed; it is not an arbitrary reward clip.
  Different historical UE identities can share the same maximum. HOL pressure
  and active-overdue inputs do not enter this reward directly.

One failed batch contributes to the direct event term once, then can contribute
to rolling worst-risk shaping while its arrival cohort remains in the100ms
window. That is two deliberate objectives, not duplicate event-ledger counting.
With2/10ms budgets, failed cohorts can continue influencing risk for roughly
98/90ms after expiry. Successful service cannot retroactively make them timely.
The relevant service actions need not match the action at arrival or first-known
failure; measuring this lag does not demonstrate a transition-timing bug.

## What the saved data can identify

C3 has72 training and432 validation episode records,24 candidate checkpoints.
For every record, identify exactly (up to floating-point summation tolerance):

    C_failure = (known_failures-bootstrap_new_failures)/80
    C_terminal = terminal_unresolved_charges/80
    C_reserve = round(mean_R_full*physical_steps)/9000
    C_switch = .01*switches
    C_worst + C_eMBB = -recorded_return - the four identified terms.

C3 initializes with zero reserve and no action before tick10, so the first10
physical occupancy samples contribute zero; the learned and full physical
reserve PRB-time totals coincide. Full replays independently check this identity.
The rounding tolerance concerns recorded float averaging; true PRB-time is integer.
Terminal charges equal the existing physical **unresolved-not-due** partition.
Consequently direct failure+terminal costs equal the upper failed-offered count
minus excluded bootstrap failures, divided by80. Overdue/overflow traffic is
not charged again at closure. Administrative charges remain distinct labels.

If no physical failure occurred, historical worst is exactly0, and eMBB is the
residual. Otherwise the separate worst/eMBB totals are **not identified** by the
logs. Do not substitute mean throughput into a per-tick clipped deficit or infer
mean worst risk from saturated decision counts. C3's evolving training weights,
actions, exploration/replay RNG history and continuation state were not saved;
reproducing those transitions would require prohibited training. No such
reconstruction is attempted.

Bounds are explicit, not guessed point estimates. C_eMBB<=29.9. Full history
cannot be valid before100ms, so C_worst<=.025*2901=72.525 over C3's3s horizon.
The residual, these nonnegative caps and a conservative service-total Jensen
lower bound constrain the split. Paid service<=whole-episode service implies
C_eMBB>=.01*max(0,2990-whole_episode_eMBB_bytes/12000). Without tick histories
this bound may be loose. `saved_episode_components.csv` preserves null values
for unidentifiable components and identifies complete no-failure cases explicitly.

## Minimum replay sample and read-only instrumentation

`protocol.json` pins18 validation checkpoints/entries before replay:9 A4 grid
cases (seed12026, first placement in each regime, j0/10/20);3 severe seed22026
replications;3 A1/A2/A3 matched contested severe comparators;2 A4 worst-first
checkpoint comparators;1 A2 overloaded comparator for inadequate-learning checks.
The sample is diagnostic, deliberately selected from known development cases;
it is not an independent performance test or an estimate of all432 tick-level
component distributions. Grid distributions are shown separately from extras.

At the adapter instance, capture the real prepared service outcome, original
risk and switch flag; recompute the pure unchanged `tick_reward` and require
exact equality with the original return. Return the original result untouched.
Record separate parts, interval sums, actual service, observed quota/grants and
per-UE snapshots. Wrap the ledger's first-failure bookkeeping only to record
arrival-to-charge lag and action/mode history; the original method executes
unchanged exactly once. No queues, ledger flags, actions or callbacks advance
time recursively. Agents are frozen, remember/learn are forbidden, RNG/update
counters/fingerprints must not change. Every replay must match C3 control/return,
per-UE/class/overall physical outcomes, recovery, resources and exogenous hashes.

`used_extra_lower_bound = sum_s max(0,allocated_s-baseline_s)` is an accounting
fact about the observed allocations. Activated minus that quantity is reserve
capacity unused by those grants. It does **not** predict the service outcome
if a controller removed reserve or took another action. Low quota versus current
load/integer-drain certificates likewise does not prove that an already-expired
batch could have been saved by a new command. No unavailable counterfactual
network outcome is reconstructed from correlations.

## Terminal horizon sensitivity and ordering

Every saved C3 physical episode is3s. Checkpoint episodes3/6/9 indicate training
budgets, **not different physical horizons**. At the already-observed prefixes
100/500/1000/2000/3000ms, count still-live ledger records not already failed;
without closing/mutating the ledger, report the charge that the existing
administrative equation would assess there relative to observed accumulated
nonterminal cost. This is prefix **accounting sensitivity only**, not an
alternative policy rollout, a new learned horizon or inferred physical failures.
At actual3s closure it must equal the recorded administrative charge.

Ranking compares all24 candidates on the same18 validation scenarios using
mean original return and the exact C2 aggregate-first metric hierarchy. These
are different scalar versus lexicographic objectives; disagreement is not by
itself a code error. Compare physical mean worst-UE failure over a whole episode
with time-integrated **rolling** maximum risk carefully: changing worst-UE identity,
maturation, warmup and a hard horizon can differ. eMBB clipped slot deficits also
differ from aggregate served/offered ratios. A higher-cost learned policy under
the very same reward is evidence against blaming the reward alone. Observational
switch/component associations cannot identify the causal effect of a weight.

Resources:CPU-only, affinity6,7,11,14, numerical threads1, sequential; warning2GiB,
monitored RSS stop2.5GiB, system MemAvailable stop4GiB. No delegated hard memory
cap. No training, weight changes, new scenarios or future-model holdout use.
