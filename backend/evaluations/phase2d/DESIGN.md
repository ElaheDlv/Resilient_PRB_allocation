# Phase 2D prospective design and source audit

Base eafcea7. This document specifies the candidate before development comparisons.
No DQN learning, state/reward/action changes or future-model holdout evaluation.

## Existing algorithms and the audited gap

- `proportional_scheduler.proportional_allocation`: capped proportional water
  filling on useful backlog PRB requirements; floor shares, redistribute by
  largest remainders, repeat at saturated caps. Stable dictionary-order ties.
  No budget or miss-history input. Growing backlogs can increase future shares,
  but adequate service rates or timely completion are not guaranteed.
- `deadline_scheduler.deadline_allocation(feasibility=False)`: first one useful
  quantum to each eligible UE excluded for at least the starvation limit, then
  minimum slack D - HOL_age - head_remaining_bytes*dt/bytes_per_PRB. Integer
  bundles must produce an additional whole byte. Virtual FIFO accounting keeps
  timestamps and reflects partial head service without changing real queues.
- The same function with `feasibility=True` (F): after the same zero-service
  guard, rank locally potentially feasible heads before infeasible ones, then
  EDF, smallest remaining PRB work, longest exclusion age, UE ID. Feasibility
  assumes current rate and the entire current quota/per-UE cap over remaining
  inclusive service slots. It ignores competition with other UEs. Expired heads
  remain eligible as fallback, never deleted. A tiny positive service allocation
  resets the zero-service guard even when sustained arrivals exceed service.
- Phase1.2 `fair_service_credit.FairServiceCredit`: infer admitted arrivals from
  queue increments since actual last service, EWMA smoothing .05, scale requested
  per-UE PRB costs to the current quota; accrue/debit byte credits capped at
  +/-20 entitlement ticks. `deadline_allocation` first spends up to
  floor(alpha*quota) on positive normalized credits, after zero-service protection,
  then runs the original F ordering. No deadline-miss-history term. This fairness
  loop can protect an expired-only queue; its credits do not test whether any
  still-timely completion can benefit. It already addresses *rate* deprivation,
  so a new ungated additive fairness bonus would largely duplicate it.
- Dispatch: `Cell.step` builds QueueViews and calls the allocator; actual service
  is `BaseStation.pull_dl_bytes` and `Cell` service accounting. Existing
  `deadline_fair` debits actual bytes in the Cell service path and removes credit
  state when a UE leaves. Phase1.2's report documents that credit clipping bounds
  accounting state, not backlog or deadline harm. Its old recovery failures are
  prior evidence, not new C2 validation results.

The specific gap is **how to spend service protection when FIFO heads have
already failed**. Pure F demotes them; the old fair rule can repay them blindly.
A prefix leading to an attainable fully admitted timely batch is different from
expired-only work. This motivates one gated extension, not a claim of superiority.
The unchanged old fair rule at alpha=.35 is retained as an explicit comparator.

## Experimental gated protection (G_lambda)

Service dt = 1 ms; fixed retrospective window W = 100 ms. At service completion
s, retain cumulative offered bytes A_i(s), actual served bytes S_i(s), and queued
bytes Q_i(s) for each UE. For the next allocation t=s+dt use differences on
(s-W,s], after a full window; no current-interval outcome is available. Retain
W plus one endpoint in bounded deques, reset with the episode. Overflow is part
of offered workload, not fabricated accepted queue demand.

Let a_i=A_i(s)-A_i(s-W), b_i=S_i(s)-S_i(s-W), and

    d_i = max(0, a_i-b_i)/max(1,a_i) in [0,1].

Let m_i be the existing causal per-UE matured failed/offered batch fraction in
its 100-ms arrival-cohort window, only with full history and >=20 matured batches;
otherwise use a neutral multiplier m_i=0 (not a claim of zero failure).

Before service t, inspect the real measurement FIFO without changing it. For
each fully admitted, not-expired batch j, let B_ij be *all remaining accepted
bytes through j*, including older expired or overflow-affected prefixes.
The estimated work is n_ij=ceil(B_ij/r_i) PRBs at current rate r_i bytes/PRB/tick.
Remaining inclusive service slots are

    L_ij = floor((arrival_ij+D_i-t+1e-12)/dt)+1.

A target is locally optimistic-reachable if L_ij>0 and n_ij<=L_ij*c_i,
where c_i=min(slice quota, per-UE cap). Choose the first such target; none
means no fairness credit. This is an optimistic test, **not** a multi-UE
schedulability certificate or predictor. Cumulative bytes larger than
(floor(D_i/dt)+1)*c_i*r_i terminate the search: no later currently-arrived
batch can be reached before its deadline. Thus long expired backlogs do not
create an unbounded scan. Rates/caps/quota are assumed constant only for this
local estimate; no future channels, traffic or jammer transitions are accessed.

For a reachable target and a full service window define protection credit

    C_i = min(B_ij, max(0,a_i-b_i)*(1+m_i)) bytes,
    E_i = max(1,a_i) bytes,
    normalized priority = (C_i - floor(g_i*r_i))/E_i.

Otherwise C_i=0. Priority is initially bounded by 2. Credits are recomputed,
not accumulated for historical failures. m_i cannot earn credit when d_i=0.
Already-failed work earns protection only as a necessary FIFO prefix to a
currently reachable timely target. Actual expiry/drops/service rules never change.

Reuse the **unchanged** `deadline_allocation` fair portion with these credits:
retain its zero-service guard, spend at most floor(lambda*quota) useful PRBs on
positive normalized credit (deterministic exclusion-age/UE-ID ties), then apply
its unchanged feasibility/EDF/remaining-work ordering and work-conserving
fallback. Partial service updates the virtual FIFO exactly as before. A final
quantum may cover a few bytes past a target owing to integer PRB granularity;
it remains bounded by total useful demand. Lambda is a **protected service-budget
share**, not a fairness penalty or a blend with proportional allocation.
It is not a per-UE minimum-service guarantee.
Lambda=0 exactly reproduces F, including its guard and fallback; it is not P.

In feasible load, recent deficits disappear and protection vanishes. In contested
load, short reachable prefixes may receive repayment before persistent demotion.
Under physical overload, no rule can promise all deadlines; unreachability gates
protection and unchanged guard/fallback still serve accepted expired traffic.
The gate may also fail to repair an already irrecoverable backlog. Miss weighting
can redistribute harm, and protecting one UE consumes PRBs another could use;
no all-UE/Pareto improvement is assumed.

The only shared-code change is an optional `Cell.experimental_urllc_allocator`
callback in its deadline branch. P/F defaults and scheduler source algorithms
are preserved. Experimental state is owned by the fresh app/cell, never added
to the 30-feature DQN state or joint action enum.

## Frozen bounded diagnostic protocol

See `protocol.json` for exact identities and hashes, frozen before experiments.
42 saved-checkpoint replays: aggregate-selected A4 for both NN seeds on all 18
validation entries (36); worst-first A4 on contested severe cases (4); one A1
and one A2 replay for the largest recorded UE-harm examples. Every replay must
match C3 control digest, parameter and exogenous hashes, budgets and physical
outcomes. Extra telemetry and pure shadow allocations may not affect control.

For each selected A4 decision, shadow P/F on the **first actual service tick**
after its RIC command, using the same queues, quota, current rates and prior
starvation ages. Compare immediate fully admitted timely completions, not a
future-rollout counterfactual. Track reversals and changes in existing state
and retrospective service deficit. A deterministic policy can oscillate on
small feature changes; that does not identify measurement noise or causality.
100-ms cohort risks mix effects of earlier modes. Subsequent 10-ms outcomes
are correlations and must not be attributed to the most recent switch alone.

Nonlearning development: P, F, old fair .35, G_{0,.25,.5,1} on all six contested
validation entries plus severe cases for both feasible and overloaded placements:
10 scenarios x 7 policies = 70 episodes. Use the **same frozen ReactiveReserve**
for every policy and assert equality of full quota trajectory hashes, arrivals,
CQI/MCS/positions and budgets. Test G0/F whole outcomes as well as unit equality.
No parameter tuning, learner selection, additional test pool or confidence
intervals from two placements. Report every lambda, including negative/equal
results. These are exploratory development observations.

Resources: CPUs6,7,11,14, inherited affinity, CPU-only, numerical threads1,
one sequential worker/no DataLoader workers. Watchdog warning2GiB, stop2.5GiB,
MemAvailable stop4GiB. No delegated hard memory cap; report sampled peaks and
limitations. Regression tests which perform DQN training are explicitly excluded.

196 actions represent 49 reserve pairs x 4 scheduler choices. Retaining P/F
plus G.25/G.5/G1 gives 5 distinct choices (245); G0 duplicates F. No action-space
expansion is implemented or inferred from arithmetic alone.
