# Phase 1: fixed UE budgets and deadline-aware URLLC scheduling

Base: `phase0/simulator-validation` (`4f6e0e6`). Feature branch:
`phase1/deadline-aware-ue-scheduler`.

The new `deadline_slack` mode changes only intra-URLLC allocation. eMBB uses the
same Phase 0 work-conserving proportional allocator as the comparison baseline.
The legacy scheduler and all Phase 0 modes remain selectable. No predictor,
deadline-aware DQN input/reward, new traffic generator, channel model, expiry
policy, or dynamic reserve sizing is introduced.

## Budget configuration

Add fields to a slice's existing episode configuration:

```json
{
  "URLLC": {
    "ue_count": 8,
    "queue_budget_classes_s": {"tight": 0.002, "relaxed": 0.010},
    "queue_budget_assignment_seed": 2026
  }
}
```

Retain that slice's existing trace fields. These numbers are illustrative **BS
queue-residence budgets**, not validated industrial end-to-end requirements.
Class names and values are configurable. The pilot uses two classes; the
assignment utility supports arbitrary positive finite class budgets.

Class labels are balanced across UE indices (counts differ by at most one), then
shuffled by a private RNG seeded from the assignment seed and episode ID using
SHA-256. Assignment neither reads channel conditions nor consumes the placement,
traffic, jammer, or controller RNG. Assignments repeat for the same configuration
and episode ID. Changing placement seed does not determine a UE's budget class.

Phase 0's `queue_budget_s` default and `queue_budgets_s` per-index overrides still
work. Explicit overrides take precedence over classes and are labeled `override`.
Queues retain their original arrival timestamps, service order, and overflow
behavior. Deadline scheduling requires a positive budget for every active URLLC
UE with a positive service rate; it fails clearly if one is missing.

## Exact allocation rule

At each simulator service step of duration `dt`, let:

- `Q_i` be currently queued bytes for UE i.
- `b_i = c_i * dt / 8` be current service bytes per PRB, using the existing
  MCS-derived bitrate `c_i` in bits/s/PRB.
- `D_i` be its fixed queueing-delay budget in seconds.
- `g_i` be the number of PRBs tentatively granted in this allocation round.

Useful demand is `ceil(Q_i / b_i)`, bounded by the optional per-UE cap. Empty
queues and zero/nonfinite rates are ineligible. The integer service model also
requires a useful quantum:

```text
q_i(g_i) = max(1, ceil((floor(g_i*b_i) + 1) / b_i) - g_i)
```

This avoids spending a fractional-byte-capacity grant that cannot deliver any
whole byte. A UE is eligible only when this quantum fits its remaining useful
demand/cap and the remaining slice quota. Under ordinary modeled radio rates the
quantum is one PRB. Resource accounting is integer throughout.

First allocate one useful quantum to each eligible UE whose consecutive
zero-service-grant age is at least `DEADLINE_STARVATION_TICKS` (default 5), ordered
by decreasing age and then UE identifier. Allocate remaining useful quanta by
minimum **one-PRB-equivalent head slack**:

```text
s_i = D_i - max(0, now - virtual_head_arrival_i)
          - virtual_head_remaining_bytes_i * dt / b_i
```

After each grant, a read-only virtual FIFO accounts for `floor(g_i*b_i)` bytes;
it advances the virtual head as those bytes cover queued batches. Actual queues
are untouched until the original service routine runs. Equal slack is broken by
decreasing starvation age, then UE identifier.

This is a heuristic based on current occupancy, head age, budget and MCS rate.
The last term is an ordering normalization at the current one-PRB rate, **not** a
claim that multiple-PRB delivery takes that amount of wall time. There is no
forecast of future arrivals, channels or jamming. Already-overdue batches keep
negative slack and are still served; there is no dropping or expiry. This choice
can spend resources on deadlines that can no longer be met, and is a possible
source of degradation under sustained overload.

The starvation age resets on a grant estimated to deliver at least one byte and
increments for nonempty positive-rate queues otherwise. Under persistent demand,
a stable set of N eligible UEs, and quota/caps sufficient for at least one useful
quantum every step, the guard prevents indefinite starvation; with unit quanta
and one available PRB, the consecutive zero-grant age is bounded by S+N-1, where
S is the configured threshold. It is not a throughput or deadline guarantee.
No such bound applies during zero quota, zero rate, or infeasible per-UE caps.

The allocator is work-conserving with respect to these eligible useful quanta:
it stops only when no further useful quantum fits, and never exceeds quota or
per-UE caps. It may leave PRBs unused if all useful demand is met or no remaining
bundle can produce a byte. Its greedy implementation costs O(P*N) comparisons
per allocation round for P useful PRBs and N UEs, plus virtual FIFO traversal.

## Controlled comparisons

The new validation policy `fixed` sets a configured reserve split once after
episode reset, before service, then keeps it constant through the existing
jammer's baseline shrink/restore behavior. Its slice quotas therefore vary with
jamming as in Phase 0, but are **identical across each scheduler pair**. Reserve
activation does not depend on UE queues or urgency.

`RES_VALIDATE_FIXED_E` and `RES_VALIDATE_FIXED_U` specify nonnegative integer
reserve PRBs with sum at most 18. The pilot defaults to `(0, 3)`. The original
masked-DQN's action list, 15-state features, architecture and reward are unchanged.
Optional `dqn` runs load the same frozen checkpoint under causal timing. They are
transfer tests of a coupled controller/scheduler, not independently optimized or
matched-reserve-cost controller comparisons.

The pilot compares three placement seeds, severities 0/10/20 lost PRBs, and both
schedulers. The first original test episode supplies unchanged traffic, 8 eMBB
UEs and (by default) 8 URLLC UEs. `--urllc-ues` changes only population for a
separately labeled load check; each UE still uses the original trace/radio model.
The default five-second horizon ends before trace exhaustion and retains any
unfinished traffic. It contains two jammer-OFF windows in attacked scenarios.

Two hashes check each pair:

- Exogenous scenario hash: every tick's UE positions, CQI/MCS, offered arrival
  bytes, jammer-effective capacity and baseline quotas. Admission losses are
  excluded because they may be scheduler-dependent.
- Service-quota hash: the actual slice quotas **before the RIC's next action**.

Budget assignments are checked separately. Fixed-policy pairs must match all
three. Frozen-DQN pairs must match exogenous scenarios and budgets; their quota
sequences may differ due to feedback.

## Measurements and units

- **Served-byte violation:** residence time exceeds the assigned budget for bytes
  actually served. Denominator is budgeted served bytes.
- **Completed-batch miss:** final accepted-byte completion exceeds its budget.
  Denominator is completed budgeted batches. Partial admissions remain explicitly
  identified and are not successful original-batch delivery.
- **Offered-batch outcomes:** fully admitted on-time completion; known failure
  (overflow, fully admitted late completion, or overdue unfinished batch); and
  unfinished/not-yet-due batches. These categories partition offered batches.
  Report lower/upper failure bounds by excluding/including the unresolved group.
  Overflow is counted once, including partial admission; it is not double-counted
  again as a late accepted-batch completion.
- **Sampled HOL violation:** after-service HOL age exceeds budget on a nonempty
  queue. Fractions include all observed UE service ticks. A late batch can finish
  on a tick whose post-service queue is empty; batch misses can therefore be
  nonzero while sampled HOL violations are zero.
- **Violation duration:** lengths of consecutive post-service HOL-violation
  samples. Mean and longest observed runs include right-censored terminal runs
  as observed lower-bound durations; the censor flag is retained per UE.
- **Recovery:** following an actual jammer-OFF transition, time until all URLLC
  post-service HOLs are within their respective budgets, confirmed by 10 ms of
  continuous compliance. Report the start of the qualifying run relative to OFF.
  New jamming or horizon termination before confirmation censors the window;
  censored recovery is null, never zero. No-jam scenarios have no recovery windows.
- **Resource/cost:** actual eMBB served throughput; mean/peak activated reserve;
  allocated and unused slice PRBs. Resource integrals use PRB-seconds and averages
  use `mean_*_prbs`. Inactive reserve and unused active slice quota are distinct.
- **Worst UE:** maximum known offered-batch failure fraction among URLLC UEs;
  per-UE and per-class counters and censoring bounds are also retained.
- **Runtime:** mean/p95 URLLC allocation-call CPU time includes useful-demand/view
  construction and scheduler bookkeeping; common MCS/demand computation is outside
  that timer for both methods. Whole-run wall time includes measurements/hashing.

These remain byte/batch queueing measurements, not packet-level reliability,
end-to-end deadlines, plant stability or safety results. The original 1 ms replay
timestamp quantization and immediate simulator service semantics still apply.

## Reproduce

```bash
python -m pytest -q

# Main paired pilot; add the optional frozen causal-DQN transfer comparison.
python backend/tools/phase1_pilot.py --output /tmp/phase1-pilot \
  --checkpoint backend/checkpoints/res_dqn/xAppResilientDQNAllocator_20260123_215931_final.pt

# Separately labeled higher-population check, same per-UE traces.
python backend/tools/phase1_pilot.py --output /tmp/phase1-stress \
  --urllc-ues 12 --seeds 1234

# Alternate configurable class budgets or per-UE caps.
python backend/tools/phase1_pilot.py --output /tmp/phase1-alternate \
  --budget-classes '{"tight":0.005,"relaxed":0.020}' --per-ue-cap 20

# Phase 0's original-core compatibility check.
python backend/tools/phase0_smoke.py --output /tmp/phase1-compat
```

The driver writes generated catalogs, raw per-run reports, logs and an aggregate
summary under the requested output directory. It does not rewrite original traces
or catalogs. For a single run, the Phase 0 evaluator now also accepts
`--scheduler deadline_slack --phase1-metrics --policy fixed --fixed-u 3` with a
budget-configured catalog. Use `--events PATH` for complete batch event history.

See `PILOT_REPORT.md`, `pilot_summary.json`, `pilot_runs.json` and
`compatibility_summary.json` for the measured results, exact scope and limitations.
