# Phase 2A — Joint RIC infrastructure

Integration base: `1a29280` (`phase1.2/fairness-feasibility`). History inspection confirmed it contains `4f6e0e6` (Phase 0), `cb35202` (Phase 1), and `6bce0d8` (Phase 1.1), plus the optional Phase 1.2 work. No merge/cherry-pick was needed. Implementation branch: `phase2a/joint-ric-infrastructure`.

This is infrastructure and a scripted timing demonstration, not a new learned controller or a scheduler-performance study. No DQN state, action catalog, network, reward, checkpoint, scheduler mathematics, source traffic, radio model, jammer model, or physical reserve partition was changed. The default pool remains 18 PRBs; activation/distribution within it can change. Existing optional Phase 1.2 scheduling remains available when joint control is disabled.

## Optional activation and scope

`JOINT_RIC_ENABLE` defaults false. Existing evaluation scripts do not instantiate the new controller. Cell-local override defaults to `None`, preserving every existing global scheduler configuration. The demo explicitly deploys only `xAppScriptedJointRIC`, with one gNB/cell, eight URLLC and eight eMBB UEs, the original periodic downlink CSVs and heterogeneous budget configuration. It requires causal mode and a 1-ms service step. The new harness rejects simultaneous deployment with other xApps so resource commands have one owner; this is a deliberate prototype limitation, not an O-RAN arbitration implementation.

No runtime global scheduler mutations occur. `Cell._joint_urllc_mode` is the local override, exposed through a read-only property. Only URLLC dispatch uses it. `PROPORTIONAL` maps to the existing `proportional_allocation`; `FEASIBILITY_AWARE` maps to the existing `deadline_allocation(..., feasibility=True)`. eMBB retains its originally configured allocator. The existing zero-service-age bookkeeping is maintained through proportional intervals so a subsequent feasibility interval cannot inherit stale exclusion ages. Neither allocator's mathematical function was edited.

## Read-only telemetry API and units

`UETelemetryCollector(cell, window_s=0.100, episode_start_s=0.0)` subscribes to measurement events through a new additive listener interface. It neither overwrites the existing diagnostic event sink nor consumes the 256-event ring. Collection prunes its own arrival records only; it never changes source queues, measurement counters, quotas or grants. Reports, UE rows and window counters are frozen dataclasses.

| Field | Units / meaning |
|---|---|
| `TelemetryReport.time_s`, window endpoints | Seconds on the gNB simulation clock, never wall clock |
| `service_step` | Completed global service-tick index |
| `cell_id`, `ue_id`, `slice_id`, `budget_class` | Identifiers/labels |
| `budget_s` | Fixed UE queue-residence budget in seconds; `None` for unbudgeted eMBB |
| `queued_bytes` | Actual remaining downlink bytes |
| `hol_wait_s` | Current physical FIFO head residence in seconds; zero for an empty queue |
| `capacity_bps_per_prb` | Estimated bits/s obtainable with one PRB from current MCS; independent of current grant. Known zero capacity is `0`; missing MCS information is `None` |
| `window.offered_batches` | Offered arrival batches in the specified trailing arrival cohort, including dropped batches |
| `window.known_failures` | Known failed offered batches in that cohort as of this report, counted once |
| `window.matured_offered_batches` | Cohort arrivals whose budget has strictly elapsed |
| `window.matured_failed_batches` | Known failures among those matured arrivals |
| `window.pending_overdue_batches` | All currently pending accepted batches already overdue, including arrivals older than the observation window |
| `history_available` | False for missing/inconsistent measurement state or unavailable pre-subscription history |
| `full_window` | Whether a complete configured window has elapsed since episode initialization |
| `slice_quota_prbs` | Current integer quota for this UE's slice |
| `slice_activated_reserve_prbs` | Slice quota minus current jammer-adjusted slice baseline |
| `cell_activated_reserve_prbs` | Total activated reserve occupancy across slices |
| `urllc_scheduler_mode` | Current URLLC mode, repeated in every UE row; it does not describe eMBB's allocator |

Quota/reserve attributes repeat per UE and must not be summed across UE rows. Outside joint mode, compatible preconfigured global policies are identified as P/F; unsupported legacy/slack/fair names are reported literally and invalidate the binary mode feature.

Missing sidecars, missing budget tracking, inconsistent physical-versus-measurement queues, and incomplete subscription history produce `None` counters, not invented zero failure rates. Empty, monitored queues can legitimately report zero offered batches, but a zero denominator still invalidates rate features. Old pending batches are obtained from current sidecar pending state, not final-episode reports. Late subscription waits until unavailable arrival cohorts leave the window. Backward snapshot queries or queries earlier than already-observed events raise rather than leak future information.

## Exact causal window and features

Default observation window W=0.100 seconds; default minimum matured exposure m=20 offered batches **per connected UE**. Tight/relaxed class labels are encoder arguments, defaulting to the existing `tight`/`relaxed` configuration. Budgets are not inferred from channel quality.

At post-service time t, define the arrival cohort for UE i:

A_i(t) = {j : max(episode_start,t-W) < a_ij <= t}.

The code uses the same 1e-12-second equality tolerance as the existing budget measurements. A known failure indicator K_ij(t) is true if any of the following holds:

- Some offered bytes overflowed (including full drops); or
- The admitted batch completed after its stored budget; or
- Accepted traffic remains pending and its stored budget has elapsed.

This is a Boolean union: overflow followed by late completion is one failure. On-time completed batches stay successful as they age. Pending traffic not yet due is not called successful or failed. A fully dropped arrival has no pending accepted traffic but remains an offered failure.

Telemetry reports |A_i| and sum K_ij over A_i. For rate features, define matured arrivals:

M_i(t) = {j in A_i(t) : t-a_ij > D_ij+epsilon}.

Let n_i=|M_i|, f_i=sum K_ij over M_i. These denominators do **not** equal the Phase 1 all-offered failure lower bound: maturity deliberately removes unresolved not-yet-due arrivals and avoids early-completion selection bias. Completed late and still-pending-overdue members are failures; on-time completed members are successes. The budget equality boundary is excluded from maturity until it has strictly elapsed.

The separate `encode_deadline_risk` returns exactly six positions plus validity/reason masks and class exposure counts:

| Position | Definition |
|---|---|
| 1 | max HOL_i/D_i over active (queued>0) URLLC UEs. Empty active set gives 0 if connected budgets are known; missing budget or no connected URLLC gives unavailable. No clipping is applied. |
| 2 | Number of active URLLC UEs with HOL_i>D_i+epsilon divided by number active. Empty active set gives 0 when budgets are known. |
| 3 | sum f_i / sum n_i over tight UEs |
| 4 | sum f_i / sum n_i over relaxed UEs |
| 5 | max_i(f_i/n_i) over every connected URLLC UE |
| 6 | 0 for proportional; 1 for feasibility-aware; unsupported/missing mode gives unavailable |

Class rate features require the class to exist, valid budgets, full W of history, and n_i>=m for **every** connected class member. Worst-UE rate requires those conditions for **every** connected URLLC UE; it never reports the worst of a selectively observed subset. No exposure, absent class, warmup, missing history or unknown measurements yields `None` with validity false. Neural consumers must explicitly handle that mask; no neural consumer was added here. A valid measured zero remains distinct from unavailable.

URLLC rates only are encoded. eMBB offered batches remain reportable, but eMBB deadline counters are `None` when no budget exists. Original traffic/batch/served-byte semantics remain unchanged. None of these metrics is packet reliability or application end-to-end deadline satisfaction.

## Joint command contract

```python
JointRICAction(delta_embb_prb, delta_urllc_prb, urllc_scheduler_mode)
```

Deltas are signed integer PRB counts, normalized to builtin integers. Modes are the two enum values. `JointRICBridge.apply` requires a post-service RIC callback at the configured decision boundary and permits at most one valid command there. Invalid commands raise and do not consume the boundary or modify either component. The caller can retain/retry the previous configuration; the supplied harness fails loudly on a bad script rather than silently changing the experiment.

Before committing anything it validates:

1. Fixed configured physical reserve and physical cell capacity are unchanged.
2. The mode is supported; feasibility mode has valid positive budgets for all URLLC UEs.
3. Both proposed quotas pass the existing `validate_quotas`, including integer quotas, current baseline floors, effective capacity and the shared reserve pool.

Only after every check passes does the synchronous single-threaded commit assign both quotas and local mode. No per-UE grant is written by the RIC. This is atomic discrete-event application, not a distributed/concurrent transaction guarantee. A receipt records before/after quotas, before/after mode, current activated occupancy, `applied_after_step=k` and `first_service_step=k+1`.

Initialization sets the initial policy before the first service tick. Episode policy reset requires a new episode token at a RIC boundary; it cannot be used to switch mid-interval. Deregistration clears UE scheduler history as before; the scripted controller resets telemetry exposure and its command cycle on episode reset. Shutdown releases the local override to the original configuration.

## Two-timescale execution

```text
1-ms tick:
  UE update → fixed-pool guard → jammer/baseline update → DL arrivals
  → local per-UE allocation → actual FIFO service → measurements
  → RIC callback

Every 10 service ticks in that callback:
  finalize previous interval's actual service outcome
  → collect frozen causal telemetry
  → encode six features and validity
  → scripted command → validate both components → synchronous commit
  → next service tick uses new mode/quotas
```

The original DQN causal collector is untouched. New scripted interval logs are not DQN replay entries. Actual interval served bytes are differences of live measurement counters, not achievable rates or final-episode statistics; missing counters yield unavailable service totals. Configuration stored at interval issue is not a promise that baseline quotas stay constant: the unchanged jammer can adjust baselines during the interval. Per-tick service history records actual quotas/mode used. Policy remains selected across jammer transitions until the next valid command/reset.

Default scripted cycle, after successive 10-ms boundaries:

| Boundary | Delta eMBB | Delta URLLC | Next mode | Activated reserve after command |
|---|---:|---:|---|---:|
| 10 | 0 | +3 | F | 3 |
| 20 | +1 | 0 | P | 4 |
| 30 | 0 | -3 | F | 1 |
| 40 | -1 | 0 | P | 0 |

It then repeats. These deltas also belong to the original action-level vocabulary; the DQN action catalog itself remains unchanged. A command at the final demo horizon may have zero subsequent service ticks: it is retained as a pending, unobserved interval, never labeled a completed outcome.

This is a direct in-process simulator interface, not implemented O-RAN E2 signaling. Computation and transport latency are not charged to simulation time. The 10-ms RIC does not guarantee preventing a 2-ms budget violation after an unexpected event.

## Exact reproduction

Run from repository root in the existing environment:

```bash
python -m pytest -q
python backend/tools/phase2a_demo.py --steps 3000 --output /tmp/joint-demo.json
python backend/tools/phase2a_validate.py --output /tmp/phase2a-validation
```

The last command runs six Phase 0 smoke scenarios, all eighteen original fixed-quota Phase 1 scenarios, two old-versus-local fixed-policy equivalence pairs, and two scripted demos. It checks saved hashes, equivalence, deterministic replay and causal switching. Configurations, traces and checkpoints are not rewritten.

Fixed-policy equivalence can also be reproduced separately, e.g.:

```bash
python backend/tools/phase2a_demo.py --fixed-mode FEASIBILITY_AWARE \
  --seed 3456 --steps 3000 --output /tmp/joint-fixed-f.json
DEADLINE_STARVATION_TICKS=3 python backend/tools/phase0_evaluate.py \
  --config backend/evaluations/phase1/main_scenarios/scenario_j20.json \
  --policy fixed --fixed-u 0 --scheduler deadline_feasible --transitions causal \
  --phase1-metrics --seed 3456 --steps 3000 --output /tmp/old-fixed-f.json
```

Compare the new `physical_trajectory_sha256` to the old `trajectory_sha256`. The separate joint digest includes modes/control history and is used for joint replay. Input hashes are recorded only as provenance after execution, never exposed to the policy.

## Recorded validation

See [REPORT.md](REPORT.md), `regression_summary.json`, `test_results.txt` and `provenance.json`. Full scripted history is stored in `demo.json.gz`; `replay_summary.json` records the independent replay check.
