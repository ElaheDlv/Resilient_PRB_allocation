# Phase 0 simulator validation

Phase 0 implements optional causal DQN collection, a work-conserving proportional
scheduler, measurement-only queue budgets, resource invariants, and an unpaced
offline evaluator. It does not implement deadline-aware control, forecasting,
expiry/drop policies, or adaptive reserve sizing. The default pool remains 18 PRBs.

## Modes

| Setting | Default | Optional behavior |
|---|---|---|
| `RES_DQN_TRANSITION_MODE` | `legacy` | `causal`: collect subsequent service outcomes |
| `PRB_UE_SCHEDULER` | `legacy` | `proportional_wc`: demand-proportional integer allocation bounded by backlog and per-UE cap |
| `PHASE0_MEASUREMENTS` | `0` | `1`: parallel FIFO batch accounting, no queue-policy changes |
| `PHASE0_MEASUREMENT_PATH` | empty | Stream complete measurement events to JSONL |

Legacy timing and scheduling remain selectable. Physical quota validation and
zero-capacity observation fixes apply in both modes. Invalid actions now raise
instead of relying on a caller's mask or silently clamping baseline violations.
This intentionally changes behavior for invalid configurations/actions.

## Causal timing contract

`SimulationEngine.step()` still updates UEs, then BS jamming/arrivals/service, then
the RIC. At the first post-service callback the causal controller observes the
network and issues an action. That bootstrap tick is not a learned transition.
The action governs subsequent service ticks; at the next decision boundary the
collector completes `(state, action, reward, next_state, done, next_mask)` before
selecting another action. Previous-action features identify the action that just
governed service. No callback advances the simulator recursively.

The interval reward is the **sum of the existing per-tick reward**, including
reserve costs during the actual interval. Gamma remains per decision. Legacy
reward curves and causal reward curves are therefore not directly comparable.
Short terminal intervals have no bootstrap because `done=True`.

`duration_steps` retains the catalog's existing decision-count interpretation;
with period 10 and dt 1 ms, 999 decisions represent 9.99 seconds of controlled
service. A causal run additionally has its bootstrap tick. Trace exhaustion may
end an episode before all bytes are served. Such bytes are recorded as unfinished,
not silently treated as successful delivery. Terminal queues are finalized before
cleanup. A run cut off mid-interval reports its unfinished interval ticks rather
than inventing a transition. Finish requested episodes for controller comparisons.

Causal mode currently requires exactly one cell and consistent fixed
jammer/controller pool sizes. Synchronous `_run_eval()` is rejected in causal
training (`RES_DQN_EVAL_INTERVAL` must be 0); use external evaluation. Legacy
in-process evaluation retains its historical behavior and is not validated as a
service-driven evaluation loop.

## Queue measurements

The original BS FIFO `(arrival_time, remaining_bytes)` and raw traces are retained.
The observer receives the same enqueue/service events in a sidecar. Optional
budgets are BS queue-residence budgets in **seconds**, not end-to-end deadlines:

```json
{
  "URLLC": {
    "ue_count": 8,
    "queue_budget_s": 0.010,
    "queue_budgets_s": {"0": 0.005, "1": 0.020}
  }
}
```

Add these keys to a slice's existing episode configuration, retaining its trace
fields. Per-index overrides take precedence over the slice default. These numbers
are illustrative test budgets, not claims about an industrial application.

Each accepted enqueue is an identifiable **byte batch**, not an original packet.
Arrival times are the existing replay's BS enqueue timestamps. This preserves its
tick quantization and any prior CSV aggregation. Service occurs at the existing
simulator timestamp; serialization and other end-to-end components are not added.
Equality to the budget is compliant, with a 1e-12 s numerical tolerance.

Outputs distinguish:

- Offered, accepted, overflowed, served and unfinished bytes. For each UE,
  `offered = served + unfinished + overflow`.
- Served-byte residence-time violations, counted only when those bytes are served.
- Accepted-batch completion violations, counted only on final accepted-byte service.
- Partially admitted completed batches, separately identified by overflow counts.
- Unresolved batches/bytes, including those already over budget at termination.
- HOL waiting-time samples and budgeted denominators.

A partially admitted batch completing its accepted portion does not imply the
original offered batch was successfully delivered. Fully rejected batches appear
as overflow events, not completions. No batch-level counter is packet reliability.
No queue is expired or reprioritized by instrumentation.

The in-memory diagnostic event ring retains 256 recent events; cumulative counters
and unresolved batch records remain available. Use `--events` or the environment
path for a **complete** event history. JSONL can be large at millisecond cadence.

## Running

From the repository root:

```bash
python -m pytest -q

python backend/tools/phase0_evaluate.py \
  --config backend/assets/episodes/test_config/gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json \
  --policy dqn \
  --checkpoint backend/checkpoints/res_dqn/xAppResilientDQNAllocator_20260123_215931_final.pt \
  --transitions causal --scheduler proportional_wc --measurements \
  --steps 10020 --output /tmp/phase0-evaluation.json
```

The evaluator defaults to unpaced execution; add `--paced` for wall-clock sleeps.
For full event logging add `--events /tmp/phase0-events.jsonl`. Omit optional flags
and choose legacy transitions/scheduler to reproduce legacy logic. The regular
backend also accepts `--mode headless --unpaced`; its default remains paced.

The standalone evaluator explicitly loads just the validation xApp, using the
actual simulation engine and network classes. It does not load unrelated live UI
or AI-service xApps. An optional SDK import was moved into the chat function so
offline simulation does not require the agent SDK. The full server may still
require its normal UI/knowledge dependencies.

The evaluator fixes seed, 1 ms step, 400 m UE spawn radius, simple topology and the
18-PRB pool. A fresh process is required for each settings configuration. Its
trajectory hash covers every tick's quotas, baseline, effective capacity, per-UE
queue/rate/latency/PRB allocation. Reported runtime includes the stepping loop,
measurement overhead and hashing, but excludes initialization/checkpoint loading
and final JSON serialization. Timings are single-run CPU measurements, not a
statistical performance benchmark.

To repeat the source-regression smoke comparison:

```bash
python backend/tools/phase0_smoke.py --output /tmp/phase0-smoke
```

This restores five core source files from commit `eef3ad6` in a temporary tree and
uses the same offline bootstrap. It does not check out or alter the user's tree.
See `smoke_summary.json` and `SMOKE_REPORT.md` for the measured scope and results.

## Remaining comparison limitations

- The smoke regression is one configuration, seed, compatible checkpoint and
  episode. Historical paper figures and all other checkpoints are not reproduced.
- A January 14 checkpoint inspected during validation has 13 inputs; the current
  model requires 15. It correctly fails loading; do not pad it silently.
- Old weights in causal mode are a transfer/smoke test, not evidence of a policy
  trained with the corrected timing. Future comparisons require retraining on a
  common causal substrate and matched reward/cadence conventions.
- The existing jammer is unchanged: phase follows the global simulator clock,
  and variable severity is selected per window. There is no new per-episode
  attacker reset or physical interference model.
- The original validator's `partial` helper is commented out; the offline evaluator
  excludes that unsupported policy. DQN, QR-DQN evaluation, aggressive, random and
  idle are exposed; QR-DQN training timing was not changed by this work.
- No industrial safety, end-to-end reliability, or scheduler performance gain is
  claimed. Work-conserving scheduling changes the transition dynamics and needs
  separate evaluation before it is used as a baseline for new controller claims.
