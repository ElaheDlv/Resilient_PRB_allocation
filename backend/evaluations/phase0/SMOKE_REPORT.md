# Phase 0 smoke-test report

Date: November 12, 2026. Audited source baseline: `eef3ad6`.
Implementation branch: `phase0/simulator-validation`.

## Verification

`python -m pytest -q`: **15 passed** (9.78 s in the recorded run).

Coverage includes causal action/outcome alignment, actual trainer replay storage
and learning, terminal handling and episode cleanup, fixed per-UE budget assignment,
partial FIFO service, overflow and byte conservation, full event logging,
zero-capacity observations, invalid-action rejection, constant and variable jammer
ON/OFF transitions, integer work-conserving allocation, and deterministic
paced/unpaced replay with instrumentation enabled/disabled. Allocation tests also
enumerate bounded integer inputs and feasible reserve allocations.

`python backend/tools/phase0_smoke.py --output /tmp/phase0-smoke-final`
completed all six runs and all four comparison checks passed.

## Representative evaluation

The existing `gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json` catalog's first
episode was used: one cell, 8 eMBB and 8 URLLC UEs, 1 ms simulation step, 10-tick
decision period, baseline 140/60 PRBs, fixed 18-PRB reserve, periodic 10-PRB jamming
with a 2.5 s period and 50% duty factor. Traffic CSVs were used unchanged.
Seed: 1234.

Checkpoint:
`backend/checkpoints/res_dqn/xAppResilientDQNAllocator_20260123_215931_final.pt`.

SHA-256:
`57968f350effc428ad0cdd6914f6a281fd410ba276d9674f6b509e8c2f5484a3`.

| Run | Simulated steps | Loop wall time |
|---|---:|---:|
| Original core, legacy timing/scheduler | 9,990 | 2.416 s |
| Phase 0, legacy timing/scheduler | 9,990 | 2.450 s |
| Legacy plus measurement instrumentation | 9,990 | 3.211 s |
| Causal timing, legacy scheduler, measured, unpaced | 9,991 | 3.520 s |
| Same causal configuration, paced | 9,991 | 15.809 s |
| Causal timing, work-conserving scheduler, measured | 9,991 | 3.515 s |

These are individual runs, not confidence intervals. Wall time covers stepping,
trajectory hashing and enabled measurements, and excludes initialization,
checkpoint/trace loading and final JSON serialization. All runs used CPU inference
with one PyTorch thread. Machine/runtime versions and full precision timing values
are in `smoke_summary.json`.

The causal run contains the one bootstrap service tick before the first action.
The catalog's 999 decision intervals account for its 9.99 s controlled horizon;
the causal run ends at 9.991 s. This difference is intentional and should not be
mistaken for an equal-horizon performance comparison between modes.

## Exact trajectory comparisons

The original-core, Phase 0 legacy, and instrumented legacy runs all produced:

`7604facf361157a511702e0978589797c9266ba7c18ff83a0ba3c34352454c59`.

The paced and unpaced causal runs both produced:

`51ea00a445827a2794dc8912a06a42fdd9d3d52610525e49970d7cf0e5975d2f`.

The work-conserving scheduler produced its own trajectory:

`3eea24c87d48e7b1e8e3c42112eb655ac39c3ac1f759b20b50121a237aca817f`.

Hashes cover every tick's slice quotas, current baseline, effective capacity,
per-UE queue bytes, served bitrate, latency indicator and PRB grants. They compare
complete tick trajectories, not just final averages. Causal and legacy trajectories
are expected to differ; no unchanged-results claim is made across those modes.

## Scope of legacy confirmation

The reference run restores the original BS, Cell, UE, resilient DQN allocator and
validator source files from `eef3ad6` in a temporary backend tree. It retains the
Phase 0 explicit-xApp bootstrap and deferred optional SDK import so the core can
run without unrelated UI/AI dependencies. This checks the original experimental
core in a common offline harness; it is not a complete server-stack regression.

For this specific checkpoint, seed, profile and episode, legacy core behavior is
reproduced exactly and measurement instrumentation does not perturb it. Historical
paper figures, the full 50-episode catalog, all severities/profiles, and other
checkpoints have **not** been reproduced.

## Remaining validity issues

1. A January 14 checkpoint initially inspected has 13 inputs, while the current
   state schema has 15. It was rejected; the compatible January 23 checkpoint above
   was used instead. Historical checkpoint/config/schema provenance remains needed.
2. The old checkpoint's causal execution is a functional smoke test, not evidence
   of optimality or training under corrected timing. Future controller comparisons
   must use the same causal timing, warm-up, horizon and reward aggregation.
3. The existing replay stamps enqueue time at simulation ticks and can combine
   multiple source packets into one batch. Measurements are queue-residence and
   batch-completion metrics, not packet reliability or end-to-end latency.
4. The smoke workload contains no configured per-UE budgets. Budget and overflow
   semantics are tested with deterministic unit/integration cases; this smoke test
   is not an industrial-deadline experiment.
5. The jammer remains unchanged and follows global simulation time. Phase and
   variable-severity selection must be paired across future policy comparisons.
6. Legacy synchronous internal evaluation remains unsuitable for service-driven
   comparison; causal mode rejects it. The existing unsupported `partial` policy
   is excluded from the new evaluator.
7. Work-conserving scheduling is enabled independently and changes network
   dynamics. No throughput, fairness or deadline-performance benefit is claimed.

## Exact Phase 0 changed-file list

Modified:

```text
backend/main.py
backend/network_layer/base_station.py
backend/network_layer/cell.py
backend/network_layer/ric.py
backend/network_layer/simulation_engine.py
backend/network_layer/ue.py
backend/network_layer/xApps/xapp_resilient_dqn_allocator.py
backend/network_layer/xApps/xapp_resilient_policy_validator.py
backend/settings/__init__.py
backend/utils/websocket_utils.py
```

Added:

```text
backend/evaluations/phase0/README.md
backend/evaluations/phase0/SMOKE_REPORT.md
backend/evaluations/phase0/smoke_summary.json
backend/network_layer/prb_accounting.py
backend/network_layer/proportional_scheduler.py
backend/network_layer/queue_measurements.py
backend/network_layer/resilient_interval.py
backend/settings/phase0_config.py
backend/tests/test_phase0.py
backend/tools/phase0_evaluate.py
backend/tools/phase0_smoke.py
pytest.ini
```

The three pre-existing untracked research-review Markdown files at repository root
were not changed and are not part of Phase 0.
