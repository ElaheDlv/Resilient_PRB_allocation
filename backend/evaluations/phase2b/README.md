# Phase 2B joint deadline-aware masked DQN

Base commit `afa5225`; branch `phase2b/joint-deadline-masked-dqn`. This optional controller uses the existing simulator and Phase 2A bridge/telemetry, and adapts the original Phase 0 `CausalInterval` without modifying it. Existing source files, legacy checkpoints, schedulers, traffic, channels and jammer are unchanged. Physical reserve remains 18 PRBs. No full training was performed.

[DESIGN.md](DESIGN.md) defines the action, state and reward equations specified before implementation. [REPORT.md](REPORT.md) records measured results, resource limits, limitations and Phase 2C recommendations. [CHANGED_FILES.md](CHANGED_FILES.md) lists every file.

## Activation and interfaces

The new `xAppJointDeadlineDQN` is disabled by default (`enabled=False`). It is not added to existing deployments. Explicit deployment uses the same one-xApp partial-constructor pattern as the Phase 2A demo. It requires one gNB/cell, causal transitions, queue measurements, positive URLLC budgets, a 1-ms service step, `RES_DQN_DECISION_PERIOD=10`, and the unchanged 18-PRB pool. The supplied smoke harness configures these settings in a fresh process with 8+8 UEs. eMBB remains work-conserving proportional in that harness; cell-local switching affects only URLLC.

- `joint_dqn_schema.py`: `ACTIONS`, strict `encode_action`/`decode_action`, `joint_action_mask(bridge)` and `encode_joint_state(obs, previous_action)`.
- `joint_reward.py`: read-only measurement listener `FailureLedger`, immutable `RewardConfig` and tick reward computation. The ledger retains only live batches and counts each failure once across overflow, overdue pending traffic and late completion.
- `joint_dqn_agent.py`: 30→256→256→98 CPU network, bounded replay, masked Bellman targets, Huber loss, gradient clipping, target updates and explicit checkpoint compatibility checking.
- `xapp_joint_deadline_dqn.py`: adapter to the unchanged `CausalInterval`, actual service-outcome reward attribution, episode resets and bounded transition/service diagnostics.
- `phase2b_guard.py`: sequential workload launcher with inherited CPU affinity, numerical thread limits, GPU disablement, process-group monitoring and conservative memory-stop threshold. It is a monitor, **not a hard aggregate memory cap**.
- `phase2b_smoke.py`: bounded training/test harness; at most 400 decision intervals per episode and three sequential episodes. It validates checkpoint round-trip and that CUDA remains uninitialized. Default replay capacity 512, batch 16, no DataLoaders or worker processes.

`train=False` disables learning and exploration in the new xApp. To evaluate a joint checkpoint, explicitly load it with `app.agent.load(path)` after network setup and before service starts, using matching training/observation/reward metadata. Legacy checkpoints require the original controller. Loading restores model, target and optimizer, not replay contents or stochastic continuation state; this is not an exact interrupted-training resume mechanism.

## Input positions and units

Positions are zero-based. Every neural input is dimensionless. The 14 retained legacy features preserve the exact original encoder's normalization/configuration and ordering: original indices 0–9 and 11–14. Original aggregate latency-excess index 10 is replaced by UE budget-risk information.

| New positions | Meaning |
|---|---|
| 0–1 | eMBB and URLLC demand PRBs / inherited maximum-PRB denominator, capped at 1 |
| 2–3 | eMBB and URLLC quotas / that denominator, capped at 1 |
| 4–5 | eMBB and URLLC jammer-adjusted baseline quotas / that denominator, capped at 1 |
| 6–7 | eMBB and URLLC queued bytes / inherited queue references, capped at 1 |
| 8–9 | Actual last-tick eMBB and URLLC throughput / inherited throughput references |
| 10 | Inherited effective-capacity ratio |
| 11 | Activated reserve / 18 |
| 12–13 | Previous issued eMBB and URLLC reserve increments / 18 |
| 14 | Maximum active URLLC HOL/budget ratio h, transformed to h/(1+h); h=1 maps to 0.5 |
| 15 | Active URLLC overdue-HOL fraction |
| 16–18 | Tight, relaxed and worst-UE matured-arrival failure fractions |
| 19 | Current URLLC scheduler identity: P=0, F=1 |
| 20–25 | Validity bits for positions 14–19 |
| 26–27 | Tight/relaxed matured exposure divided by (class UE count × minimum exposure), capped at 1 |
| 28–29 | Known-exposure bits for positions 26–27 |

Unavailable risk values are numeric zero **with a zero validity bit**, distinct from a valid measured zero. Raw Phase 2A validity/reason/exposure data is retained in bounded transition examples. The default window is 100 ms, with at least 20 matured offered arrivals per connected class UE; worst-UE risk requires all URLLC UEs to meet exposure conditions. Pending not-yet-due batches are not counted as successful service. See Phase 2A definitions for exact arrival cohorts and strict deadline equality tolerance.

The action index is `2*(7*e_index+u_index)+mode_index`, using ascending increments `(-9,-3,-1,0,1,3,9)` and mode order `(P,F)`. Index 48 is no-op P, index 49 no-op F. Invalid physical increments stay in the nominal 98-action catalog but are masked both at selection and application. The mask reuses quota validation and the Phase 2A positive-budget rule; the bridge commits both quotas and policy atomically.

## Timing and finite horizons

Ticks 1–10 use initialized P scheduling and are explicitly initialization service, excluded from learned transitions. The first action is issued after tick 10 and affects ticks 11–20. Subsequent transitions close before the next action is selected. Episode `duration_steps` is measured in 10-tick intervals and includes initialization, so 300 intervals produce 299 learned transitions. Trace exhaustion is recognized at a decision boundary, with service continuing to that boundary. No recursive simulator advance occurs.

Terminal reward uses current live pending batches before clearing UEs. Previously known failures are never charged again. Not-yet-failed unresolved batches receive a separate administrative terminal charge; this is not an observed deadline miss. Per-UE live sidecar counters independently verify the event ledger's failure union at every terminal boundary. Terminal next masks/states belong to that same physical boundary; TD targets ignore bootstrapping when done.

## Resource-safe reproduction

Inspect current CPU load and RAM first. CPUs 8,9,10,15 were selected for this machine; change `--cpus` if appropriate for another run. Execute the following **sequentially**, never in parallel:

```bash
python backend/tools/phase2b_guard.py --report /tmp/joint-tests-resources.json -- \
  python -m pytest -q

python backend/tools/phase2b_guard.py --report /tmp/joint-smoke-resources.json -- \
  python backend/tools/phase2b_smoke.py --intervals 300 \
  --output /tmp/joint-smoke.json --checkpoint /tmp/joint-smoke.pt

python backend/tools/phase2b_guard.py --report /tmp/joint-replay-resources.json -- \
  python backend/tools/phase2b_smoke.py --intervals 300 \
  --output /tmp/joint-replay.json --checkpoint /tmp/joint-replay.pt

python backend/tools/phase2b_guard.py --report /tmp/joint-regressions-resources.json -- \
  python backend/tools/phase2a_validate.py --output /tmp/joint-regressions
```

Compare smoke/replay `trajectory_sha256`, `final_parameter_sha256`, `losses`, `first_examples`, `last_examples`, `episode_summaries` and `reward_sum` for exact equality. Wall-clock timing is intentionally excluded from determinism comparisons. The saved regression summary additionally checks the scripted Phase 2A hash against its committed reference.

Affinity applies to the guard and subprocesses. OMP, MKL, OpenBLAS, NumExpr, Accelerate and BLIS safeguards are set to one thread per process; PyTorch intra-op and smoke inter-op threads are one. CUDA visibility is empty and all learning tensors use CPU. Monitoring samples aggregate RSS and CPU every 50 ms, including the launcher; RSS conservatively sums shared resident pages. A 3-GiB RSS threshold stops only the launched process group. Cgroup writes and the user systemd bus were unavailable, so the requested 4-GiB ceiling is monitored with headroom rather than hard-enforced. Sampling can miss short peaks; this mechanism should not be used for unconstrained workloads. The guard also refuses launch if available RAM is below 4 GiB. No system settings or existing training jobs are modified.
