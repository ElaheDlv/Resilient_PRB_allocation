# Phase 2B validation report

Branch: `phase2b/joint-deadline-masked-dqn`. Base: `afa5225` (full hash in `provenance.json`). The commit containing this report is the implementation commit on that branch. All implementation files are new; protected Phase 0–2A source files are byte-identical to the base. No full-scale training, hyperparameter search, predictive control or Phase 2C work was performed.

## Implemented controller and definitions

The optional controller selects all 98 nominal `(eMBB increment, URLLC increment, URLLC mode)` combinations. Increments are `(-9,-3,-1,0,1,3,9)` and modes are `(P,F)`. Index `2*(7*e_index+u_index)+mode_index` is deterministic. A joint feasibility mask enforces current baseline floors, physical capacity, shared reserve occupancy and positive budgets for feasibility scheduling. Both components are applied through the existing atomic Phase 2A bridge. Physical reserve remains 18 PRBs; the underlying local schedulers and eMBB dispatch are unchanged.

The new versioned network has **30 inputs, two 256-unit ReLU hidden layers and 98 outputs**. The primary state has 14 legacy inputs (drop only original aggregate-latency feature index 10) plus six Phase 2A risk inputs. Additional inputs are six validity bits, two normalized class exposures and two exposure-known bits. HOL pressure h uses h/(1+h), with the deadline boundary at 0.5. Previous reserve deltas remain in the retained legacy features; current scheduler identity is a risk feature. Unknown measurements have explicit invalid masks and never masquerade as measured zero failures. Full mathematical definitions, units, thresholds, positions and normalization are in [DESIGN.md](DESIGN.md) and [README.md](README.md).

The dimensionless tick reward is:

`r = -(new_known_failures + terminal_unresolved_charges)/80`

`    -(dt/0.01)*(0.25*valid_worst*worst_risk + 0.10*eMBB_shortfall + 0.02*activated_reserve/18)`

`    -0.01*first_served_tick_after_mode_switch`.

Here dt=0.001 s; 80 is the fixed 8-UE × 1,000-batches/(UE s) × 10-ms reference workload. eMBB shortfall is `max(0,1-actual_served_bytes*8/(dt*96e6))`. Worst risk is the absolute maximum causal per-UE matured-arrival failure fraction, with the unchanged 100-ms window and minimum 20 matured arrivals per UE. It does not reward equally poor service. Repeated worst-risk cost is explicitly risk-duration shaping; the newly-known event term is charged once per batch across the union of overflow, expiry and late completion. An independent live sidecar identity verifies that union per UE at termination.

Pending not-yet-due traffic is not successful. At the terminal observation, previously unfailed pending batches receive separately labeled administrative charges; already failed traffic is not charged twice. These charges are not observed deadline misses. This is explicit finite-horizon shaping and may charge traffic that could have met its deadline after the chosen horizon. No future arrivals, jamming states or final-episode statistics are reward inputs. The original legacy reward remains unchanged.

The original `CausalInterval` is reused through a joint adapter. An action issued after tick k receives only outcomes from k+1 onward. Replay stores the pre-action state, action index, subsequent reward, next state/mask and done flag. The first 10 ticks are initialized proportional service, excluded from training. Terminal state/mask is collected before UE clearing. Episode reset clears telemetry exposure, live failure identity and previous joint command. Trace exhaustion closes at the next decision boundary without recursive stepping.

## Tests and historical regressions

The final full suite passed **78 tests in 88.08 seconds**: 62 existing tests plus 16 new Phase 2B cases. No tests were skipped or reduced because of resource limits. New coverage includes all 98 action round-trips, exhaustive mask/application agreement at empty/full reserve under jammed/unjammed baselines, invalid budgets, all 30 state positions and warmup semantics, partial service, equality boundaries, failure union accounting, terminal unresolved charges, masked TD targets, real parameter updates, bounded replay, checkpoint schema rejection and metadata round-trip, causal policy switching, episode resets and deterministic learning.

Real functional integrations use the original no/moderate/severe jamming models with a test-only shortened period, paired replay at each severity, and four actual neural updates per case. These are functional checks with scripted actions, not policy comparisons. The independent smoke below uses learned epsilon-greedy action selection and the original jamming period.

| Historical check | Result |
|---|---:|
| Saved Phase 0 smoke trajectories, including legacy/causal/paced variants | 6/6 match |
| Saved Phase 1 fixed-quota scenarios | 18/18 match |
| Original global allocator versus local fixed policy | Both P and F match |
| Phase 2A scripted replay | Exact match |
| Phase 2A scripted digest versus committed reference | Exact match |

`regression_summary.json`, `test_results.txt` and `provenance.json` contain the recorded evidence. These checks do not reproduce every historical publication or Phase 1.2 experiment. Locally supplied CSVs remain unmodified; their hashes are saved, but the inputs were not tracked at the integration base. Original legacy encoder, NN/reward/checkpoint code, scheduler code and Phase 2A infrastructure hashes are unchanged.

## CPU-only training smoke

Configuration: seed 3456; one episode; 8 URLLC and 8 eMBB UEs; original severe-jamming scenario and periodic CSVs; 3,000 service ticks (3 simulated seconds); 300 decision intervals including initialization. Replay capacity 512, batch 16, Adam learning rate 1e-4, gamma 0.99, epsilon 0.25, target synchronization every 20 updates, Huber loss and gradient norm limit 10. All tensors and checkpoint mapping use CPU.

| Measurement | Result |
|---|---:|
| Learned transitions / actual optimizer updates | 299 / 284 |
| Stored replay entries | 299 |
| Simulation-loop wall time | 4.1944 s |
| Guarded command wall time, including startup/save/load | 6.27 s |
| Decision-boundary computation median / p95 | 4,012.148 / 5,161.312 µs |
| Reward sum | -120.03161420103095 |
| Loss minimum / maximum | 0.0089833271 / 0.2164189965 |
| Scheduler switches | 128 |
| Parameters changed / checkpoint round-trip | Yes / passed |
| Fixed-seed second run | Exact trajectory, reward, loss, parameter and terminal-accounting match |
| CUDA initialized | False |

Decision-boundary time includes observation/encoding, transition processing, learning, mask construction, selection, bridge application and bounded diagnostics; initialization has no learning update. Non-boundary service work is excluded from that boundary statistic. Wall-clock compute/transport latency is not charged to simulated time. These measurements provide no real-time deployment guarantee.

Examples from the saved trace:

| Issued after tick | Service outcome ends at tick | Action index | Reward | Valid next actions | Terminal |
|---|---:|---:|---:|---:|---|
| 10 | 20 | 67 | -0.02741778 | 58 | No |
| 20 | 30 | 67 | -0.02118556 | 54 | No |
| 30 | 40 | 67 | -0.02476167 | 66 | No |
| 2990 | 3000 | 74 | -1.19429778 | 58 | Yes |

Every saved example verifies the receipt's first affected tick is issue+1, a 10-tick observed interval, 30 finite state inputs, 98 next-mask bits and reward equal to minus the recorded costs. Terminal TD targets ignore that mask's bootstrap value. Unit/integration tests additionally verify every service tick's actual mode during switching.

The ledger observed 24,000 offered URLLC batches and 5,954 known failures, including initialization; initialization contained 80 offers and zero failures. At the terminal boundary, 71 batches / 43,798 bytes remained unresolved. Of those, 36 received administrative charges; 35 had already been penalized as known failures. Per-UE failure totals independently match live sidecar overflow + fully admitted late completions + non-overflow overdue pending batches. These are accounting checks, not reliability or performance claims.

`smoke.json.gz` contains bounded first/last transition examples, masks, raw feature validity/exposures, the last 512 service records and terminal checks. `replay_summary.json` records exact deterministic comparisons. `smoke_checkpoint.pt` is a functional smoke artifact, not a converged policy: it contains model/target/optimizer state, state/action/reward versions, observation/training configuration, seed, resolved catalog and input hashes. Incompatible schemas/configurations and legacy state dictionaries are explicitly rejected. Replay and random-stream continuation are not saved, so exact interrupted-training resume is unsupported.

## Mandatory resource-limit report

Initial inspection found about 25 GiB available RAM. CPUs **8,9,10,15** were selected among the least busy from a per-core sample. The guard reapplied affinity and inspected assigned-core utilization and available RAM before each expensive command. All test, regression and training workloads were sequential.

| Requirement | Observed/enforced behavior |
|---|---|
| Logical CPUs | 8,9,10,15; inherited `sched_setaffinity`, at most four simultaneously available logical CPUs |
| Maximum sampled aggregate CPU utilization | 138.64% (100% equals one logical CPU; includes guard) |
| Maximum sampled aggregate resident memory, across all eight guarded commands | 1.5132 GiB |
| Smoke aggregate resident memory | Approximately 0.706 GiB |
| Memory enforcement | **Monitored, not hard-capped**; no writable delegated cgroup and user systemd bus inaccessible |
| Conservative stop condition | Kill only our launched process group at sampled RSS ≥3 GiB, leaving headroom below 4 GiB |
| Numerical threads | One per process for OMP/MKL/OpenBLAS/NumExpr/Accelerate/BLIS; PyTorch intra-op one, smoke inter-op one |
| Training/evaluation concurrency | One heavy training or evaluation process at a time; no parallel searches or xdist |
| DataLoader workers | Zero |
| GPU | CUDA visibility empty; tensors CPU; CUDA initialization false in both smoke runs |
| Skipped/reduced tests | None; no memory-stop event |

`resource_reports.json` preserves all eight guarded command reports, including preliminary checks and final runs. The RSS sum conservatively includes shared pages multiple times. Monitoring is sampled every 50 ms and can miss short peaks; it is not a substitute for cgroup enforcement on large workloads. A guard also refuses launch below 4 GiB available RAM. No system-wide CPU settings, GPU configuration or existing process was modified or terminated. Full-scale training was deliberately excluded by scope.

## Remaining research risks and Phase 2C recommendation

1. The reward weights, 96-Mbps eMBB reference and 1,000-batch/(UE s) reference rate are initial documented choices, not tuned values. A different offered workload needs an explicitly versioned reference rather than a completion-dependent denominator.
2. Finite-horizon administrative penalties can dominate short episodes and charge newly arrived feasible traffic. Full studies should test horizon sensitivity and use a consistently defined arrival-admission cutoff plus a draining/accounting protocol if selected in a future design. This phase does not change traffic or add draining.
3. The 100-ms worst-risk window mixes outcomes from multiple policies. The 30-input state remains an aggregate, partially observed representation; validity masks do not make it Markov. Credit assignment and reward-shaping sensitivity require investigation.
4. Absolute worst-risk cost avoids rewarding universal poor service, but cannot guarantee every UE's fairness or feasibility. Existing poor-channel/overload and recovery tradeoffs remain. Analyze per-UE outcome attribution before interpreting aggregate gains.
5. A 10-ms RIC cannot prevent the first unexpected 2-ms deadline miss. Local 1-ms scheduling remains essential. E2 signaling and transport/computation delays are idealized, and CPU timings are not deployment measurements.
6. Live backlog scans and measurement maintenance scale with congestion. Replay and diagnostics are bounded, but long-run profiling and resource monitoring remain necessary. The current environment cannot hard-enforce aggregate memory, so no unconstrained experiment should be launched here.
7. The smoke confirms working updates, not convergence, superiority, a useful switching policy or industrial end-to-end reliability. The checkpoint is not an exact resume artifact.

For a separately authorized Phase 2C: first fix scenario splits, horizons, objective weights and stopping rules. Train fresh matched controls: reserve-only deadline-aware DQN with P fixed; reserve-only DQN with F fixed; and the joint 98-action DQN. Include fixed-reserve P/F and switch-only ablations to isolate the two decisions. Keep observation/reward definitions, environment realizations, interaction budgets and network capacity comparable; preserve physical 18-PRB reserve and both default traffic traces. Frozen legacy DQN may be a diagnostic baseline, not an independently optimized joint controller.

Use disjoint tuning and held-out paired seeds, equal simulator-service interaction budgets, multiple independently trained seeds, and evaluate greedily with identical exogenous scenarios. Report aggregate/class/per-UE failure decomposition, worst-UE service, unresolved/dropped traffic, recovery, eMBB throughput, reserve/PRB use, switching and runtime, with paired uncertainty across independent seeds. Select hyperparameters on validation scenarios only. Retain the four-CPU/4-GiB/CPU-only restrictions unless the user explicitly changes them; run one process at a time. **Phase 2C has not been started.**
