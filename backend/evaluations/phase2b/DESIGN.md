# Phase 2B design, specified before implementation

Base: `afa5225`. Optional, CPU-only joint controller; legacy code and checkpoints remain separate. No full training or policy-superiority experiment is authorized in this phase.

## Action and state

Action index is `((e_index * 7) + u_index) * 2 + mode_index`, with increments `(-9,-3,-1,0,1,3,9)` and modes `(PROPORTIONAL, FEASIBILITY_AWARE)`. All 98 nominal actions are retained. The mask checks exactly the Phase 2A bridge's mode/budget and quota feasibility constraints, including returning only activated reserve above current baseline. Commands go through that bridge. No dynamic physical reserve sizing.

Remove legacy feature 10 (aggregate latency excess, zero-based) from the original 15 features. The other 14 retain their original order/normalization, including the previous two reserve increments. Append the Phase 2A six features. HOL pressure h is transformed to h/(1+h), so the budget boundary h=1 maps to 0.5 and overdue remains distinguishable. Other risk features retain their [0,1] values. This gives 20 primary features. Append six risk-validity bits, two normalized class-exposure counts and two exposure-known bits: **30 network inputs**. A missing primary value has numeric placeholder zero and validity zero. Exposure counts are scaled by (minimum matured exposure × connected class UE count), clipped to [0,1]; unknown exposure is zero with its known bit zero. Full original reasons/raw exposures are retained alongside encoded state in bounded transition examples. The previous mode is represented by the current mode feature; there is no redundant previous mode input. The previous deltas describe the last issued command, not total activated reserve (which has its own retained feature).

## Reward (version joint-reward-v1)

Let service tick duration be dt=0.001 s, nominal control interval T=0.010 s, U be the fixed number of connected URLLC UEs, and the original reference offered-batch frequency be f=1000 batches/(UE s). The fixed denominator N_ref=U*f*T is 80 for the default workload; it is never the number of timely completions or a policy-dependent denominator.

Per observed service tick:

`r = - w_failure*(new_known_failures + terminal_unresolved_charges)/N_ref`

`    - (dt/T) * [w_worst*V_worst*F_worst + w_embb*max(0,1-B_embb*8/(dt*target_bps)) + w_reserve*activated_reserve/18]`

`    - w_switch*I(first tick after a mode-changing command)`.

Defaults: w_failure=1, w_worst=0.25, w_embb=0.10, w_reserve=0.02, w_switch=0.01; target_bps=96e6 (legacy eMBB throughput reference); f=1000. These are functional initial values, not tuned research results. All terms and reward are dimensionless. dt/T makes continuous costs integrate over an interval; event costs use the fixed nominal-interval workload normalization, including shortened final intervals. All configuration is versioned with checkpoints.

A newly known failure is the Boolean union of any overflow, late completion or accepted pending traffic strictly older than its stored budget (+1e-12 s tolerance). An event ledger marks each active batch once; overflow followed by expiry or completion never charges twice. Fully dropped batches are immediately failed and need no retained active record. Completed records are removed, bounding ledger memory by live backlog. Pending-not-yet-due traffic is neither successful nor failed during the episode.

At the declared terminal observation, every accepted pending batch not already failed receives one administrative unresolved charge with the same coefficient. This is recorded **separately** from known deadline failures and is not called an observed deadline miss. Already-failed pending traffic receives no additional event penalty. All unresolved batches/bytes, including previously failed batches, are reported. This explicit finite-horizon closure prevents avoiding failure costs by leaving difficult traffic unfinished; it can penalize genuinely still-feasible end arrivals and must be treated as horizon-dependent shaping in subsequent research. No final-episode summary or future event is consulted.

F_worst is the Phase 2A matured-arrival maximum per-UE failure fraction, using its full-window and per-UE minimum-exposure rule. V_worst is its validity bit. Missing fairness history contributes no invented failure or success, and the state includes validity. This is an absolute worst failure risk, not a disparity measure that rewards equally poor service. Its repeated cost is explicitly risk-duration shaping; the newly-known-failure event term remains exactly once. Class feature/exposure definitions and maturity rules are unchanged from Phase 2A. The 100-ms risk window spans multiple control intervals.

B_embb is actual eMBB served bytes in this tick from live measurement counter differences, not achievable radio rate. Reserve occupancy is the allocation actually used during that service tick. The switch cost is charged only after the action has had its first subsequent service tick. No rewards are assigned at command issue. Cumulative reward is the sum of tick rewards over the following service interval.

## Causality, episodes and learning

Reuse the original `CausalInterval` implementation unchanged through an adapter exposing joint actions/state/masks/reward. First 10-ms interval is initialization service under proportional scheduling; the first learned action is issued after tick 10. The initialized interval is excluded from learned transitions and its event counters are explicitly discarded, while live batch failure identity is retained. Configured episode duration includes that interval. Trace exhaustion is recognized at the next 10-tick boundary, allowing existing queued traffic to receive service until that boundary; this keeps subsequent episode initialization aligned without advancing time recursively. A catalog episode must contain at least two decision intervals. Each learned action affects ticks k+1 onward and has no recursive simulator stepping. Terminal state/mask is collected before UE clearing; reward closure is performed at that same observation, then the terminal transition is stored, then the next episode resets telemetry, ledger and previous command.

Two 256-unit ReLU hidden layers; 30 inputs, 98 outputs. Bounded replay, masked target-network max, Huber loss and gradient clipping. Terminal TD targets have no bootstrap, including all-false next masks; nonterminal all-false masks are errors. All learning tensors and checkpoint mapping use CPU. New checkpoints carry exact state/action/reward schemas, training configuration and seed; incompatible or legacy files are rejected by the joint loader. Legacy loader remains untouched.

## Resource policy

Workload affinity is limited to CPUs 8,9,10,15 selected from a current utilization sample. Numerical thread counts are 1, CUDA is disabled, no DataLoader workers or parallel test/training processes. Cgroups are not writable and user systemd is inaccessible here, so aggregate memory is **monitored, not hard-capped**. The standard-library workload launcher samples its own process tree and stops its own group at a conservative 3-GiB aggregate RSS threshold, below the 4-GiB ceiling. Small sequential tests, replay capacity 512 and batch 16 are used. No full training or parallel searches.
