# Phase2E prospective independent validation and efficiency protocol

Base fbafd09. `protocol.json` was frozen before G1 profiling or independent
screening. The Phase2D formulation in `feasibility_gated_fairness.py`, lambda=1,
100ms post-service offered/served window, matured exposure20, credit arithmetic,
FIFO-prefix gate, 3-tick zero-service guard and F kernel are fixed. Input hashes
protect that source and P/F. No neural model is constructed by this study;
no reward/state/action/interface/model/jammer/reserve changes occur.

## Independent membership

Candidate indices15001–15024; actual scenario seed=index*4096. The unchanged UE
registration initializer uses scenario seed+100000+slice_index*1000+UE_index.
Compare all384 new registration streams against all1152 streams of the prior
C2 candidate blocks12001–12024,13001–13024,14001–14024. The old discovery protocol
is read only for block bounds; the frozen joint-DQN holdout manifest, scenarios
and outcomes are never evaluated. All Phase2D development placements lie in
the old C2 validation block, so the new block is separate from those too.

Run exactly24 reference-P/zero-reserve20ms calibrations; retain positive static
radio rates, original arrival sizes/periodicity, budgets and positions. No G1
outcomes are available during membership selection. In ascending seed order,
select first three original650-byte placements satisfying the existing C2
feasible certificate, then first three unused satisfying its contested
certificate. Use first three remaining placements for the **existing C2
normalized-overload definition**: copy the original URLLC CSV changing only its
length column to floor(85.8/sum_i(1/r_i)) bytes. Do not change any source CSV.
Feasible/contested traffic remains650-byte/ms. Overload remains the prior
placement-normalized synthetic workload definition, not a new traffic process.
Budgets2/10ms, budget assignment seed2026, radio models, baseline quotas,
jamming severities0/10/20 and timing, 18PRB physical reserve and3s horizons remain.

No replacement based on P/F/G1 performance, no extension if certificates yield
fewer than three placements. Freeze `independent_manifest.json`, config/trace
hashes and membership audit before independent evaluation. The discovered pool
contains3/3/3 placements,27 scenario entries. `exogenous_probes.json` contains
only exogenous calibration records; it excludes reference outcome measurements.

## Frozen bounded execution budget

- Profile original G1 on the first existing C2 validation j20 scenario of each
  regime;3 episodes. These are development inputs, not independent tests.
- Profile optimized implementation on those same3 entries after tests. Do not
  use profiled timing for feasibility conclusions: cProfile overhead is large.
- Run original and optimized G1 separately for all10 Phase2D prototype scenarios:
  20 full3s episodes. Require exact per-UE allocation and service trajectory hashes,
  control digests, physical outcomes/classes/UEs/recovery, administrative rewards,
  causal service-deficit summaries and exogenous/budget/resource trajectories.
- Evaluate P,F,fixed G1 on all27 new entries,81 episodes. Use the identical frozen
  ReactiveReserve rule and assert complete quota-trajectory equality.
- Replay original G1 on the three independent contested severe entries, comparing
  with optimized independent runs:3 additional episodes. No coefficient tuning.

Maximum110 full episodes plus24 short probes, excluding small unit tests. No
training, search, automatic budget growth or future-model holdout evaluation.
All independent placement results and individual harms are retained, not just
means. Three placements per stratum do not justify a strong population/rare-event
reliability or significance claim; milliseconds/UEs/severities are not independent
placement replicates. Report paired differences and counts/ranges instead.

## Optimization proof and scope

cProfile identifies repeated full collection and100ms arrival-cohort processing
as the principal overhead. The original credit is

    C_i=min(prefix_i, max(0,a_i-b_i)*(1+m_i))

only with a full service window and reachable timely FIFO target; otherwise0.
Therefore all credits are exactly0 if no UE has a full window, a_i>b_i and a
reachable target. Reorder only pure service-window/target computations, then
skip the full telemetry query in that case. The original zero-service guard
cannot make a zero credit positive, so the zero-credit fair loop is also a no-op.
Otherwise query the original collector at the original current timestamp and
use exactly the old arithmetic, entitlement, target helper and allocation kernel.
No stale10ms caching, approximate time bins, new thresholds or extra heuristic.

`equivalent_g1.OptimizedG1Allocator` is a separate implementation; the original
remains untouched. Strict **scheduling and numerical-outcome** equivalence is
required; private collector query counts differ. A diagnostic unused miss
fraction is null when deliberately not queried, rather than being falsely
reported as zero. It does not enter allocations, rewards, the DQN state or
published service/physical outcomes. Internal query-watermark/cache identity is
not claimed. Causal next-time collection prunes the same old arrival cohorts;
extra deletion timing cannot change queues or current-window failures. Full
replays test that claim in addition to the mathematical zero-credit condition.
If equivalence fails, do not substitute it as G1; preserve original G1 evaluation
and label the differing implementation a separate candidate.

## Timing and decision interpretation

Measure wall and process CPU around **entire `engine.step(.001)`**, including
traffic/queue/radio simulation, P/F/G allocations, service, all reference RIC work,
ordinary measurement/telemetry and G post-service credit/window updates. Baseline
P/F have a read-only service window for comparable deficit measurements. Exclude
setup/reset and post-timer trajectory hashing/CSV export. Report per-tick
median/p95/p99/max and >1ms counts for all ticks, nonterminal ticks, RIC boundaries
and ordinary ticks; report allocator+mandatory G post-service update as a
separate transaction scope. The full simulation includes instrumentation and
is not a production radio implementation. Wall time includes OS preemption;
CPU time is also supplied. Terminal report construction is visible in all-tick
results and separately excluded from nonterminal results, never hidden.

Use nearest-rank quantiles. A1ms period is the necessary benchmark, not an
invented application SLA or hard real-time certification. Quantify misses and
per-placement variability. There is no justified universal acceptable aggregate
loss threshold; publish aggregate/class/UE trade-offs without inventing one.
No Pareto improvement unless all required service objectives and evaluated UEs
satisfy it. Without reproducible benefits and defensible runtime evidence,
recommend retaining P/F rather than expanding the controller.

Resources: inherited affinity6,7,11,14; numerical threads1; CPU only; sequential;
warning2GiB, RSS-watchdog stop2.5GiB, MemAvailable stop4GiB. No delegated hard
memory cap is available; report sampled peaks/limits. Existing training tests
are explicitly deselected, not silently executed. Stop after reporting.
