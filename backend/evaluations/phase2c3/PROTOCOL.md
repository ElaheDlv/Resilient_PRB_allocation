# Frozen matched-objective pilot

Base: `phase2c2/workload-feasibility-design`, commit `64a861e`.
The machine-readable protocol was written before the first training run. Its
input hashes cover only the frozen training and validation manifests, referenced
configs and original/derived URLLC traffic traces. The original eMBB trace is
verified against the preexisting C2 provenance hash and separately recorded
in final provenance. The original CSVs are ignored by Git; identical compressed
copies are included in this delivery for clean-checkout reproduction. No
future-model holdout is loaded.

All methods use `JointDQNAgent`: 30→256→256→98 with ReLU, CPU Adam at 1e-4,
Huber loss, global gradient norm clipping at 10, gamma 0.99, 2,048-entry replay,
32-sample minibatches and target copies every 100 optimizer updates. The Phase
2B reward, state, risk window and 18-PRB physical reserve are unchanged.

The physical action mask is intersected with:

| Method | Additional restriction |
| --- | --- |
| A1 | PROPORTIONAL mode only |
| A2 | FEASIBILITY_AWARE mode only |
| A3 | Current observed-state reactive reserve increment, either scheduler mode |
| A4 | No additional restriction |

For A3, compute the existing `ReactiveReserve` choice on the current physical
49-pair mask, retain its two matching entries in the 98-action catalog, and let
the learner choose the mode. Commit exactly one complete joint command through
`JointRICBridge.apply`. Recompute this restriction for each subsequent observation
and replay next-state mask. The heuristic receives current quotas and effective
capacity only. This is not two sequential commands.

NN seeds 12026 and 22026 are independent initialization/exploration/replay RNG
seeds. Scenario seeds remain the frozen C2 identities. Every learner sees the
same nine scenario entries, listed in `protocol.json`; both placement seeds are
represented in each regime. Each consecutive block of three episodes contains
all three regimes and all three jammer severities. Nine of the 18 training
entries are used per learner to retain C1's conservative per-network budget.

Each episode has 3,000 physical 1-ms ticks. The first 10 ms initialize the
observation window, then 299 causal 10-ms transitions enter replay. This gives
2,691 transitions and 2,660 optimizer steps per learner. Epsilon decreases
linearly from 1 to .05 over the first 80% of that interaction budget. All methods
use the same schedule; no budget extensions or hyperparameter searches occur.
A2 starts in F; the other methods start in P. Initialization service is included
in physical measurements and excluded from learned reward, consistently with C1.

After episodes 3, 6 and 9 (897/1,794/2,691 transitions), save and greedily evaluate
on all 18 validation entries. Validation must preserve policy parameters,
interaction/update counts, replay and the agent's random generator. The two NN
seeds are replications, not extra independent placement samples. No confidence
interval or statistical superiority claim is justified from two initializations
and two placements per regime. Validation-selected performance remains a
selection-biased pilot estimate, not a held-out generalization estimate.

Selection 1 uses the exact C2 `checkpoint_key`, with nine equally weighted
regime/severity strata and two equally weighted placements per stratum:
aggregate known failed/offered batches; mean per-scenario worst-UE failure;
offered-failure upper bound; eMBB served/offered deficit; normalized reserve;
switches/decision; earlier episode. Selection 2 swaps the first two terms,
retaining all tie breaks. This worst-first preference is transparent and has
no invented hard fairness threshold. Both selections and every checkpoint's
aggregate/worst coordinates are published; the criterion is not chosen from
results. Each NN replicate is selected independently.

Physical known failure counts are the disjoint union of overflow-affected
batches, fully admitted late completions and fully admitted overdue unfinished
batches. The upper bound adds fully admitted unfinished batches not yet due.
Terminal training penalties for unresolved traffic are administrative reward
charges, not additional physical known failures. These are batch/byte queueing
measurements, not packet reliability or end-to-end industrial control deadlines.

Resources: four-core inherited affinity, one numerical thread per process,
CUDA disabled, no workers, one simulation/training process at a time. The
inherited RSS watchdog warns at 2 GiB and stops its process tree at 2.5 GiB;
MemAvailable below 4 GiB also stops it. Sampling cannot impose a hard cap or
catch every instantaneous peak. No delegated writable cgroup is available.
