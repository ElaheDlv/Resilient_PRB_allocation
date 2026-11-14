# Proposed reward ablation — not executed

## Hypothesis and scope

Test whether absolute100ms historical worst-UE risk makes learning less
responsive to useful current service than a direct per-UE event signal. This
is a credit-assignment hypothesis, not a finding that removing fairness is safe.
No alternative reward returns, ablation training or ablation evaluation were
computed in Phase2F. Retain the original reward until a subsequent authorized
matched experiment supports a change.

Use three arms, all with the **full joint A4**30-state /98-action interface and
its original causal mask. P/F scheduler equations, quotas, traffic, channel,
jammer, physical18-PRB reserve and observation values stay identical.

| Arm | Fairness term per10ms transition | Purpose |
| --- | --- | --- |
| R0 | Original sum of `.025 * valid_worst_risk` over10 service ticks | Reproducible reference. |
| R1 | Zero historical worst-risk term; leave all other components unchanged | Diagnostic removal; may sacrifice worst-UE performance. |
| R2 | `.25 * max_u(newly_known_physical_failures_u_in_interval / 10)` | Test recent per-UE events instead of repeated historical failure fractions. |

R2 counts exactly-once physical ledger failures in the interval, including expiry,
late completion and overflow; **exclude administrative terminal charges** from
this fairness term. Ten is the fixed per-UE reference exposure
`1000 batches/s * .010s`. Aggregate failure and terminal components remain exactly
original. Do not divide by variable observed offered counts or add hidden state.
**Do not clip** R2: more than10 failures first becoming known for one UE in an
interval may exceed.25, and must be reported as such. This prespecified event
normalization preserves the original steady10-failures-per-UE fairness scale;
it deliberately changes transient duration and validity behavior. It does not
purport to be a matured failure-probability estimate. It requires causal per-UE
ledger counters in a future implementation, with conservation tests before use.

Do not simultaneously adjust switch, reserve, eMBB or terminal weights. Do not
add a fourth arm from pilot outcomes. Keep gamma unchanged; otherwise the timing
comparison would be confounded. R1/R2 are reward changes requiring new training,
not claims based on rescoring a frozen policy or inventing reserve counterfactuals.

## Matched bounded pilot protocol

Before running, freeze an experiment manifest with code/config/input hashes and
initial network fingerprints. For each of NN seeds12026/22026, construct one
initialization and copy its policy/target parameters into all three arms. Use the
same independent replay/action RNG seeds. Exploration actions may diverge once
policies learn; do not force different policies to take identical actions.

- Architecture: original two256-unit layers,98 outputs;30 inputs and identical
  action masks. CPU only; original Huber loss, gradient clipping10, Adam1e-4.
- Replay2048, batch32, gamma.99, target interval100. Preserve original epsilon
  schedule1→.05 over.8 of the budget.
- Nine3s episodes in the exact C3 `training_order`;2691 paid transitions per
  learner, checkpoints after3/6/9 episodes. No automatic budget extension.
- Six sequential learners total (three reward arms ×two initialization seeds).
  Pair all exogenous placements/traffic/channel/jamming and episode ordering.
- Use the same C2 training/validation scenarios already used in C3, without
  changing loads to favor an arm. These are development data. Do not touch the
  reserved future-model holdout or claim independent confirmation.
- If that short budget is uninformative, report an inconclusive pilot; do not
  spend more training selectively on a promising arm.

## Prespecified selection and metrics

Use the original C2 aggregate-first physical hierarchy for the primary checkpoint
selection in **every arm**, including its original tie-break rules. Report the
original worst-first selection as a separately labeled secondary sensitivity
analysis for every arm. Do not select one arm with aggregate-first and another
with worst-first. Do not select by each arm's training return: those objectives
are different and raw return magnitudes are not comparable across arms.

Prespecify the identical18 validation entries, pairing by scenario and NN seed.
Report aggregate known physical failure fraction, unresolved-not-due fraction,
upper failed-offered fraction, worst UE, every individual UE, each deadline class,
eMBB deficit/throughput, reserve PRB-time, mode switches and recovery. Keep
administrative charges separate from actual failures. Include reward-component
distributions, saturation, history carry and learning curves, but physical metrics
remain primary. No new composite metric selected after results.

Show per-scenario paired differences and both initialization outcomes. There are
only two independent NN seeds and two placements/regime; avoid confidence claims
from thousands of dependent ticks or checkpoint rows. Report ties, disagreements
between primary/secondary selections, and harms alongside gains. No Pareto claim
unless every stated objective and each evaluated UE meets its criterion under
explicit uncertainty. No arbitrary industrial acceptance threshold is inferred
from this audit; deployment SLAs require separate application evidence.

## Decision rule and safeguards

R2 is promising only if its matched physical aggregate-failure improvement is
supported across both initialization seeds, its worst/individual-UE harms are
explicitly acceptable under a stated application trade-off, and gains do not
come solely from additional reserve or lost eMBB. If results are mixed or small,
retain R0 and mark the causal hypothesis unresolved. R1 improvement alone would
show a trade-off, not justify removing fairness from an industrial controller.

Use at most four affinity-bound logical CPUs, sequential jobs, CPU-selected
PyTorch, one numerical thread, zero DataLoader workers,2GiB RSS warning and
2.5GiB process-tree stop, plus system-available-RAM monitoring. Freeze/check input
hashes before each run and stop on any ledger/resource accounting failure. This
plan does not authorize its execution in Phase2F.
