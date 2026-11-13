# Phase 2C-1 paired training pilot

Branch `phase2c1/paired-training-pilot`, integration base `ed267de`. This is a bounded exploratory study, not a journal-scale evaluation. [REPORT.md](REPORT.md) separates software checks, measured pilot findings and remaining uncertainty. The key result is that all held-out policies, including no activated reserve, have zero known deadline failures; the pilot does not establish a deadline advantage for joint learning.

## Frozen protocol and data separation

[protocol.json](protocol.json) and [AUDIT.md](AUDIT.md) were specified before training. Scenario seeds are independent of network initialization:

| Purpose | Scenario seeds | Jamming severities | Use |
|---|---|---|---|
| Training | 1101, 1102, 1103 | 0,10,20 | Nine 3-second episodes per learner; seed-major order |
| Validation | 2101 | 0,10,20 | Checkpoint/rule selection only |
| Held-out | 3101, 3102 | 0,10,20 | One frozen paired pilot; no subsequent retuning |
| Functional tests | 9091 | Test-specific | Software correctness, excluded from study |

One initialization/exploration/replay seed, 12026, is used for each learner. Each episode is a fresh single-entry catalog with canonical `phase1_pilot` UE identifiers and catalog index zero. Placement seed, simulator RNGs, trace arrivals, 2/10-ms budget assignment and initial resource state are therefore independent of controller order and checkpoint. Complete per-tick exogenous hashes plus deadline/position assignment hashes must match across methods within each pair. This avoids the catalog-index/placement coupling in a naive multi-episode use of the original smoke harness.

The unchanged default configuration has one gNB/cell, 8 URLLC and 8 eMBB UEs, original periodic downlink CSVs, 1-ms service, 10-ms decisions, unchanged channel/jammer and a fixed 18-PRB physical reserve. Original Phase 1.2 capacity-conditioned regimes require generated/scaled traces and were not used here. Mean offered-fluid demand/current baseline is reported as a diagnostic, not a proof of deadline feasibility.

## Controllers and learning

| Method | Reserve controller | URLLC local mode |
|---|---|---|
| B0 | No activation, pool still physically reserved | P |
| B1 | Existing validator's aggressive reactive reserve heuristic | P |
| B2 | Same reference heuristic | F |
| B3 | Retrained original 15-state/49-action DQN, original reward | P |
| B4 | Same original learning objective/observations | F |
| B5 | Separately retrained original reserve DQN | Causal hysteretic rule |
| P | Phase 2B 30-state/98-action joint DQN | Learned with reserve action |

P/F are the existing work-conserving proportional and feasibility-aware schedulers, unchanged. B1/B2 reuse the reference current-baseline-loss target and nearest feasible signed increment; no future jammer settings enter the policy. Every reserve action and mode change goes through the Phase 2A atomic bridge. B2/B4 initialize in F; the others initialize in P. The first ten service ticks are initialized service, excluded from replay but included in every method's physical metrics. Every learner then receives 299 transitions per 300-interval episode.

B5 enters F when valid maximum HOL/budget exceeds its selected threshold; it returns to P only after two consecutive decisions at or below half that threshold. Missing pressure retains mode and resets the release count. Candidate entries .75 and 1.25 are evaluated on validation using selected B3 reserve weights, with a declared lower-threshold tie-break. Entry .75 is frozen before B5 is trained from scratch. The final B5 is not merely a frozen transfer baseline.

All learners use two 256-unit ReLU layers, CPU Adam 1e-4, gamma .99, replay 2048, batch 32, one update per transition after 32 entries, and target synchronization every 100 optimizer updates. Exploration decreases linearly from 1 to .05 over 80% of the fixed 2691-transition budget, then stays .05. Joint learning retains Phase 2B Huber loss and norm clip 10; legacy learners retain the source MSE loss without clipping. State, action and reward definitions/weights remain unchanged. The generic agent gained overridable dimensions/network/loss and an explicit epsilon argument; default Phase 2B behavior and checkpoint metadata remain compatible.

Candidates are saved after episodes 3,6,9 and evaluated greedily on validation. Selection minimizes mean physical offered-failure **upper bound**, then mean worst-UE upper bound, then negative eMBB Mbps, then earlier episode. Every candidate ties at zero failure and 96 Mbps, so all selected checkpoints are episode 3: **897 transitions / 866 updates**, despite the full 2691-transition / 2660-update training run for each method. No held-out result changes that choice.

## Metrics and censoring

Independent physical reporting reuses Phase 1 `Phase1Metrics`, including per-class/UE batch metrics, actual eMBB service, reserve/PRB use and HOL-based recovery. All initialization and controlled service ticks are measured. Offered batch outcomes are mutually exclusive: on-time completion, known failure, or unresolved-not-yet-due. Known failure is the union of overflow, fully admitted late completion and non-overflow pending overdue traffic. Lower bound is known failures/offered; upper bound adds unresolved-not-yet-due/offered. No policy receives an extra draining interval. Administrative terminal reward charges are separately labeled and never added to observed physical deadline failures.

eMBB satisfaction is actual served bytes/offered bytes over the same horizon; raw Mbps is also reported. Per-UE resource diagnostics give mean granted PRBs, capacity bytes/(PRB tick), and offered-fluid PRB demand. Recovery requires 10 ms of simultaneous all-UE queue-budget compliance after an OFF transition; failed recovery is right-censored and no OFF event is N/A. Batch statistics, served-byte residence and sampled HOL are distinct; none measures packet reliability or application end-to-end latency.

Training returns have different objectives/scales across P and B3/B4/B5 and are not compared as performance scores. Training episode variability mixes scenario changes and exploration, not independent trained-seed uncertainty. The held-out results contain two paired scenario seeds per severity; no significance or generalization claim is made.

## Resource-safe reproduction

Inspect current CPU load and RAM first. CPUs 6,8,9,15 were chosen among relatively idle cores for these runs; change `--cpus` if appropriate. Run the following commands **sequentially** from repository root:

```bash
python backend/tools/phase2b_guard.py --cpus 6,8,9,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c1-tests-resources.json -- python -m pytest -q

python backend/tools/phase2b_guard.py --cpus 6,8,9,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c1-train-resources.json -- python backend/tools/phase2c1_pilot.py \
  --stage train --output /tmp/c1-run

python backend/tools/phase2b_guard.py --cpus 6,8,9,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c1-heldout-resources.json -- python backend/tools/phase2c1_pilot.py \
  --stage test --output /tmp/c1-run

python backend/tools/phase2b_guard.py --cpus 6,8,9,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c1-analysis-resources.json -- python backend/tools/phase2c1_analyze.py \
  --input /tmp/c1-run --output /tmp/c1-analysis --plots
```

`--protocol path.json` supports another explicitly specified bounded configuration, but doing so defines a new experiment. The pilot CLI rejects missing/overlapping seed splits, excessive budgets, missing controllers, unguarded execution and unsafe CPU/GPU settings. Evaluation requires complete frozen selection matching the supplied protocol. `--stage selftest` runs tiny independent software scenarios, not training/test study seeds.

For historical checks, run the existing `backend/tools/phase2a_validate.py --output /tmp/c1-regressions` through the same guard. The saved regression summary also compares its demo against the committed Phase 2A digest. A guarded 300-interval Phase 2B smoke was compared with the committed Phase 2B trajectory, parameters, loss sequence and reward: exact equality.

The guard sets inherited affinity, one thread per numerical library, empty CUDA visibility, and samples its own memory plus the entire launched process tree every 50 ms. It warns at 2 GiB and stops its own workload at 2.5 GiB or if system available RAM drops below 4 GiB. Descendant workers starting a new session are tracked and terminated too. This Python build lacks pidfd support, so tested tracked-PID/group termination is used. No DataLoaders, parallel training/evaluation jobs, full sweeps or GPU operations are used. Cgroups are not writable here: memory limits are **monitored, not hard-enforced**. Sampling can miss short peaks; headroom and bounded replay/logging are deliberate safeguards.

## Recorded artifacts

`training.json`, `validation.json`, `rule_validation.json` and `heldout.json` contain episode-level measurements with bounded summaries, not full packet/event histories. `selection.json` identifies frozen selected models; `checkpoints/` contains only those four models, each with protocol/schema/config/seed metadata. Candidate episodes 6/9 are reproducible through training but not duplicated in this artifact directory. Replay/RNG continuation state is not saved: these are evaluation checkpoints, not exact interrupted-training resume files.

`pilot_table.csv`, `pilot_summary.json`, `paired_differences.json`, `per_ue_differences.json`, `learning_diagnostics.json` and `validation_learning.png` are generated by the analyzer. `resource_reports.json`, `test_results.txt`, `regression_summary.json` and `provenance.json` preserve validation/provenance. [CHANGED_FILES.md](CHANGED_FILES.md) is the exact inventory. The three pre-existing root research notes are untouched.
