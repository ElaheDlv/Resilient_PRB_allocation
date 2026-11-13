# Phase 2C-2 workload feasibility and validation design

Branch `phase2c2/workload-feasibility-design`; base `b2007c3`. **Final protocol is v2.** No DQN model was created, loaded, evaluated or trained. State, reward, actions, source queues, local schedulers, channel/jammer and the physical 18-PRB reserve are unchanged.

[REPORT.md](REPORT.md) explains results/readiness. [DESIGN.md](DESIGN.md) defines capacity certificates, discovery and prospective checkpoint ranking. [FUTURE_ABLATIONS.md](FUTURE_ABLATIONS.md) audits learning interpretation and specifies later ablations without implementing them. [CHANGED_FILES.md](CHANGED_FILES.md) is the exact inventory.

## Important correction to scenario independence

The legacy episode-index-zero initializer derives UE seeds as `scenario_seed + 100000 + slice_index*1000 + UE_index`. Adjacent base seeds reuse 14 of 16 registration seeds. C1 held-out seeds3101/3102 share seven of eight URLLC positions; their distinct scenario labels did not imply independent geometry draws. Paired within-scenario results remain measured observations, but their placement diversity and independent-seed interpretation were overstated. The same concern applies to other consecutive-base-seed cohorts using that initializer; uncertainty claims require reassessment.

The preliminary C2 v1 discovery inherited that issue and was retired. Its reference job was interrupted after 149 completed records. `retired_v1_protocol.json` and `retired_v1_audit.json` document why; v1 performance is excluded from final results. This revision corrected a source-level independence fault, not an outcome-based choice of better scenarios. Definitions, targets and zero-learning scope were unchanged.

Final **v2 maps candidate index to scenario seed as `4096*index`**, preserving the original initialization algorithm and radio model. Index blocks are training12001–12024, validation13001–13024 and future_holdout14001–14024. All **1152 derived registration seeds** across the 72 probes are checked nonoverlapping, including cross-slice collisions, with 32-bit wrap rejected. Fresh single-entry catalogs use canonical UE IDs and episode index zero. Distinct seeds remain pseudorandom model replicates, not proof of real industrial population representativeness.

## Discovery and final pools

`discovery_protocol.json` is the prespecified final v2 design. Probe performance is discarded; only exogenous rate/placement/budget/arrival diagnostics are retained. Select the first two qualifying original-trace seeds per regime and pool. Original coverage found feasible/contested candidates in all pools, but no full-pool overload candidate in these bounded blocks; this is not proof that none exists elsewhere.

Overload alone uses separate synthetic copies from the existing Phase 1.2 length-column scaler. Targets change because its old 57/63-PRB fixed-controller overload criterion is not an overload proof against a controller that can activate all 18 PRBs. Six generated CSVs preserve every non-length field and all arrival timestamps; original files are hash-checked unchanged. Each manifest labels `original` versus `phase12_scaled`.

| Regime | Exogenous definition at severe jamming |
|---|---|
| Feasible | Sum of per-UE one-tick integer requirements <=54 baseline PRBs: constructive fresh-batch capacity condition |
| Contested | Strict due-work bound exceeds 54-PRB ON baseline; one-tick integer requirement <=72 full-pool ON capacity; fluid demand <60 OFF baseline |
| Overloaded | Strict due-work bound exceeds even 78-PRB optimistic full-pool OFF capacity |

Positive static rates, one batch/UE/ms, budgets 2/10 ms and no extra per-UE cap are verified. Per-UE floors/ceilings and finite deadline windows matter: fluid load below quota is not a deadline certificate. Constructions assume fresh traffic/empty queues at fixed available quota and do not promise an arbitrary scheduler's success, clearing inherited backlog, a 10-ms RIC response, eMBB throughput, or end-to-end reliability.

Each pool has six distinct placement seeds (two/regime), each with j0/j10/j20. Thus **18 placement seeds / 54 scenario configurations / 162 paired B0/B1/B2 reference episodes**. Complete manifests were frozen before full reference characterization and were not replaced based on outcomes. All 18 scenario-level severe-jamming coverage gates pass. Feasible/contested use original 650-byte periodic traffic; scaled overload targets approximately 85.8 fluid PRBs/tick, above even full-pool OFF capacity.

`training_manifest.json`, `validation_manifest.json` and `future_holdout_manifest.json` are future scenario pools, **not authorization to increase a training budget**. Later fixed-budget sampling must be separately specified. Future holdout is **reference-characterized, model-unseen**, not untouched blind test data. Do not retune from future proposed-model evaluations on it; publication-level generalization needs an independently preregistered untouched cohort.

## Checkpoint discrimination proposal

The new pure `checkpoint_key` function uses regime/severity-balanced lexicographic metrics:

1. Mean known-failed/offered batches.
2. Mean scenario worst-UE known-failure fraction.
3. Offered-failure upper bound, adding unresolved-not-yet-due traffic.
4. eMBB actual served/offered throughput deficit.
5. Activated reserve PRB-ticks divided by `18*service_ticks`.
6. Mode switches per completed decision.
7. Earlier checkpoint only after every quality/cost term ties.

Count terms use rational fractions; tiny numerical accumulation noise is removed below actual byte/PRB-tick resolution. Resource/switch terms separate equal-deadline candidates that C1 treated as equivalent. `prospective_rule_reference_demo.json` demonstrates discrimination using references only; **no DQN checkpoint was reranked**. Equal stress-stratum weights are a design choice, not industrial prevalence. Aggregate-first ranking can accept worse individual UEs when aggregate failure improves. Upper bounds can favor horizon draining. No hard fairness or Pareto guarantee follows; application-based guardrails must be justified before learning if needed.

## Reproduction under resource limits

Inspect utilization first. This run assigned logical CPUs **7,8,10,15**, selected among relatively idle cores. Change affinity if needed on another run. From repository root, execute these commands **sequentially**:

```bash
python backend/tools/phase2b_guard.py --cpus 7,8,10,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c2-probe-resources.json -- python backend/tools/phase2c2_design.py \
  --stage probe --output /tmp/c2-design

python backend/tools/phase2b_guard.py --cpus 7,8,10,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c2-freeze-resources.json -- python backend/tools/phase2c2_design.py \
  --stage freeze --output /tmp/c2-design

python backend/tools/phase2b_guard.py --cpus 7,8,10,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c2-reference-resources.json -- python backend/tools/phase2c2_design.py \
  --stage reference --output /tmp/c2-design

python backend/tools/phase2b_guard.py --cpus 7,8,10,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c2-summary-resources.json -- python backend/tools/phase2c2_design.py \
  --stage summarize --output /tmp/c2-design --plots
```

Optional `--stage compatibility` reruns nine archived C1 B0/B1/B2 episodes and verifies exact original control/exogenous/deadline hashes and physical outcomes. It does not evaluate a learned controller. For source tests without learning:

```bash
python backend/tools/phase2b_guard.py --cpus 7,8,10,15 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c2-tests-resources.json -- python -m pytest -q \
  backend/tests/test_phase0.py backend/tests/test_phase1.py backend/tests/test_phase11.py \
  backend/tests/test_phase12.py backend/tests/test_phase2c2.py -k 'not trainer and not training'
```

C2 rejects learned-controller names, overlapping derived streams, unsupported resource settings and expanded screening blocks. `run_reference` forces `agent=None, train=False`; tests forbid neural-agent construction. Pure freeze/summary stages avoid loading Torch. Numerical threads are one; CUDA visibility is empty and simulator episodes assert CUDA remains uninitialized. Workers are sequential, no DataLoaders. The guard warns at 2 GiB, stops at 2.5 GiB or system available RAM below 4 GiB, and includes descendant processes. Memory is **monitored, not hard-capped**, because delegated cgroups are not writable. No existing job, GPU setting or system-wide CPU configuration is changed.

## Files and portability

`freeze.json` is the immutable pre-reference snapshot; its `reference_characterization_complete=false` describes that instant. `coverage_gates.json` records the completed current status. Manifest hashes verify membership did not change. Catalogs under `configs/` use `@scenario/...` for generated paths, resolved relative to the artifact/output root by `resolve_catalog`. Original dataset paths resolve from repository root. Do not pass symbolic catalogs directly to old runners without this resolver.

`reference_runs.json.gz` contains all 162 episode summaries, per-UE outcomes, segment service and sparse after-service trace frames. The summary stage accepts either plain or compressed input. `reference_summary.json`, `reference_table.csv` and `per_ue_metrics.csv` provide readable aggregate/class/worst/throughput/reserve/recovery and individual analyses; the PNG shows severe-jamming validation results. No full per-packet history is stored. Source CSVs, DQN code and schedulers are unchanged and hash-checked in `provenance.json`.

`prior_workload_audit.json` compares C1 with previous harm cases and records the seed-overlap correction. `compatibility.json`, `test_results.txt` and `resource_reports.json` preserve checks. The retired interrupted v1 run lost its final numeric peak report; its guard was active and emitted no warning before interruption. That measurement gap is disclosed rather than inventing a peak. The final v2 data/peaks are complete. No subsequent training has been launched.
