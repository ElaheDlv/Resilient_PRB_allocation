# Phase2E reproduction

Read `REPORT.md` for the decision, `PROTOCOL.md`/`protocol.json` for the prospective
design, `CHANGED_FILES.md` for the exact file inventory, and `figures/CAPTIONS.md`
for the six editable SVG/PDF figure pairs and their underlying data references.
All results are for frozen Phase2D G1, lambda=1, with a decision-equivalent lazy
telemetry implementation. No DQN or frozen joint-DQN holdout evaluation occurs.

Use the Python/packages recorded in `environment.json`. Restore original traffic
CSV dependencies if absent using the snapshots and instructions in the Phase2C-3
README; the original CSVs are ignored by Git. Input hashes are checked at every
stage. No source CSV, older experiment artifact or checkpoint is overwritten.

## Tests, without training

From repository root, run sequentially under the guard:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2e-regression-resources.json -- python -m pytest backend/tests -q \
  -k 'not actual_updates_bounded and not real_causal_learning and not episode_resets_no_leaked_exposure and not actual_sequential_controllers_training and not actual_matched_learning'
```

The five Phase2E checks can be run alone by selecting
`backend/tests/test_phase2e.py` with the same guarded pytest command. Seven
existing training tests are deliberately excluded from full regression.

## Regenerate analysis only

No new simulation or model loading:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2e-report-resources.json -- env MPLCONFIGDIR=/tmp/phase2e-mpl \
  python backend/tools/phase2e_report.py
```

## Reproduce the frozen study in a separate output directory

The optional `--output` path avoids overwriting this delivery. Copy the frozen
protocol without reselecting seeds or running `freeze` again:

```bash
mkdir -p /tmp/phase2e-reproduction
cp backend/evaluations/phase2e/protocol.json /tmp/phase2e-reproduction/protocol.json
```

Then execute the predeclared stages **sequentially** (the shell loop waits for
each command). Existing stage markers prevent an accidental repeated completed
stage. Do not run other heavy jobs from this study concurrently.

```bash
for phase2e_stage in profile_original discover equivalence profile_optimized evaluate; do
  python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
    --report "/tmp/phase2e-reproduction/${phase2e_stage}_resources.json" -- \
    python backend/tools/phase2e_study.py "$phase2e_stage" --output /tmp/phase2e-reproduction \
    > "/tmp/phase2e-reproduction/${phase2e_stage}.log" 2>&1 || break
done
```

After all stages succeed, regenerate tables and vectors with:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2e-reproduction/report_resources.json -- env MPLCONFIGDIR=/tmp/phase2e-mpl \
  python backend/tools/phase2e_report.py --output /tmp/phase2e-reproduction
```

Budget:24 short exogenous probes,6 profiled development episodes,20 unprofiled
development equivalence episodes,81 independent P/F/G1 episodes and3 independent
original G1 checks. No training, tuning, new scenarios or automatic growth.
Profiles include large cProfile overhead; only unprofiled runs assess timing.
Numeric outcomes are deterministic under the frozen model/environment; CPU
load, measured timing, compressed-stream timestamps and vector metadata are
not bitwise reproducible. Scratch-output manifest paths reflect that directory.

## Data guide

- `independent_manifest.json`, `manifest_freeze.json`, `exogenous_probes.json`,
  `configs/`, `traces/`:pre-performance membership, disjoint stream audit and
  exogenous certificates. No reference performance is used for selection.
- `profile_original.csv`, `profile_optimized.csv`, `profile_*.prof`,
  `profile_components.csv`:full callable profiles and selected components.
  Cumulative times overlap and must not be added.
- `equivalence_runs.jsonl.gz`, `equivalence.json`, `optimization_timings.csv`:
  ten exact original/optimized pairs, including complete per-UE trajectory hashes.
- `independent_original_checks.jsonl.gz`:three additional exact new-placement
  checks, all severe contested cases.
- `independent_runs.jsonl.gz`, `scenarios.csv`:81 complete independent records,
  physical outcomes and all timing scopes/quantiles/exceedances.
- `per_ue.csv`, `paired_per_ue.csv`, `service_windows.csv.gz`:individual batch
  partitions, service deficits, allocations, paired harms and retrospective
  window samples. Deficits alone are not a starvation or deadline diagnosis.
- `paired_differences.csv`, `consistency.csv`:G1 minus P/F within identical
  scenarios. Positive failure differences are harm. Counts are clustered on
  three placements per regime, not independent UE/tick samples.
- `summary.csv`, `regime_means.csv`:every regime/severity, plus clearly labeled
  means across severities. Per-run quantiles are averaged, not pooled quantiles.
- `recovery_windows.csv`:OFF recovery times/censoring; no-jam has no recovery window.
- `tick_times.csv.gz`:all243000 independent per-tick wall/CPU/allocator/update
  timings. Full tick includes the reference simulator/metrics/RIC path; outside
  audit/export is excluded. RIC-boundary and terminal cost are separately visible.
- `validation_checks.json`, `provenance.json`, logs and `*_resources.json`:
  equivalence/pairing/source/input checks, software/resource limits and caveats.

Memory is watchdog-monitored, not a delegated hard cgroup cap. Affinity is not
exclusive isolation. This study does not certify real-time industrial deployment.
