# Phase 2G controlled reward experiment

See `REPORT.md` for measured outcomes, [METHODS.md](METHODS.md) for exact equations
and [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) for prespecified interpretation.
The original Phase2F [ablation plan](../phase2f/ABLATION_PLAN.md) remains authoritative.

The archived experiment is a **single bounded run**: six sequential learners,
2691 transitions each;18 checkpoints;324 fixed validation evaluations. Do not
rerun into this directory, add a reward arm or extend training based on outcomes.
The script refuses to overwrite training outputs. Checkpoints contain optimizer
snapshots, not replay/continuation state; resuming training is not supported.

## Files

- `protocol.json`: frozen arms, original inputs/source hashes, initialization
  fingerprints, exact C3 scenario order/settings and resource limits.
- `initializations/`: two source policy/target/empty optimizer/RNG snapshots copied
  into every paired reward arm. No learner continues from an earlier trained model.
- `correctness_gate.json`, `correctness.log`, `regression.log`: passing pretraining
  gate; archived `correctness_attempt*.log` retain test-harness failures corrected
  before any training. Seven existing learning tests were deliberately excluded.
- `checkpoints/`:18 versioned model/target/optimizer checkpoints; `checkpoint_audit.json`
  checks schema, budgets, evaluated fingerprints and R0 historical reproducibility.
- `training.jsonl.gz`, `validation.jsonl.gz`:54/324 complete original physical
  summaries, arm costs and control/parameter identities. Returns across arms
  optimize different objectives and must not be used as common performance scores.
- `intervals.jsonl.gz`:113,022 ten-tick records, streamed to disk; physical failure
  counters, six learned costs, and original reference costs on the **same observed**
  trajectory. Reference accounting does not predict a different policy's service.
- `updates.jsonl.gz`:15,960 original optimizer diagnostics. `optimizer_curves.csv`
  uses prespecified nonoverlapping100-update bins, with a final60-update bin.
- `validation_metrics.csv`, `per_ue_validation.csv`, `per_class_validation.csv`:
  all324 evaluations with primary/secondary checkpoint selection flags.
- `learning_curves.csv`: fixed checkpoint means by arm, individual NN seed / mean,
  regime and severity. Severity `all` includes0/10/20; regime `all` includes all
  three loads. Mean columns are descriptive averages of the two NN seeds.
- `selection.json`, `selected_summary.csv`: exact aggregate-first and worst-first
  choices and separately labeled results. Conditional recovery delay includes
  recovered windows only; window/recovered counts expose censoring.
- `paired_checkpoint_differences.csv`: all216 R1/R2 minus R0 scenario comparisons
  at the **same** checkpoint interaction count.
- `paired_selected_differences.csv`: R1/R2 minus matched R0 for each criterion,
  initialization seed and scenario. Failure changes are **percentage points**;
  throughput changes are Mbit/s, reserve changes PRBs, mode occupancy points.
- `per_ue_paired_differences.csv`: every UE's signed differences, not only harmed UEs.
- `interval_signal_diagnostics.csv`: actual fairness variability, original historical
  carry and event-based costs on each observed episode. These diagnostics are not
  an isolated causal explanation of learned outcomes.
- `figures/`: SVG/PDF exports and captions. `analysis.json`: machine-readable summaries.
- `*_resources.json`: guard commands, affinity, sampled aggregate RSS/CPU and stop
  status; `runtime.json`: experiment wall time. `provenance.json` and
  `CHANGED_FILES.md` record input integrity and the exact changed-file list.

## Reproduction and resources

Use the existing environment, from repository root. Inspect CPU/RAM first. The
archived runs used CPUs6,7,11,14, but choose up to four currently suitable allowed
CPUs if reproducing later. CPU affinity is inherited by child processes; numerical
threads are1, CUDA is disabled, DataLoader workers0. The guard's2GiB warning and
2.5GiB aggregate RSS stop are **monitored, not hard memory caps**; no writable
cgroup was available. It also stops below4GiB system available RAM.

Archived commands (run sequentially, never parallel):

```bash
free -h
ps -eo pid,psr,pcpu,rss,comm --sort=-pcpu | head -16
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/freeze_resources.json \
  -- python backend/tools/phase2g_ablation.py freeze
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/correctness_resources.json \
  -- python -m pytest backend/tests/test_phase2g.py -q
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/regression_resources.json \
  -- python -m pytest backend/tests -q \
  -k 'not actual_updates_bounded and not real_causal_learning and not episode_resets_no_leaked_exposure and not actual_sequential_controllers_training and not actual_matched_learning'
```

After passing, a gate pins the protocol and successful regression log hash.
Training requires that gate, matching source/input hashes and absent prior outputs.
The gate was written only after both guarded commands exited successfully:

```python
from pathlib import Path
import hashlib, json
p = Path('backend/evaluations/phase2g')
assert json.loads((p/'correctness_resources.json').read_text())['exit_code'] == 0
assert json.loads((p/'regression_resources.json').read_text())['exit_code'] == 0
assert '10 passed' in (p/'correctness.log').read_text()
assert '126 passed, 7 deselected' in (p/'regression.log').read_text()
gate = dict(passed=True,
    protocol_sha256=hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest(),
    correctness_tests=10, regression_passed=126, training_tests_deselected=7,
    training_updates_in_gate=0,
    regression_log_sha256=hashlib.sha256((p/'regression.log').read_bytes()).hexdigest(),
    note='Initial test-harness failures corrected before any training; source/experiment hashes frozen and verified.')
(p/'correctness_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
```

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/experiment_resources.json \
  -- python backend/tools/phase2g_ablation.py run
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/audit_resources.json \
  -- python backend/tools/phase2g_ablation.py audit
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/report_resources.json \
  -- env MPLCONFIGDIR=/tmp/phase2g-mpl python backend/tools/phase2g_report.py
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 \
  --warning-gib 2 --stop-gib 2.5 --report backend/evaluations/phase2g/integrity_resources.json \
  -- python backend/tools/phase2g_report.py --verify-only
```

These commands document the original execution, not permission to overwrite
archived artifacts. Reproducing training requires an isolated checkout at the
specified base with output paths changed consistently to a fresh directory and
a separately frozen, explicitly identified replication protocol. The `freeze`
command checks the requested base; do not bypass integrity or correctness gates.
The reporter and checkpoint auditor do not train/evaluate fresh scenarios.
