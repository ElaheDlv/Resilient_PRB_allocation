# Phase 2C-3 matched-objective bounded pilot

See `REPORT.md` for measured findings, `PROTOCOL.md` and `protocol.json` for the
prospective design, and `figures/CAPTIONS.md` for all seven SVG/PDF figure pairs.
Raw episode records are compressed JSON lines; no future holdout data is used.
The pilot stops after its frozen budget, regardless of the results.

From repository root, with the same Python environment and no other Phase C3
workload running:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c3-tests-resources.json -- python -m pytest backend/tests -q

python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c3-report-resources.json -- python backend/tools/phase2c3_report.py
```

To reproduce training, first use a fresh output directory, preserving the frozen
`protocol.json`. The runner deliberately refuses to overwrite existing raw
training logs. It reads only the C2 training/validation manifests and verifies
all referenced input hashes. Because the output location is fixed, reproduce in
a separate checkout with the generated C3 results removed (keep the protocol):

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report backend/evaluations/phase2c3/pilot_resources.json \
  -- python backend/tools/phase2c3_pilot.py run \
  > backend/evaluations/phase2c3/pilot.log 2>&1
```

Do not rerun `freeze`: it was run once before experiments and refuses to overwrite
the protocol. Checkpoints preserve weights/optimizer/target and run metadata,
not replay/RNG continuation state; they are evaluation snapshots, not exact
training-resume checkpoints. Reproduction starts each learner from its frozen
initialization seed. Select the same Python/package environment; runtime/scheduler
computation timings and SVG/PDF metadata are not bitwise deterministic.

`validation_scenarios.csv` and `per_ue.csv` contain every candidate, while
`selected_scenarios.csv` and `selected_per_ue.csv` contain both selection rules.
`summary.csv` separates regime and severity. `paired_differences.csv` and
`paired_per_ue.csv` subtract A1/A2/A3 from A4 within identical scenario/NN seeds;
positive failure differences indicate harm. `recovery_windows.csv` preserves
jammer-OFF recovery times and censoring. `updates.csv` contains every optimizer
update. Checkpoint curves use two-initialization ranges, never confidence bands.

To verify all saved checkpoint metadata, update counts and validation parameter
fingerprints without running any new simulation:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/c3-audit-resources.json -- python backend/tools/phase2c3_pilot.py audit
```

The original traffic CSVs are ignored by Git in this repository. Identical
compressed copies are included in `original_traffic/`. On a clean checkout,
restore only missing originals before running tests or training:

```bash
python - <<'PY'
import gzip
from pathlib import Path
root = Path.cwd()
for name in ('embb_dl.csv', 'urllc_dl.csv'):
    target = root / 'backend/notebooks/Resilient_data' / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.decompress((root / 'backend/evaluations/phase2c3/original_traffic' / (name + '.gz')).read_bytes()))
PY
```

The runner checks the original URLLC hash in the frozen protocol and the eMBB
hash against the prior C2 provenance. Existing originals are never overwritten.
