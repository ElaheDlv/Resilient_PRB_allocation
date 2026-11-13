# Phase2D reproduction and artifact guide

`REPORT.md` contains measured findings and the decision. `DESIGN.md` contains the
prospective source audit, mathematical candidate and experiment plan.
`protocol.json` pins all validation inputs and saved C3 checkpoints before the
study. The study never opens the frozen future-model holdout manifest or trains
a network. It refuses to overwrite an existing `runs.jsonl.gz`.

From repository root, use the environment recorded in `environment.json` and
restore missing original traffic CSVs from the snapshots/instructions in
`../phase2c3/README.md` if needed. Keep one numerical thread and at most four CPUs
through the guard. Other heavy commands must not run concurrently.

Run regression **without the seven tests that train DQN**:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2d-regression-resources.json -- python -m pytest backend/tests -q \
  -k 'not actual_updates_bounded and not real_causal_learning and not episode_resets_no_leaked_exposure and not actual_sequential_controllers_training and not actual_matched_learning'
```

Run only the ten Phase2D nonlearning checks:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2d-tests-resources.json -- python -m pytest backend/tests/test_phase2d.py -q
```

Regenerate all analysis/plots **without any new simulation**:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report /tmp/phase2d-report-resources.json -- env MPLCONFIGDIR=/tmp/phase2d-mpl \
  python backend/tools/phase2d_report.py
```

To reproduce the frozen study, use a separate checkout/output directory with the
Phase2D generated results absent, retaining `protocol.json` and all frozen C2/C3
inputs. Do not run `freeze` again: it was executed once prospectively and refuses
to overwrite its file. Then run:

```bash
python backend/tools/phase2b_guard.py --cpus 6,7,11,14 --warning-gib 2 --stop-gib 2.5 \
  --report backend/evaluations/phase2d/study_resources.json -- \
  python backend/tools/phase2d_study.py run > backend/evaluations/phase2d/study.log 2>&1
```

Budget:42 saved-checkpoint replays,70 nonlearning episodes,3s simulated time each.
No optimizer updates, searches, final-test pool or automatic extension. Learner
seeds refer only to the existing C3 initialization checkpoints. Every replay is
checked against C3; all nonlearning policies use the same reactive reserve rule.
G0 must match F on all ten prototype scenarios. Every lambda is development data,
not an independently selected final method. Wall-clock timings are not bitwise
reproducible across CPU load and instrumentation.

## Recorded data

- `runs.jsonl.gz`:112 episode summaries with exact batch accounting, UE outcomes,
  radio/PRB diagnostics, recovery and1ms deprivation **counts**, not consecutive
  wall-time duration claims. It retains accepted/overflow/unresolved distinctions.
- `decisions.jsonl.gz`:all observed RIC decisions and unchanged30-feature states.
- `shadows.jsonl.gz`:pure same-state P/F grants at the first service tick after
  each replay command. These are immediate completion comparisons, not rollouts.
- `ue_windows.csv.gz`:per-UE100ms retrospective service/queue/deadline exposures at
  10ms RIC samples. Positive deficits can include benign rounding effects; combine
  magnitude, queue slope, HOL and failures before diagnosing pathological harm.
- `harm_ticks.csv.gz`:1ms service frames for the exact94.667pp harm case (A1,
  aggregate A4 and worst-first A4). Figure03 thins to10ms for readability.
- `interval_outcomes.csv.gz`:cumulative counters at10ms boundaries; their differences
  count observed failure events/completions, not arrival-cohort probabilities.
- `a4_recorded_candidates.csv`:all108 original C3 A4 candidate/scenario records.
- `decision_diagnostics.csv`, `switch_correlations.csv`, `switch_strata.csv`:
  correlations, reversals, condition changes and following-interval events.
- `state_discrimination.csv`:ranges/stds conditional on valid saturated/non-saturated
  worst risk for each input, plus measured complementary service-deficit values.
- `prototype_scenarios.csv`, `prototype_summary.csv`, `prototype_paired.csv` and
  `per_ue.csv`:all fixed settings, comparisons, class/UE and computation outcomes.
- `service_segments.csv`:distinct initialON/OFF/secondON service statistics.
  Fields explicitly suffixed `same_onoff_group` are grouped deprivation totals
  repeated as context, not per-segment counts; do not sum them across ON segments.
  Use `deprivation_counts.csv` for unique counts and `sampled_deprivation.csv` for
  consecutive *sampled*10ms observations, not continuous-time guarantees.
- `figures/`:six editable SVG/PDF pairs; `CAPTIONS.md` identifies underlying numbers.
- `*_resources.json`, logs, `provenance.json`, `validation_checks.json`:limits,
  test/replay/pairing evidence and exact hashes. Memory is watchdog-monitored,
  not hard-enforced; no unrelated process was modified.
