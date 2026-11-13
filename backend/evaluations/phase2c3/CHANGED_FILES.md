# Exact changed-file inventory

Base commit: `64a861e`. Three unrelated user research notes were left untouched.

## Code

- `backend/network_layer/matched_control.py` — New action-mask intersections and causal atomic A3 adapter; shared joint encoder/reward/learner.
- `backend/tools/phase2c1_pilot.py` — Optional app_class injection hook; original default preserved; no existing algorithm changes.
- `backend/tools/phase2c3_pilot.py` — Frozen training/validation-only protocol, matched sequential training, checkpoint selection/logging and snapshot audit.
- `backend/tools/phase2c3_report.py` — Measured paired/per-UE tables, two selection rules, seven SVG/PDF figures and numeric data; no additional simulations.
- `backend/tests/test_phase2c3.py` — Nine focused tests for masks, causal A3, real replay/updates/reset/evaluation, schema/reward identity and fairness selection.

## Delivered research artifacts

Every path below is new. Checkpoints are evaluation snapshots; gzipped JSON lines retain the full episode measurements. Original traffic snapshots preserve existing CSV bytes.

- `backend/evaluations/phase2c3/CHANGED_FILES.md`
- `backend/evaluations/phase2c3/PROTOCOL.md`
- `backend/evaluations/phase2c3/README.md`
- `backend/evaluations/phase2c3/REPORT.md`
- `backend/evaluations/phase2c3/audit.log`
- `backend/evaluations/phase2c3/audit_resources.json`
- `backend/evaluations/phase2c3/candidate_scores.csv`
- `backend/evaluations/phase2c3/checkpoint_audit.json`
- `backend/evaluations/phase2c3/checkpoints/A1_seed12026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A1_seed12026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A1_seed12026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A1_seed22026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A1_seed22026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A1_seed22026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed12026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed12026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed12026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed22026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed22026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A2_seed22026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed12026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed12026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed12026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed22026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed22026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A3_seed22026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed12026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed12026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed12026_ep9.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed22026_ep3.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed22026_ep6.pt`
- `backend/evaluations/phase2c3/checkpoints/A4_seed22026_ep9.pt`
- `backend/evaluations/phase2c3/environment.json`
- `backend/evaluations/phase2c3/figures/01_training_reward.pdf`
- `backend/evaluations/phase2c3/figures/01_training_reward.svg`
- `backend/evaluations/phase2c3/figures/02_validation_failure.pdf`
- `backend/evaluations/phase2c3/figures/02_validation_failure.svg`
- `backend/evaluations/phase2c3/figures/03_validation_worst.pdf`
- `backend/evaluations/phase2c3/figures/03_validation_worst.svg`
- `backend/evaluations/phase2c3/figures/04_validation_reserve.pdf`
- `backend/evaluations/phase2c3/figures/04_validation_reserve.svg`
- `backend/evaluations/phase2c3/figures/05_mode_selection.pdf`
- `backend/evaluations/phase2c3/figures/05_mode_selection.svg`
- `backend/evaluations/phase2c3/figures/06_learning_diagnostics.pdf`
- `backend/evaluations/phase2c3/figures/06_learning_diagnostics.svg`
- `backend/evaluations/phase2c3/figures/07_aggregate_worst_tradeoff.pdf`
- `backend/evaluations/phase2c3/figures/07_aggregate_worst_tradeoff.svg`
- `backend/evaluations/phase2c3/figures/CAPTIONS.md`
- `backend/evaluations/phase2c3/figures/learning_bins.csv`
- `backend/evaluations/phase2c3/figures/training_data.csv`
- `backend/evaluations/phase2c3/figures/validation_curves.csv`
- `backend/evaluations/phase2c3/final_tests.log`
- `backend/evaluations/phase2c3/final_tests_resources.json`
- `backend/evaluations/phase2c3/original_traffic/embb_dl.csv.gz`
- `backend/evaluations/phase2c3/original_traffic/urllc_dl.csv.gz`
- `backend/evaluations/phase2c3/paired_differences.csv`
- `backend/evaluations/phase2c3/paired_per_ue.csv`
- `backend/evaluations/phase2c3/pareto_frontier.csv`
- `backend/evaluations/phase2c3/per_ue.csv`
- `backend/evaluations/phase2c3/pilot.log`
- `backend/evaluations/phase2c3/pilot_resources.json`
- `backend/evaluations/phase2c3/protocol.json`
- `backend/evaluations/phase2c3/provenance.json`
- `backend/evaluations/phase2c3/recovery_windows.csv`
- `backend/evaluations/phase2c3/regression.log`
- `backend/evaluations/phase2c3/regression_resources.json`
- `backend/evaluations/phase2c3/report.log`
- `backend/evaluations/phase2c3/report_resources.json`
- `backend/evaluations/phase2c3/runtime.json`
- `backend/evaluations/phase2c3/selected_per_ue.csv`
- `backend/evaluations/phase2c3/selected_scenarios.csv`
- `backend/evaluations/phase2c3/selection.json`
- `backend/evaluations/phase2c3/summary.csv`
- `backend/evaluations/phase2c3/summary.json`
- `backend/evaluations/phase2c3/tests.log`
- `backend/evaluations/phase2c3/tests_resources.json`
- `backend/evaluations/phase2c3/training.jsonl.gz`
- `backend/evaluations/phase2c3/updates.csv`
- `backend/evaluations/phase2c3/validation.jsonl.gz`
- `backend/evaluations/phase2c3/validation_scenarios.csv`
