# Exact Phase 2C-1 changed-file inventory

32 files: 2 modified implementation files, 4 new implementation/test/tool files and 26 new documentation/result artifacts. The three pre-existing untracked root research notes are excluded.

| File | Change |
|---|---|
| `backend/network_layer/joint_dqn_agent.py` | Modified: add reusable network/dimension/loss hooks, optional explicit epsilon, interaction counts and read-only TD/Q diagnostics; preserve default Phase 2B schema and numerical behavior. |
| `backend/tools/phase2b_guard.py` | Modified: configure warning/stop thresholds, monitor system pressure and complete descendant trees, stop only owned workers with a tested PID fallback. |
| `backend/network_layer/pilot_control.py` | New: original-objective CPU agent adapter, reference reactive reserve reuse, causal hysteretic switching and all seven paired pilot controllers with independent metrics. |
| `backend/tools/phase2c1_pilot.py` | New: bounded sequential train/validation/test/selftest runner, split validation, independent scenario/network seeds, exploration schedule and frozen checkpoint/rule selection. |
| `backend/tools/phase2c1_analyze.py` | New: complete-grid/exogenous checks, independent physical summaries, paired per-UE differences, learning diagnostics and validation plot. |
| `backend/tests/test_phase2c1.py` | New: seven focused tests for split/exploration rules, source network/encoder/loss reuse, masks/schema compatibility, causal hysteresis, real paired runs and nested-worker memory termination. |
| `backend/evaluations/phase2c1/AUDIT.md` | New artifact: Pre-training source/objective audit and frozen baseline/protocol rationale. |
| `backend/evaluations/phase2c1/README.md` | New artifact: Controller, learning, timing, censoring, resource policy and exact sequential reproduction commands. |
| `backend/evaluations/phase2c1/REPORT.md` | New artifact: Measured software checks, training behavior, paired pilot, per-UE analysis, resource limits and research recommendation. |
| `backend/evaluations/phase2c1/CHANGED_FILES.md` | New artifact: This exact file-by-file inventory. |
| `backend/evaluations/phase2c1/protocol.json` | New artifact: Predeclared budgets, splits, seeds, checkpoints, rule candidates, objectives and resource limits. |
| `backend/evaluations/phase2c1/selection.json` | New artifact: Frozen chosen candidates, physical validation scores, full training budgets and rule threshold. |
| `backend/evaluations/phase2c1/checkpoints/B3_episode3.pt` | New artifact: Selected original DQN/P evaluation checkpoint: 897 transitions, 866 updates, full schema/protocol metadata. |
| `backend/evaluations/phase2c1/checkpoints/B4_episode3.pt` | New artifact: Selected original DQN/F evaluation checkpoint with the same interaction budget and metadata. |
| `backend/evaluations/phase2c1/checkpoints/B5_episode3.pt` | New artifact: Selected independently trained original DQN/rule-switching checkpoint and frozen rule metadata. |
| `backend/evaluations/phase2c1/checkpoints/P_episode3.pt` | New artifact: Selected joint deadline-aware checkpoint and complete 30-state/98-action training metadata. |
| `backend/evaluations/phase2c1/heldout.json` | New artifact: All 42 held-out episode-level records including physical per-UE metrics and paired fingerprints. |
| `backend/evaluations/phase2c1/learning_diagnostics.json` | New artifact: Per-learner returns, variability, loss/TD/Q behavior and physical validation results. |
| `backend/evaluations/phase2c1/paired_differences.json` | New artifact: Joint-minus-reference differences for each of six held-out scenario pairs. |
| `backend/evaluations/phase2c1/per_ue_differences.json` | New artifact: Individual UE failure, capacity, service-demand, grant and unresolved-byte comparisons. |
| `backend/evaluations/phase2c1/pilot_summary.json` | New artifact: All method/severity means, class and worst-UE bounds, switching, reserve, throughput and recovery. |
| `backend/evaluations/phase2c1/pilot_table.csv` | New artifact: Human-readable 21-row held-out comparison table. |
| `backend/evaluations/phase2c1/provenance.json` | New artifact: Base/source/input/model hashes, pairing checks and precise experiment scope. |
| `backend/evaluations/phase2c1/regression_summary.json` | New artifact: Saved Phase 0/1 and committed Phase 2A/2B compatibility evidence. |
| `backend/evaluations/phase2c1/resource_reports.json` | New artifact: All 13 guarded-command resource measurements, limits and notes for corrected failed test attempts. |
| `backend/evaluations/phase2c1/rule_validation.json` | New artifact: Six validation-only rule-candidate probe records; no held-out tuning. |
| `backend/evaluations/phase2c1/test_results.txt` | New artifact: Actual 85-test full-suite and final seven-test focused pytest outputs. |
| `backend/evaluations/phase2c1/train_runtime.json` | New artifact: Measured sequential training-plus-validation stage runtime and protocol hash. |
| `backend/evaluations/phase2c1/test_runtime.json` | New artifact: Measured frozen held-out stage runtime and protocol hash. |
| `backend/evaluations/phase2c1/training.json` | New artifact: All 36 training episode summaries, rewards, physical outcomes, losses, Q and exploration diagnostics. |
| `backend/evaluations/phase2c1/validation.json` | New artifact: All 36 greedy checkpoint-validation episode records. |
| `backend/evaluations/phase2c1/validation_learning.png` | New artifact: Validation failure/worst-UE curves showing all candidates overlap at zero. |
