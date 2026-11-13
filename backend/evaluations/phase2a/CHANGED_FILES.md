# Exact Phase 2A changed-file inventory

20 files: 4 modified source files, 8 new source/test/tool files and 8 new documentation/result artifacts. The three pre-existing untracked root research notes are excluded.

| File | Change |
|---|---|
| `backend/network_layer/cell.py` | Add optional cell-local URLLC allocator dispatch and maintain starvation ages across proportional intervals; preserve configured eMBB dispatch. |
| `backend/network_layer/queue_measurements.py` | Add independent read-only event listeners without replacing diagnostics or changing queue counters. |
| `backend/network_layer/simulation_engine.py` | Add optional before-service hook and explicit post-service RIC callback context. |
| `backend/settings/__init__.py` | Import the optional Phase 2A setting. |
| `backend/settings/phase2a_config.py` | Define disabled-by-default joint control flag. |
| `backend/network_layer/joint_control.py` | Define immutable joint actions/receipts and validate/apply atomic quota-mode commands with physical-pool and timing guards. |
| `backend/network_layer/ue_telemetry.py` | Collect frozen causal UE snapshots and trailing arrival/maturity/failure counters with missing-history handling. |
| `backend/network_layer/deadline_risk.py` | Encode the six independent deadline-risk features with exposure counts, validity masks and reasons. |
| `backend/network_layer/xApps/xapp_scripted_joint_ric.py` | Implement optional single-owner scripted controller, interval accounting, local mode switching and episode lifecycle. |
| `backend/tools/phase2a_demo.py` | Run reproducible offline scripted or fixed-mode demonstrations with histories, physical hashes and measured timing. |
| `backend/tools/phase2a_validate.py` | Run bounded saved regressions, fixed-policy equivalence, causal switching and exact replay checks. |
| `backend/tests/test_phase2a.py` | Add 17 unit/integration cases covering telemetry, controls, causality, resets, replay and compatibility. |
| `backend/evaluations/phase2a/README.md` | Document interfaces, units, equations, exposure rules, command/timing contract and reproduction. |
| `backend/evaluations/phase2a/REPORT.md` | Record measured validation, demo, timing, limitations and Phase 2B recommendation. |
| `backend/evaluations/phase2a/CHANGED_FILES.md` | Provide this exact file-by-file inventory. |
| `backend/evaluations/phase2a/demo.json.gz` | Save complete compressed scripted telemetry, control, service and interval trace. |
| `backend/evaluations/phase2a/regression_summary.json` | Save six Phase 0 and eighteen Phase 1 comparisons, two equivalence pairs and representative demo metadata. |
| `backend/evaluations/phase2a/replay_summary.json` | Save exact replay checks and final pending-command accounting. |
| `backend/evaluations/phase2a/provenance.json` | Save base ancestry, unchanged protected-source hashes, input hashes and dependency versions. |
| `backend/evaluations/phase2a/test_results.txt` | Save final full-suite pytest result. |
