# Exact Phase 2B changed-file inventory

18 new files: 7 implementation/test/tool files and 11 documentation/result artifacts. No existing tracked file was modified. The three pre-existing untracked root research notes are excluded.

| File | Change |
|---|---|
| `backend/network_layer/joint_dqn_schema.py` | Add the deterministic 98-action catalog, exact bridge-consistent mask, separate 30-input encoder and raw state metadata. |
| `backend/network_layer/joint_reward.py` | Add immutable reward configuration, exactly-once causal failure ledger, separately labeled terminal charges and dimensionless tick costs. |
| `backend/network_layer/joint_dqn_agent.py` | Add CPU-only two-256-layer masked DQN, bounded replay, TD targets, finite updates and versioned checkpoint compatibility checks. |
| `backend/network_layer/xApps/xapp_joint_deadline_dqn.py` | Add optional joint xApp and adapter to unchanged CausalInterval, post-service attribution, atomic commands, resets, bounded diagnostics and independent terminal accounting. |
| `backend/tools/phase2b_guard.py` | Add sequential CPU-affinity launcher, numerical-thread/GPU safeguards, sampled aggregate memory/CPU monitoring and conservative process-group stop threshold. |
| `backend/tools/phase2b_smoke.py` | Add bounded CPU-only smoke harness, deterministic scenario setup, real updates, checkpoint round-trip and metadata/runtime reporting. |
| `backend/tests/test_phase2b.py` | Add 16 unit/integration cases for actions, masks, state validity, failure/reward/terminal attribution, TD targets, updates, checkpoints, jamming, switching, resets and replay. |
| `backend/evaluations/phase2b/DESIGN.md` | Record state/action/reward equations and units, causal horizon handling and resource policy specified before implementation. |
| `backend/evaluations/phase2b/README.md` | Document optional interfaces, input positions, timing, resource limitations and exact sequential reproduction commands. |
| `backend/evaluations/phase2b/REPORT.md` | Record final tests/regressions, smoke configuration, measured runtime/resources, causal examples and Phase 2C recommendation. |
| `backend/evaluations/phase2b/CHANGED_FILES.md` | Provide this exact file inventory. |
| `backend/evaluations/phase2b/provenance.json` | Save integration base, unchanged protected-source hashes, trace hashes, versions and smoke checkpoint hash. |
| `backend/evaluations/phase2b/regression_summary.json` | Save Phase 0/1 trajectory checks, fixed-policy equivalence and committed Phase 2A digest equality. |
| `backend/evaluations/phase2b/replay_summary.json` | Save exact learned replay checks, finite loss range, rewards, runtime and terminal accounting. |
| `backend/evaluations/phase2b/resource_reports.json` | Save all eight guarded command measurements and aggregate resource-limit reporting. |
| `backend/evaluations/phase2b/smoke.json.gz` | Save compressed bounded transition examples, validity/exposure metadata, service tail and live terminal checks. |
| `backend/evaluations/phase2b/smoke_checkpoint.pt` | Save the CPU functional-smoke model/target/optimizer with versioned schemas, training/observation/reward config, seed and scenario/input metadata. |
| `backend/evaluations/phase2b/test_results.txt` | Save actual final 78-test pytest output. |
