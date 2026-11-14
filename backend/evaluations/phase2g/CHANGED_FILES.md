# Exact Phase 2G changed-file inventory

Every listed file is **new** relative to `2c8c438`. No original tracked file,
reward weight/configuration, DQN update/architecture, state/action definition,
scheduler, physical model or earlier experiment artifact was modified.
Three preexisting untracked root research notes were preserved and not staged.

**90 new files**: four source/test files and 86 experiment artifacts.

| File | Purpose |
| --- | --- |
| `backend/network_layer/reward_ablation.py` | Optional per-UE ledger, exact R0/R1/R2 adapter and version-only agent subclass; originals unchanged. |
| `backend/tools/phase2g_ablation.py` | Frozen initialization/protocol, six sequential learners,324 validations, gating and checkpoint audit. |
| `backend/tools/phase2g_report.py` | Read-only streaming analyses, paired tables and nine SVG/PDF figure pairs. |
| `backend/tests/test_phase2g.py` | Ten pretraining correctness tests; no optimizer training in the gate. |
| `backend/evaluations/phase2g/ANALYSIS_PLAN.md` | Prespecified physical metrics, paired comparisons and interpretation criteria. |
| `backend/evaluations/phase2g/METHODS.md` | Exact three reward equations, counter timing, versions and matched protocol. |
| `backend/evaluations/phase2g/README.md` | Data dictionary, exact guarded commands, correctness gate and replication constraints. |
| `backend/evaluations/phase2g/REPORT.md` | Measured selected/matched-budget outcomes, UE harms, signal diagnostics and recommendation. |
| `backend/evaluations/phase2g/analysis.json` | Machine-readable summaries and largest primary per-UE increases. |
| `backend/evaluations/phase2g/audit.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/audit_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/checkpoint_audit.json` | 18 checkpoint metadata/update checks;324 evaluated fingerprints; R0 historical comparisons. |
| `backend/evaluations/phase2g/checkpoints/R0_seed12026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R0_seed12026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R0_seed12026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R0_seed22026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R0_seed22026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R0_seed22026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed12026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed12026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed12026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed22026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed22026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R1_seed22026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed12026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed12026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed12026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed22026_ep3.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed22026_ep6.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/checkpoints/R2_seed22026_ep9.pt` | One of18 arm/version-tagged CPU policy/target/optimizer checkpoints. |
| `backend/evaluations/phase2g/correctness.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/correctness_attempt1.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/correctness_attempt1_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/correctness_attempt2.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/correctness_attempt2_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/correctness_gate.json` | Successful pretraining tests/regression and pinned protocol hash. |
| `backend/evaluations/phase2g/correctness_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/environment.json` | CPU/GPU/thread/memory safeguards, software versions and observed peaks. |
| `backend/evaluations/phase2g/experiment.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/experiment_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/figures/01_aggregate_failures.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/01_aggregate_failures.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/02_worst_ue_failures.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/02_worst_ue_failures.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/03_reserve_use.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/03_reserve_use.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/04_embb_throughput.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/04_embb_throughput.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/05_modes_and_switching.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/05_modes_and_switching.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/06_reward_components.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/06_reward_components.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/07_td_loss_and_q.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/07_td_loss_and_q.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/08_per_ue_distributions.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/08_per_ue_distributions.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/09_selected_severity_strata.pdf` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/09_selected_severity_strata.svg` | Standalone scientific figure; exact scope/source data in figures/CAPTIONS.md. |
| `backend/evaluations/phase2g/figures/CAPTIONS.md` | Figure scope, sampling, source tables and interpretive limits. |
| `backend/evaluations/phase2g/freeze.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/freeze_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/initializations/seed12026.pt` | One of two copied source initializations. |
| `backend/evaluations/phase2g/initializations/seed22026.pt` | One of two copied source initializations. |
| `backend/evaluations/phase2g/integrity.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/integrity_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/interval_signal_diagnostics.csv` | 378 streamed episode summaries of fairness variability and original history/event accounting. |
| `backend/evaluations/phase2g/intervals.jsonl.gz` | 113022 causal intervals with physical per-UE deltas and actual/reference reward components. |
| `backend/evaluations/phase2g/learning_curves.csv` | Fixed-validation means at897/1794/2691 transitions, separate NN seeds/mean and regime/severity. |
| `backend/evaluations/phase2g/optimizer_curves.csv` | 100-update binned original TD loss/error/Q/target magnitudes; final bin60. |
| `backend/evaluations/phase2g/paired_checkpoint_differences.csv` | 216 R1/R2-minus-R0 pairs at identical interaction counts. |
| `backend/evaluations/phase2g/paired_selected_differences.csv` | 144 selected-checkpoint pairs, separating primary and secondary criteria. |
| `backend/evaluations/phase2g/per_class_validation.csv` | All648 class/checkpoint/scenario outcomes; batch/byte/HOL measures remain distinct. |
| `backend/evaluations/phase2g/per_ue_paired_differences.csv` | 1152 signed individual-UE selected-policy differences, including all harms and gains. |
| `backend/evaluations/phase2g/per_ue_validation.csv` | All2592 UE/checkpoint/scenario outcomes, classes, service and resource observations. |
| `backend/evaluations/phase2g/protocol.json` | Frozen arms, core/source/data hashes, initializations and exact bounded training settings. |
| `backend/evaluations/phase2g/provenance.json` | Archived artifact hashes, input integrity and unchanged base evidence. |
| `backend/evaluations/phase2g/regression.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/regression_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/report.log` | Archived command/test output; initial test-harness failures explicitly retained. |
| `backend/evaluations/phase2g/report_resources.json` | Guard command, affinity, elapsed time, sampled RSS/CPU and stop status (attempts retain pre-gate failures). |
| `backend/evaluations/phase2g/result_integrity.json` | Independent count, seed/initialization, per-UE conservation and exact selection recomputation. |
| `backend/evaluations/phase2g/runtime.json` | End-to-end bounded experiment elapsed time. |
| `backend/evaluations/phase2g/selected_summary.csv` | Primary and secondary selected physical/component metrics by NN seed, regime and severity. |
| `backend/evaluations/phase2g/selection.json` | Exact aggregate-first and worst-first candidates/selections, separate for all six learners. |
| `backend/evaluations/phase2g/training.jsonl.gz` | 54 complete training physical summaries and original-arm costs. |
| `backend/evaluations/phase2g/training_checks.json` | Six exact learning budgets, frozen validations, matched exogenous hashes and CPU-only checks. |
| `backend/evaluations/phase2g/updates.jsonl.gz` | 15960 original optimizer diagnostic records, no algorithm changes. |
| `backend/evaluations/phase2g/validation.jsonl.gz` | 324 complete greedy validation physical summaries, counts and learned/control identities. |
| `backend/evaluations/phase2g/validation_metrics.csv` | All324 fixed validation evaluations with six costs and primary/secondary selection flags. |
| `backend/evaluations/phase2g/CHANGED_FILES.md` | This exact file-by-file inventory. |
