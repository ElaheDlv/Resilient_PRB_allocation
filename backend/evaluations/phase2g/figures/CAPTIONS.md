# Phase 2G figures

**01_aggregate_failures.** Identical fixed validation scenarios at each checkpoint, six episodes/regime/NN seed. Columns separate initialization seeds and their descriptive mean; rows are workload regimes. All three jamming severities included. No exploratory-training return is used. Data: learning_curves.csv; severity strata remain available there.

**02_worst_ue_failures.** Identical fixed validation scenarios at each checkpoint, six episodes/regime/NN seed. Columns separate initialization seeds and their descriptive mean; rows are workload regimes. All three jamming severities included. No exploratory-training return is used. Data: learning_curves.csv; severity strata remain available there.

**03_reserve_use.** Identical fixed validation scenarios at each checkpoint, six episodes/regime/NN seed. Columns separate initialization seeds and their descriptive mean; rows are workload regimes. All three jamming severities included. No exploratory-training return is used. Data: learning_curves.csv; severity strata remain available there.

**04_embb_throughput.** Identical fixed validation scenarios at each checkpoint, six episodes/regime/NN seed. Columns separate initialization seeds and their descriptive mean; rows are workload regimes. All three jamming severities included. No exploratory-training return is used. Data: learning_curves.csv; severity strata remain available there.

**05_modes_and_switching.** Fixed validation18-scenario means. Mode occupancy includes the shared initial10ms P interval. Switches count mode changes, not reserve-command changes. Data: learning_curves.csv.

**06_reward_components.** Actual learned-objective costs for fixed validation evaluations. Fairness has the versioned R0/R1/R2 definition, so total heights are not a common physical performance measure. Data: learning_curves.csv and validation_metrics.csv.

**07_td_loss_and_q.** Original unchanged learner; nonoverlapping100-update bin means (last bin60), starting after31 replay warmup transitions. Different reward scales imply different TD/Q scales; lower loss is not a common physical performance ranking. Data: optimizer_curves.csv; raw updates.jsonl.gz.

**08_per_ue_distributions.** Aggregate-first selected checkpoints. Each NN panel shows48 UE/scenario observations per arm/regime; mean panels average the two NN outcomes within each matched scenario/UE. These dependent observations are not independent replication. Boxes show quartiles/median, whiskers1.5IQR; points retain all observations. Exact per-UE paired differences and deadline classes are in the CSVs.

**09_selected_severity_strata.** Aggregate-first selected checkpoints, two placements per regime/severity/NN. Both seeds shown separately; no scenario replacement or tuning. Data: selected_summary.csv.
