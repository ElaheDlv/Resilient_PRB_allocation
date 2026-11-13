**01_training_reward.** Matched reward for all methods. Markers distinguish regimes; white/gray/method-color fills denote j=0/10/20 PRBs. Lines distinguish NN seeds 12026/22026; connected points change scenario difficulty and are not a convergence curve. Data: training_data.csv.

**02_validation_failure.** Greedy fixed-validation mean over 18 paired scenarios (nine equally weighted strata), at checkpoints 3/6/9. Lines: mean of NN seeds 12026/22026; shaded band: their range, not a confidence interval. No untrained baseline was evaluated. Data: validation_curves.csv.

**03_validation_worst.** Greedy fixed-validation mean over 18 paired scenarios (nine equally weighted strata), at checkpoints 3/6/9. Lines: mean of NN seeds 12026/22026; shaded band: their range, not a confidence interval. No untrained baseline was evaluated. Data: validation_curves.csv.

**04_validation_reserve.** Greedy fixed-validation mean over 18 paired scenarios (nine equally weighted strata), at checkpoints 3/6/9. Lines: mean of NN seeds 12026/22026; shaded band: their range, not a confidence interval. No untrained baseline was evaluated. Data: validation_curves.csv.

**05_mode_selection.** Fixed validation mode occupancy and switching for all matched controllers; A4 is the joint controller. Mode occupancy includes the initial 10-ms bootstrap. Two-seed means and ranges; data: validation_curves.csv.

**06_learning_diagnostics.** Nonoverlapping 100-update means (last bin may be shorter). Q magnitude is the minibatch maximum absolute selected-action Q before its optimizer step, not a convergence guarantee. Same Huber loss for every method. Data: learning_bins.csv and ../updates.csv.

**07_aggregate_worst_tradeoff.** All 24 method/NN-seed/checkpoint points, with episode numbers. Circles/squares denote NN seeds 12026/22026; dotted lines follow training. Both axes are minimized. No fairness SLA is assumed. Data: validation_curves.csv and ../selection.json.
