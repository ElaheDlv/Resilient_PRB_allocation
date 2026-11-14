**01_saved_episode_accounting.** All72 training and432 validation episodes. Four costs are exactly identified; gray worst+eMBB residual is not split by estimation. No training trajectories are regenerated. Data: ../saved_episode_components.csv and ../saved_component_distributions.csv.

**02_interval_components.** Primary nine-case A4 grid: one learner seed and one placement/regime, all severities,897 correlated intervals/regime. Bars=mean, points=p95; all six original components directly measured, with no new weights. Extras remain separately tabulated. Data: ../interval_component_distributions.csv and reward_intervals.csv.gz.

**03_service_recovery_and_history.** Observed case A4_22026_ep3_validation_contested_seed53260288_j20, centered at1620ms. No hypothetical action outcome: improved current service coexists with retrospective cohort penalty. Worst UID can change/tie; data retains UID/validity. History is not itself an erroneous charge. Data: ../strong_history_carry_examples.csv and reward_intervals.csv.gz.

**04_switch_magnitudes.** Left: all recorded C3 validation A3/A4 episodes, not the single-seed grid alone. Right: exact algebraic exchange rates, not simulated counterfactual outcomes or a weight ablation. Correlations do not establish causal influence. Data: ../saved_episode_components.csv and MATHEMATICAL_AUDIT.md.

**05_reserve_and_observed_service.** Primary grid only. Failure/headroom associations do not show that another reserve command would save expired traffic. Unused means activation beyond minimum extra needed to make observed grants fit baseline quotas; it is accounting, not a reconstructed no-reserve rollout. Data: ../temporal_and_resource_diagnostics.csv and reward_ticks.csv.gz.

**06_reward_objective_order.** All24 frozen candidate policies on the same18 validation entries. Six of276 pairwise orders disagree; best five agree and there is no mean six-criterion dominance reversal. This is descriptive, not an independent-sample inference or proof of globally correct action incentives. Data: ../candidate_ordering.csv and ordering_comparisons.csv.

**07_terminal_accounting.** Left: same unchanged physical prefix, hypothetical administrative accounting only; not different-horizon controllers, new outcomes or an ablation. Markers denote j0/10/20. Right: actual3s closure relative to whole episode and last10ms. Undefined zero-total shares omitted. Data: ../observed_prefix_accounting.csv.gz and actual_terminal_intervals.csv.
