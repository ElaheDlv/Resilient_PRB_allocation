**01_profile_cost.** One predeclared development j20 episode per regime and implementation. cProfile overhead inflates these times; nested cumulative times are not additive and are not deadline-budget measurements. Full function/call data: ../profile_components.csv and profile_*.prof.

**02_equivalent_optimization.** Ten paired development scenarios, separate unprofiled deterministic original/optimized runs with exact allocation/outcome equivalence. Points below diagonal are faster. Wall-clock variation and fixed original-first order limit timing inference. Data: ../optimization_timings.csv.

**03_independent_placements.** Every independent placement, regime and severity, with fixed P/F/G1 and identical reactive reserves. Three placements per stratum; no pooled-sample confidence interval. Placement indices map to actual seeds in placement_indices.csv. Data: ../scenarios.csv.

**04_independent_tradeoffs.** All27 paired independent scenarios per comparator. Lower-left improves both displayed metrics; this is not an all-UE or all-objective Pareto test. Overlapping zero points retained in numerical data. Data: ../paired_differences.csv.

**05_individual_ue_harms.** Each independent contested severe placement/UE; red means G1 harms that UE, blue improves it. Mean worst-UE improvement cannot establish individual protection. Data for every regime/class/severity: ../paired_per_ue.csv and ../per_ue.csv.

**06_complete_tick_cost.** Unprofiled independent execution. Transaction=URLLC allocator plus mandatory G1 post-service credit update; tick=entire engine.step including reference RIC/metrics and physics, excluding external audit/export. Baseline P/F service windows are measurement overhead in the full tick. Dashed line=1ms reference period, not real-time certification. Descriptive CDF across correlated ticks; no independent-sample inference. Data: ../tick_times.csv.gz and ../scenarios.csv.
