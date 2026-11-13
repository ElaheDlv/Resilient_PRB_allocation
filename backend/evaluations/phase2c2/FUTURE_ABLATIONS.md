# Later learning interpretation and ablations — recommendations only

No ablation or learner is implemented or trained in Phase 2C-2. Model parameters, reward, state and action catalog remain unchanged.

The Phase 2C-1 checkpoints were all chosen at episode 3: 897 observed transitions and 866 updates, although complete training continued to 2691/2660. Those selected weights had seen only placement seed1101. Validation seed2101 had zero known/upper failures and 96-Mbps eMBB for every candidate; earlier-episode tie-breaking discarded later, harder experiences. This does not identify convergence, optimum interaction budget or superior learning. Full training was not independently repeated end-to-end.

Adjacent legacy scenario seeds also share registration streams: C1 held-out seeds3101/3102 share 7/8 URLLC placements. Distinct integer labels were not independent geometry draws. Paired within-scenario results remain valid, but prior independent-seed wording/uncertainty assumptions require correction. Final v2 uses nonoverlapping registration streams without changing the radio model.

One network initialization (12026) gives no variance estimate across learned policies. Multiple scenario placements do not replace independent optimizer/network seeds. A future pilot should predeclare at least several independent training initializations and report each one; no such run is authorized here. Keep equal simulator interactions, not merely wall time or optimizer steps, across comparisons.

Original 15-state learners optimize the original average-latency/eMBB/resource reward with MSE/no clipping. The proposed 30-state controller optimizes exactly-once deadline events, absolute worst-UE risk duration, actual eMBB shortfall, reserve and mode-switch costs with Huber/norm clip 10. Differences in returns, losses, observations and robustness cannot be attributed only to joint action selection. Loss magnitudes across those objectives are not comparable convergence scores. Retain original-objective baselines as historical references, not the sole mechanism-isolation controls.

Prospective matched ablations should share the same 30-feature state, six validity bits/two exposure-known bits, unchanged joint reward and weights, Huber loss, optimizer, exploration schedule, replay, target updates, horizon/censoring and interaction budget:

1. **Reserve-only/P**: physically valid reserve increments; URLLC mode fixed P.
2. **Reserve-only/F**: same, mode fixed F.
3. **Switch-only**: mode P/F is learned while a frozen, declared causal reserve rule (for example the existing B1 reactive rule) determines reserve increments. Its reserve trajectory can react to the current baseline but cannot be optimized by the switch learner. A zero-reserve version may be a separate diagnostic, not a silently stronger/weaker substitute.
4. **Full joint**: both decisions learned.

One way to match parameter counts is the same 98-output network with additional ablation action masks: only the fixed mode for reserve-only, or only the predetermined current reserve increment pair for switch-only. This is a specification, not code in this phase. Mask/replay/TD tests are required before implementing it. Reserve-only/F is particularly important if F is already a strong static scheduler; do not select scenarios to manufacture an advantage for the joint model.

The 100-ms matured-arrival risk window overlaps ten consecutive 10-ms decisions. Repeated observations/rewards are correlated and mix outcomes from several earlier policies. They are causal history, not independent samples of the most recent action's effect, nor proof of a Markov state. The worst-risk cost is duration shaping; failure-event costs must stay exactly once. Report raw exposures and unavailable warmup separately. Do not treat windows, UEs, ticks or batches as independent trained seeds.

The 2-ms/10-ms queueing budgets are at or below one RIC interval. A surprise jam or burst can miss a tight budget before the next command; a slower controller cannot retroactively repair that batch. One-ms local scheduling still matters. Capacity constructions do not remove control latency, initial zero reserve, inherited backlog or idealized in-process signaling. eMBB has a separate throughput objective, not those queueing budgets. These synthetic downlink batch budgets do not establish packet or industrial end-to-end reliability.

Use the prospective validation metric hierarchy in DESIGN/README, balanced over all regimes/severities and multiple placement seeds, and report each regime separately. Reserve/switching terms discriminate equal-deadline candidates; earlier checkpoint is only the final tie-break. Aggregate-first ranking can still trade one UE against others: no per-UE nondegradation or Pareto claim follows. If an industrial application demands a hard worst-UE constraint, justify and freeze that constraint from application requirements before training; do not invent its numerical SLA after seeing outcomes.

Discovery pools are reference-characterized and future-model-unseen, not untouched test data. Freeze membership and do not retune from subsequent learned-model holdout results. A publication-level generalization claim should include a separately preregistered untouched evaluation cohort. No new cohort search, reward retuning, prediction model, LLM integration, dynamic physical reserve, training-budget increase or journal experiment is started by this recommendation. Retain the four-core, CPU-only, 2-GiB warning/2.5-GiB monitored-stop safeguards.
