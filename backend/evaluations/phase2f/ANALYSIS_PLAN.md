# Prereplay diagnostic analysis plan

Do not adjust weights or run any reward ablation. Use all504 saved episodes for
identified totals/bounds and the exact24-candidate ordering. No significance
claim from correlated checkpoint/episode/tick comparisons.

Use the nine predeclared `grid` replays as the primary interval-component
distribution sample (three severities per regime, one placement/NN seed). Report
all extra comparisons separately and label the narrow diagnostic sampling.
Six components are original positive costs; return is their negative sum.

Descriptive saturation: worst risk=1 within1e-12, with validity denominators.
Descriptive high risk: worst>=.9; this threshold is not an SLA or reward change.
Distinguish (a) service catches up while fresh failures continue, from (b) no new
known failures and positive historical worst cost. Strong favorable-interval
criterion: no new failures and on-time completed batches>=newly offered batches.
Show longest observed runs/durations and exact adjacent intervals. If no such
case exists in this sample, say so; do not add replays to obtain a desired example.
Track paid failure events charged under a later action/mode than their arrival;
this is temporal attribution, not proof of incorrect causal transition timing.
Historical failed-cohort arrival modes likewise do not identify causal blame.

Switch influence: report absolute component quantiles, sums/shares and elementary
fixed-weight exchange rates; compare switch/stay observations without causality.
No cost share is defined if total cost is zero.

Resource interpretation: report unused activated capacity in **observed** grants,
and low quotas versus existing exogenous fresh-demand certificates during known
failures. Do not assert an unobserved reserve action would have saved a batch.
Use paired already-observed controllers/checkpoints to illustrate reward ordering;
keep frozen-policy inferiority under the very same reward distinct from a
scalar-versus-lexicographic objective disagreement. Do not infer hypothetical
alternative outcomes by correlations.

All saved physical horizons are3s. Analyze exact terminal charges across training
checkpoints and regimes. Prefix administrative accounting at100/500/1000/2000/
3000ms is a calculation on the **observed prefix**, not a different-horizon
experiment. Recommend only a small matched prospective ablation plan; do not
compute new reward-arm returns or train any model here.

Schema clarification during analysis: the recorded issue-step gap detects a later
decision interval, not necessarily a changed action value. The public fields are
`cross_decision_interval_failures` and `later_decision_interval_fraction`; all
reported percentages use that literal definition. An exact header-only rename
clarified the recorded interval table; no values or trajectories changed.
