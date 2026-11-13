# Phase 1.2: fairness, useful service, and workload feasibility

Base `6bce0d8`; branch `phase1.2/fairness-feasibility`. The new optional intra-URLLC scheduler is `deadline_fair`. Legacy, proportional, original slack, and feasibility policies remain selectable. The DQN, radio/jammer models, source traces, 18-PRB physical reserve pool and fixed activated reserve of three URLLC PRBs are unchanged.

## Causal bounded service-credit policy

This candidate adds a bounded priority for recent service deficits to the Phase 1.1 feasibility allocator. It is a heuristic for trading timely completion against service deprivation, not a proven max-min deadline optimizer or a guaranteed minimum service rate.

For UE i at interval t, let B_i(t) be observed pre-service queued bytes, B_i^+(t-1) its actual post-service queue last interval, b_i(t) estimated bytes/PRB, and Q(t) the fixed slice quota. Infer newly **admitted** bytes as a_i(t)=max(0,B_i(t)-B_i^+(t-1)). No future trace data, future jammer state, prediction, DQN state, or measurement-only completion counters enter the policy.

1. Track recent admitted bytes per interval, r_i(t)=0.95*r_i(t-1)+0.05*a_i(t). On first observation initialize r_i=a_i. This is a causal historical average, not a forecast. Overflowed bytes are excluded because they are not queue-serviceable traffic.
2. Estimated attainable PRB demand x_i=min(r_i/b_i,Q,per-UE cap), excluding zero/nonfinite b_i. Capacity scaling h=min(1,Q/sum_i x_i), with h=0 when the sum is zero. Service entitlement in bytes is e_i=h*x_i*b_i, or zero for excluded rates.
3. Maintain byte credits z_i. Before scheduling, z_i=clip(z_i+e_i, -L_i,+L_i), where L_i=20*max(1,e_i). After actual network service s_i, z_i=clip(z_i-s_i,-L_i,+L_i). Actual served bytes, including partial batches, are debited, not achievable capacity or granted PRBs. State is removed when a UE is deregistered/reset.
4. Apply the existing zero-service starvation guard first (three ticks for both feasibility variants). Then use **at most** floor(alpha*Q) additional PRBs, limited by remaining quota, to repay positive credits. At every useful integer quantum choose the largest normalized remaining credit:

   (z_i-floor(g_i*b_i))/max(1,e_i),

   with descending starvation age then ascending UE identifier as deterministic ties. g_i is the current round's cumulative grant, including guard service. A quantum must fit the fairness allowance, current slice quota, useful demand and per-UE cap.
5. Use every remaining eligible PRB under the unchanged Phase 1.1 feasibility/EDF/remaining-work rank. Protected/fallback service can drain expired or infeasible FIFO heads; nothing is discarded or reprioritized within a UE queue. Partially served batches retain original arrival time. Empty/zero-rate queues receive no grants; inability to fit a useful subbyte bundle is not eligible idle capacity.

For useful-demand ceilings, virtual FIFO updates, inclusive deadline boundaries and the optimistic feasibility formula, see `../phase11/README.md`. The fairness allowance is a bounded intervention, not a separate physical reserve and not a hard minimum allocation. Setting alpha=0 reproduces the feasibility allocator's grants. Credits cannot guarantee deadline fairness in overload, erase accumulated FIFO delay, or give a hard service-share bound. The old conditional zero-service bound remains; a short no-service bound is different from adequate throughput or timely completion. Inactive queues can retain decaying historical load/credit; burst responsiveness beyond these traces is not established.

Environment: `DEADLINE_FAIR_FRACTION`, default 0.35, valid[0,1]. Global scheduler default remains legacy. The credit window 20 ticks and smoothing 0.05 are fixed design choices; only alpha is tuned. Clipping the credit state does not bound real unserved workload or actual delay.

## Operating regions and pairing

Each evaluation uses 8 eMBB and 8 URLLC UEs, fixed 2/10-ms URLLC queue budgets, 1-ms simulator intervals, fixed eMBB/URLLC reserve activation 0/3, and unchanged jammer period 2500 ticks/duty 0.5. Evaluate jamming 0 and 20; the latter gives URLLC quotas 57 while ON and 63 while OFF. No-jamming quota is 63. Jammed trials begin ON. Three-second episodes include an initial 1249-tick ON segment (service ticks 1–1249), a complete 1250-tick OFF segment (ticks 1250–2499), and 501 ticks of the next ON segment; there is one OFF recovery window per independent seed. These horizons are explicitly shorter than the five-second Phase 1.1 diagnosis and are not pooled with it.

To separate capacity regimes from placement variation, obtain each seed's static radio capacities b_i from 20-tick exogenous calibration trials under the same channel model. Calibration performance is discarded; only rate/quota/assignment fields define workloads. The same common batch size lambda is assigned to every URLLC UE in that seed/regime:

lambda = floor(rho*57 / sum_i(1/b_i)),

for rho=0.60 (nominal),0.96 (near),1.15 (overloaded). Scale only the CSV `length` column by lambda/650, with integer flooring; retain timestamps, addresses, slice, direction and original replay binning. Source CSV files are never edited. The supplied source contains constant 650-byte one-ms batches. eMBB replay is untouched. Generated configs/traces are saved in the output directory, with batch-size/rate/trace hashes in summaries.

Normalization is an experimental conditioning device using fixed channel capacities; it is not information given to the scheduler and is not a deployment load-adaptation controller. These are synthetic, capacity-conditioned variations of the existing trace, **not proof of realistic industrial traffic or application-calibrated deadlines**. The near regime can exhibit integer contention even when fluid utilization is below one; whether this produces measurable policy differences is an experimental result, not assumed.

For every seed/regime/jammer setting, all four policies must match complete exogenous, budget-assignment and actual service-quota hashes. Every UE's capacity must also match its calibration. Channels are static under the preserved frozen-mobility setup. Settings affect volume alone, not deadline assignment or placement.

## Feasibility certificates and limitations

Let C=sum_i(lambda/b_i) be fluid PRBs per arrival interval and R=sum_i ceil(lambda/b_i) the one-interval integer drain requirement. R<=minimum quota is a strong constructive certificate for these constant synchronous arrivals: every batch can finish in its arrival interval. C<=Q alone is only a necessary long-run condition, not a deadline-schedulability proof. Batch granularity, FIFO backlog and finite budgets matter.

For T constant-quota service timestamps starting with an empty arrival window, count n_i=max(0,floor(T-D_i/dt+epsilon)) arrivals whose deadlines fall inside that window. Any zero-miss schedule necessarily requires sum_i(n_i*lambda/b_i)<=T*Q. A strict excess proves at least some failures unavoidable even with ideal cross-UE scheduling; it does not prove the observed failure rate optimal or identify which UE must fail. Use the first 1249-tick ON window for severe trials and the full 3000-tick horizon for no-jamming trials. Per-UE lambda<=floor(Q_min*b_i) certifies that this UE alone has sufficient one-interval capacity; this does not imply joint feasibility.

Existing Phase 1.1 harmed cases are analyzed from their committed counters, five-second horizon and original 650-byte load. Their long-run and finite-window conditions are distinguished. A missing impossibility certificate is not a feasibility proof. Service starvation is reported separately as (a) positive queued demand but zero actual service and (b) cumulative service materially below arrivals. The latter can persist despite service every interval.

## Validation and held-out protocol

Validation seeds 6001/6002 use separate budget assignment 2028; held-out seeds 7001–7006 use assignment 2026. No seed overlaps previous studies. Validation evaluates proportional and feasibility baselines plus alpha in{0.15,0.35,0.65} across all three regimes and both jammer settings:60 episodes. Select the lowest mean worst-UE failed-offered fraction subject to retaining at least 50% of the mean aggregate feasibility-versus-proportional improvement; ties use aggregate failure then smaller alpha. If no parameter meets the constraint, choose aggregate failure then worst-UE failure and report constraint failure. All conditions have equal weight.

The selected alpha is 0.35. Validation mean aggregate failures: proportional 32.1962%, feasibility 4.0701%; retention threshold 18.1332%. Alpha 0.15/0.35/0.65 scores 8.1038%/14.1524%/22.1958% aggregate and 32.5556%/32.5500%/32.9472% mean worst-UE. The tiny worst-UE difference selecting 0.35 is weak tuning evidence; there is no claimed statistically established parameter optimum. The parameter was frozen before primary held-out evaluation;144 primary held-out episodes were run once.

Validation at rho 0.96 produced no known misses. A validation-only screening at rho 0.995 (16 episodes, same 6001/6002 validation seeds, frozen alpha) likewise produced no known misses. To examine a boundary where finite budgets and integer scheduling can matter, an additional threshold at rho 1.0005 was selected using those validation outcomes only:16 validation episodes, then 48 once-evaluated paired held-out episodes using new seeds 9001–9006 and budget seed 2026. Alpha remains 0.35, with no retuning. The original rho 0.96 regime and its zero-result screening are retained; these additions are labeled supplementary rather than claimed predeclared before the primary evaluation.

An initial threshold-validation launch incorrectly selected 8000-series seeds. It was stopped; its partial results are excluded, and **all 8001–8006 seeds are excluded from the held-out cohort**. The corrected validation summary is asserted to contain only 6001/6002. A regression test locks validation/held-out separation. These aborted trials do not count in completed study episodes or parameter selection. The threshold extension combines 144 primary episodes and 48 threshold episodes, each cell with six independent seeds.

A final recoverable-overload control uses target rho 1.05: fluid demand exceeds the 57-PRB ON quota but remains below the 63-PRB OFF quota. The original rho 1.15 overload cannot recover even OFF, so it cannot establish that a scheduler impairs recovery versus simply lacking capacity. This extra control was added after that operating-region limitation was identified; it uses 16 validation episodes on 6001/6002, then 48 once-evaluated held-out episodes on untouched 10001–10006, with alpha still 0.35. No further load regimes or parameter searches were added. Final included totals:108 validation/screening episodes plus 240 held-out episodes. Calibration-only trials and the aborted launch are excluded from those totals.

Confidence intervals use six independent seed values or within-seed paired differences, Student t df 5, descriptive 95%, without multiplicity correction. UEs, batches, service ticks and windows within a seed are not independent replications. Means across different operating regions are protocol summaries, not estimates of an industrial workload distribution. No Pareto improvement is asserted.

## Measurement and timing

Offered-batch failures use the existing disjoint precedence: any overflow, fully admitted late completion, or fully admitted unfinished-overdue traffic are known failures; fully admitted unfinished not-yet-due traffic is censored. Lower fraction=known failures/offered; upper adds not-yet-due unresolved. Completed-admitted-batch lateness and served-byte residence lateness remain separate from sampled after-service HOL violations. Equality is on time. No batch measurement is called packet reliability.

Recovery requires 10 ms of sustained **all-UE** compliant post-service HOL after jammer OFF. Nonrecovering windows are censored rather than omitted. No-jamming recovery is undefined, not zero success. Individual first compliant-hold recovery windows are also exported in per_ue_recovery.json. Individual recoveries at disjoint times do not imply simultaneous all-UE recovery. Reports include per-UE and class failures, unresolved traffic, actual bytes/PRBs, rate/CQI/MCS, sampled HOL runs, zero-service runs, resources and recovery windows. Phase 1.1 sidecars retain horizon censor flags.

Scheduler times are measured with the existing wall-clock perf_counter_ns around allocation, including credit preparation but excluding post-service credit debit, network service and gzip diagnostics. These are grant-computation timings, not total-policy implementation overhead. Median, p 95 and p 99 are reported as the mean of each seed's per-interval quantiles, not percentiles of seed means or a real-time guarantee. Loop runtime includes diagnostics/I/O and is host-dependent. Preserve original p 95/mean fields for compatibility.

## Exact commands

From the repository root in the existing Python environment:

```bash
python -m pytest -q
python backend/tools/phase0_smoke.py --output /tmp/phase12-compat
python backend/tools/phase12_study.py --output /tmp/phase12 --stage validation
python backend/tools/phase12_study.py --output /tmp/phase12 --stage heldout
python backend/tools/phase12_study.py --output /tmp/phase12 --stage boundary_validation
python backend/tools/phase12_study.py --output /tmp/phase12 --stage threshold_validation
python backend/tools/phase12_study.py --output /tmp/phase12 --stage threshold_heldout
python backend/tools/phase12_study.py --output /tmp/phase12 --stage recovery_validation
python backend/tools/phase12_study.py --output /tmp/phase12 --stage recovery_heldout
python backend/tools/phase12_analyze.py --source /tmp/phase12 --output /tmp/phase12-analysis
```

The heldout command requires validation's frozen selection.json. All evaluations run in fresh processes, unpaced, with the same causal fixed-quota controller. No checkpoint is loaded. Full gzip allocation/event streams and raw configs/scaled traces are regenerable in the output directory. Committed summaries retain per-run/per-UE/class results, inputs, hashes, recovery, timing and diagnostic examples; CSV exports supply all phases. `existing_candidate_harms.json` derives only from the committed Phase 1.1 data. `REPORT.md` gives the measured decision; `CHANGED_FILES.md` lists the exact implementation/artifacts.

Zero-variance t intervals in all-zero cells describe these six observed seeds only; they are not reliability guarantees or bounds on rare unobserved failures.

Representative Phase 1.1 trajectory regression commands (compare `trajectory_sha256` with the matching seed4008/severity20/policy records in `backend/evaluations/phase11/heldout_summary.json`):

```bash
python backend/tools/phase0_evaluate.py \
  --config backend/evaluations/phase1/main_scenarios/scenario_j20.json \
  --scheduler proportional_wc --policy fixed --fixed-u 3 --transitions causal \
  --seed 4008 --steps 5000 --phase1-metrics --output /tmp/original-wc.json
DEADLINE_STARVATION_TICKS=3 python backend/tools/phase0_evaluate.py \
  --config backend/evaluations/phase1/main_scenarios/scenario_j20.json \
  --scheduler deadline_feasible --policy fixed --fixed-u 3 --transitions causal \
  --seed 4008 --steps 5000 --phase1-metrics --output /tmp/original-feasible.json
```
