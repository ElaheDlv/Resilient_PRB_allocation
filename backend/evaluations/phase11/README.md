# Phase 1.1: scheduler diagnosis and fixed-quota experiment

Base: `cb35202`, branch `phase1.1/scheduler-diagnosis`. Only the intra-URLLC allocation rule and opt-in read-only diagnostics are extended. No DQN retraining, prediction, reward/state/architecture change, jammer change, trace change, or physical reserve partition change.

## Candidate rule

All queues remain FIFO; partially served batches retain their original arrival timestamp and their remaining bytes. There is no expiry discard. As in Phase 1, BS service occurs at the current simulator timestamp, so service at exactly the budget is on time. These are queue residence budgets, not serialization or end-to-end deadlines.

For UE i, current per-PRB service capacity is b_i = c_i * dt / 8 bytes per interval; c_i is the unchanged radio model estimate. Its budget is D_i. At each useful integer grant quantum, the read-only virtual FIFO supplies arrival a_i and remaining head bytes h_i. Given cumulative current grants g_i and quota Q:

- Useful total grant limit: L_i = ceil(B_i / b_i), further limited by the existing per-UE cap.
- Useful next quantum: q_i = max(1, ceil((floor(g_i*b_i)+1)/b_i) - g_i). Eligibility requires g_i+q_i <= L_i and q_i <= remaining slice quota. Empty and nonpositive/nonfinite-rate queues are excluded.
- Inclusive remaining service timestamps: K_i = floor((D_i-max(0,t-a_i)+1e-12)/dt)+1.
- Estimated additional PRBs to finish the virtual head: R_i = max(1, ceil((floor(g_i*b_i)+h_i)/b_i)-g_i).
- Current per-interval maximum: C_i = min(Q, per-UE cap), or Q without a cap.
- Optimistic remaining attainable PRBs: A_i = max(0,C_i-g_i) + max(0,K_i-1)*C_i.
- A head is *potentially feasible* if K_i>0 and R_i<=A_i. This is a local screening estimate, not a joint schedulability guarantee: it optimistically assumes unchanged current rate and full current slice quota for this UE at future timestamps. Future shared competition and per-interval byte rounding can make it optimistic. No arrival, channel, or jammer prediction is used.

First grant at most one useful quantum to each protected UE (zero-byte-service age >= S), ordered by descending age then UE identifier. Then repeatedly choose:

1. Potentially feasible heads before infeasible/expired heads.
2. Among potentially feasible heads: earliest absolute deadline a_i+D_i, then smallest R_i, then largest starvation age, then UE identifier.
3. Among infeasible/expired heads: smallest R_i, then largest starvation age, then identifier. They remain work-conserving fallback traffic and can also receive protected service.

Advance only the virtual FIFO by the incremental whole service bytes; recompute eligibility and rank after each quantum. No actual queue is changed until normal simulator service. This gives completion preference at equal deadlines while preserving the shared useful-demand bounds. Unused PRBs are allowed only when no eligible useful quantum fits quota/caps. A residual fractional byte in the last allocated PRB is unavoidable granularity, not an excess grant.

The guard bounds **zero-service starvation**, not queue delay or completion fairness. For persistent eligible unit-quantum queues with at least one PRB available per interval, the conditional bound remains S+N-1 intervals. Subbyte bundles, zero capacity and binding caps need feasibility conditions; no unconditional guarantee is claimed. Expired heads can block later viable batches because FIFO is preserved.

The original `deadline_slack` formula is unchanged. `deadline_feasible` selects this candidate; eMBB continues to use the Phase 0 work-conserving proportional allocator. Repository defaults remain legacy.

## Protocol fixed before held-out evaluation

All runs: 5 seconds, 1 ms intervals, 8 eMBB and 8 URLLC UEs, original replay traces and radio model, frozen placements, two queue budgets (2/10 ms), fixed reserve activation eMBB=0/URLLC=3 from the unchanged pool of 18. Jamming severities 0/10/20 use the exact Phase 1 jammer configuration and paired episode ID. Actual service-quota, complete exogenous-trajectory, and budget-assignment hashes must agree across policies.

Reproduction uses original seeds 1234,2345,3456 and budget assignment seed 2026. It reproduces the 18 fixed-quota Phase 1 runs involved in the scheduler discrepancy; frozen-DQN and extra-load Phase 1 runs are outside this diagnosis's controlled comparison.

Validation uses seeds 3001,3002 and separate budget assignment seed 2027. Candidate starvation threshold S is selected from {3,5,10} by mean known-failed offered fraction across validation severities/seeds, then mean worst-UE fraction, then smaller S. All validation scores tied, selecting S=3; validation therefore supplies no evidence that guard tuning itself improves performance. The original slack guard remains 5.

Held-out seeds 4001,...,4008 and budget seed 2026 are used once after freezing S. All three policies receive identical paired inputs and quotas. No parameter is selected using these results. Independent units for confidence intervals are seeds, not UE samples, time ticks, or jammer windows. Descriptive 95% Student t intervals use eight seed values, df=7; paired differences use within-seed differences. Rare bad placements and eight seeds make these exploratory intervals; no multiplicity-adjusted significance claim is made. Unbounded difference intervals are retained rather than clipped.

## Accounting and diagnostic semantics

Each offered batch enters exactly one final category, in this precedence order:

1. Any overflow: **overflow failure**, even if the admitted part later completes late. Fully dropped and partially admitted batches are separately counted.
2. Fully admitted and completed within budget: **on-time**.
3. Fully admitted and completed after budget: **late-only failure**.
4. Fully admitted, unfinished at horizon, already overdue: **unfinished-overdue failure**.
5. Fully admitted, unfinished, not yet overdue: **unresolved-not-due** (censored).

N = overflow + on-time + late-only + unfinished-overdue + unresolved-not-due is asserted per UE. Lower failed-offered fraction = (overflow+late-only+unfinished-overdue)/N. Upper = (overflow+late-only+unfinished-overdue+unresolved-not-due)/N. Unresolved traffic is not silently omitted or automatically called a failure. Raw late-completed counts include completions of partially admitted batches and therefore overlap descriptive overflow counts; **only the disjoint late-only counter is used in failure sums**. Unfinished accepted traffic can similarly overlap overflow descriptively; the outcome partition gives overflow precedence. Both event-derived counters and original sidecar counters are checked independently. Offered bytes = served + remaining + overflow remains enforced.

Served-byte lateness is measured at each byte-service event. Completed-batch lateness is measured when the entire admitted batch completes. HOL violation frequency is sampled after service; a late departing batch can leave a compliant head, so zero sampled HOL violations does not imply zero late completions. Consecutive sampled runs are observed durations, right-censored at the horizon when applicable; Phase 1 reports retain censor flags. A service-starvation run is consecutive intervals with positive pre-service queue and positive estimated rate but zero actual bytes served. It does not measure receiving too little service to meet a deadline.

CSV `phase=before/during/after` resource/HOL/starvation statistics describe observation-time segments. Batch outcomes in the same rows are **arrival cohorts**, followed until the common horizon, including completions in a later segment. Repeated jam windows are pooled as during/after, but consecutive runs are reset across segment transitions. Jamming starts ON in these configurations, so there is no observed pre-jamming baseline for jammed runs; before rows are empty. The paired no-jamming condition provides a separate reference. No pre-jamming results are invented.

Raw gzip JSONL traces contain every pre-service UE FIFO head/rate/CQI/MCS/grant and all URLLC arrivals, partial-service events and completions. A retained pair of full moderate-jamming seed3456 traces and selected evidence ticks support direct inspection. `expired_head_grant_prbs` and `infeasible_grant_prbs` describe the **pre-allocation head**, not all bytes within a grant: a grant may also serve later viable batches. They must not be interpreted as precise wasted-PRB counts. Offered/drop/completion accounting uses actual events, not these proxies. Diagnostics currently target the study's single-cell, single-episode configuration.

## Exact reproduction commands

From repository root with the same Python environment as Phase 1:

```bash
python -m pytest -q
python backend/tools/phase0_smoke.py --output /tmp/phase11-compat
python backend/tools/phase11_study.py --output /tmp/phase11 --stage reproduction
python backend/tools/phase11_study.py --output /tmp/phase11 --stage validation
python backend/tools/phase11_study.py --output /tmp/phase11 --stage heldout
python backend/tools/phase11_analyze.py --source /tmp/phase11 --output /tmp/phase11-analysis
```

Validation must finish before heldout; selection.json records the frozen parameter and seeds. The scripts run every evaluation in a fresh process, unpaced, with causal fixed reserve application. Full diagnostics add wall-clock overhead; scheduler computation timing excludes the diagnostic callback. Single-run example:

```bash
DEADLINE_STARVATION_TICKS=3 python backend/tools/phase0_evaluate.py \
  --config backend/evaluations/phase1/main_scenarios/scenario_j20.json \
  --policy fixed --fixed-e 0 --fixed-u 3 --transitions causal \
  --scheduler deadline_feasible --seed 4001 --steps 5000 \
  --phase1-metrics --diagnostics /tmp/example.jsonl.gz --output /tmp/example.json
```

`per_ue_attribution.csv` and `per_class_attribution.csv` cover reproduction, validation, and heldout, including each segment's outcomes. `heldout_analysis.json` supplies seed-level intervals and paired UE harms. Stage summaries retain all per-run per-UE/class data, recovery windows, trajectory hashes, and concrete diagnostic examples. Large regenerable full traces/reports are not all duplicated in version control.
