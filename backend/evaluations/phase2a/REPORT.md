# Phase 2A validation report

Implementation branch: `phase2a/joint-ric-infrastructure`. Integration base: `1a29280` (full hash in `provenance.json`). It already includes Phase 0, Phase 1, Phase 1.1 and subsequent Phase 1.2 changes; no integration was necessary. The commit containing this report is the implementation commit (`git log -1` on that branch).

## Result

The optional telemetry, cell-local policy selection, atomic reserve/policy command, separate six-feature encoder and scripted 10-ms controller are implemented. No learning/training was performed. The original 15-dimensional encoder, DQN action catalog, network, reward and checkpoint-loading implementation remain unchanged. No checkpoint was created or rewritten. Existing proportional and feasibility scheduler source files are byte-identical to the base. eMBB retains its configured scheduler, including when that configuration is legacy.

See [README.md](README.md) for every telemetry field/unit, exact feature equations, denominator/exposure rules, activation instructions and timing contract. See [CHANGED_FILES.md](CHANGED_FILES.md) for the complete inventory.

## Tests and reproduction

Run from repository root:

```bash
python -m pytest -q
python backend/tools/phase2a_validate.py --output /tmp/phase2a-validation
python backend/tools/phase2a_demo.py --steps 3000 --output /tmp/joint-demo.json
```

The final full suite passed **62 tests in 69.08 seconds**, including 17 new Phase 2A tests. Unit/integration coverage includes frozen telemetry, measurement subscription, trailing causal cohorts, budget equality, partial service and overflow union accounting, missing and zero capacity, no-offer/missing history, exposure masks, invalid/atomic combined commands, fixed-pool guards, scheduler initialization/switching, episode resets, future CSV/jammer isolation, deterministic replay, legacy eMBB and disabled-xApp compatibility. Short integration cases exercise no/moderate/severe jamming and ON/OFF transitions. `test_results.txt` contains the actual pytest output.

| Verification | Measured result |
|---|---:|
| Saved Phase 0 smoke hashes, including legacy, causal, paced and work-conserving cases | 6/6 match |
| Saved fixed-quota Phase 1 hashes: 3 severities × 3 seeds × 2 schedulers | 18/18 match |
| Fixed local policy versus original global policy physical trajectories | 2/2 match (P and F) |
| Scripted replay: full telemetry/control history, interval outcomes and service history | Exact match |
| Commands issued after tick k; first effect at k+1 | All 300 boundaries verified |

This is a bounded historical regression claim. It does not establish reproduction of every published or Phase 1.2 experiment. `regression_summary.json` records scenario results and equivalence hashes. `provenance.json` records protected-source hashes, dependency versions and runtime input hashes. The CSV inputs are locally supplied and were not tracked at the base commit, so their base-commit identity cannot be established through Git; saved trajectory equality is the tested regression evidence.

## Scripted demonstration

The representative run uses seed 3456, the original severe-jamming configuration, 8+8 UEs, 1-ms service, 10-ms decisions and the fixed physical 18-PRB reserve. The cycle below repeats; service at a boundary uses the preceding policy. Physical reserve occupancy varies only through activation/distribution inside the fixed pool.

| Applied after tick | eMBB delta | URLLC delta | Selected mode | First affected tick | Activated reserve |
|---|---:|---:|---|---:|---:|
| 10 | 0 | +3 | Feasibility | 11 | 3 |
| 20 | +1 | 0 | Proportional | 21 | 4 |
| 30 | 0 | -3 | Feasibility | 31 | 1 |
| 40 | -1 | 0 | Proportional | 41 | 0 |

The 3,000-tick run contains 300 completed control intervals, 300 issued commands and a final pending command with zero subsequent ticks. That final command is explicitly unobserved, not credited with a completed outcome. The original jammer still updates baselines inside intervals. No quota/policy change retroactively affects completed service. Frozen telemetry has 16 UE rows at every decision. The eMBB scheduler remains work-conserving proportional throughout this demo; a separate test verifies preservation of legacy eMBB.

The compressed [demo.json.gz](demo.json.gz) contains full telemetry, validity masks, receipts, interval service outcomes and per-tick allocation history. Read it with Python's `gzip.open(..., 'rt')` and `json.load`. `replay_summary.json` records the exact replay checks without duplicating the full trace. Failure-rate features are unavailable during the 100-ms warmup; the first full window has 388 matured tight-class and 356 matured relaxed-class exposures. Unavailable is represented separately from measured zero.

## Execution time

| Measurement | Result |
|---|---:|
| Simulated time / service ticks | 3 seconds / 3,000 |
| Unpaced simulation-loop wall time | 1.9501 seconds |
| Joint RIC boundary computation median | 620.019 microseconds |
| Joint RIC boundary computation p95 | 753.089 microseconds |

These are one-run measurements in the provided environment, not real-time guarantees. Loop time excludes process startup and output serialization. Boundary time includes telemetry, encoding, scripting, validation and recording; it excludes gNB service, neural inference and real transport. Tests and environment versions are recorded separately. Direct synchronous commands have no simulated transport delay, and computation time does not advance the simulation clock.

## Remaining risks and Phase 2B recommendation

- The 100-ms matured-arrival failure fraction has a different denominator from Phase 1's all-offered failure lower bound. It combines several 10-ms policy intervals, so learning credit assignment must account for history. Older overdue pending traffic is separately reported rather than silently omitted.
- Sparse UEs may not meet the per-UE minimum of 20 matured arrivals. A future learning interface must consume validity/exposure masks explicitly. Filling unavailable features with zero would be invalid.
- These six features are an independent encoder, not a complete learning state. Phase 2B must specify any state extension, normalization, masks, joint action catalog, replay semantics and reward before training. Changed dimensions would require an appropriate new model/checkpoint; the old DQN is not optimized for policy switching.
- This is one-cell, single-owner, single-threaded in-process control. It provides no E2 implementation, xApp arbitration or distributed transaction guarantee. A 10-ms controller cannot ensure prevention of unexpected 2-ms budget misses.
- Pending-queue consistency/overdue scans scale with backlog, and full history logging grows with horizon. Long-run computation and memory should be profiled before large experiments.
- Switching preserves the existing starvation-age semantics, but fairness, recovery and application utility are not established by a scripted cycle. Previously observed feasibility-policy worst-UE tradeoffs remain relevant. No improvement or Pareto claim follows from this phase.

The infrastructure is ready for a separately specified Phase 2B experiment. Before training, define state/validity handling and causal outcome attribution, then use paired exogenous scenarios and held-out seeds to compare against fixed-policy baselines. **Phase 2B has not been started.**
