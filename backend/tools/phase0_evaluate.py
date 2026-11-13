"""Reproducible, optionally unpaced offline evaluation of existing policies.

Run from the repository root. The explicit xApp list excludes UI/agent services;
all UE, BS, jammer and RIC stepping still uses SimulationEngine.step().
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def run_evaluation(config, *, policy="idle", checkpoint="", seed=1234,
                   transitions="legacy", scheduler="legacy", paced=False,
                   measurements=False, steps=10000, period=10, dt=0.001,
                   stop_after_episodes=1, event_path="", phase1=False,
                   fixed_reserve=(0, 0), per_ue_cap=None, diagnostics_path=""):
    # settings are imported once. For another configuration use a fresh process.
    if "settings" in sys.modules:
        raise RuntimeError("Use a fresh process for each evaluation configuration")
    measurements = measurements or phase1
    os.environ.update(RAN_TOPOLOGY_PRESET="simple", SIM_FREEZE_MOBILITY="1",
        SIM_STEP_TIME_DEFAULT=str(dt), UE_SPAWN_RADIUS_M="400", UE_DEFAULT_MAX_COUNT="100",
        PRB_GYM_CONFIG_PATH=str(config), RES_VALIDATE_CONFIG_PATH=str(config),
        RES_VALIDATE_ENABLE="1", RES_VALIDATE_POLICY=policy, RES_VALIDATE_SEED=str(seed),
        RES_VALIDATE_DECISION_PERIOD=str(period), RES_VALIDATE_DQN_MODEL=checkpoint,
        RES_VALIDATE_QRDQN_MODEL=checkpoint, RES_VALIDATE_DASH_ENABLE="0",
        RES_VALIDATE_TB_ENABLE="0", RES_DQN_TRANSITION_MODE=transitions,
        PRB_UE_SCHEDULER=scheduler, PHASE0_MEASUREMENTS="1" if measurements else "0",
        PHASE0_MEASUREMENT_PATH=str(event_path),
        RES_VALIDATE_FIXED_E=str(fixed_reserve[0]), RES_VALIDATE_FIXED_U=str(fixed_reserve[1]),
        JAM_RESERVED_PRBS="18", RES_DQN_TOTAL_RESILIENCY_PRB="18")
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small MLP inference is dominated by thread-pool overhead on many-core hosts.
    torch.set_num_threads(1)
    from network_layer.simulation_engine import SimulationEngine
    from network_layer.xApps.xapp_resilient_policy_validator import xAppResilientPolicyValidator

    engine = SimulationEngine()
    engine.reset_network()
    engine.network_setup(xapp_classes=[xAppResilientPolicyValidator])
    for cell in engine.cell_list.values():
        cell.prb_per_ue_cap = per_ue_cap
    observer = None
    if phase1:
        from network_layer.phase1_metrics import Phase1Metrics
        observer = Phase1Metrics()
        engine.after_service_observer = observer.observe
    diagnostic = None
    if diagnostics_path:
        if not phase1:
            raise ValueError("Diagnostics require Phase 1 metrics")
        from network_layer.scheduler_diagnostics import SchedulerDiagnostics
        diagnostic = SchedulerDiagnostics(diagnostics_path)
        for cell in engine.cell_list.values():
            cell.allocation_diagnostic = diagnostic.allocation
        for bs in engine.base_station_list.values():
            bs.queue_measurements.event_sink = diagnostic.event
        previous_observe = engine.after_service_observer
        def observe(engine, dt):
            previous_observe(engine, dt)
            diagnostic.observe(engine, dt)
        engine.after_service_observer = observe
    app = next(iter(engine.ric.xapp_list.values()))
    digest = hashlib.sha256()
    records = []
    started = time.perf_counter()
    for _ in range(steps):
        engine.sim_step += 1
        engine.step(dt)
        cells = []
        for cell in engine.cell_list.values():
            bs = cell.base_station
            cells.append(dict(quotas=dict(cell.slice_dl_prb_quota),
                baseline=dict(getattr(cell, "base_dl_prb_quota", {})),
                effective=cell._effective_max_dl_prb,
                ues=[dict(imsi=u.ue_imsi, queued=bs.get_dl_buf_bytes(u.ue_imsi),
                    served_bps=u.served_downlink_bitrate, latency_s=u.downlink_latency,
                    prbs=cell.prb_ue_allocation_dict[u.ue_imsi]["downlink"])
                    for u in cell.connected_ue_list.values()]))
        record = dict(step=engine.sim_step, cells=cells)
        digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
        if len(records) < 3 or engine.sim_step % 1000 == 0:
            records.append(record)
        if paced:
            time.sleep(dt)
        if app._episode_counter >= stop_after_episodes:
            break
    elapsed = time.perf_counter() - started
    reports = []
    for bs in engine.base_station_list.values():
        if getattr(bs, "queue_measurements", None) is not None:
            reports.extend(bs.queue_measurements.closed)
            for imsi in list(bs._dl_replay):
                reports.append(bs.queue_measurements.finalize(imsi, bs._time_s))
            bs.queue_measurements.close()
    diagnostics = diagnostic.finish(reports) if diagnostic is not None else None
    pending = getattr(app, "_causal_interval", None)
    pending = pending.pending if pending is not None else None
    result = dict(seed=seed, transitions=transitions, scheduler=scheduler, paced=paced,
        policy=policy, steps=engine.sim_step, simulation_seconds=engine.sim_step*dt,
        python_version=sys.version.split()[0], numpy_version=np.__version__,
        torch_version=torch.__version__, device=app.device, torch_threads=torch.get_num_threads(),
        completed_episodes=app._episode_counter, elapsed_seconds=elapsed,
        steps_per_second=engine.sim_step/elapsed, trajectory_sha256=digest.hexdigest(),
        samples=records, batch_measurements=reports,
        unfinished_control_interval_ticks=pending["ticks"] if pending else 0,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None,
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest())
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    if observer is not None:
        result["phase1"] = observer.finish(reports)
        result["fixed_reserve"] = list(fixed_reserve) if policy == "fixed" else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=["idle", "random", "aggressive", "dqn", "qrdqn", "fixed"], default="idle")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--transitions", choices=["legacy", "causal"], default="legacy")
    parser.add_argument("--scheduler", choices=["legacy", "proportional_wc", "deadline_slack", "deadline_feasible", "deadline_fair"], default="legacy")
    parser.add_argument("--diagnostics", default="", help="Gzip per-tick allocation and batch-event trace")
    parser.add_argument("--phase1-metrics", action="store_true")
    parser.add_argument("--fixed-e", type=int, default=0)
    parser.add_argument("--fixed-u", type=int, default=0)
    parser.add_argument("--per-ue-cap", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--measurements", action="store_true")
    parser.add_argument("--events", default="", help="Optional full batch-event JSONL stream; requires --measurements")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.events and not args.measurements:
        parser.error("--events requires --measurements")
    result = run_evaluation(args.config, policy=args.policy, checkpoint=args.checkpoint,
        transitions=args.transitions, scheduler=args.scheduler, seed=args.seed, paced=args.paced,
        measurements=args.measurements, steps=args.steps, period=args.period, dt=args.dt,
        stop_after_episodes=args.episodes, event_path=args.events, phase1=args.phase1_metrics,
        fixed_reserve=(args.fixed_e,args.fixed_u),per_ue_cap=args.per_ue_cap, diagnostics_path=args.diagnostics)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"samples", "batch_measurements", "phase1", "diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
