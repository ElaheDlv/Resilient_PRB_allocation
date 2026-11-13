"""Short reproducible joint-control demonstration; no training or E2 transport."""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(
            ROOT / "backend/evaluations/phase1/main_scenarios/scenario_j20.json"
        ),
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=3456)
    p.add_argument("--window", type=float, default=0.100)
    p.add_argument("--minimum-matured", type=int, default=20)
    p.add_argument(
        "--base-scheduler",
        choices=["legacy", "proportional_wc"],
        default="proportional_wc",
    )
    p.add_argument("--fixed-mode", choices=["PROPORTIONAL", "FEASIBILITY_AWARE"])
    p.add_argument("--output", required=True)
    a = p.parse_args()
    os.environ.update(
        RAN_TOPOLOGY_PRESET="simple",
        SIM_FREEZE_MOBILITY="1",
        SIM_STEP_TIME_DEFAULT="0.001",
        UE_SPAWN_RADIUS_M="400",
        UE_DEFAULT_MAX_COUNT="100",
        PRB_GYM_CONFIG_PATH=a.config,
        RES_DQN_TRANSITION_MODE="causal",
        PRB_UE_SCHEDULER=a.base_scheduler,
        PHASE0_MEASUREMENTS="1",
        PHASE0_MEASUREMENT_PATH="",
        JAM_RESERVED_PRBS="18",
        RES_DQN_TOTAL_RESILIENCY_PRB="18",
        DEADLINE_STARVATION_TICKS="3",
    )
    import numpy as np
    import torch

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    from network_layer.simulation_engine import SimulationEngine
    from network_layer.xApps.xapp_scripted_joint_ric import xAppScriptedJointRIC

    engine = SimulationEngine()
    engine.reset_network()
    extra = {}
    if a.fixed_mode:
        from network_layer.joint_control import JointRICAction, URLLCSchedulerMode

        mode = URLLCSchedulerMode(a.fixed_mode)
        extra = dict(
            initial_mode=mode,
            script=lambda report, features, index: JointRICAction(0, 0, mode),
        )
    engine.network_setup(
        xapp_classes=[
            partial(
                xAppScriptedJointRIC,
                enabled=True,
                window_s=a.window,
                minimum_matured=a.minimum_matured,
                spawn_seed=a.seed,
                **extra,
            )
        ]
    )
    app = next(iter(engine.ric.xapp_list.values()))
    physical_digest = hashlib.sha256()
    started = time.perf_counter()
    for _ in range(a.steps):
        engine.sim_step += 1
        engine.step(0.001)
        cells = []
        for cell in engine.cell_list.values():
            bs = cell.base_station
            cells.append(
                dict(
                    quotas=dict(cell.slice_dl_prb_quota),
                    baseline=dict(cell.base_dl_prb_quota),
                    effective=cell._effective_max_dl_prb,
                    ues=[
                        dict(
                            imsi=u.ue_imsi,
                            queued=bs.get_dl_buf_bytes(u.ue_imsi),
                            served_bps=u.served_downlink_bitrate,
                            latency_s=u.downlink_latency,
                            prbs=cell.prb_ue_allocation_dict[u.ue_imsi]["downlink"],
                        )
                        for u in cell.connected_ue_list.values()
                    ],
                )
            )
        physical_digest.update(
            json.dumps(
                dict(step=engine.sim_step, cells=cells), sort_keys=True, allow_nan=False
            ).encode()
        )
        if app.finished:
            break
    elapsed = time.perf_counter() - started
    result = app.result()
    times = result.pop("boundary_computation_ns")
    result.update(
        seed=a.seed,
        physical_trajectory_sha256=physical_digest.hexdigest(),
        fixed_mode=a.fixed_mode,
        base_scheduler=a.base_scheduler,
        steps=engine.sim_step,
        simulation_seconds=engine.sim_step * 0.001,
        elapsed_seconds=elapsed,
        service_tick_s=0.001,
        ric_period_s=0.010,
        window_s=a.window,
        minimum_matured=a.minimum_matured,
        physical_reserve_prbs=18,
        boundary_median_us=statistics.median(times) / 1000 if times else None,
        boundary_p95_us=(
            sorted(times)[max(0, int(0.95 * len(times)) - 1)] / 1000 if times else None
        ),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        torch_version=torch.__version__,
    )
    catalog = json.loads(Path(a.config).read_text())
    entries = catalog["episodes"] if isinstance(catalog, dict) else catalog
    trace_paths = {
        cfg["trace"]
        for entry in entries
        for cfg in entry["slices"].values()
        if cfg.get("trace")
    }
    result["config_sha256"] = hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    result["trace_sha256"] = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in sorted(trace_paths)
    }
    app.close()
    for bs in engine.base_station_list.values():
        bs.queue_measurements.close()
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"history", "intervals", "service_history"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
