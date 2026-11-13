"""Optional single-cell scripted control harness. No neural network is created."""

from dataclasses import asdict
import hashlib
import json
import time

import settings
from .xapp_base import xAppBase
from .xapp_resilient_dqn_allocator import ResilientDQNEnv
from network_layer.joint_control import (
    JointRICAction,
    JointRICBridge,
    URLLCSchedulerMode,
)
from network_layer.ue_telemetry import UETelemetryCollector
from network_layer.deadline_risk import encode_deadline_risk


def default_script(report, features, command_index):
    # Signed changes alternate independently of policy. Net change per cycle=0.
    return [
        JointRICAction(0, 3, URLLCSchedulerMode.FEASIBILITY_AWARE),
        JointRICAction(1, 0, URLLCSchedulerMode.PROPORTIONAL),
        JointRICAction(0, -3, URLLCSchedulerMode.FEASIBILITY_AWARE),
        JointRICAction(-1, 0, URLLCSchedulerMode.PROPORTIONAL),
    ][command_index % 4]


class xAppScriptedJointRIC(xAppBase):
    def __init__(
        self,
        ric=None,
        *,
        enabled=None,
        period_ticks=10,
        window_s=0.100,
        minimum_matured=20,
        script=default_script,
        initial_mode=URLLCSchedulerMode.PROPORTIONAL,
        spawn_seed=3456
    ):
        super().__init__(ric)
        self.enabled = bool(settings.JOINT_RIC_ENABLE if enabled is None else enabled)
        self.period_ticks = period_ticks
        self.window_s = window_s
        self.minimum_matured = minimum_matured
        self.script = script
        self.initial_mode = URLLCSchedulerMode(initial_mode)
        self.spawn_seed = spawn_seed
        self.history = []
        self.intervals = []
        self.service_history = []
        self.timings_ns = []
        self._episode_counter = 0
        self.finished = False
        self._digest = hashlib.sha256()
        self._pending = None

    def start(self):
        if not self.enabled:
            return
        if len(self.ric.xapp_list) != 1:
            raise ValueError(
                "Scripted joint control requires explicit sole-xApp deployment"
            )
        if settings.RES_DQN_TRANSITION_MODE != "causal":
            raise ValueError("Joint control requires causal execution mode")
        self.service_dt_s = float(settings.SIM_STEP_TIME_DEFAULT)
        if self.service_dt_s != 0.001:
            raise ValueError(
                "Phase 2A scripted evaluation requires a 1-ms service tick"
            )
        engine = self.ric.simulation_engine
        if len(engine.base_station_list) != 1 or len(engine.cell_list) != 1:
            raise ValueError("Phase 2A requires one gNB and one cell")
        self.engine = engine
        self.cell = next(iter(engine.cell_list.values()))
        self.env = ResilientDQNEnv(self.ric, spawn_seed=self.spawn_seed)
        self.env.reset()
        if self.env.exhausted or not self.env.ready():
            raise ValueError(
                "Joint control requires a configured, populated episode catalog"
            )
        self.collector = UETelemetryCollector(self.cell, self.window_s)
        self.bridge = JointRICBridge(engine, self.cell, self.period_ticks)
        self._before = engine.before_service_observer
        engine.before_service_observer = self.before_service
        self._begin_episode()

    def _served_total(self):
        measurements = self.cell.base_station.queue_measurements
        return (
            sum(t["served_bytes"] for t in measurements.totals.values())
            if measurements is not None
            else None
        )

    def _begin_episode(self):
        self.collector.reset_episode()
        self.bridge.begin_episode(
            self.initial_mode, episode_token=self.env._episode_idx
        )
        self.command_index = 0
        self._interval_start_served = self._served_total()
        self._pending = dict(
            start_step=self.engine.sim_step,
            ticks=0,
            served_bytes=0,
            configuration=dict(
                quotas=dict(self.cell.slice_dl_prb_quota), mode=self.initial_mode.value
            ),
        )
        self.history.append(
            dict(
                kind="episode_initialized",
                episode_id=self.env._current_episode["id"],
                after_step=self.engine.sim_step,
                first_service_step=self.engine.sim_step + 1,
                mode=self.initial_mode.value,
            )
        )

    def before_service(self, engine, dt):
        if self._before is not None:
            self._before(engine, dt)
        self.bridge.check_physics()
        if (
            abs(dt - self.service_dt_s) > 1e-12
            or float(settings.SIM_STEP_TIME_DEFAULT) != self.service_dt_s
        ):
            raise RuntimeError("Joint control service step changed")

    def step(self):
        if not self.enabled or self.finished:
            return
        step = self.engine.sim_step
        # Capture completed service before collecting telemetry or applying commands.
        service = dict(
            step=step,
            quotas=dict(self.cell.slice_dl_prb_quota),
            mode=self.cell.last_scheduler_stats[settings.NETWORK_SLICE_URLLC_NAME][
                "mode"
            ],
            embb_mode=(
                "legacy" if settings.PRB_UE_SCHEDULER == "legacy" else "proportional_wc"
            ),
            effective_prbs=self.cell._effective_max_dl_prb,
            activated_reserve_prbs=sum(self.cell.slice_dl_prb_quota.values())
            - sum(self.cell.base_dl_prb_quota.values()),
            allocated_prbs={
                sl: sum(
                    self.cell.prb_ue_allocation_dict[uid]["downlink"]
                    for uid, u in self.cell.connected_ue_list.items()
                    if u.slice_type == sl
                )
                for sl in self.cell.slice_dl_prb_quota
            },
        )
        self.service_history.append(service)
        physical = dict(
            service=service,
            ues=[
                dict(
                    uid=u.ue_imsi,
                    queue=self.cell.base_station.get_dl_buf_bytes(u.ue_imsi),
                    served=u.served_downlink_bitrate,
                    prbs=self.cell.prb_ue_allocation_dict[u.ue_imsi]["downlink"],
                )
                for u in self.cell.connected_ue_list.values()
            ],
        )
        self._digest.update(json.dumps(physical, sort_keys=True).encode())
        self._pending["ticks"] += 1
        served = self._served_total()
        self._pending["served_bytes"] = (
            served - self._interval_start_served
            if served is not None and self._interval_start_served is not None
            else None
        )
        if self._pending["ticks"] < self.period_ticks:
            return
        if not self.engine.in_ric_callback:
            raise RuntimeError("Joint xApp must run in post-service RIC callback")
        started = time.perf_counter_ns()
        report = self.collector.collect(step)
        features = encode_deadline_risk(report, self.minimum_matured)
        self.intervals.append(
            dict(
                **self._pending,
                end_step=step,
                end_report=asdict(report),
                risk_features=asdict(features)
            )
        )
        self.env._steps_in_episode += 1
        done = (
            self.env._steps_in_episode >= self.env._episode_step_limit
            or self.env._traces_consumed()
        )
        if done:
            self._episode_counter += 1
            self.env.reset()
            if self.env.exhausted:
                self.finished = True
                self.bridge.release()
                self._pending = None
                self.timings_ns.append(time.perf_counter_ns() - started)
                return
            self._begin_episode()
        else:
            action = self.script(report, features, self.command_index)
            receipt = self.bridge.apply(action)
            self.history.append(
                dict(
                    kind="joint_command",
                    episode_id=self.env._current_episode["id"],
                    telemetry=asdict(report),
                    risk_features=asdict(features),
                    action=asdict(action),
                    receipt=asdict(receipt),
                )
            )
            self.command_index += 1
            self._interval_start_served = self._served_total()
            self._pending = dict(
                start_step=step,
                ticks=0,
                served_bytes=0,
                configuration=dict(
                    quotas=dict(self.cell.slice_dl_prb_quota),
                    mode=self.cell.urllc_scheduler_mode.value,
                ),
            )
        self.timings_ns.append(time.perf_counter_ns() - started)

    def result(self):
        return dict(
            history=self.history,
            intervals=self.intervals,
            service_history=self.service_history,
            trajectory_sha256=self._digest.hexdigest(),
            complete_episodes=self._episode_counter,
            incomplete_interval_ticks=self._pending["ticks"] if self._pending else 0,
            pending_interval=self._pending,
            boundary_computation_ns=self.timings_ns,
        )

    def close(self):
        if not self.enabled:
            return
        self.collector.close()
        self.bridge.release()
        self.engine.before_service_observer = self._before
        self.enabled = False
