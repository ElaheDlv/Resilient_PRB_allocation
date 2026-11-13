"""Optional synchronous RIC-to-gNB commands; no E2 transport or DQN changes."""

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral

import settings
from .prb_accounting import validate_quotas


class URLLCSchedulerMode(str, Enum):
    PROPORTIONAL = "PROPORTIONAL"
    FEASIBILITY_AWARE = "FEASIBILITY_AWARE"

    @property
    def allocator(self):
        return {
            self.PROPORTIONAL: "proportional_wc",
            self.FEASIBILITY_AWARE: "deadline_feasible",
        }[self]


@dataclass(frozen=True)
class JointRICAction:
    delta_embb_prb: int
    delta_urllc_prb: int
    urllc_scheduler_mode: URLLCSchedulerMode

    def __post_init__(self):
        for value in (self.delta_embb_prb, self.delta_urllc_prb):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("PRB deltas must be integers")
        object.__setattr__(self, "delta_embb_prb", int(self.delta_embb_prb))
        object.__setattr__(self, "delta_urllc_prb", int(self.delta_urllc_prb))
        object.__setattr__(
            self, "urllc_scheduler_mode", URLLCSchedulerMode(self.urllc_scheduler_mode)
        )


@dataclass(frozen=True)
class JointCommandReceipt:
    applied_after_step: int
    first_service_step: int
    quotas_before: tuple
    quotas_after: tuple
    mode_before: URLLCSchedulerMode | None
    mode_after: URLLCSchedulerMode
    activated_reserve_prbs: int


class JointRICBridge:
    """Validate everything before the single-threaded quota/mode commit.

    A command changes configuration for the next service tick. It never edits
    already-computed UE grants or any queue. Only this cell is controlled.
    """

    def __init__(self, engine, cell, period_ticks=10):
        if (
            not isinstance(period_ticks, int)
            or isinstance(period_ticks, bool)
            or period_ticks < 1
        ):
            raise ValueError("Decision period must be a positive integer")
        if len(engine.cell_list) != 1 or cell not in engine.cell_list.values():
            raise ValueError("Phase 2A supports exactly one controlled cell")
        self.engine, self.cell, self.period_ticks = engine, cell, period_ticks
        self.pool = int(settings.JAM_RESERVED_PRBS)
        self.physical_prbs = cell.max_dl_prb
        self.episode_token = None
        self.origin_step = None
        self.last_command_step = None
        self.check_physics()

    def check_physics(self):
        if (
            settings.JAM_RESERVED_PRBS != self.pool
            or settings.RES_DQN_TOTAL_RESILIENCY_PRB != self.pool
            or self.cell.max_dl_prb != self.physical_prbs
        ):
            raise RuntimeError(
                "Physical reserve/PRB configuration changed during joint control"
            )

    def begin_episode(
        self, initial_mode=URLLCSchedulerMode.PROPORTIONAL, *, episode_token=None
    ):
        self.check_physics()
        step = self.engine.sim_step
        if self.origin_step is None:
            if step != 0:
                raise RuntimeError(
                    "Joint control must be initialized before first service"
                )
        elif (
            not getattr(self.engine, "in_ric_callback", False)
            or step <= self.origin_step
            or (step - self.origin_step) % self.period_ticks
            or episode_token is None
            or episode_token == self.episode_token
        ):
            raise RuntimeError(
                "Episode policy reset requires a new episode at a RIC boundary"
            )
        mode = URLLCSchedulerMode(initial_mode)
        self._validate_mode(mode)
        validate_quotas(self.cell, self.cell.slice_dl_prb_quota, self.pool)
        self.episode_token = episode_token
        self.origin_step = self.engine.sim_step
        self.last_command_step = None
        # Initialization is explicit; it does not consume a resource action.
        self.cell._joint_urllc_mode = mode

    def _validate_mode(self, mode):
        if mode is URLLCSchedulerMode.FEASIBILITY_AWARE:
            for ue in self.cell.connected_ue_list.values():
                if ue.slice_type == settings.NETWORK_SLICE_URLLC_NAME:
                    budget = getattr(ue, "queue_budget_s", None)
                    if budget is None or not math.isfinite(budget) or budget <= 0:
                        raise ValueError(
                            "Feasibility scheduling requires positive URLLC UE budgets"
                        )

    def apply(self, action):
        if not isinstance(action, JointRICAction):
            raise TypeError("Expected JointRICAction")
        self.check_physics()
        step = self.engine.sim_step
        if (
            self.origin_step is None
            or step <= self.origin_step
            or (step - self.origin_step) % self.period_ticks
            or not getattr(self.engine, "in_ric_callback", False)
        ):
            raise RuntimeError(
                "Joint commands require a post-service RIC decision boundary"
            )
        if self.last_command_step == step:
            raise RuntimeError(
                "A valid joint command was already applied at this boundary"
            )
        self._validate_mode(action.urllc_scheduler_mode)
        before = dict(self.cell.slice_dl_prb_quota)
        quotas = dict(before)
        for name, delta in [
            (settings.NETWORK_SLICE_EMBB_NAME, action.delta_embb_prb),
            (settings.NETWORK_SLICE_URLLC_NAME, action.delta_urllc_prb),
        ]:
            quotas[name] = quotas.get(name, 0) + int(delta)
        occupancy = validate_quotas(self.cell, quotas, self.pool)
        old_mode = self.cell.urllc_scheduler_mode
        # No await, callback, or validation after the commit starts.
        self.cell.slice_dl_prb_quota, self.cell._joint_urllc_mode = (
            quotas,
            action.urllc_scheduler_mode,
        )
        self.last_command_step = step
        return JointCommandReceipt(
            step,
            step + 1,
            tuple(sorted(before.items())),
            tuple(sorted(quotas.items())),
            old_mode,
            action.urllc_scheduler_mode,
            occupancy,
        )

    def release(self):
        self.cell._joint_urllc_mode = None
