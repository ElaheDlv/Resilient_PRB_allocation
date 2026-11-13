"""Causal read-only UE snapshots with arrival-cohort deadline counters.

The observation window is (max(episode_start,t-W),t]. Retained records are
measurement events, not future traffic or final-episode outcome labels.
"""

from collections import defaultdict
from dataclasses import dataclass
import math

import settings
from utils import estimate_throughput
from .joint_control import URLLCSchedulerMode

EPS = 1e-12


@dataclass(frozen=True)
class BatchWindow:
    start_s: float
    end_s: float
    full_window: bool
    history_available: bool
    offered_batches: int | None
    known_failures: int | None
    matured_offered_batches: int | None
    matured_failed_batches: int | None
    pending_overdue_batches: int | None


@dataclass(frozen=True)
class UETelemetry:
    ue_id: str
    slice_id: str
    budget_s: float | None
    budget_class: str | None
    queued_bytes: int
    hol_wait_s: float
    capacity_bps_per_prb: float | None
    window: BatchWindow
    slice_quota_prbs: int
    slice_activated_reserve_prbs: int
    cell_activated_reserve_prbs: int
    urllc_scheduler_mode: URLLCSchedulerMode | str | None


@dataclass(frozen=True)
class TelemetryReport:
    time_s: float
    service_step: int
    cell_id: str
    window_s: float
    ues: tuple[UETelemetry, ...]


@dataclass
class _Arrival:
    arrival_s: float
    budget_s: float | None
    overflow: bool
    terminal: bool = False
    late: bool = False

    def overdue(self, now):
        return self.budget_s is not None and now - self.arrival_s > self.budget_s + EPS

    def failed(self, now):
        return self.overflow or self.late or (not self.terminal and self.overdue(now))


class UETelemetryCollector:
    def __init__(self, cell, window_s=0.100, *, episode_start_s=0.0):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("Observation window must be positive seconds")
        self.cell, self.window_s = cell, window_s
        self.measurements = getattr(cell.base_station, "queue_measurements", None)
        self.records = defaultdict(dict)
        self.origin_s = float(episode_start_s)
        self.attached_s = float(cell.base_station._time_s)
        self._event_time_s = self.attached_s
        self._collected_time_s = self.attached_s
        if (
            not math.isfinite(self.origin_s)
            or self.origin_s < 0
            or self.origin_s > self.attached_s + EPS
        ):
            raise ValueError("Invalid episode start timestamp")
        self._missing_history = set()
        if self.measurements is not None:
            self._missing_history = {
                uid
                for uid, total in self.measurements.totals.items()
                if total["offered_batches"] > 0
            }
        if self.measurements is not None:
            self.measurements.add_listener(self._event)

    def reset_episode(self):
        self.records.clear()
        self.origin_s = float(self.cell.base_station._time_s)
        self.attached_s = self.origin_s
        self._event_time_s = self.origin_s
        self._collected_time_s = self.origin_s
        self._missing_history = (
            {
                uid
                for uid, total in self.measurements.totals.items()
                if total["offered_batches"] > 0
            }
            if self.measurements is not None
            else set()
        )

    def _event(self, event):
        self._event_time_s = max(self._event_time_s, event["time_s"])
        uid = event["imsi"]
        if event["kind"] == "arrival":
            self.records[uid][event["batch_id"]] = _Arrival(
                event["time_s"],
                event["budget_s"],
                event["overflow_bytes"] > 0,
                terminal=event["accepted_bytes"] == 0,
            )
        elif event["kind"] == "batch_completion":
            batch = self.records.get(uid, {}).get(event["batch_id"])
            if batch is not None:
                batch.terminal = True
                batch.late = event["over_budget"]
        elif event["kind"] == "unfinished":
            self.records.pop(uid, None)
            self._missing_history.discard(uid)

    def _window(self, uid, now, budget):
        start = max(self.origin_s, now - self.window_s)
        qm = self.measurements
        # Existing arrivals before subscription cannot be reconstructed from a
        # 256-event ring. Wait for their entire arrival cohort to leave W.
        pending_batches = qm.pending.get(uid, ()) if qm is not None else ()
        consistent = qm is not None and sum(
            b.remaining_bytes for b in pending_batches
        ) == self.cell.base_station.get_dl_buf_bytes(uid)
        available = consistent and (
            uid not in self._missing_history or start + EPS >= self.attached_s
        )
        full = now - self.origin_s + EPS >= self.window_s
        if not consistent:
            pending = None
        elif budget is None:
            pending = None
        else:
            current_pending = qm.pending.get(uid, ())
            pending = (
                sum(
                    b.budget_s is not None and now - b.arrival_s > b.budget_s + EPS
                    for b in current_pending
                )
                if qm.budgets.get(uid) == budget
                and all(b.budget_s is not None for b in current_pending)
                else None
            )
        known = self.records.get(uid, {})
        # Only recent cohorts are needed; current pending state comes directly
        # from the measurement sidecar, including batches older than W.
        for bid in [bid for bid, b in known.items() if b.arrival_s <= start + EPS]:
            del known[bid]
        if not available:
            return BatchWindow(start, now, full, False, None, None, None, None, pending)
        batches = [b for b in known.values() if b.arrival_s <= now + EPS]
        if (
            budget is None
            or qm.budgets.get(uid) != budget
            or any(b.budget_s != budget for b in batches)
        ):
            return BatchWindow(
                start, now, full, True, len(batches), None, None, None, None
            )
        matured = [b for b in batches if b.overdue(now)]
        return BatchWindow(
            start,
            now,
            full,
            True,
            len(batches),
            sum(b.failed(now) for b in batches),
            len(matured),
            sum(b.failed(now) for b in matured),
            pending,
        )

    def collect(self, service_step):
        cell = self.cell
        bs = cell.base_station
        now = float(bs._time_s)
        if now + EPS < max(self._event_time_s, self._collected_time_s):
            raise RuntimeError(
                "Telemetry cannot query before already-observed events or snapshots"
            )
        self._collected_time_s = now
        base = getattr(cell, "base_dl_prb_quota", {})
        occupancy = sum(cell.slice_dl_prb_quota.values()) - sum(base.values())
        mode = cell.urllc_scheduler_mode
        if mode is None:
            configured = settings.PRB_UE_SCHEDULER
            mode = {
                "proportional_wc": URLLCSchedulerMode.PROPORTIONAL,
                "deadline_feasible": URLLCSchedulerMode.FEASIBILITY_AWARE,
            }.get(configured, configured)
        rows = []
        for uid, ue in sorted(cell.connected_ue_list.items()):
            budget = getattr(ue, "queue_budget_s", None)
            mcs = getattr(ue, "downlink_mcs_data", None)
            if (
                not mcs
                or mcs.get("modulation_order") is None
                or mcs.get("target_code_rate") is None
            ):
                capacity = 0.0 if getattr(ue, "downlink_cqi", None) == 0 else None
            else:
                capacity = float(
                    estimate_throughput(
                        mcs["modulation_order"], mcs["target_code_rate"], 1
                    )
                )
                if not math.isfinite(capacity) or capacity < 0:
                    capacity = None
            rows.append(
                UETelemetry(
                    uid,
                    ue.slice_type,
                    budget,
                    getattr(ue, "queue_budget_class", None),
                    bs.get_dl_buf_bytes(uid),
                    bs.get_dl_head_wait_s(uid),
                    capacity,
                    self._window(uid, now, budget),
                    cell.slice_dl_prb_quota.get(ue.slice_type, 0),
                    max(
                        0,
                        cell.slice_dl_prb_quota.get(ue.slice_type, 0)
                        - base.get(ue.slice_type, 0),
                    ),
                    occupancy,
                    mode,
                )
            )
        return TelemetryReport(
            now, int(service_step), cell.cell_id, self.window_s, tuple(rows)
        )

    def close(self):
        if self.measurements is not None:
            self.measurements.remove_listener(self._event)
