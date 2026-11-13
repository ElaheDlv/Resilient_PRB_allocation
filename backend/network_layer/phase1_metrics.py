"""After-service pilot measurements, independent of scheduling decisions."""

from collections import defaultdict
import hashlib
import json
import statistics
import settings


def batch_outcomes(report):
    """Mutually exclusive offered-batch outcomes, including overflow and censoring."""
    unresolved_full = [b for b in report["unresolved"] if b["overflow_bytes"] == 0]
    overdue = sum(b["over_budget"] for b in unresolved_full)
    failed = (
        report["overflowed_batches"] + report["fully_admitted_late_batches"] + overdue
    )
    unresolved = len(unresolved_full) - overdue
    offered = report["offered_batches"]
    on_time = report["fully_admitted_on_time_batches"]
    if report["budget_s"] is not None and offered != on_time + failed + unresolved:
        raise AssertionError("Offered-batch outcome partition is inconsistent")
    return dict(
        offered=offered,
        on_time=on_time,
        known_failed=failed,
        unresolved_not_due=unresolved,
        unfinished_overdue=overdue,
        failure_fraction_lower=failed / offered if offered else None,
        failure_fraction_upper=(failed + unresolved) / offered if offered else None,
    )


class RecoveryTracker:
    def __init__(self, hold_s=0.010):
        if hold_s <= 0:
            raise ValueError("Recovery hold time must be positive")
        self.hold_s = hold_s
        self.previous_jammed = None
        self.pending = None
        self.windows = []

    def observe(self, now, dt, jammed, compliant):
        if jammed and self.pending is not None:
            self.pending.update(censored=True, end_s=now - dt, reason="next_jam")
            self.pending = None
        if self.previous_jammed is True and not jammed:
            self.pending = dict(
                off_s=now,
                recovery_s=None,
                censored=True,
                end_s=None,
                compliant_run_s=0.0,
            )
            self.windows.append(self.pending)
        if self.pending is not None:
            p = self.pending
            p["compliant_run_s"] = p["compliant_run_s"] + dt if compliant else 0.0
            if p["compliant_run_s"] + 1e-12 >= self.hold_s:
                p.update(
                    recovery_s=max(0.0, now - p["compliant_run_s"] + dt - p["off_s"]),
                    censored=False,
                    end_s=now,
                    reason="recovered",
                )
                self.pending = None
        self.previous_jammed = jammed

    def finish(self, now):
        if self.pending is not None:
            self.pending.update(end_s=now, reason="horizon")
            self.pending = None
        return self.windows


class Phase1Metrics:
    def __init__(self, hold_s=0.010):
        self.recovery = RecoveryTracker(hold_s)
        self.exogenous_hash = hashlib.sha256()
        self.quota_hash = hashlib.sha256()
        self.previous_offered = {}
        self.resources = defaultdict(
            lambda: dict(
                quota_prb_s=0.0, allocated_prb_s=0.0, unused_prb_s=0.0, scheduler_ns=[]
            )
        )
        self.elapsed_s = 0.0
        self.embb_bytes = 0.0
        self.reserve_prb_s = 0.0
        self.max_reserve = 0
        self.assignments = {}

    def observe(self, engine, dt):
        self.elapsed_s += dt
        now = engine.sim_step * dt
        exogenous, control = [], []
        compliant = True
        jammed = False
        for cell in engine.cell_list.values():
            bs = cell.base_station
            measurements = bs.queue_measurements
            if measurements is None:
                raise ValueError("Phase 1 measurements require PHASE0_MEASUREMENTS")
            baseline = dict(cell.base_dl_prb_quota)
            occupancy = sum(cell.slice_dl_prb_quota.values()) - sum(baseline.values())
            self.reserve_prb_s += occupancy * dt
            self.max_reserve = max(self.max_reserve, occupancy)
            nominal = cell.max_dl_prb - int(settings.JAM_RESERVED_PRBS)
            jammed |= cell._effective_max_dl_prb < nominal
            users = []
            for uid, ue in sorted(cell.connected_ue_list.items()):
                offered = measurements.totals[uid]["offered_bytes"]
                delta = offered - self.previous_offered.get(uid, 0)
                self.previous_offered[uid] = offered
                users.append(
                    dict(
                        uid=uid,
                        position=[ue.position_x, ue.position_y],
                        cqi=ue.downlink_cqi,
                        mcs=ue.downlink_mcs_index,
                        offered_bytes=delta,
                    )
                )
                if ue.slice_type == "eMBB":
                    self.embb_bytes += ue.served_downlink_bitrate * dt / 8
                else:
                    if ue.queue_budget_s is None:
                        raise ValueError(
                            "Phase 1 metrics require a budget for each URLLC UE"
                        )
                    self.assignments[uid] = dict(
                        budget_s=ue.queue_budget_s,
                        budget_class=ue.queue_budget_class,
                        position=[ue.position_x, ue.position_y],
                        cqi=ue.downlink_cqi,
                    )
                    wait = bs.get_dl_head_wait_s(uid)
                    compliant &= (
                        bs.get_dl_buf_bytes(uid) == 0
                        or wait <= ue.queue_budget_s + 1e-12
                    )
            for slice_name, stats in cell.last_scheduler_stats.items():
                total = self.resources[slice_name]
                total["quota_prb_s"] += stats["quota"] * dt
                total["allocated_prb_s"] += stats["allocated"] * dt
                total["unused_prb_s"] += stats["unused"] * dt
                total["scheduler_ns"].append(stats["elapsed_ns"])
            exogenous.append(
                dict(
                    cell=cell.cell_id,
                    effective=cell._effective_max_dl_prb,
                    baseline=baseline,
                    users=users,
                )
            )
            control.append(
                dict(cell=cell.cell_id, quotas=dict(cell.slice_dl_prb_quota))
            )
        self.exogenous_hash.update(json.dumps(exogenous, sort_keys=True).encode())
        self.quota_hash.update(json.dumps(control, sort_keys=True).encode())
        self.recovery.observe(now, dt, jammed, compliant)

    def finish(self, reports):
        per_ue = {}
        groups = defaultdict(list)
        for r in reports:
            if r["budget_s"] is None:
                continue
            outcomes = batch_outcomes(r)
            served = r["budgeted_served_bytes"]
            completed = r["budgeted_completed_batches"]
            row = dict(
                budget_s=r["budget_s"],
                budget_class=r["budget_class"],
                batch_outcomes=outcomes,
                completed_batches=completed,
                completed_batch_late_fraction=(
                    r["completed_over_budget_batches"] / completed
                    if completed
                    else None
                ),
                completed_over_budget_batches=r["completed_over_budget_batches"],
                served_bytes=served,
                served_over_budget_bytes=r["served_over_budget_bytes"],
                served_byte_late_fraction=(
                    r["served_over_budget_bytes"] / served if served else None
                ),
                overflow_bytes=r["overflow_bytes"],
                unfinished_bytes=r["unfinished_bytes"],
                hol=r["hol_statistics"],
            )
            per_ue[r["imsi"]] = row
            groups[r["budget_class"] or "unclassified"].append(row)
        per_class = {name: self.aggregate(rows) for name, rows in groups.items()}
        overall = self.aggregate(list(per_ue.values()))
        resources = {}
        for name, total in self.resources.items():
            times = total["scheduler_ns"]
            resources[name] = {k: v for k, v in total.items() if k != "scheduler_ns"}
            for key in ["quota", "allocated", "unused"]:
                resources[name][f"mean_{key}_prbs"] = (
                    total[f"{key}_prb_s"] / self.elapsed_s
                )
            resources[name]["scheduler_mean_us"] = (
                statistics.mean(times) / 1000 if times else None
            )
            resources[name]["scheduler_median_us"] = (
                statistics.median(times) / 1000 if times else None
            )
            resources[name]["scheduler_p99_us"] = (
                sorted(times)[max(0, int(0.99 * len(times)) - 1)] / 1000
                if times
                else None
            )
            resources[name]["scheduler_p95_us"] = (
                sorted(times)[max(0, int(0.95 * len(times)) - 1)] / 1000
                if times
                else None
            )
        worst = (
            max(
                per_ue,
                key=lambda uid: per_ue[uid]["batch_outcomes"]["failure_fraction_lower"]
                or 0,
            )
            if per_ue
            else None
        )
        return dict(
            per_ue=per_ue,
            per_class=per_class,
            overall=overall,
            worst_ue=worst,
            worst_ue_failure_fraction=(
                per_ue[worst]["batch_outcomes"]["failure_fraction_lower"]
                if worst
                else None
            ),
            recovery=self.recovery.finish(self.elapsed_s),
            embb_throughput_mbps=self.embb_bytes * 8 / self.elapsed_s / 1e6,
            mean_reserve_prbs=self.reserve_prb_s / self.elapsed_s,
            max_reserve_prbs=self.max_reserve,
            resources=resources,
            assignments=self.assignments,
            starvation_limit_ticks=settings.DEADLINE_STARVATION_TICKS,
            recovery_hold_s=self.recovery.hold_s,
            exogenous_sha256=self.exogenous_hash.hexdigest(),
            quota_sha256=self.quota_hash.hexdigest(),
        )

    @staticmethod
    def aggregate(rows):
        offered = sum(r["batch_outcomes"]["offered"] for r in rows)
        failed = sum(r["batch_outcomes"]["known_failed"] for r in rows)
        unresolved = sum(r["batch_outcomes"]["unresolved_not_due"] for r in rows)
        completed = sum(r["completed_batches"] for r in rows)
        late = sum(r["completed_over_budget_batches"] for r in rows)
        served = sum(r["served_bytes"] for r in rows)
        late_bytes = sum(r["served_over_budget_bytes"] for r in rows)
        time = sum(r["hol"]["observed_s"] for r in rows)
        violating = sum(r["hol"]["violating_s"] for r in rows)
        runs = sum(r["hol"]["runs"] for r in rows)
        return dict(
            offered_batches=offered,
            known_failed_batches=failed,
            completed_batches=completed,
            completed_over_budget_batches=late,
            served_bytes=served,
            served_over_budget_bytes=late_bytes,
            unresolved_not_due_batches=unresolved,
            failure_fraction_lower=failed / offered if offered else None,
            failure_fraction_upper=(failed + unresolved) / offered if offered else None,
            completed_batch_late_fraction=late / completed if completed else None,
            served_byte_late_fraction=late_bytes / served if served else None,
            hol_violation_fraction=violating / time if time else None,
            mean_observed_violation_s=violating / runs if runs else 0.0,
            longest_observed_violation_s=max(
                (r["hol"]["longest_run_s"] for r in rows), default=0
            ),
            unfinished_bytes=sum(r["unfinished_bytes"] for r in rows),
            overflow_bytes=sum(r["overflow_bytes"] for r in rows),
        )
