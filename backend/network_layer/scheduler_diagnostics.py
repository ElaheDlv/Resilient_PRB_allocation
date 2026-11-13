"""Optional read-only, event-complete diagnostics for paired scheduler studies."""

from collections import defaultdict
import gzip
import json
import math


def outcome_partition(batches, now):
    counts = dict(
        offered=len(batches),
        completed=0,
        late_completed=0,
        overflow=0,
        on_time=0,
        late_only=0,
        unfinished_overdue=0,
        unresolved_not_due=0,
        unfinished=0,
        partially_admitted=0,
        fully_dropped=0,
    )
    for b in batches:
        complete = b.get("completion_s") is not None
        late = complete and b["completion_s"] - b["time_s"] > b["budget_s"] + 1e-12
        counts["completed"] += int(complete)
        counts["late_completed"] += int(late)
        counts["unfinished"] += int(not complete and b["accepted_bytes"] > 0)
        overflow = b["overflow_bytes"] > 0
        counts["partially_admitted"] += int(overflow and b["accepted_bytes"] > 0)
        counts["fully_dropped"] += int(overflow and b["accepted_bytes"] == 0)
        if overflow:
            counts["overflow"] += 1
        elif complete:
            counts["late_only" if late else "on_time"] += 1
        elif now - b["time_s"] > b["budget_s"] + 1e-12:
            counts["unfinished_overdue"] += 1
        else:
            counts["unresolved_not_due"] += 1
    assert counts["offered"] == sum(
        counts[k]
        for k in [
            "overflow",
            "on_time",
            "late_only",
            "unfinished_overdue",
            "unresolved_not_due",
        ]
    )
    counts["known_failed"] = (
        counts["overflow"] + counts["late_only"] + counts["unfinished_overdue"]
    )
    n = counts["offered"]
    counts["failure_lower"] = counts["known_failed"] / n if n else None
    counts["failure_upper"] = (
        (counts["known_failed"] + counts["unresolved_not_due"]) / n if n else None
    )
    return counts


class SchedulerDiagnostics:
    def __init__(self, path):
        self.file = gzip.open(path, "wt", encoding="utf-8")
        self.pending_events = []
        self.inputs = {}
        self.batches = {}
        self.stats = defaultdict(
            lambda: dict(
                ticks=0,
                prbs=0,
                served_bytes=0,
                rate_sum=0.0,
                cqi_sum=0.0,
                mcs_sum=0.0,
                violating_ticks=0,
                violation_runs=0,
                longest_violation_s=0.0,
                starvation_max_s=0.0,
                no_service_ticks=0,
                infeasible_grant_prbs=0,
                expired_grant_prbs=0,
                eligible_unused_ticks=0,
                over_demand_prbs=0,
            )
        )
        self.runs = defaultdict(lambda: [0.0, 0.0])
        self.examples = {}
        self.classes = {}
        self.seen_jam = False
        self.now = 0.0
        self.dt = None

    def event(self, event):
        if event["kind"] in {"arrival", "service", "batch_completion"}:
            self.pending_events.append(event)

    def allocation(self, cell, ids, grants, quota, dt, requirements):
        bs = cell.base_station
        self.inputs = {}
        for uid in ids:
            ue = cell.connected_ue_list[uid]
            queue = bs._dl_buf_queue.get(uid, ())
            head = queue[0] if queue else None
            rate = requirements.get(uid, {}).get("dl_throughput_per_prb", 0) * dt / 8
            cap = (
                min(quota, cell.prb_per_ue_cap)
                if cell.prb_per_ue_cap is not None
                else quota
            )
            wait = max(0, bs._time_s - head[0]) if head else 0
            slots = math.floor((ue.queue_budget_s - wait + 1e-12) / dt) + 1
            need = math.ceil(head[1] / rate) if head and rate > 0 else None
            useful = math.ceil(bs.get_dl_buf_bytes(uid) / rate) if rate > 0 else 0
            self.inputs[uid] = dict(
                uid=uid,
                budget_class=ue.queue_budget_class,
                budget_s=ue.queue_budget_s,
                queued_bytes=bs.get_dl_buf_bytes(uid),
                head_arrival_s=head[0] if head else None,
                head_bytes=head[1] if head else 0,
                head_wait_s=wait,
                bytes_per_prb=rate,
                head_needed_prbs=need,
                slots_including_now=slots,
                attainable_cap=cap,
                optimistic_feasible=bool(
                    head and need is not None and slots > 0 and need <= slots * cap
                ),
                expired=bool(head and wait > ue.queue_budget_s + 1e-12),
                useful_prbs=useful,
                grant=grants[uid],
                quota=quota,
                cqi=ue.downlink_cqi,
                mcs=ue.downlink_mcs_index,
            )

    def observe(self, engine, dt):
        if self.dt is not None and abs(self.dt - dt) > 1e-15:
            raise ValueError("Diagnostic step size changed within an episode")
        self.dt = dt
        self.now = engine.sim_step * dt
        cells = list(engine.cell_list.values())
        if len(cells) != 1:
            raise ValueError("Diagnostic pilot currently requires exactly one cell")
        cell = cells[0]
        import settings

        jammed = (
            cell._effective_max_dl_prb < cell.max_dl_prb - settings.JAM_RESERVED_PRBS
        )
        phase = "during" if jammed else "after" if self.seen_jam else "before"
        self.seen_jam |= jammed
        rows = []
        for uid, row in self.inputs.items():
            self.classes[uid] = row["budget_class"]
            ue = cell.connected_ue_list[uid]
            row = dict(
                row,
                served_bytes=round(ue.served_downlink_bitrate * dt / 8),
                hol_after_s=cell.base_station.get_dl_head_wait_s(uid),
                queue_after_bytes=cell.base_station.get_dl_buf_bytes(uid),
            )
            late = (
                row["queue_after_bytes"] > 0
                and row["hol_after_s"] > row["budget_s"] + 1e-12
            )
            starved = (
                row["queued_bytes"] > 0
                and row["bytes_per_prb"] > 0
                and row["served_bytes"] == 0
            )
            for segment in ["all", phase]:
                key = (uid, segment)
                s = self.stats[key]
                runs = self.runs[key]
                s["ticks"] += 1
                s["prbs"] += row["grant"]
                s["served_bytes"] += row["served_bytes"]
                s["rate_sum"] += row["bytes_per_prb"]
                s["cqi_sum"] += row["cqi"]
                s["mcs_sum"] += row["mcs"]
                s["violating_ticks"] += int(late)
                s["violation_runs"] += int(late and runs[0] == 0)
                runs[0] = runs[0] + dt if late else 0
                runs[1] = runs[1] + dt if starved else 0
                s["longest_violation_s"] = max(s["longest_violation_s"], runs[0])
                s["starvation_max_s"] = max(s["starvation_max_s"], runs[1])
                s["no_service_ticks"] += int(starved)
                s["infeasible_grant_prbs"] += row["grant"] * int(
                    not row["optimistic_feasible"] and row["queued_bytes"] > 0
                )
                s["expired_grant_prbs"] += row["grant"] * int(row["expired"])
                s["over_demand_prbs"] += max(0, row["grant"] - row["useful_prbs"])
                unused = row["quota"] - sum(r["grant"] for r in self.inputs.values())
                capacity = row["attainable_cap"]
                extra_useful = (
                    row["useful_prbs"] > row["grant"]
                    and capacity > row["grant"]
                    and math.floor((row["grant"] + unused) * row["bytes_per_prb"])
                    > math.floor(row["grant"] * row["bytes_per_prb"])
                )
                s["eligible_unused_ticks"] += int(unused > 0 and extra_useful)
            rows.append(row)
        # Segment runs end at transitions; do not concatenate disjoint jam windows.
        for (uid, segment), runs in self.runs.items():
            if segment not in {"all", phase}:
                runs[:] = [0.0, 0.0]
        events = [e for e in self.pending_events if e["imsi"] in self.inputs]
        for e in events:
            bid = e.get("batch_id")
            if e["kind"] == "arrival":
                self.batches[bid] = dict(e, phase=phase)
            elif e["kind"] == "batch_completion":
                self.batches[bid]["completion_s"] = e["time_s"]
        tick = dict(
            time_s=self.now, jammed=jammed, phase=phase, ues=rows, events=events
        )
        self.file.write(json.dumps(tick, separators=(",", ":"), allow_nan=False) + "\n")
        for mechanism, predicate in {
            "infeasible_receives_service": lambda r: r["grant"] > 0
            and not r["optimistic_feasible"],
            "expired_receives_service": lambda r: r["grant"] > 0 and r["expired"],
            "relaxed_no_service": lambda r: r["budget_class"] == "relaxed"
            and r["queued_bytes"] > 0
            and r["served_bytes"] == 0,
            "late_completion": lambda r: any(
                e["kind"] == "batch_completion"
                and e["over_budget"]
                and e["imsi"] == r["uid"]
                for e in events
            ),
        }.items():
            if mechanism not in self.examples and any(predicate(r) for r in rows):
                self.examples[mechanism] = tick
        self.pending_events.clear()

    def finish(self, reports):
        self.file.close()
        result = {}
        groups = defaultdict(list)
        for uid, label in self.classes.items():
            rows = [b for b in self.batches.values() if b["imsi"] == uid]
            segments = {}
            for segment in ["all", "before", "during", "after"]:
                batches = (
                    rows
                    if segment == "all"
                    else [b for b in rows if b["phase"] == segment]
                )
                s = dict(self.stats[(uid, segment)])
                n = s["ticks"]
                s.update(
                    outcomes=outcome_partition(batches, self.now),
                    hol_frequency=s["violating_ticks"] / n if n else None,
                    mean_violation_s=(
                        s["violating_ticks"] * self.dt / s["violation_runs"]
                        if s["violation_runs"]
                        else 0
                    ),
                    mean_bytes_per_prb=s["rate_sum"] / n if n else None,
                    mean_cqi=s["cqi_sum"] / n if n else None,
                    mean_mcs=s["mcs_sum"] / n if n else None,
                )
                segments[segment] = s
            result[uid] = dict(budget_class=label, segments=segments)
            groups[label].extend(rows)
            r = next(r for r in reports if r["imsi"] == uid)
            a = segments["all"]["outcomes"]
            assert a["offered"] == r["offered_batches"]
            assert a["overflow"] == r["overflowed_batches"]
            assert a["late_only"] == r["fully_admitted_late_batches"]
            assert segments["all"]["served_bytes"] == r["served_bytes"]
        class_stats = {}
        for label in groups:
            class_stats[label] = {}
            for segment in ["all", "before", "during", "after"]:
                members = [
                    v["segments"][segment]
                    for v in result.values()
                    if v["budget_class"] == label
                ]
                n = sum(v["ticks"] for v in members)
                class_stats[label][segment] = dict(
                    prbs=sum(v["prbs"] for v in members),
                    served_bytes=sum(v["served_bytes"] for v in members),
                    hol_frequency=(
                        sum(v["violating_ticks"] for v in members) / n if n else None
                    ),
                    starvation_max_s=max(
                        (v["starvation_max_s"] for v in members), default=0
                    ),
                    longest_violation_s=max(
                        (v["longest_violation_s"] for v in members), default=0
                    ),
                    mean_violation_s=(
                        sum(v["violating_ticks"] for v in members)
                        * self.dt
                        / sum(v["violation_runs"] for v in members)
                        if sum(v["violation_runs"] for v in members)
                        else 0
                    ),
                    mean_bytes_per_prb=(
                        sum(v["rate_sum"] for v in members) / n if n else None
                    ),
                    mean_cqi=sum(v["cqi_sum"] for v in members) / n if n else None,
                    mean_mcs=sum(v["mcs_sum"] for v in members) / n if n else None,
                )
        return dict(
            per_ue=result,
            class_service=class_stats,
            per_class={
                label: {
                    segment: outcome_partition(
                        (
                            rows
                            if segment == "all"
                            else [b for b in rows if b["phase"] == segment]
                        ),
                        self.now,
                    )
                    for segment in ["all", "before", "during", "after"]
                }
                for label, rows in groups.items()
            },
            examples=self.examples,
        )
