"""Measurement-only sidecar for FIFO byte batches, never packet reliability.

    Budget endpoints are BS enqueue and simulator service timestamps. Completion
    does not include air-interface serialization or other end-to-end delays.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math


@dataclass
class Batch:
    batch_id: int
    arrival_s: float
    accepted_bytes: int
    remaining_bytes: int
    budget_s: float | None
    overflow_bytes: int


class QueueMeasurements:
    def __init__(self, event_path=""):
        self.pending = defaultdict(deque)
        self.budgets = {}
        self.totals = defaultdict(lambda: dict(offered_bytes=0, accepted_bytes=0,
            overflow_bytes=0, served_bytes=0, served_over_budget_bytes=0,
            budgeted_served_bytes=0, completed_batches=0, budgeted_completed_batches=0,
            completed_over_budget_batches=0, overflowed_batches=0,
            partially_admitted_completed_batches=0))
        self.events = deque(maxlen=256)
        self.closed = []
        self._next_id = 0
        self._file = open(event_path, "a", encoding="utf-8") if event_path else None

    def set_budget(self, imsi, budget_s):
        if budget_s is not None and (not math.isfinite(budget_s) or budget_s <= 0):
            raise ValueError("queue_budget_s must be positive finite seconds or null")
        self.budgets[imsi] = budget_s

    def emit(self, kind, imsi, now, **values):
        event = dict(kind=kind, imsi=imsi, time_s=now, **values)
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event, allow_nan=False) + "\n")

    def arrival(self, imsi, now, offered, accepted):
        total = self.totals[imsi]
        total["offered_bytes"] += offered
        total["accepted_bytes"] += accepted
        total["overflow_bytes"] += offered - accepted
        total["overflowed_batches"] += int(offered > accepted)
        batch_id = self._next_id
        self._next_id += 1
        if accepted:
            self.pending[imsi].append(Batch(batch_id, now, accepted, accepted,
                self.budgets.get(imsi), offered - accepted))
        self.emit("arrival", imsi, now, batch_id=batch_id, offered_bytes=offered,
                  accepted_bytes=accepted, overflow_bytes=offered-accepted,
                  budget_s=self.budgets.get(imsi))

    def service(self, imsi, now, count):
        remaining = count
        while remaining:
            if not self.pending[imsi]:
                raise AssertionError("Measurement FIFO disagrees with service FIFO")
            batch = self.pending[imsi][0]
            used = min(remaining, batch.remaining_bytes)
            age = max(0.0, now - batch.arrival_s)
            late = batch.budget_s is not None and age > batch.budget_s + 1e-12
            self.totals[imsi]["served_bytes"] += used
            self.totals[imsi]["budgeted_served_bytes"] += used * int(batch.budget_s is not None)
            self.totals[imsi]["served_over_budget_bytes"] += used * int(late)
            batch.remaining_bytes -= used
            remaining -= used
            self.emit("service", imsi, now, batch_id=batch.batch_id, bytes=used,
                      residence_s=age, over_budget=late)
            if batch.remaining_bytes == 0:
                self.pending[imsi].popleft()
                self.totals[imsi]["completed_batches"] += 1
                self.totals[imsi]["budgeted_completed_batches"] += int(batch.budget_s is not None)
                self.totals[imsi]["partially_admitted_completed_batches"] += int(batch.overflow_bytes > 0)
                self.totals[imsi]["completed_over_budget_batches"] += int(late)
                self.emit("batch_completion", imsi, now, batch_id=batch.batch_id,
                          residence_s=age, over_budget=late,
                          accepted_bytes=batch.accepted_bytes, overflow_bytes=batch.overflow_bytes)

    def snapshot(self, imsi, now):
        pending = [dict(batch_id=b.batch_id, arrival_s=b.arrival_s,
                        remaining_bytes=b.remaining_bytes, budget_s=b.budget_s,
                        over_budget=b.budget_s is not None and now-b.arrival_s > b.budget_s+1e-12)
                   for b in self.pending[imsi]]
        result = dict(self.totals[imsi], budget_s=self.budgets.get(imsi),
                      unfinished_bytes=sum(b["remaining_bytes"] for b in pending),
                      unfinished_batches=len(pending), unresolved=pending,
                      hol_wait_s=max(0.0, now-self.pending[imsi][0].arrival_s) if pending else 0.0)
        if result["offered_bytes"] != result["served_bytes"] + result["unfinished_bytes"] + result["overflow_bytes"]:
            raise AssertionError("Queue byte conservation failed")
        return result

    def sample(self, imsi, now):
        queue = self.pending[imsi]
        self.emit("hol", imsi, now, hol_wait_s=max(0.0, now-queue[0].arrival_s) if queue else 0.0,
                  queued_bytes=sum(b.remaining_bytes for b in queue))

    def finalize(self, imsi, now):
        result = dict(imsi=imsi, time_s=now, **self.snapshot(imsi, now))
        self.closed.append(result)
        self.emit("unfinished", imsi, now, summary=result)
        self.pending.pop(imsi, None)
        self.totals.pop(imsi, None)
        self.budgets.pop(imsi, None)
        return result

    def close(self):
        if self._file:
            self._file.close()
            self._file = None
