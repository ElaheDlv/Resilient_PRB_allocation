"""Current-state slack / optional optimistic-feasibility FIFO scheduling.

The virtual FIFO is used only to account for grants within this allocation round.
It does not consume queues, forecast arrivals, or change service/expiry rules.
"""

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class QueueView:
    queued_bytes: int
    bytes_per_prb: float
    budget_s: float | None
    fifo: Iterable[tuple[float, int]]


def deadline_allocation(
    queues,
    quota,
    now_s,
    dt,
    starvation_ticks,
    starvation_limit=5,
    per_ue_cap=None,
    *,
    feasibility=False,
):
    if quota < 0 or int(quota) != quota or dt <= 0 or starvation_limit < 1:
        raise ValueError("Invalid quota, step or starvation limit")
    if per_ue_cap is not None and (per_ue_cap < 0 or int(per_ue_cap) != per_ue_cap):
        raise ValueError("Invalid per-UE cap")
    grants = dict.fromkeys(queues, 0)
    limits, heads, iterators = {}, {}, {}
    for uid, q in queues.items():
        rate = q.bytes_per_prb
        if q.queued_bytes <= 0 or not math.isfinite(rate) or rate <= 0:
            continue
        if q.budget_s is None or not math.isfinite(q.budget_s) or q.budget_s <= 0:
            raise ValueError(f"Active URLLC UE {uid} needs a positive queueing budget")
        limit = math.ceil(q.queued_bytes / rate)
        limits[uid] = min(limit, per_ue_cap) if per_ue_cap is not None else limit
        iterators[uid] = iter(q.fifo)
        heads[uid] = list(next(iterators[uid], (now_s, 0)))
        if heads[uid][1] <= 0:
            raise AssertionError("Nonempty UE buffer lacks a FIFO head")

    remaining = int(quota)

    def quantum(uid):
        rate = queues[uid].bytes_per_prb
        # A bundle must produce at least one additional whole service byte.
        target = math.floor(grants[uid] * rate) + 1
        return max(1, math.ceil(target / rate) - grants[uid])

    def eligible(uid):
        n = quantum(uid)
        return grants[uid] + n <= limits[uid] and n <= remaining

    def give(uid):
        nonlocal remaining
        count = quantum(uid)
        before = math.floor(grants[uid] * queues[uid].bytes_per_prb)
        grants[uid] += count
        remaining -= count
        used = math.floor(grants[uid] * queues[uid].bytes_per_prb) - before
        while used > 0 and heads[uid][1] > 0:
            consumed = min(used, heads[uid][1])
            heads[uid][1] -= consumed
            used -= consumed
            if heads[uid][1] == 0:
                heads[uid] = list(next(iterators[uid], (now_s, 0)))

    # At most one useful quantum per protected UE, oldest starvation age first.
    protected = sorted(
        (u for u in limits if starvation_ticks.get(u, 0) >= starvation_limit),
        key=lambda u: (-starvation_ticks.get(u, 0), u),
    )
    for uid in protected:
        if eligible(uid):
            give(uid)

    def slack(uid):
        arrival, head_bytes = heads[uid]
        q = queues[uid]
        value = (
            q.budget_s - max(0.0, now_s - arrival) - head_bytes * dt / q.bytes_per_prb
        )
        if feasibility:
            # Optimistic local feasibility: unchanged current rate and quota,
            # inclusive instantaneous service timestamps (the simulator model).
            slots = (
                math.floor((q.budget_s - max(0.0, now_s - arrival) + 1e-12) / dt) + 1
            )
            cap = min(int(quota), per_ue_cap) if per_ue_cap is not None else int(quota)
            current = math.floor(grants[uid] * q.bytes_per_prb)
            need = max(
                1, math.ceil((current + head_bytes) / q.bytes_per_prb) - grants[uid]
            )
            attainable = max(0, cap - grants[uid]) + max(0, slots - 1) * cap
            feasible = slots > 0 and need <= attainable
            # EDF among potentially feasible FIFO heads, then shortest service.
            # Infeasible/expired heads remain eligible as fallback; never dropped.
            return (
                0 if feasible else 1,
                arrival + q.budget_s if feasible else 0,
                need,
                -starvation_ticks.get(uid, 0),
                uid,
            )
        return (value, -starvation_ticks.get(uid, 0), uid)

    while remaining:
        active = [uid for uid in limits if eligible(uid)]
        if not active:
            break
        give(min(active, key=slack))
    return grants
