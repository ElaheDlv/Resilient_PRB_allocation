"""Bounded, work-conserving proportional integer allocation (no deadlines)."""

import math


def proportional_allocation(demands, quota, per_ue_cap=None):
    """Allocate min(quota, useful demand) by capped proportional water filling.

    Dictionary order deterministically breaks equal remainders. Zero-demand UEs
    receive nothing. Caps and integer rounding never create extra PRBs.
    """
    if quota < 0 or int(quota) != quota:
        raise ValueError("quota must be a nonnegative integer")
    if per_ue_cap is not None and (per_ue_cap < 0 or int(per_ue_cap) != per_ue_cap):
        raise ValueError("per_ue_cap must be a nonnegative integer")
    limits = {uid: min(max(0, int(d)), int(per_ue_cap)) if per_ue_cap is not None
              else max(0, int(d)) for uid, d in demands.items()}
    result = dict.fromkeys(demands, 0)
    remaining = min(int(quota), sum(limits.values()))
    while remaining:
        active = [u for u in demands if result[u] < limits[u]]
        weight = sum(demands[u] for u in active)
        shares = {u: remaining * demands[u] / weight for u in active}
        additions = {u: min(limits[u] - result[u], math.floor(shares[u])) for u in active}
        used = sum(additions.values())
        for u in active:
            result[u] += additions[u]
        remaining -= used
        if not remaining:
            break
        # Recompute after saturating caps; otherwise allocate largest remainders.
        if any(result[u] == limits[u] for u in active):
            continue
        order = sorted(active, key=lambda u: -(shares[u] - math.floor(shares[u])))
        for u in order[:remaining]:
            result[u] += 1
            remaining -= 1
    return result
