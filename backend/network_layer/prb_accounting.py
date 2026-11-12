"""Integer quota invariants shared by action selection, application and service."""

import math

import settings


def require(condition, message):
    # Keep physical checks active even when Python is run with -O.
    if not condition:
        raise AssertionError(message)


def quota_limits(cell, pool_size, fallback_base=None):
    base = getattr(cell, "base_dl_prb_quota", None)
    if base is None:
        base = fallback_base or dict(cell.slice_dl_prb_quota)
    base = dict(base)
    require(all(math.isfinite(v) and v >= 0 and int(v) == v for v in base.values()),
            "Baseline quotas must be nonnegative integers")
    require(int(pool_size) == pool_size and pool_size >= 0, "Invalid reserve pool size")
    physical = int(cell.max_dl_prb)
    effective = getattr(cell, "_effective_max_dl_prb", None)
    effective = physical if effective is None else int(effective)
    reserved = int(getattr(settings, "JAM_RESERVED_PRBS", 0))
    require(0 <= reserved <= physical, "Reserved capacity outside physical capacity")
    require(0 <= effective <= physical, "Effective capacity outside physical capacity")
    cap = min(physical, effective + reserved, sum(base.values()) + int(pool_size))
    return base, cap


def validate_quotas(cell, quotas, pool_size, fallback_base=None):
    base, cap = quota_limits(cell, pool_size, fallback_base)
    require(all(math.isfinite(v) and v >= 0 and int(v) == v for v in quotas.values()),
            "PRB quotas must be nonnegative integers")
    require(all(quotas.get(s, 0) >= v for s, v in base.items()), "Quota below current baseline")
    require(sum(quotas.values()) <= cap, f"PRB quota {sum(quotas.values())} exceeds cap {cap}")
    occupancy = sum(quotas.values()) - sum(base.values())
    require(0 <= occupancy <= pool_size, "Reserve occupancy outside pool")
    return occupancy
