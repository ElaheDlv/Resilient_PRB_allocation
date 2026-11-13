"""Decision-equivalent execution of the frozen Phase2D G1 equations.

The original implementation is untouched. Only evaluation order and work proven
irrelevant to zero credits change. No approximate temporal caching is used.
"""
from .deadline_scheduler import deadline_allocation
from .feasibility_gated_fairness import ExperimentalFairAllocator, reachable_target


class OptimizedG1Allocator(ExperimentalFairAllocator):
    """Fixed lambda=1. Same service windows, credit arithmetic, target and kernel.

    A nonzero credit requires a full service window, positive offered-minus-served
    bytes and a reachable target. If no UE meets all three, every original credit
    is zero for every possible miss fraction. Omit the full telemetry query and
    zero-credit fair loop. Otherwise query the original current-time collector.

    `last_protection.miss_fraction` is None when an unused diagnostic fraction
    was not queried; it never enters control, rewards, queues or exported outcomes.
    Scheduling/physical equivalence, not private query-cache identity, is claimed.
    """
    def __init__(self, cell, collector):
        super().__init__(cell, collector, 1., legacy=False)
        self.telemetry_calls = 0
        self.zero_credit_fast_paths = 0

    def __call__(self, queues, quota, now_s, dt, starvation_ticks,
                 starvation_limit=5, per_ue_cap=None, *, feasibility=True, **unused):
        if not feasibility:
            raise ValueError('Experimental protection requires feasibility mode')
        windows = {u:self.window.view(u) for u in queues}
        pending = self.cell.base_station.queue_measurements.pending
        targets = {u:reachable_target(pending.get(u,()),q.bytes_per_prb,q.budget_s,
                    now_s,dt,quota,per_ue_cap) for u,q in queues.items()}
        nonzero_possible = any(w['full'] and w['offered']>w['served'] and targets[u]
                               for u,w in windows.items())
        self.last_protection = {}
        if not nonzero_possible:
            self.zero_credit_fast_paths += 1
            self.last_protection = {u:dict(window=w,miss_fraction=None,target=targets[u],credit_bytes=0.)
                                    for u,w in windows.items()}
            # Original guard runs before the fair loop. Zero credits cannot
            # become positive after a guard grant, so its no-op loop is removable.
            return deadline_allocation(queues,quota,now_s,dt,starvation_ticks,
                starvation_limit,per_ue_cap,feasibility=True)
        self.telemetry_calls += 1
        report = self.collector.collect(self.cell.base_station.simulation_engine.sim_step)
        telemetry = {u.ue_id:u for u in report.ues}
        credits,entitlements = {},{}
        for uid in queues:
            w,u,target = windows[uid],telemetry[uid],targets[uid]
            mature = u.window.matured_offered_batches
            miss = (u.window.matured_failed_batches/mature
                    if u.window.full_window and mature is not None and mature>=20 else 0.)
            entitlements[uid] = max(1,w['offered'])
            credit = (min(target['prefix_bytes'],max(0,w['offered']-w['served'])*(1+miss))
                      if target and w['full'] else 0.)
            credits[uid] = credit
            self.last_protection[uid] = dict(window=w,miss_fraction=miss,target=target,credit_bytes=credit)
        return deadline_allocation(queues,quota,now_s,dt,starvation_ticks,
            starvation_limit,per_ue_cap,feasibility=True,fair_fraction=1.,
            fair_credits=credits,fair_entitlements=entitlements)
