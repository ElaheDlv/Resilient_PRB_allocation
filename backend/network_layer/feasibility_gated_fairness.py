"""Isolated experimental service-deficit protection; no controller/state changes.

The existing feasibility allocator is reused, including its virtual FIFO,
zero-service guard, integer quanta, fallback and useful-demand limits.
"""
from collections import deque
import math

from .deadline_scheduler import deadline_allocation
from .fair_service_credit import FairServiceCredit

EPS = 1e-12


class ServiceWindow:
    """Post-service cumulative observations, retaining W seconds plus one endpoint."""
    def __init__(self, window_s=.100):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError('Positive finite window required')
        self.window_s = window_s
        self.history = {}

    def observe(self, uid, now, offered, served, queued):
        if min(now, offered, served, queued) < 0:
            raise ValueError('Negative observation')
        h = self.history.setdefault(uid, deque([(0., 0, 0, 0)]))
        if now <= h[-1][0]:
            raise ValueError('Service observations must advance in time')
        if offered < h[-1][1] or served < h[-1][2]:
            raise ValueError('Counters reset without episode reset')
        h.append((now, offered, served, queued))
        while len(h) > 2 and h[1][0] <= now-self.window_s+EPS:
            h.popleft()

    def view(self, uid):
        h = self.history.get(uid)
        if not h:
            return dict(full=False, offered=0, served=0, deficit=0., queue_growth=0)
        first,last = h[0],h[-1]
        offered,served = last[1]-first[1],last[2]-first[2]
        return dict(full=last[0]+EPS >= self.window_s,
            offered=offered,served=served,
            deficit=max(0.,min(1.,(offered-served)/offered)) if offered else 0.,
            queue_growth=last[3]-first[3])


def reachable_target(pending, rate, budget, now, dt, quota, cap=None):
    """Find first fully admitted, not-expired batch with attainable FIFO prefix.

    Optimistic local test: current rate and cap persist; other UEs are ignored.
    Includes the current service timestamp and equality at the deadline. An
    early byte bound limits scanning even when the expired backlog is huge.
    """
    if rate <= 0 or not math.isfinite(rate) or not budget or budget <= 0 or quota <= 0:
        return None
    cap = min(quota,cap) if cap is not None else quota
    if cap <= 0:
        return None
    max_slots = math.floor((budget+EPS)/dt)+1
    prefix = 0
    for b in pending:
        prefix += b.remaining_bytes
        need = math.ceil(prefix/rate)
        if need > max_slots*cap:
            return None
        slack = b.arrival_s+budget-now
        slots = math.floor((slack+EPS)/dt)+1
        if b.overflow_bytes == 0 and slack >= -EPS and need <= max(0,slots)*cap:
            return dict(prefix_bytes=prefix,need_prbs=need,slack_s=max(0.,slack),slots=slots)
    return None


class ExperimentalFairAllocator:
    """Quota-fraction protection with causal deficit and feasible-prefix gating.

    lambda is a protected PRB share, not a priority bonus. Historical misses
    only scale a positive recent service deficit attached to a reachable timely
    target. Credits are recomputed each round, never a debt for past failures.
    legacy=True runs the unchanged Phase1.2 credit rule as an explicit comparator.
    """
    def __init__(self, cell, collector, fraction, *, legacy=False):
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Fraction outside [0,1]')
        self.cell,self.collector,self.fraction = cell,collector,fraction
        self.legacy = legacy
        self.window = ServiceWindow()
        self.old_credit = FairServiceCredit() if legacy else None
        self.previous_served = {}
        self.last_protection = {}

    def observe_service(self, now):
        bs=self.cell.base_station
        for uid,u in self.cell.connected_ue_list.items():
            if u.slice_type != 'URLLC':continue
            total=bs.queue_measurements.totals[uid]
            self.window.observe(uid,now,total['offered_bytes'],total['served_bytes'],bs.get_dl_buf_bytes(uid))
            if self.old_credit is not None:
                served=total['served_bytes']-self.previous_served.get(uid,0)
                self.old_credit.service(uid,served,bs.get_dl_buf_bytes(uid))
            self.previous_served[uid]=total['served_bytes']

    def __call__(self, queues, quota, now_s, dt, starvation_ticks,
                 starvation_limit=5, per_ue_cap=None, *, feasibility=True, **unused):
        if not feasibility:
            raise ValueError('Experimental protection requires feasibility mode')
        args={}
        self.last_protection={}
        if self.fraction:
            if self.legacy:
                credits,entitlements=self.old_credit.prepare(queues,quota,per_ue_cap)
            else:
                report=self.collector.collect(self.cell.base_station.simulation_engine.sim_step)
                telemetry={u.ue_id:u for u in report.ues}
                credits,entitlements={},{}
                pending=self.cell.base_station.queue_measurements.pending
                for uid,q in queues.items():
                    w=self.window.view(uid);u=telemetry[uid]
                    mature=u.window.matured_offered_batches
                    miss=(u.window.matured_failed_batches/mature
                          if u.window.full_window and mature is not None and mature >= 20 else 0.)
                    target=reachable_target(pending.get(uid,()),q.bytes_per_prb,q.budget_s,now_s,dt,quota,per_ue_cap)
                    entitlements[uid]=max(1,w['offered'])
                    credit=(min(target['prefix_bytes'],max(0,w['offered']-w['served'])*(1+miss))
                            if target and w['full'] else 0.)
                    credits[uid]=credit
                    self.last_protection[uid]=dict(window=w,miss_fraction=miss,target=target,credit_bytes=credit)
            args=dict(fair_fraction=self.fraction,fair_credits=credits,fair_entitlements=entitlements)
        return deadline_allocation(queues,quota,now_s,dt,starvation_ticks,
            starvation_limit,per_ue_cap,feasibility=True,**args)
