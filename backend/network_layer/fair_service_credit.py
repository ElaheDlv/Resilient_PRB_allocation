"""Bounded causal service credits; no traffic/channel prediction or queue changes."""

import math


class FairServiceCredit:
    """Credits are bytes owed relative to a capacity-scaled recent admitted load.

    Queue increments since the previous actual service identify admitted arrivals.
    Offered overflow is deliberately not interpreted as serviceable queue demand.
    State is per UE and is removed when the UE leaves the episode.
    """

    def __init__(self, window_ticks=20, smoothing=0.05):
        if window_ticks < 1 or not 0 < smoothing <= 1:
            raise ValueError("Invalid credit window or smoothing")
        self.window_ticks = window_ticks
        self.smoothing = smoothing
        self.states = {}

    def prepare(self, queues, quota, per_ue_cap=None):
        costs = {}
        for uid, q in queues.items():
            state = self.states.setdefault(
                uid, dict(after=0, rate=None, credit=0.0, bound=0.0)
            )
            arrivals = max(0, q.queued_bytes - state["after"])
            state["rate"] = (
                arrivals
                if state["rate"] is None
                else ((1 - self.smoothing) * state["rate"] + self.smoothing * arrivals)
            )
            rate = q.bytes_per_prb
            cap = min(quota, per_ue_cap) if per_ue_cap is not None else quota
            costs[uid] = (
                min(cap, state["rate"] / rate)
                if math.isfinite(rate) and rate > 0
                else 0.0
            )
        factor = (
            min(1.0, quota / sum(costs.values())) if sum(costs.values()) > 0 else 0.0
        )
        credits, entitlements = {}, {}
        for uid, q in queues.items():
            state = self.states[uid]
            entitlement = (
                costs[uid] * factor * q.bytes_per_prb if costs[uid] > 0 else 0.0
            )
            state["bound"] = self.window_ticks * max(1.0, entitlement)
            state["credit"] = max(
                -state["bound"], min(state["bound"], state["credit"] + entitlement)
            )
            credits[uid] = state["credit"]
            entitlements[uid] = max(1.0, entitlement)
        return credits, entitlements

    def service(self, uid, served_bytes, queued_after):
        if uid in self.states:
            state = self.states[uid]
            state["after"] = queued_after
            state["credit"] = max(
                -state["bound"], min(state["bound"], state["credit"] - served_bytes)
            )

    def remove(self, uid):
        self.states.pop(uid, None)
