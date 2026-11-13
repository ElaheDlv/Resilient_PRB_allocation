"""Causal exactly-once failure ledger and explicit finite-horizon reward shaping."""

from dataclasses import asdict, dataclass
import math

REWARD_VERSION = 'joint-reward-v1'
EPS = 1e-12


@dataclass(frozen=True)
class RewardConfig:
    failure: float = 1.0
    worst: float = 0.25
    embb: float = 0.10
    reserve: float = 0.02
    switch: float = 0.01
    embb_target_bps: float = 96e6
    reference_batches_per_ue_s: float = 1000.0
    period_s: float = 0.01

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Reward settings must be finite and nonnegative')
        if min(self.embb_target_bps, self.reference_batches_per_ue_s, self.period_s) <= 0:
            raise ValueError('Reward normalizations must be positive')


class FailureLedger:
    """Retain only live batches; never infer new failures from rolling rate changes."""

    def __init__(self, measurements, ue_ids):
        self.qm = measurements
        self.ue_ids = frozenset(ue_ids)
        if not self.ue_ids or any(measurements.totals.get(u, {}).get('offered_batches', 0)
                                  for u in self.ue_ids):
            raise ValueError('Failure ledger must attach before episode arrivals')
        self.live = {}
        self.new_failures = 0
        self.offered = 0
        self.total_failures = 0
        self.terminal_charges = 0
        self.by_ue = {u: dict(offered=0, failures=0, terminal_charges=0) for u in self.ue_ids}
        self.closed = False
        self.qm.add_listener(self.on_event)

    def _fail(self, uid, record):
        if not record['failed']:
            record['failed'] = True
            self.new_failures += 1
            self.total_failures += 1
            self.by_ue[uid]['failures'] += 1

    def on_event(self, event):
        uid = event['imsi']
        if uid not in self.ue_ids or self.closed:
            return
        key = (uid, event.get('batch_id'))
        if event['kind'] == 'arrival':
            budget = event['budget_s']
            if budget is None or not math.isfinite(budget) or budget <= 0:
                raise ValueError('Joint reward requires measured positive URLLC budgets')
            record = dict(arrival=event['time_s'], budget=budget, failed=False)
            self.offered += 1
            self.by_ue[uid]['offered'] += 1
            if event['overflow_bytes'] > 0:
                self._fail(uid, record)
            if event['accepted_bytes']:
                self.live[key] = record
        elif event['kind'] == 'batch_completion':
            record = self.live.pop(key)
            if event['over_budget']:
                self._fail(uid, record)

    def sample(self, now):
        if self.closed:
            raise RuntimeError('Cannot sample a closed reward ledger')
        for (uid, _), record in self.live.items():
            if now-record['arrival'] > record['budget']+EPS:
                self._fail(uid, record)

    def terminal(self, now):
        self.sample(now)
        if len(self.live) != sum(len(self.qm.pending.get(u, ())) for u in self.ue_ids):
            raise AssertionError('Reward ledger differs from live measured backlog')
        unresolved = len(self.live)
        charged = 0
        for (uid, _), record in self.live.items():
            if not record['failed']:
                charged += 1
                self.by_ue[uid]['terminal_charges'] += 1
        self.terminal_charges += charged
        self.closed = True
        return dict(unresolved_batches=unresolved,
                    unresolved_bytes=sum(b.remaining_bytes for u in self.ue_ids
                                         for b in self.qm.pending.get(u, ())),
                    terminal_unresolved_charges=charged)

    def consume(self):
        counts = dict(new_known_failures=self.new_failures, offered_batches=self.offered)
        self.new_failures = self.offered = 0
        return counts

    def close(self):
        self.qm.remove_listener(self.on_event)


def tick_reward(*, counts, terminal_charges, risk, embb_bytes, dt, reserve_prbs,
                switched, ue_count, config):
    if dt <= 0 or ue_count < 1 or embb_bytes < 0 or not 0 <= reserve_prbs <= 18:
        raise ValueError('Invalid service outcome for joint reward')
    nref = ue_count * config.reference_batches_per_ue_s * config.period_s
    scale = dt/config.period_s
    parts = dict(
        failure_cost=config.failure*counts['new_known_failures']/nref,
        terminal_cost=config.failure*terminal_charges/nref,
        worst_cost=scale*config.worst*(risk.values[4] if risk.valid[4] else 0.0),
        embb_cost=scale*config.embb*max(0.0, 1-embb_bytes*8/(dt*config.embb_target_bps)),
        reserve_cost=scale*config.reserve*reserve_prbs/18,
        switch_cost=config.switch*int(switched),
    )
    reward = -sum(parts.values())
    if not math.isfinite(reward):
        raise ValueError('Nonfinite joint reward')
    return reward, parts
