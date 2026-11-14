"""Predeclared Phase 2G rewards; original reward, learner and physics untouched.

R2 is assessed only at the end of the action's observed service interval. Counters
contain exactly-once physical failures, never administrative terminal charges.
"""
from collections import Counter

from .joint_dqn_agent import JointDQNAgent
from .joint_dqn_schema import encode_joint_state
from .joint_reward import FailureLedger, REWARD_VERSION
from .matched_control import MatchedAdapter, xAppMatchedPilot
from .resilient_interval import CausalInterval
from .xApps.xapp_joint_deadline_dqn import JointCausalAdapter

ARMS = ('R0', 'R1', 'R2')
VERSIONS = {'R0': REWARD_VERSION, 'R1': 'joint-reward-no-history-v1',
            'R2': 'joint-reward-interval-events-v1'}
DEFINITIONS = {
    'R0': 'Original sum of valid historical worst-risk cost over ten service ticks',
    'R1': 'Remove only historical worst-risk cost',
    'R2': '0.25 * max_u(interval_new_known_physical_failures_u / 10); no clipping',
}
COMPONENTS = ('failure_cost', 'terminal_cost', 'fairness_cost', 'embb_cost',
              'reserve_cost', 'switch_cost')


def check_arm(arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown predeclared reward arm: {arm}')


class PerUEFailureLedger(FailureLedger):
    """Expose per-consume deltas from the original exactly-once failure union."""

    def __init__(self, measurements, ue_ids):
        self.pending_by_ue = Counter()
        self.last_consumed_by_ue = {}
        super().__init__(measurements, ue_ids)

    def _fail(self, uid, record):
        was_failed = record['failed']
        super()._fail(uid, record)
        if not was_failed:
            self.pending_by_ue[uid] += 1

    def consume(self):
        counts = super().consume()
        self.last_consumed_by_ue = {u: self.pending_by_ue[u] for u in sorted(self.ue_ids)}
        if sum(self.last_consumed_by_ue.values()) != counts['new_known_failures']:
            raise AssertionError('Per-UE physical failure deltas do not conserve total')
        self.pending_by_ue.clear()
        return counts


def event_fairness_cost(counts):
    """Fixed per-UE reference exposure ten; deliberately unbounded on bursts."""
    if not counts or any(isinstance(v, bool) or not isinstance(v, int) or v < 0
                         for v in counts.values()):
        raise ValueError('Expected nonnegative integer physical failure counts')
    return .25 * (max(counts.values()) / 10)


class RewardArmAgent(JointDQNAgent):
    """Only checkpoint metadata differs; inherited learning is unchanged."""

    def __init__(self, config, *, arm):
        check_arm(arm)
        self.arm = arm
        super().__init__(config)

    def metadata(self):
        result = super().metadata()
        result.update(reward_version=VERSIONS[self.arm], reward_arm=self.arm,
                      fairness_definition=DEFINITIONS[self.arm])
        return result


class RewardAblationAdapter(MatchedAdapter):
    def __init__(self, app):
        super().__init__(app)
        self.totals = dict.fromkeys(COMPONENTS, 0.)
        self.reference_totals = dict.fromkeys(COMPONENTS, 0.)
        self.paid_failure_by_ue = Counter()
        self.interval_count = 0

    def _apply_action(self, action):
        super()._apply_action(action)
        self.service_ticks = 0
        self.failures = Counter()
        self.parts = dict.fromkeys(COMPONENTS, 0.)
        self.reference_parts = dict.fromkeys(COMPONENTS, 0.)
        self.switch_in_interval = bool(self.switched)
        self.valid_risk_ticks = self.saturated_risk_ticks = 0

    def _compute_reward(self, obs, action, previous):
        # Call the original reward exactly once, with its real service outcome.
        original, original_parts = JointCausalAdapter._compute_reward(self, obs, action, previous)
        self.monitor_return += original
        self.service_ticks += 1
        deltas = self.app.ledger.last_consumed_by_ue
        self.failures.update(deltas)
        self.paid_failure_by_ue.update(deltas)
        reference = dict(original_parts)
        reference['fairness_cost'] = reference.pop('worst_cost')
        parts = dict(reference)
        if self.app.reward_arm != 'R0':
            parts['fairness_cost'] = (event_fairness_cost(self.failures)
                if self.app.reward_arm == 'R2' and self.service_ticks == 10 else 0.)
        # Preserve R0's original floating-point sum and reward exactly.
        reward = original if self.app.reward_arm == 'R0' else -sum(parts.values())
        for k in COMPONENTS:
            self.totals[k] += parts[k]
            self.parts[k] += parts[k]
            self.reference_totals[k] += reference[k]
            self.reference_parts[k] += reference[k]
        self.valid_risk_ticks += int(obs['risk'].valid[4])
        self.saturated_risk_ticks += int(obs['risk'].valid[4] and obs['risk'].values[4] >= 1-1e-12)
        if self.service_ticks > 10:
            raise AssertionError('Action interval exceeded ten service ticks')
        if self.service_ticks == 10:
            self.interval_count += 1
            row = dict(start_step=self.app.engine.sim_step-10, end_step=self.app.engine.sim_step,
                action_index=self.interval_action_index(action), mode=action.urllc_scheduler_mode.value,
                delta_e=action.delta_embb_prb, delta_u=action.delta_urllc_prb,
                switched=self.switch_in_interval, failures_by_ue=dict(self.failures),
                new_failures=sum(self.failures.values()),
                valid_risk_ticks=self.valid_risk_ticks, saturated_risk_ticks=self.saturated_risk_ticks,
                reward_components=dict(self.parts), reference_components=dict(self.reference_parts))
            if self.app.interval_sink is not None:
                self.app.interval_sink(row)
        return reward, parts

    def interval_action_index(self, action):
        return self.actions.index(action)


class xAppRewardAblation(xAppMatchedPilot):
    def __init__(self, ric=None, *, reward_arm, interval_sink=None, **kwargs):
        check_arm(reward_arm)
        self.reward_arm = reward_arm
        self.interval_sink = interval_sink
        super().__init__(ric, **kwargs)
        if self.ablation != 'A4' or getattr(self.agent, 'arm', None) != reward_arm:
            raise ValueError('Reward ablation requires full A4 and matching versioned agent')

    def start(self):
        super().start()
        qm = self.cell.base_station.queue_measurements
        ids = self.ledger.ue_ids
        self.ledger.close()
        self.ledger = PerUEFailureLedger(qm, ids)
        self.adapter = RewardAblationAdapter(self)
        self.interval = CausalInterval(self.adapter, 10, encode_joint_state)

    def _finish(self):
        super()._finish()
        if self.adapter.service_ticks != 10:
            raise AssertionError('Ablation requires complete ten-tick intervals')
        if self.adapter.interval_count != self.transition_count:
            raise AssertionError('Lost interval reward instrumentation')
        paid = sum(self.adapter.paid_failure_by_ue.values())
        if paid != self.ledger.total_failures-self.bootstrap_counts['new_known_failures']:
            raise AssertionError('Bootstrap/paid physical failures do not conserve')
        if abs(sum(self.adapter.totals.values())+self.reward_sum) > 1e-7:
            raise AssertionError('Episode components do not sum to learned reward')
        self.report.update(reward_arm=self.reward_arm, return_objective=VERSIONS[self.reward_arm],
            reward_components=dict(self.adapter.totals),
            reference_reward_components=dict(self.adapter.reference_totals),
            paid_failures_by_ue=dict(self.adapter.paid_failure_by_ue),
            physical_ledger_by_ue=self.ledger.by_ue,
            reward_interval_count=self.adapter.interval_count)
