"""Paired pilot controllers; physical models and original scheduler functions unchanged."""

from collections import Counter, defaultdict
from dataclasses import asdict
import json

import numpy as np
import torch
import settings
from utils import estimate_throughput
from .joint_control import JointRICAction, JointRICBridge, URLLCSchedulerMode as Mode
from .joint_dqn_agent import JointDQNAgent
from .joint_dqn_schema import encode_joint_state, joint_action_mask
from .joint_reward import FailureLedger
from .phase1_metrics import Phase1Metrics
from .resilient_interval import CausalInterval
from .ue_telemetry import UETelemetryCollector
from .xApps.xapp_joint_deadline_dqn import JointCausalAdapter, xAppJointDeadlineDQN
from .xApps.xapp_resilient_dqn_allocator import ACTION_LIST, ResilientDQNEnv, _DQN, build_state
from .xApps.xapp_resilient_policy_validator import xAppResilientPolicyValidator


class LegacyPilotAgent(JointDQNAgent):
    """Original 15→256→256→49 network, encoder and reward; original MSE/no clipping."""

    state_dimension = 15
    action_count = len(ACTION_LIST)
    gradient_clip = None

    def make_network(self):
        return _DQN(self.state_dimension, self.action_count)

    def loss_function(self, prediction, targets):
        return torch.nn.functional.mse_loss(prediction, targets)

    def metadata(self):
        return dict(state_schema='original-15-v1', action_schema='original-49-v1',
            reward_version='original-source-ed267de', network='15-256-256-49-ReLU',
            loss='MSE', gradient_clip=None, training_config=asdict(self.config),
            reward_settings={k: getattr(settings, k) for k in (
                'RES_DQN_W_E', 'RES_DQN_W_U', 'RES_DQN_LAMBDA_LAT', 'RES_DQN_MU_RESERVE',
                'RES_DQN_URLLC_LATENCY_DEADLINE')}, device='cpu', seed=self.config.seed)


class HystereticModeRule:
    def __init__(self, entry=1.25, release_ratio=0.5, release_decisions=2):
        if entry <= 0 or not 0 < release_ratio < 1 or release_decisions < 1:
            raise ValueError('Invalid causal switching rule')
        self.entry, self.release_ratio = entry, release_ratio
        self.release_decisions = release_decisions
        self.mode, self.low_count = Mode.PROPORTIONAL, 0

    def select(self, risk):
        if not risk.valid[0]:
            self.low_count = 0
            return self.mode
        pressure = risk.values[0]
        if self.mode is Mode.PROPORTIONAL:
            if pressure > self.entry:
                self.mode = Mode.FEASIBILITY_AWARE
        elif pressure <= self.entry*self.release_ratio:
            self.low_count += 1
            if self.low_count >= self.release_decisions:
                self.mode, self.low_count = Mode.PROPORTIONAL, 0
        else:
            self.low_count = 0
        return self.mode


class ReactiveReserve:
    """Reuse reference current-baseline-loss target and nearest feasible increment."""

    _aggressive_deltas = xAppResilientPolicyValidator._aggressive_deltas
    _closest_action = xAppResilientPolicyValidator._closest_action

    def __init__(self, env):
        self._env = env
        self._nojam_base = dict(env._target_cell().base_dl_prb_quota)

    def select(self, obs, mask):
        return self._closest_action(mask, *self._aggressive_deltas(obs))


class PilotAdapter(JointCausalAdapter):
    def __init__(self, app):
        super().__init__(app)
        if app.method != 'P':
            self.actions = tuple(JointRICAction(e, u, Mode.PROPORTIONAL) for e,u in ACTION_LIST)
        self.prev_action = JointRICAction(0, 0, app.initial_mode)
        self.monitor_return = 0.0

    def _action_mask(self):
        mask = joint_action_mask(self.app.bridge)
        if self.app.method == 'P':
            return mask
        fixed = int(self.app.initial_mode is Mode.FEASIBILITY_AWARE)
        return mask[fixed::2]

    def _apply_action(self, action):
        if self.app.method != 'P':
            mode = (self.app.rule.select(self._collect_obs()['risk'])
                    if self.app.rule is not None else self.app.initial_mode)
            action = JointRICAction(action.delta_embb_prb, action.delta_urllc_prb, mode)
        super()._apply_action(action)

    def _compute_reward(self, obs, action, previous):
        monitor, _ = super()._compute_reward(obs, action, previous)
        self.monitor_return += monitor
        if self.app.method == 'P':
            # monitor is precisely the unchanged Phase 2B joint reward.
            return monitor, {'joint_reward': monitor}
        return self.app.physical._compute_reward(obs['legacy'],
            (action.delta_embb_prb, action.delta_urllc_prb),
            (previous.delta_embb_prb, previous.delta_urllc_prb))


def legacy_pilot_state(obs, previous):
    return build_state(obs['legacy'], (previous.delta_embb_prb, previous.delta_urllc_prb))


class xAppPairedPilot(xAppJointDeadlineDQN):
    """One fresh scenario per app, persistent learner supplied by the sequential runner."""

    def __init__(self, ric=None, *, method, agent, scenario_seed, train=False,
                 exploration=None, rule_entry=1.25, rule_release_ratio=0.5, rule_release_decisions=2):
        super().__init__(ric, enabled=True, train=train)
        if method not in {'B0','B1','B2','B3','B4','B5','P'}:
            raise ValueError('Unknown pilot method')
        self.method, self.agent, self.scenario_seed = method, agent, scenario_seed
        self.exploration = exploration or (lambda step: 0.0)
        self.initial_mode = Mode.FEASIBILITY_AWARE if method in {'B2','B4'} else Mode.PROPORTIONAL
        self.rule = HystereticModeRule(rule_entry, rule_release_ratio, rule_release_decisions) if method == 'B5' else None
        self.mode_ticks = Counter()
        self.valid_fractions, self.q_magnitudes, self.selected_q = [], [], []
        self.worst_saturated_ticks = self.worst_valid_ticks = 0
        self.ue_resources = defaultdict(lambda: dict(prbs=0.,capacity_bytes_per_prb=0.,fluid_prbs=0.))
        self.fluid_load_sum = self.fluid_load_ticks = 0
        self.episode_losses = []
        self.mask_checks = 0
        self.steps_seen = 0

    def start(self):
        if len(self.ric.xapp_list) != 1 or settings.RES_DQN_TRANSITION_MODE != 'causal':
            raise ValueError('Pilot requires one causal control owner')
        if settings.JAM_RESERVED_PRBS != 18 or settings.RES_DQN_TOTAL_RESILIENCY_PRB != 18:
            raise ValueError('Pilot requires fixed 18-PRB physical reserve')
        if float(settings.SIM_STEP_TIME_DEFAULT) != .001 or settings.RES_DQN_DECISION_PERIOD != 10:
            raise ValueError('Pilot requires 1-ms service and 10-ms decisions')
        self.engine = self.ric.simulation_engine
        if len(self.engine.cell_list) != 1 or len(self.engine.base_station_list) != 1:
            raise ValueError('Pilot supports a single gNB/cell')
        self.cell = next(iter(self.engine.cell_list.values()))
        self.physical = ResilientDQNEnv(self.ric, spawn_seed=self.scenario_seed)
        self.physical.reset()
        if not self.physical.ready() or self.physical._episode_step_limit < 2:
            raise ValueError('Pilot requires a populated scenario of >=2 intervals')
        self.collector = UETelemetryCollector(self.cell, self.window_s)
        self.bridge = JointRICBridge(self.engine, self.cell, 10)
        self.bridge.begin_episode(self.initial_mode, episode_token=0)
        self._before = self.engine.before_service_observer
        self.engine.before_service_observer = self.before_service
        self.metrics = Phase1Metrics()
        self.engine.after_service_observer = self.observe_service
        self._previous_offered = {}
        self.ledger = FailureLedger(self.cell.base_station.queue_measurements,
            [uid for uid,u in self.cell.connected_ue_list.items() if u.slice_type == 'URLLC'])
        self.adapter = PilotAdapter(self)
        self.interval = CausalInterval(self.adapter, 10,
            encode_joint_state if self.method == 'P' else legacy_pilot_state)
        self.origin = self.engine.sim_step
        self.last_embb_served = self.embb_served()
        self.reactive = ReactiveReserve(self.physical)
        self.initial_fingerprint = self.agent.fingerprint() if self.agent else None
        self.updates_before = self.agent.updates if self.agent else 0
        self.report = None

    def observe_service(self, engine, dt):
        self.metrics.observe(engine, dt)
        self.steps_seen += 1
        self.mode_ticks[self.cell.urllc_scheduler_mode.value] += 1
        # Exogenous offered-fluid PRBs/current baseline is a capacity diagnostic,
        # not an input to any policy, nor a proof of deadline feasibility.
        demand, known = 0.0, True
        qm = self.cell.base_station.queue_measurements
        for uid,u in self.cell.connected_ue_list.items():
            if u.slice_type != 'URLLC':
                continue
            total = qm.totals[uid]['offered_bytes']
            arrivals = total-self._previous_offered.get(uid, 0)
            self._previous_offered[uid] = total
            mcs = u.downlink_mcs_data or {}
            rate = (estimate_throughput(mcs.get('modulation_order',0),mcs.get('target_code_rate',0),1)*dt/8
                    if mcs.get('modulation_order',0)>0 and mcs.get('target_code_rate',0)>0 else 0)
            resource = self.ue_resources[uid]
            resource['prbs'] += self.cell.prb_ue_allocation_dict[uid]['downlink']
            resource['capacity_bytes_per_prb'] += rate
            resource['fluid_prbs'] += arrivals/rate if rate > 0 else 0.
            if rate <= 0 and arrivals:
                known = False
            elif rate > 0:
                demand += arrivals/rate
        if known:
            self.fluid_load_sum += demand/max(1,self.cell.base_dl_prb_quota['URLLC'])
            self.fluid_load_ticks += 1

    def step(self):
        if self.finished:
            return
        if not self.engine.in_ric_callback:
            raise RuntimeError('Pilot requires a post-service RIC callback')
        step = self.engine.sim_step
        self.ledger.sample(self.cell.base_station._time_s)
        if self.interval.pending is None:
            if step-self.origin < 10:
                return
            self.bootstrap_counts = self.ledger.consume()
            self.last_embb_served = self.embb_served()
            self.physical._steps_in_episode = 1
        else:
            p = self.interval.pending
            terminal = ((self.physical._steps_in_episode+1 >= self.physical._episode_step_limit
                        and p['ticks']+1 >= 10) or self.adapter._traces_consumed())
            self.adapter.prepare_tick(terminal)
            transition = self.interval.observe()
            if transition is None:
                return
            if self.train:
                self.agent.remember(transition)
                loss = self.agent.learn()
                if loss is not None:
                    self.episode_losses.append(self.agent.last_update.copy())
            self.transition_count += 1
            self.reward_sum += transition.reward
            if transition.done:
                self._finish()
                return
        obs = self.adapter._collect_obs()
        risk = obs['risk']
        self.worst_valid_ticks += int(risk.valid[4])
        self.worst_saturated_ticks += int(risk.valid[4] and risk.values[4] >= 1-1e-12)
        state = (encode_joint_state if self.method == 'P' else legacy_pilot_state)(obs, self.adapter.prev_action)
        mask = self.adapter._action_mask()
        self.valid_fractions.append(sum(mask)/len(mask))
        if self.method == 'B0':
            index = ACTION_LIST.index((0,0))
        elif self.method in {'B1','B2'}:
            index = self.reactive.select(obs['legacy'], mask)
        else:
            epsilon = self.exploration(self.agent.interactions) if self.train else 0.0
            index = self.agent.select(state,mask,explore=self.train,epsilon=epsilon)
            with torch.no_grad():
                q = self.agent.policy(torch.as_tensor(state,dtype=torch.float32))
                self.q_magnitudes.append(float(q[torch.tensor(mask)].abs().max()))
                self.selected_q.append(float(q[index]))
        if not mask[index]:
            raise AssertionError('Pilot selected an infeasible action')
        self.mask_checks += 1
        self.interval.begin(index)
        self.switch_count += int(self.adapter.switched)
        receipt = self.adapter.receipt
        self.digest.update(json.dumps(dict(step=step,action=index,mode=receipt.mode_after.value,
            state=state.tolist(),next_effect=receipt.first_service_step),sort_keys=True).encode())

    def _finish(self):
        qm = self.cell.base_station.queue_measurements
        now = self.cell.base_station._time_s
        reports = [dict(imsi=uid, **qm.snapshot(uid,now)) for uid in self.cell.connected_ue_list]
        physical = self.metrics.finish(reports)
        embb = [r for r in reports if r['budget_s'] is None]
        embb_offered = sum(r['offered_bytes'] for r in embb)
        embb_served = sum(r['served_bytes'] for r in embb)
        accounting = self._terminal_accounting()
        if physical['overall']['known_failed_batches'] != self.ledger.total_failures:
            raise AssertionError('Pilot physical and causal failure counts differ')
        mean = lambda values: float(np.mean(values)) if values else None
        self.report = dict(method=self.method,scenario_seed=self.scenario_seed,
            physical=physical,
            embb_offered_mbps=embb_offered*8/(self.steps_seen*.001)/1e6,
            embb_served_offered_fraction=embb_served/embb_offered if embb_offered else None,
            ue_resources={u:{k:v/self.steps_seen for k,v in row.items()}
                for u,row in self.ue_resources.items()},return_value=self.reward_sum,
            return_objective='joint-reward-v1' if self.method=='P' else 'original-legacy',
            common_joint_monitor_return=self.adapter.monitor_return,
            transitions=self.transition_count,updates=(self.agent.updates-self.updates_before if self.agent else 0),
            terminal=self.adapter.terminal_summary,live_accounting=accounting,
            known_failures=self.ledger.total_failures,bootstrap=self.bootstrap_counts,
            losses=dict(mean=mean([v['loss'] for v in self.episode_losses]),
                max=max((v['loss'] for v in self.episode_losses),default=None),
                mean_abs_td=mean([v['mean_abs_td'] for v in self.episode_losses]),
                max_abs_target=max((v['max_abs_target'] for v in self.episode_losses),default=None)),
            q=dict(max_abs=max(self.q_magnitudes,default=None),mean_selected=mean(self.selected_q)),
            mean_valid_action_fraction=mean(self.valid_fractions),
            valid_action_checks=self.mask_checks,mode_service_ticks=dict(self.mode_ticks),
            switches=self.switch_count,switches_per_decision=self.switch_count/max(1,self.mask_checks),
            worst_risk_valid_decisions=self.worst_valid_ticks,
            worst_risk_saturated_fraction=(self.worst_saturated_ticks/self.worst_valid_ticks if self.worst_valid_ticks else None),
            mean_exogenous_fluid_load_ratio=(self.fluid_load_sum/self.fluid_load_ticks if self.fluid_load_ticks else None),
            control_digest=self.digest.hexdigest(),
            parameter_sha256=self.agent.fingerprint() if self.agent else None)
        self.finished = True
        self.ledger.close()
