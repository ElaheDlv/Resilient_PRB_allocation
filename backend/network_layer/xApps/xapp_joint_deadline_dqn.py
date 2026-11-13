"""Optional single-cell causal joint DQN. No recursive stepping or GPU use."""

from collections import deque
from dataclasses import asdict
import hashlib
import json
import time

import settings
from .xapp_base import xAppBase
from .xapp_resilient_dqn_allocator import ResilientDQNEnv
from network_layer.deadline_risk import encode_deadline_risk
from network_layer.joint_control import JointRICAction, JointRICBridge, URLLCSchedulerMode as Mode
from network_layer.joint_dqn_agent import JointDQNAgent, TrainingConfig
from network_layer.joint_dqn_schema import ACTIONS, encode_joint_state, joint_action_mask, state_metadata
from network_layer.joint_reward import FailureLedger, RewardConfig, tick_reward
from network_layer.resilient_interval import CausalInterval
from network_layer.ue_telemetry import UETelemetryCollector


class JointCausalAdapter:
    """Adapt joint interfaces to the unchanged Phase 0 CausalInterval collector."""

    def __init__(self, app):
        self.app = app
        self.actions = ACTIONS
        self.prev_action = JointRICAction(0, 0, Mode.PROPORTIONAL)
        self.tick_outcome = None
        self.switched = False
        self.receipt = None
        self.counts = dict(new_known_failures=0, offered_batches=0)
        self.terminal_summary = None

    @property
    def _steps_in_episode(self):
        return self.app.physical._steps_in_episode

    @_steps_in_episode.setter
    def _steps_in_episode(self, value):
        self.app.physical._steps_in_episode = value

    @property
    def _episode_step_limit(self):
        return self.app.physical._episode_step_limit

    @property
    def _current_episode(self):
        return self.app.physical._current_episode

    def _traces_consumed(self):
        return (self.app.physical._traces_consumed()
                and (self.app.engine.sim_step-self.app.origin) % 10 == 0)

    def _action_mask(self):
        return joint_action_mask(self.app.bridge)

    def _collect_obs(self):
        report = self.app.collector.collect(self.app.engine.sim_step)
        return dict(legacy=self.app.physical._collect_obs(), report=report,
                    risk=encode_deadline_risk(report, self.app.minimum_matured),
                    minimum_matured=self.app.minimum_matured)

    def _apply_action(self, action):
        self.receipt = self.app.bridge.apply(action)
        self.switched = self.receipt.mode_before != self.receipt.mode_after
        self.counts = dict(new_known_failures=0, offered_batches=0)
        self.terminal_summary = None

    def prepare_tick(self, terminal):
        app = self.app
        now = app.cell.base_station._time_s
        app.ledger.sample(now)
        if terminal:
            self.terminal_summary = app.ledger.terminal(now)
        counts = app.ledger.consume()
        for name, count in counts.items():
            self.counts[name] += count
        served = app.embb_served()
        delta = served-app.last_embb_served
        app.last_embb_served = served
        self.tick_outcome = dict(counts=counts,
            terminal_charges=(self.terminal_summary or {}).get('terminal_unresolved_charges', 0),
            embb_bytes=delta, reserve_prbs=sum(app.cell.slice_dl_prb_quota.values())
                - sum(app.cell.base_dl_prb_quota.values()))

    def _compute_reward(self, obs, action, previous):
        if self.tick_outcome is None:
            raise RuntimeError('Reward requires an observed subsequent service tick')
        result = tick_reward(**self.tick_outcome, risk=obs['risk'], dt=0.001,
            switched=self.switched, ue_count=len(self.app.ledger.ue_ids),
            config=self.app.reward_config)
        self.switched = False
        self.tick_outcome = None
        return result


class xAppJointDeadlineDQN(xAppBase):
    def __init__(self, ric=None, *, enabled=False, config=TrainingConfig(),
                 reward_config=RewardConfig(), window_s=0.100, minimum_matured=20,
                 train=True, action_selector=None):
        super().__init__(ric)
        self.enabled = enabled
        self.config, self.reward_config = config, reward_config
        self.window_s, self.minimum_matured = window_s, minimum_matured
        self.train, self.action_selector = train, action_selector
        self.finished = False
        self.examples = deque(maxlen=24)
        self.first_examples = []
        self.services = deque(maxlen=512)
        self.timings_ns = deque(maxlen=4096)
        self.losses = deque(maxlen=512)
        self.episode_summaries = deque(maxlen=32)
        self.transition_count = self.completed_episodes = self.switch_count = 0
        self.reward_sum = 0.0
        self.digest = hashlib.sha256()

    def start(self):
        if not self.enabled:
            return
        if len(self.ric.xapp_list) != 1 or settings.RES_DQN_TRANSITION_MODE != 'causal':
            raise ValueError('Joint DQN requires sole-xApp deployment and causal mode')
        if settings.JAM_RESERVED_PRBS != 18 or settings.RES_DQN_TOTAL_RESILIENCY_PRB != 18:
            raise ValueError('Phase 2B requires the fixed 18-PRB reserve')
        if float(settings.SIM_STEP_TIME_DEFAULT) != 0.001 or self.reward_config.period_s != 0.010:
            raise ValueError('Joint DQN requires 1-ms ticks and 10-ms reward intervals')
        if int(settings.RES_DQN_DECISION_PERIOD) != 10:
            raise ValueError('Episode duration must use the 10-tick RIC interval')
        self.engine = self.ric.simulation_engine
        if len(self.engine.base_station_list) != 1 or len(self.engine.cell_list) != 1:
            raise ValueError('Joint DQN requires one gNB/cell')
        self.cell = next(iter(self.engine.cell_list.values()))
        self.physical = ResilientDQNEnv(self.ric, spawn_seed=self.config.seed)
        self.physical.reset()
        if self.physical.exhausted or not self.physical.ready():
            raise ValueError('Joint DQN requires a populated episode catalog')
        self.collector = UETelemetryCollector(self.cell, self.window_s)
        self.bridge = JointRICBridge(self.engine, self.cell, 10)
        self.agent = JointDQNAgent(self.config, self.reward_config,
            dict(window_s=self.window_s, minimum_matured=self.minimum_matured))
        self.initial_fingerprint = self.agent.fingerprint()
        self._before = self.engine.before_service_observer
        self.engine.before_service_observer = self.before_service
        self._begin_episode()

    def _begin_episode(self):
        if self.physical._episode_step_limit < 2:
            raise ValueError('Joint episodes need initialization plus a learned interval')
        self.collector.reset_episode()
        self.bridge.begin_episode(Mode.PROPORTIONAL, episode_token=self.physical._episode_idx)
        qm = self.cell.base_station.queue_measurements
        if qm is None:
            raise ValueError('Joint reward requires live queue measurements')
        ids = [uid for uid, u in self.cell.connected_ue_list.items() if u.slice_type == 'URLLC']
        self.ledger = FailureLedger(qm, ids)
        self.adapter = JointCausalAdapter(self)
        self.interval = CausalInterval(self.adapter, 10, encode_joint_state)
        self.origin = self.engine.sim_step
        self.last_embb_served = self.embb_served()
        self.bootstrap_counts = None
        self.commands_in_episode = 0

    def embb_served(self):
        qm = self.cell.base_station.queue_measurements
        return sum(qm.totals.get(uid, {}).get('served_bytes', 0)
                   for uid, u in self.cell.connected_ue_list.items() if u.slice_type == 'eMBB')

    def before_service(self, engine, dt):
        if self._before is not None:
            self._before(engine, dt)
        self.bridge.check_physics()
        if abs(dt-0.001) > 1e-12 or float(settings.SIM_STEP_TIME_DEFAULT) != 0.001:
            raise RuntimeError('Service tick changed during joint control')

    def step(self):
        if not self.enabled or self.finished:
            return
        if not self.engine.in_ric_callback:
            raise RuntimeError('Joint DQN requires post-service RIC callbacks')
        started = time.perf_counter_ns()
        step = self.engine.sim_step
        mode = self.cell.last_scheduler_stats['URLLC']['mode']
        service = dict(step=step, mode=mode, quotas=dict(self.cell.slice_dl_prb_quota),
                       baseline=dict(self.cell.base_dl_prb_quota),
                       effective_prbs=self.cell._effective_max_dl_prb)
        self.services.append(service)
        self.digest.update(json.dumps(dict(service=service, queues={
            uid: self.cell.base_station.get_dl_buf_bytes(uid)
            for uid in self.cell.connected_ue_list}), sort_keys=True).encode())
        if self.interval.pending is None:
            if step-self.origin < 10:
                self.ledger.sample(self.cell.base_station._time_s)
                return
            self.ledger.sample(self.cell.base_station._time_s)
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
            self.agent.remember(transition)
            loss = self.agent.learn() if self.train else None
            if loss is not None:
                self.losses.append(loss)
            self.transition_count += 1
            self.reward_sum += transition.reward
            example = dict(start_step=step-transition.info['interval_ticks'], end_step=step,
                state=transition.state.tolist(), action_index=transition.action_idx,
                action=asdict(ACTIONS[transition.action_idx]),
                reward=transition.reward, reward_parts=transition.info['reward_parts'],
                next_state=transition.next_state.tolist(), next_mask=transition.next_mask,
                done=bool(transition.done), receipt=asdict(self.adapter.receipt),
                next_state_metadata=state_metadata(transition.info['obs']),
                counts=dict(self.adapter.counts), terminal=self.adapter.terminal_summary)
            self.examples.append(example)
            if len(self.first_examples) < 3:
                self.first_examples.append(example)
            self.digest.update(json.dumps(example, sort_keys=True).encode())
            if transition.done:
                self.episode_summaries.append(dict(episode=self.physical._current_episode['id'],
                    terminal=self.adapter.terminal_summary, by_ue=self.ledger.by_ue,
                    total_known_failures=self.ledger.total_failures,
                    live_accounting=self._terminal_accounting(),
                    bootstrap_counts=self.bootstrap_counts))
                self.completed_episodes += 1
                self.ledger.close()
                self.physical.reset()
                if self.physical.exhausted:
                    self.finished = True
                    self.bridge.release()
                    self.timings_ns.append(time.perf_counter_ns()-started)
                    return
                self._begin_episode()
                self.timings_ns.append(time.perf_counter_ns()-started)
                return
        obs = self.adapter._collect_obs()
        state = encode_joint_state(obs, self.adapter.prev_action)
        mask = self.adapter._action_mask()
        index = (self.action_selector(state, mask, self.commands_in_episode)
                 if self.action_selector else self.agent.select(state, mask, explore=self.train))
        self.interval.begin(index)
        self.commands_in_episode += 1
        self.switch_count += int(self.adapter.switched)
        self.timings_ns.append(time.perf_counter_ns()-started)

    def _terminal_accounting(self):
        """Independent live sidecar check for reporting; never a reward input."""
        qm = self.cell.base_station.queue_measurements
        now = self.cell.base_station._time_s
        result = {}
        for uid in self.ledger.ue_ids:
            total = qm.totals[uid]
            overdue_nonoverflow = sum(
                b.overflow_bytes == 0 and now-b.arrival_s > b.budget_s+1e-12
                for b in qm.pending.get(uid, ()))
            known = (total['overflowed_batches']+total['fully_admitted_late_batches']
                     +overdue_nonoverflow)
            if known != self.ledger.by_ue[uid]['failures']:
                raise AssertionError('Event ledger disagrees with live failure union accounting')
            result[uid] = dict(offered=total['offered_batches'],
                overflowed=total['overflowed_batches'],
                fully_admitted_late=total['fully_admitted_late_batches'],
                pending_overdue_without_overflow=overdue_nonoverflow,
                independently_counted_known_failures=known)
        return result

    def result(self):
        return dict(transitions=self.transition_count, completed_episodes=self.completed_episodes,
            updates=self.agent.updates, replay_size=len(self.agent.replay),
            initial_parameter_sha256=self.initial_fingerprint,
            final_parameter_sha256=self.agent.fingerprint(),
            trajectory_sha256=self.digest.hexdigest(), reward_sum=self.reward_sum,
            switches=self.switch_count, losses=list(self.losses),
            first_examples=self.first_examples, last_examples=list(self.examples),
            service_tail=list(self.services), episode_summaries=list(self.episode_summaries),
            pending_interval_ticks=(self.interval.pending or {}).get('ticks', 0),
            boundary_timings_ns=list(self.timings_ns), device=str(self.agent.device))

    def close(self):
        if self.enabled:
            self.ledger.close()
            self.collector.close()
            self.bridge.release()
            self.engine.before_service_observer = self._before
            self.enabled = False
