"""Pretraining gate: physical-event conservation, causal intervals, frozen identity."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from network_layer.joint_reward import FailureLedger, RewardConfig, tick_reward
from network_layer.queue_measurements import QueueMeasurements
from network_layer.reward_ablation import (PerUEFailureLedger, event_fairness_cost,
    RewardArmAgent, RewardAblationAdapter, VERSIONS, COMPONENTS)
from network_layer.joint_dqn_agent import TrainingConfig
from network_layer.joint_dqn_schema import ACTIONS
from network_layer.joint_control import URLLCSchedulerMode as Mode

ROOT = Path(__file__).resolve().parents[2]


def ledger():
    qm = QueueMeasurements()
    for uid in ('tight','relaxed'):
        qm.set_budget(uid, .002 if uid=='tight' else .010)
    return qm, PerUEFailureLedger(qm, ['tight','relaxed'])


def test_partial_service_overdue_and_late_completion_count_once():
    qm,l = ledger()
    qm.arrival('tight',0.,100,100)
    qm.service('tight',.001,40)
    l.sample(.002)
    assert l.consume()['new_known_failures'] == 0  # Equality remains timely.
    l.sample(.003)
    assert l.consume()['new_known_failures'] == 1
    assert l.last_consumed_by_ue == {'relaxed':0,'tight':1}
    qm.service('tight',.004,60)
    l.sample(.005)
    assert l.consume()['new_known_failures'] == 0
    assert l.total_failures == 1 and not l.live
    l.close()


def test_overflow_expiry_completion_union_and_per_ue_conservation():
    qm,l = ledger()
    qm.arrival('tight',0.,100,40)  # Partial overflow is already a batch failure.
    qm.arrival('tight',0.,100,0)
    qm.arrival('relaxed',0.,10,10)
    assert l.consume()['new_known_failures'] == 2
    assert sum(l.last_consumed_by_ue.values()) == 2
    l.sample(.020)
    assert l.consume()['new_known_failures'] == 1
    assert l.last_consumed_by_ue['relaxed'] == 1
    qm.service('tight',.021,40)
    qm.service('relaxed',.021,10)
    assert l.consume()['new_known_failures'] == 0
    assert sum(v['failures'] for v in l.by_ue.values()) == l.total_failures == 3
    l.close()


def test_terminal_unresolved_does_not_enter_event_fairness():
    qm,l = ledger()
    qm.arrival('tight',0.,10,10)
    qm.arrival('relaxed',.099,10,10)
    terminal = l.terminal(.100)
    assert terminal['terminal_unresolved_charges'] == 1
    assert terminal['unresolved_batches'] == 2
    assert l.consume()['new_known_failures'] == 1
    assert l.last_consumed_by_ue == {'relaxed':0,'tight':1}
    assert event_fairness_cost(l.last_consumed_by_ue) == .025
    assert l.by_ue['relaxed']['terminal_charges'] == 1
    assert l.by_ue['relaxed']['failures'] == 0
    l.close()


def test_reset_and_bootstrap_deltas_do_not_leak():
    qm,l = ledger()
    qm.arrival('tight',0.,5,0)
    l.consume()  # Same consume used to discard the unpaid initialization interval.
    assert l.consume()['new_known_failures'] == 0
    assert not any(l.last_consumed_by_ue.values())
    l.close()
    qm2,l2 = ledger()
    assert l2.total_failures == 0 and l2.consume()['new_known_failures'] == 0
    l2.close()


def test_r2_uses_max_of_interval_sums_not_sum_of_tick_maxima_and_is_unclipped():
    assert event_fairness_cost({'a':10,'b':10}) == .25
    assert event_fairness_cost({'a':20,'b':1}) == .5
    assert event_fairness_cost({'a':5,'b':5}) == .125
    with pytest.raises(ValueError):
        event_fairness_cost({'a':-1})


def fake_adapter(arm):
    app = NS(method='P', initial_mode=Mode.PROPORTIONAL, reward_arm=arm, ledger=NS(ue_ids=('a','b'),last_consumed_by_ue={}),
        reward_config=RewardConfig(), engine=NS(sim_step=10), interval_sink=None)
    adapter = RewardAblationAdapter(app)
    adapter.service_ticks = 0
    from collections import Counter
    adapter.failures = Counter()
    adapter.parts = dict.fromkeys(COMPONENTS,0.)
    adapter.reference_parts = dict.fromkeys(COMPONENTS,0.)
    adapter.switch_in_interval = True
    adapter.valid_risk_ticks = adapter.saturated_risk_ticks = 0
    return adapter


@pytest.mark.parametrize('arm', ['R0','R1','R2'])
def test_exact_r0_all_components_interval_boundary_and_no_future_leakage(arm):
    a = fake_adapter(arm)
    risk = NS(values=(0,0,0,0,.8,0), valid=(False,False,False,False,True,False))
    action = next(x for x in ACTIONS if x.delta_embb_prb==x.delta_urllc_prb==0
                  and x.urllc_scheduler_mode is Mode.PROPORTIONAL)
    rewards = []
    # Failures alternate between UEs; only the full-interval max is appropriate.
    for i in range(10):
        counts = {'a':2 if i<5 else 0, 'b':2 if i>=5 else 0}
        a.app.ledger.last_consumed_by_ue = counts
        outcome = dict(counts=dict(new_known_failures=2,offered_batches=2),
                       terminal_charges=4 if i==9 else 0, embb_bytes=3210+i, reserve_prbs=9)
        a.tick_outcome = outcome
        a.switched = i==0
        a.app.engine.sim_step = 11+i
        old,parts = tick_reward(**outcome,risk=risk,dt=.001,switched=i==0,ue_count=2,config=RewardConfig())
        r,p = a._compute_reward({'risk':risk},action,action)
        if arm=='R0':
            assert r == old
            assert p['fairness_cost'] == parts['worst_cost']
        else:
            assert p['fairness_cost'] == (.25 if arm=='R2' and i==9 else 0.)
            for k in ('failure_cost','terminal_cost','embb_cost','reserve_cost','switch_cost'):
                assert p[k] == parts[k]
        rewards.append(r)
    assert a.failures == {'a':10,'b':10}
    assert a.interval_count == 1
    assert sum(rewards) == pytest.approx(-sum(a.parts.values()))
    assert a.parts['switch_cost'] == .01
    assert a.parts['terminal_cost'] == .2
    assert a.parts['fairness_cost'] == pytest.approx({'R0':.2,'R1':0.,'R2':.25}[arm])
    # A subsequent action starts fresh; a new boundary failure cannot charge backward.
    previous_parts = a.parts.copy()
    a.service_ticks = 0
    a.failures.clear()
    a.parts = dict.fromkeys(COMPONENTS,0.)
    a.app.ledger.last_consumed_by_ue = {'a':1,'b':0}
    a.tick_outcome = dict(outcome,counts=dict(new_known_failures=1,offered_batches=1),terminal_charges=0)
    a._compute_reward({'risk':risk},action,action)
    assert a.failures == {'a':1,'b':0}
    assert previous_parts['failure_cost'] == pytest.approx(1.)


def test_checkpoint_versions_reject_cross_arm_and_keep_network_identical(tmp_path):
    cfg = TrainingConfig(seed=12026)
    agents = [RewardArmAgent(cfg,arm=arm) for arm in VERSIONS]
    assert len({a.fingerprint() for a in agents}) == 1
    assert all(a.state_dimension==30 and a.action_count==98 for a in agents)
    for a in agents:
        assert a.metadata()['reward_version'] == VERSIONS[a.arm]
    path = tmp_path/'r2.pt'
    agents[2].save(path)
    agents[2].load(path)
    with pytest.raises(ValueError):
        agents[0].load(path)


def test_real_frozen_arms_same_physics_masks_and_r0_identity():
    # Physics settings are import-time values: configure in a fresh subprocess.
    import subprocess
    import sys
    import os
    code = r'''
from tools.phase2c1_pilot import configure_environment
configure_environment()
import json
from network_layer.reward_ablation import RewardArmAgent, event_fairness_cost, VERSIONS
from network_layer.joint_dqn_schema import ACTIONS
from network_layer.joint_control import URLLCSchedulerMode as Mode
from tools.phase2g_ablation import OUT, run, make_agent, entries
from tools.phase2c3_pilot import run as original_run
p = json.loads((OUT/'protocol.json').read_text())
entry = next(e for e in entries('validation') if e['regime']=='overloaded' and e['severity']==20)
class Fixed(RewardArmAgent):
    def select(self,state,mask,**kwargs):
        index = next(i for i,a in enumerate(ACTIONS) if a.delta_embb_prb==a.delta_urllc_prb==0
                     and a.urllc_scheduler_mode is Mode.PROPORTIONAL)
        assert len(state)==30 and len(mask)==98 and mask[index]
        return index
    def remember(self,*args):
        raise AssertionError('No training in correctness gate')
    learn = remember
observed = []
for arm in VERSIONS:
    agent = make_agent(12026,arm,p,Fixed)
    intervals = []
    row = run(entry,arm,agent,p,intervals=32,interval_sink=intervals.append)
    assert len(intervals) == 31
    assert row['known_failures'] > 0
    assert sum(v['new_failures'] for v in intervals) == sum(row['paid_failures_by_ue'].values())
    assert all(sum(v['failures_by_ue'].values())==v['new_failures'] for v in intervals)
    if arm=='R2':
        assert all(v['reward_components']['fairness_cost']==event_fairness_cost(v['failures_by_ue']) for v in intervals)
    if arm=='R0':
        original = original_run(entry,'A4',agent,p,intervals=32)
        assert row['return_value']==original['return_value']
        assert row['control_digest']==original['control_digest']
    observed.append(row)
assert all(r['physical']['per_ue']==observed[0]['physical']['per_ue'] for r in observed)
assert all(r['control_digest']==observed[0]['control_digest'] for r in observed)
assert all(r['updates']==0 for r in observed)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, env=dict(os.environ, PYTHONPATH='backend'))
    assert result.returncode == 0, result.stdout+result.stderr
