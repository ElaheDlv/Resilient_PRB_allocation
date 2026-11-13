"""Bounded joint state/action/reward, masked TD, real causal learning and compatibility."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import settings

from network_layer.deadline_risk import DeadlineRiskFeatures
from network_layer.joint_control import JointRICAction, JointRICBridge, URLLCSchedulerMode as Mode
from network_layer.joint_dqn_agent import JointDQNAgent, TrainingConfig, masked_td_targets
from network_layer.joint_dqn_schema import ACTIONS, decode_action, encode_action, encode_joint_state, joint_action_mask
from network_layer.joint_reward import FailureLedger, RewardConfig, tick_reward
from network_layer.queue_measurements import QueueMeasurements
from network_layer.resilient_interval import Transition
from network_layer.xApps.xapp_resilient_dqn_allocator import build_state

ROOT = Path(__file__).resolve().parents[2]


def risk_fixture(valid=True):
    return DeadlineRiskFeatures((2., .5, .25 if valid else None, .5 if valid else None,
                                 .75 if valid else None, 1.),
                                (True, True, valid, valid, valid, True),
                                ('valid',)*6, (40, 40) if valid else (None, None))


def bridge_fixture(monkeypatch, extra=(0, 0), effective=180):
    monkeypatch.setattr(settings, 'JAM_RESERVED_PRBS', 18)
    monkeypatch.setattr(settings, 'RES_DQN_TOTAL_RESILIENCY_PRB', 18)
    class Cell(NS):
        @property
        def urllc_scheduler_mode(self):
            return self._joint_urllc_mode
    base = dict(eMBB=126, URLLC=54) if effective == 180 else dict(eMBB=140, URLLC=60)
    cell = Cell(max_dl_prb=218, _effective_max_dl_prb=effective,
        base_dl_prb_quota=base, slice_dl_prb_quota={k:v+d for (k,v),d in zip(base.items(), extra)},
        _joint_urllc_mode=None, connected_ue_list={'u':NS(slice_type='URLLC', queue_budget_s=.002)})
    engine = NS(cell_list={'c':cell}, sim_step=0, in_ric_callback=False)
    bridge = JointRICBridge(engine, cell)
    bridge.begin_episode(episode_token=0)
    engine.sim_step = 10
    engine.in_ric_callback = True
    return bridge


def test_all_98_action_indices_roundtrip_and_invalid_indices():
    assert len(ACTIONS) == len(set(ACTIONS)) == 98
    for e in range(7):
        for u in range(7):
            for mode in range(2):
                i = ((e*7)+u)*2+mode
                assert encode_action(decode_action(i)) == i
                assert decode_action(i).urllc_scheduler_mode == (Mode.PROPORTIONAL, Mode.FEASIBILITY_AWARE)[mode]
    for invalid in (-1, 98, True, .5):
        with pytest.raises(ValueError): decode_action(invalid)
    with pytest.raises(ValueError): encode_action(JointRICAction(2, 0, Mode.PROPORTIONAL))


@pytest.mark.parametrize('effective,extra', [(200,(0,0)),(180,(0,0)),(180,(9,9)),(200,(9,9))])
def test_joint_mask_matches_atomic_execution_all_candidates(monkeypatch, effective, extra):
    b = bridge_fixture(monkeypatch, extra, effective)
    mask = joint_action_mask(b)
    before = dict(b.cell.slice_dl_prb_quota)
    for index, action in enumerate(ACTIONS):
        b.cell.slice_dl_prb_quota = dict(before)
        b.cell._joint_urllc_mode = Mode.PROPORTIONAL
        b.last_command_step = None
        if mask[index]:
            receipt = b.apply(action)
            assert 0 <= receipt.activated_reserve_prbs <= 18
            assert b.cell.urllc_scheduler_mode == action.urllc_scheduler_mode
        else:
            with pytest.raises((ValueError, AssertionError)): b.apply(action)
            assert b.cell.slice_dl_prb_quota == before
            assert b.cell.urllc_scheduler_mode == Mode.PROPORTIONAL


def test_invalid_budget_masks_only_feasibility_mode(monkeypatch):
    b = bridge_fixture(monkeypatch)
    b.cell.connected_ue_list['u'].queue_budget_s = None
    mask = joint_action_mask(b)
    assert any(mask) and not any(mask[1::2]) and any(mask[::2])


def test_30_state_features_validity_exposures_and_legacy_compatibility():
    previous = JointRICAction(3,-1,Mode.PROPORTIONAL)
    report = NS(ues=[NS(slice_id='URLLC', budget_class=c) for c in ('tight','relaxed')])
    obs = dict(legacy={'embb_prb':140}, risk=risk_fixture(), report=report, minimum_matured=20)
    s = encode_joint_state(obs,previous)
    legacy = build_state(obs['legacy'], (3,-1))
    assert legacy.shape == (15,) and s.shape == (30,)
    np.testing.assert_array_equal(s[:14], np.delete(legacy, 10))
    assert s[14] == pytest.approx(2/3) and s[19] == 1
    np.testing.assert_array_equal(s[20:26], np.ones(6))
    obs['risk'] = risk_fixture(False)
    warmup = encode_joint_state(obs,previous)
    assert np.isfinite(warmup).all()
    np.testing.assert_array_equal(warmup[16:19], np.zeros(3))
    np.testing.assert_array_equal(warmup[22:25], np.zeros(3))
    np.testing.assert_array_equal(warmup[26:], np.zeros(4))
    # Known zero failure is a distinct neural observation from unavailable history.
    obs['risk'] = replace(risk_fixture(), values=(1.,0.,0.,0.,0.,0.))
    measured = encode_joint_state(obs,previous)
    assert measured[14] == .5 and measured[16] == 0 and measured[22] == 1
    assert not np.array_equal(warmup, measured)


def test_exactly_once_failure_ledger_partial_overflow_expiry_and_terminal():
    qm = QueueMeasurements(); qm.set_budget('u', .002)
    ledger = FailureLedger(qm, ['u'])
    qm.arrival('u', .001, 10, 5)
    qm.service('u', .002, 2)
    ledger.sample(.003)  # equality boundary, overflow already failed
    assert ledger.consume()['new_known_failures'] == 1
    ledger.sample(.004)
    qm.service('u', .005, 3)
    assert ledger.consume()['new_known_failures'] == 0  # late+overflow union
    qm.arrival('u', .006, 10, 10)
    ledger.sample(.008)
    assert ledger.consume()['new_known_failures'] == 0
    ledger.sample(.009)
    assert ledger.consume()['new_known_failures'] == 1
    ledger.sample(.010)
    assert ledger.consume()['new_known_failures'] == 0
    qm.arrival('u', .010, 10, 10)
    qm.arrival('u', .010, 10, 0)  # full drop, no live record
    end = ledger.terminal(.011)
    assert end == dict(unresolved_batches=2, unresolved_bytes=20, terminal_unresolved_charges=1)
    assert ledger.consume()['new_known_failures'] == 1
    assert ledger.total_failures == 3 and len(ledger.live) == 2
    assert ledger.by_ue['u'] == dict(offered=4, failures=3, terminal_charges=1)
    with pytest.raises(RuntimeError): ledger.sample(.012)
    ledger.close(); assert not qm._listeners


def test_late_completion_without_intermediate_sample_and_bounded_live_records():
    qm=QueueMeasurements(); qm.set_budget('u', .002)
    ledger=FailureLedger(qm,['u'])
    for i in range(300):
        t=i*.01
        qm.arrival('u',t,1,1);qm.service('u',t+.003,1)
    assert ledger.total_failures==300 and not ledger.live
    assert ledger.consume()['new_known_failures']==300
    assert ledger.terminal(3.)['terminal_unresolved_charges']==0
    ledger.close()


def test_reward_equations_normalization_fairness_and_switch_once():
    reward, p = tick_reward(counts=dict(new_known_failures=8), terminal_charges=4,
        risk=risk_fixture(), embb_bytes=6000, dt=.001, reserve_prbs=9,
        switched=True, ue_count=8, config=RewardConfig())
    assert p == pytest.approx(dict(failure_cost=.1, terminal_cost=.05,
        worst_cost=.01875, embb_cost=.005, reserve_cost=.001, switch_cost=.01))
    assert reward == pytest.approx(-sum(p.values()))
    # Equal universal failure still has maximum worst-risk cost, never a fairness bonus.
    _, bad = tick_reward(counts=dict(new_known_failures=0), terminal_charges=0,
        risk=replace(risk_fixture(),values=(2.,1.,1.,1.,1.,1.)), embb_bytes=12000,
        dt=.001, reserve_prbs=0, switched=False, ue_count=8, config=RewardConfig())
    assert bad['worst_cost'] == pytest.approx(.025)


def test_masked_td_targets_include_terminal_all_false_and_reject_nonterminal():
    q=torch.tensor([[1.,999.,3.],[999.,999.,999.]])
    masks=torch.tensor([[True,False,True],[False,False,False]])
    got=masked_td_targets(torch.tensor([2.,4.]), torch.tensor([False,True]),q,masks,.5)
    torch.testing.assert_close(got,torch.tensor([3.5,4.]))
    with pytest.raises(ValueError):
        masked_td_targets(torch.zeros(2),torch.zeros(2,dtype=torch.bool),q,masks,.5)


def test_actual_updates_bounded_replay_checkpoint_and_incompatible_rejection(tmp_path):
    cfg=TrainingConfig(seed=9,batch_size=2,replay_capacity=4)
    agent=JointDQNAgent(cfg)
    initial=agent.fingerprint()
    for i in range(8):
        s=np.ones(30,dtype=np.float32)*i/8
        agent.remember(Transition(s,48,-1.,s+.01,False,[True]*98,{}))
    assert len(agent.replay)==4
    assert np.isfinite(agent.learn()) and agent.fingerprint()!=initial
    path=tmp_path/'joint.pt';agent.save(path,run_metadata={'steps':8,'trace_sha256':'test'})
    other=JointDQNAgent(cfg);other.load(path)
    assert other.fingerprint()==agent.fingerprint() and other.updates==1
    assert other.checkpoint_run_metadata=={'steps':8,'trace_sha256':'test'}
    with pytest.raises(ValueError):JointDQNAgent(replace(cfg,seed=10)).load(path)
    legacy=tmp_path/'legacy.pt';torch.save(agent.policy.state_dict(),legacy)
    with pytest.raises(ValueError):other.load(legacy)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    for field in ('state_schema','action_schema','reward_version','observation_config'):
        bad=copy.deepcopy(payload);bad['metadata'][field]='incompatible'
        corrupt=tmp_path/'wrong-schema.pt';torch.save(bad,corrupt)
        with pytest.raises(ValueError):other.load(corrupt)
    assert agent.device.type=='cpu' and all(p.device.type=='cpu' for p in agent.policy.parameters())


def run_smoke(tmp_path, config, tag, *, intervals=8, episodes=1):
    output=tmp_path/f'{tag}.json';checkpoint=tmp_path/f'{tag}.pt'
    run=subprocess.run([sys.executable,str(ROOT/'backend/tools/phase2b_smoke.py'),
        '--config',str(config),'--intervals',str(intervals),'--episodes',str(episodes),
        '--scripted','--output',str(output),'--checkpoint',str(checkpoint)],
        cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0, run.stdout+run.stderr
    return json.loads(output.read_text())


@pytest.mark.parametrize('severity',[0,10,20])
def test_real_causal_learning_jam_switching_and_replay(tmp_path,severity):
    catalog=json.loads((ROOT/f'backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json').read_text())
    catalog['episodes'][0]['jam_var']['period_steps']=20  # test-only short ON/OFF cycle
    config=tmp_path/'catalog.json';config.write_text(json.dumps(catalog))
    a=run_smoke(tmp_path,config,'a',intervals=20)
    b=run_smoke(tmp_path,config,'b',intervals=20)
    assert a['trajectory_sha256']==b['trajectory_sha256']
    assert a['final_parameter_sha256']==b['final_parameter_sha256']
    assert a['losses']==b['losses'] and a['parameters_changed'] and a['updates']==4
    assert a['transitions']==19 and a['completed_episodes']==1 and a['checkpoint_roundtrip']
    assert not a['cuda_initialized'] and a['device']=='cpu'
    services={s['step']:s for s in a['service_tail']}
    for t in a['last_examples']:
        assert t['end_step']-t['start_step']==10
        assert t['receipt']['applied_after_step']==t['start_step']
        assert t['receipt']['first_service_step']==t['start_step']+1
        expected='proportional_wc' if t['action']['urllc_scheduler_mode']=='PROPORTIONAL' else 'deadline_feasible'
        assert all(services[k]['mode']==expected for k in range(t['start_step']+1,t['end_step']+1))
        assert len(t['next_mask'])==98 and any(t['next_mask'])
        assert np.isfinite(t['state']).all() and np.isfinite(t['next_state']).all()
    assert a['last_examples'][-1]['done'] and a['last_examples'][-1]['terminal'] is not None
    assert len({s['effective_prbs'] for s in services.values()})==(1 if severity==0 else 2)


def test_episode_resets_no_leaked_exposure_or_pending_transition(tmp_path):
    config=ROOT/'backend/evaluations/phase1/main_scenarios/scenario_j0.json'
    a=run_smoke(tmp_path,config,'two',intervals=4,episodes=2)
    assert a['completed_episodes']==2 and a['transitions']==6 and a['steps']==80
    assert a['pending_interval_ticks']==0
    assert sum(t['done'] for t in a['last_examples'])==2
    reset=next(t for t in a['last_examples'] if t['start_step']==50)
    assert reset['state'][19]==0  # P initial mode before first command
    assert reset['state'][22:25]==[0.,0.,0.]
    assert reset['next_state_metadata']['risk']['valid'][2:5]==[False]*3
