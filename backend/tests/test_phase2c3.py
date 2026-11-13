"""Matched masks, atomic causal composition, learner isolation and real replay."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from network_layer.joint_control import URLLCSchedulerMode as Mode
from network_layer.joint_dqn_agent import JointDQNAgent, TrainingConfig
from network_layer.joint_dqn_schema import ACTIONS, joint_action_mask
from network_layer.matched_control import MatchedAdapter, restricted_mask
from network_layer.pilot_control import ReactiveReserve
from tools.phase2c3_pilot import OUT, manifest, run, selection_key, verify
from test_phase2b import bridge_fixture


def test_frozen_balanced_training_and_manifest_isolation():
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    entries={e['id']:e for e in manifest('training')}
    assert len(p['network_seeds'])==2
    for start in (0,3,6):
        block=[entries[s] for s in p['training_order'][start:start+3]]
        assert {e['regime'] for e in block}=={'feasible','contested','overloaded'}
        assert {e['severity'] for e in block}=={0,10,20}
    assert all('holdout' not in name for name in p['input_sha256'])
    with pytest.raises(ValueError):manifest('future_holdout')


@pytest.mark.parametrize('method', ['A1','A2','A3','A4'])
def test_mask_is_intersection_with_physics(monkeypatch,method):
    b=bridge_fixture(monkeypatch,extra=(9,9));physical=joint_action_mask(b)
    mask=restricted_mask(method,physical,(0,0))
    assert np.all(~mask | physical)
    for index in np.flatnonzero(mask):
        a=ACTIONS[index]
        if method=='A1':assert a.urllc_scheduler_mode is Mode.PROPORTIONAL
        if method=='A2':assert a.urllc_scheduler_mode is Mode.FEASIBILITY_AWARE
        if method=='A3':assert (a.delta_embb_prb,a.delta_urllc_prb)==(0,0)
    assert mask.any()


def test_a3_mask_recomputes_next_command_and_commits_atomically(monkeypatch):
    b=bridge_fixture(monkeypatch,effective=200)
    env=NS(_target_cell=lambda:b.cell,_collect_obs=lambda:{'embb_prb':b.cell.slice_dl_prb_quota['eMBB'],
        'urllc_prb':b.cell.slice_dl_prb_quota['URLLC'],
        'embb_base_prb':b.cell.base_dl_prb_quota['eMBB'],
        'urllc_base_prb':b.cell.base_dl_prb_quota['URLLC'],
        'effective_prb_ratio':b.cell._effective_max_dl_prb/200,
        'reserve_used_prb':sum(b.cell.slice_dl_prb_quota.values())-sum(b.cell.base_dl_prb_quota.values())})
    # Exact reference computation; changes below are current observed baselines.
    app=NS(method='P',ablation='A3',initial_mode=Mode.PROPORTIONAL,
           bridge=b,physical=env,reactive=ReactiveReserve(env))
    adapter=MatchedAdapter(app)
    nojam=adapter._action_mask();assert nojam[48] and nojam[49] and sum(nojam)==2
    b.cell.base_dl_prb_quota=dict(eMBB=126,URLLC=54)
    b.cell.slice_dl_prb_quota=dict(b.cell.base_dl_prb_quota)
    b.cell._effective_max_dl_prb=180
    mask=adapter._action_mask();assert not np.array_equal(mask,nojam)
    indices=np.flatnonzero(mask);assert len(indices)==2
    a=ACTIONS[indices[1]];before=dict(b.cell.slice_dl_prb_quota)
    adapter._apply_action(a)
    assert b.cell.slice_dl_prb_quota==dict(eMBB=before['eMBB']+a.delta_embb_prb,
        URLLC=before['URLLC']+a.delta_urllc_prb)
    assert b.cell.urllc_scheduler_mode is a.urllc_scheduler_mode
    assert adapter.receipt.first_service_step==11
    b.engine.sim_step+=10
    nxt=adapter._action_mask()
    assert sum(nxt)==2 and not np.array_equal(mask,nxt)


def _assert_real_learning_and_replay():
    from tools.phase2c1_pilot import assert_paired
    p=json.loads((OUT/'protocol.json').read_text())
    entry=next(e for e in manifest('training') if e['regime']=='contested' and e['severity']==20)
    rows=[]
    for method in ('A1','A2','A3','A4'):
        class SpyAgent(JointDQNAgent):
            def remember(self, transition):
                assert transition.info['interval_ticks']==10
                super().remember(transition)
        agent=SpyAgent(TrainingConfig(seed=71,batch_size=4))
        before=agent.fingerprint()
        r=run(entry,method,agent,p,train=True,intervals=12)
        assert agent.fingerprint()!=before and agent.interactions==11 and agent.updates==8
        assert r['return_objective']=='joint-reward-v1'
        assert r['return_value']==pytest.approx(r['common_joint_monitor_return'])
        assert r['transitions']==11 and len(agent.replay)==11
        for state,action,reward,next_state,done,mask in agent.replay:
            assert state.shape==next_state.shape==(30,) and len(mask)==98
            if method=='A1':assert action%2==0 and not any(mask[1::2])
            if method=='A2':assert action%2==1 and not any(mask[::2])
            if method=='A3':assert sum(mask)==2
        # Terminal transition is included, no action/reward crosses resets.
        assert agent.replay[-1][4] and not agent.replay[0][4]
        params=agent.fingerprint();rng=agent.rng.getstate();updates=agent.updates
        a=run(entry,method,agent,p,intervals=12)
        b=run(entry,method,agent,p,intervals=12)
        assert a['control_digest']==b['control_digest']
        assert agent.fingerprint()==params and agent.updates==updates and len(agent.replay)==11
        assert agent.rng.getstate()==rng
        rows.append(r)
    assert assert_paired(rows)==1
    assert not torch.cuda.is_initialized()


def test_actual_matched_learning_and_evaluation_are_paired_and_causal():
    import os
    import subprocess
    import sys
    code = "from tools.phase2c1_pilot import configure_environment; configure_environment(); import sys; sys.path.insert(0, 'backend/tests'); from test_phase2c3 import _assert_real_learning_and_replay; _assert_real_learning_and_replay()"
    env=dict(os.environ,PYTHONPATH='backend')
    result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_fairness_selection_is_prespecified_worst_first(monkeypatch):
    from fractions import Fraction
    import network_layer.workload_feasibility as feasibility
    values=(Fraction(1,10),Fraction(4,10),Fraction(2,10),Fraction(0),Fraction(1,2),Fraction(0),3)
    monkeypatch.setattr(feasibility,'checkpoint_key',lambda rows,episode:values)
    assert selection_key([],3)==values
    assert selection_key([],3,True)==(values[1],values[0],*values[2:])


def test_same_initial_network_and_reward_for_every_ablation():
    # Action restrictions are outside the agent: no separate learner variant.
    from network_layer.matched_control import xAppMatchedPilot
    agents=[JointDQNAgent(TrainingConfig(seed=12026)) for _ in range(4)]
    assert len({a.fingerprint() for a in agents})==1
    assert all(a.metadata()==agents[0].metadata() for a in agents)
    assert agents[0].loss_function(torch.tensor([3.]),torch.tensor([0.])).item()==2.5
    assert agents[0].gradient_clip==10.
    with pytest.raises(ValueError):xAppMatchedPilot(method='B1')
