"""Nonlearning experimental allocator validation; existing P/F remain unchanged."""
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from network_layer.deadline_scheduler import QueueView,deadline_allocation
from network_layer.feasibility_gated_fairness import ServiceWindow,reachable_target,ExperimentalFairAllocator
from network_layer.queue_measurements import Batch
from network_layer.deadline_risk import encode_deadline_risk
from network_layer.joint_dqn_schema import encode_joint_state
from network_layer.joint_control import JointRICAction,URLLCSchedulerMode as Mode

ROOT=Path(__file__).resolve().parents[2]


def batch(arrival,n,overflow=0):return Batch(1,arrival,n,n,.01,overflow)


def test_window_uses_actual_offered_and_served_bounded_history_and_reset():
    w=ServiceWindow()
    for step in range(1,1001):w.observe('u',step*.001,step*100,step*60,step*40)
    r=w.view('u')
    assert r['full'] and r['deficit']==pytest.approx(.4)
    assert r['queue_growth']==4000 and len(w.history['u'])<=102
    assert ServiceWindow().view('u')['deficit']==0
    with pytest.raises(ValueError):w.observe('u',1.,100001,60001,40000)
    with pytest.raises(ValueError):w.observe('u',1.001,1,1,1)


def test_target_gates_past_failures_but_can_clear_prefix_for_timely_work():
    assert reachable_target([batch(0,20)],10,.01,.020,.001,2) is None
    p=[batch(0,10),batch(.019,10)]
    target=reachable_target(p,10,.01,.020,.001,2)
    assert target['prefix_bytes']==20 and target['need_prbs']==2
    partial=batch(.019,100);partial.remaining_bytes=3
    assert reachable_target([partial],10,.01,.020,.001,2)['prefix_bytes']==3
    # Timely target hidden behind an unattainable prefix cannot attract credit.
    assert reachable_target([batch(0,1000),batch(.019,10)],10,.01,.020,.001,2) is None
    assert reachable_target([batch(.019,10,1)],10,.01,.020,.001,2) is None
    assert reachable_target(p,0,.01,.020,.001,2) is None
    assert reachable_target([batch(.010,20)],10,.01,.020,.001,2)['slots']==1
    assert reachable_target([batch(.010,21)],10,.01,.020,.001,2) is None


def make_allocator(fraction,pending,offered=100,served=50):
    qm=NS(pending=pending)
    cell=NS(base_station=NS(simulation_engine=NS(sim_step=101),queue_measurements=qm))
    window=NS(matured_offered_batches=100,matured_failed_batches=100,full_window=True)
    report=NS(ues=[NS(ue_id=u,window=window) for u in pending])
    a=ExperimentalFairAllocator(cell,NS(collect=lambda step:report),fraction)
    for u in pending:
        for i in range(1,101):a.window.observe(u,i*.001,i*offered,i*served,i*(offered-served))
    return a


def test_failures_alone_do_not_earn_protection_and_expired_only_is_gated():
    p={'old':[batch(0,10)],'new':[batch(.099,10)]}
    q={u:QueueView(10,10,.01,[(b.arrival_s,b.remaining_bytes) for b in bs]) for u,bs in p.items()}
    a=make_allocator(1.,p,served=100)
    a(q,2,.101,.001,{},feasibility=True)
    assert all(r['credit_bytes']==0 for r in a.last_protection.values())
    a=make_allocator(1.,p)
    a(q,2,.101,.001,{},feasibility=True)
    assert a.last_protection['old']['credit_bytes']==0
    assert a.last_protection['new']['credit_bytes']>0


@pytest.mark.parametrize('fraction',[0,.25,.5,1.])
def test_integer_useful_caps_work_conservation_determinism_and_lambda_zero(fraction):
    for quota in range(7):
        p={'a':[batch(.099,13)],'b':[batch(.099,27)]}
        q={u:QueueView(sum(b.remaining_bytes for b in bs),10,.01,[(b.arrival_s,b.remaining_bytes) for b in bs]) for u,bs in p.items()}
        a=make_allocator(fraction,p)
        result=a(q,quota,.101,.001,{},per_ue_cap=2,feasibility=True)
        assert sum(result.values())==min(quota,4)
        assert all(isinstance(v,int) and 0<=v<=2 for v in result.values())
        assert result==a(q,quota,.101,.001,{},per_ue_cap=2,feasibility=True)
        if fraction==0:
            assert result==deadline_allocation(q,quota,.101,.001,{},per_ue_cap=2,feasibility=True)
    # Existing starvation protection is still served before the fair portion.
    q={'old':QueueView(20,10,.001,[(0.,20)]),'new':QueueView(20,10,.01,[(.099,20)])}
    a=make_allocator(fraction,{'old':[batch(0,20)],'new':[batch(.099,20)]})
    assert a(q,1,.101,.001,{'old':3},starvation_limit=3,feasibility=True)['old']==1


def test_current_encoder_collision_omits_per_ue_service_deficits():
    # Same snapshot and aggregate traffic; swap the failed UE inside one class.
    # The worst risk and class risk cannot identify which UE needs repayment.
    def ue(uid,failed):
        return NS(ue_id=uid,slice_id='URLLC',budget_s=.01,budget_class='relaxed',queued_bytes=100,
            hol_wait_s=.005,urllc_scheduler_mode=Mode.FEASIBILITY_AWARE,
            window=NS(history_available=True,full_window=True,matured_offered_batches=100,matured_failed_batches=failed))
    a=NS(ues=[ue('u0',100),ue('u1',0)])
    b=NS(ues=[ue('u0',0),ue('u1',100)])
    state=lambda r:encode_joint_state(dict(legacy={},report=r,risk=encode_deadline_risk(r),minimum_matured=20),JointRICAction(0,0,Mode.FEASIBILITY_AWARE))
    np.testing.assert_array_equal(state(a),state(b))
    assert state(a).shape==(30,)


def test_real_nonlearning_replay_and_lambda_zero_compatibility(tmp_path):
    code=r'''
import contextlib,json,sys
from functools import partial
from pathlib import Path
from tools.phase2c1_pilot import configure_environment
configure_environment()
from tools.phase2d_study import Streams,make_classes,validation_entries,C2,C3
from tools.phase2c1_pilot import run_episode
from tools.phase2c2_design import resolve_catalog
from network_layer.joint_dqn_agent import JointDQNAgent
JointDQNAgent.learn=lambda *a:(_ for _ in ()).throw(AssertionError('learning forbidden'))
Matched,Reference=make_classes()
p=json.loads((C3/'protocol.json').read_text());p.update(network_seed=0,training_scenario_seeds=[0,1,2],severities=[0,10,20])
e=next(e for e in validation_entries() if e['regime']=='contested' and e['severity']==20)
streams=Streams(Path(sys.argv[1]));rows=[]
for spec in ('F','G0','OLD035','G025','G05','G1'):
 identity=dict(run_id=spec,policy=spec,network_seed=0,episode=0,selection='unit',scenario_id=e['id'],regime=e['regime'],severity=20,replay=False)
 with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
  r=run_episode('B2',20,e['seed'],p,agent=None,train=False,intervals=15,catalog_override=resolve_catalog(C2,e),app_class=partial(Reference,identity=identity,streams=streams,spec=spec))
 rows.append(r)
assert rows[0]['control_digest']==rows[1]['control_digest']
assert rows[0]['physical']['overall']==rows[1]['physical']['overall']
assert len({r['physical']['quota_sha256'] for r in rows})==1
assert len({r['physical']['exogenous_sha256'] for r in rows})==1
assert all(r['updates']==0 for r in rows)
streams.close()
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,env=dict(os.environ,PYTHONPATH='backend'),capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_saved_checkpoint_forward_replay_is_noninterfering(tmp_path):
    code=r'''
import contextlib,json,sys
from functools import partial
from pathlib import Path
from tools.phase2c1_pilot import configure_environment
configure_environment()
from tools.phase2d_study import Streams,make_classes,validation_entries,C2,C3
from tools.phase2c1_pilot import run_episode
from tools.phase2c2_design import resolve_catalog
from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
JointDQNAgent.learn=lambda *a:(_ for _ in ()).throw(AssertionError('learning forbidden'))
JointDQNAgent.remember=JointDQNAgent.learn
Matched,Reference=make_classes()
p=json.loads((C3/'protocol.json').read_text());p.update(network_seed=12026,training_scenario_seeds=[0,1,2],severities=[0,10,20])
a=JointDQNAgent(TrainingConfig(seed=12026,epsilon=1.,**p['training_config']))
a.load(C3/'checkpoints/A4_seed12026_ep6.pt');fingerprint=a.fingerprint()
e=next(e for e in validation_entries() if e['regime']=='contested' and e['severity']==20)
streams=Streams(Path(sys.argv[1]))
identity=dict(run_id='saved',policy='A4',network_seed=12026,episode=6,selection='unit',scenario_id=e['id'],regime=e['regime'],severity=20,replay=True)
with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
 observed=run_episode('A4',20,e['seed'],p,agent=a,train=False,intervals=15,catalog_override=resolve_catalog(C2,e),app_class=partial(Matched,identity=identity,streams=streams))
 from network_layer.matched_control import xAppMatchedPilot
 original=run_episode('A4',20,e['seed'],p,agent=a,train=False,intervals=15,catalog_override=resolve_catalog(C2,e),app_class=xAppMatchedPilot)
assert observed['control_digest']==original['control_digest']
assert observed['physical']['overall']==original['physical']['overall']
assert a.fingerprint()==fingerprint and not a.replay
streams.close()
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,env=dict(os.environ,PYTHONPATH='backend'),capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
