"""Protocol leakage guards, baseline equivalence and bounded paired training checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from network_layer.deadline_risk import DeadlineRiskFeatures
from network_layer.joint_dqn_agent import JointDQNAgent, TrainingConfig
from network_layer.pilot_control import HystereticModeRule, LegacyPilotAgent, legacy_pilot_state
from network_layer.joint_control import JointRICAction, URLLCSchedulerMode as Mode
from network_layer.xApps.xapp_resilient_dqn_allocator import build_state, _DQN
from tools.phase2c1_pilot import validate_protocol, epsilon_schedule, assert_paired

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'backend/evaluations/phase2c1/protocol.json'


def test_disjoint_protocol_and_fixed_budget_exploration():
    p=json.loads(PROTOCOL.read_text());validate_protocol(p)
    assert epsilon_schedule(p,0)==1.
    assert epsilon_schedule(p,2691)==pytest.approx(.05)
    assert epsilon_schedule(p,1000)<epsilon_schedule(p,500)
    p['test_scenario_seeds'].append(p['training_scenario_seeds'][0])
    with pytest.raises(ValueError):validate_protocol(p)


def risk(pressure, valid=True):
    return DeadlineRiskFeatures((pressure,None,None,None,None,None),
        (valid,False,False,False,False,False),('test',)*6,(None,None))


def test_hysteretic_rule_causality_equality_missing_and_hold():
    r=HystereticModeRule(1.)
    assert r.select(risk(1.)) is Mode.PROPORTIONAL
    assert r.select(risk(1.001)) is Mode.FEASIBILITY_AWARE
    assert r.select(risk(.5)) is Mode.FEASIBILITY_AWARE
    assert r.select(risk(.5,False)) is Mode.FEASIBILITY_AWARE
    assert r.select(risk(.5)) is Mode.FEASIBILITY_AWARE
    assert r.select(risk(.5)) is Mode.PROPORTIONAL


def test_original_network_encoder_and_loss_are_retained():
    cfg=TrainingConfig(seed=123,batch_size=2,replay_capacity=4)
    agent=LegacyPilotAgent(cfg)
    previous=JointRICAction(3,-1,Mode.FEASIBILITY_AWARE)
    obs=dict(embb_demand_prb=22,urllc_latency=.003)
    np.testing.assert_array_equal(legacy_pilot_state({'legacy':obs},previous),build_state(obs,(3,-1)))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed);reference=_DQN(15,49)
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(value,agent.policy.state_dict()[key],rtol=0,atol=0)
    assert agent.loss_function(torch.tensor([3.]),torch.tensor([0.])).item()==9
    assert agent.gradient_clip is None
    assert agent.select(np.zeros(15,dtype=np.float32),[False]*48+[True],epsilon=1.)==48
    assert agent.select(np.zeros(15,dtype=np.float32),[False]*48+[True],epsilon=0.)==48


def test_explicit_epsilon_does_not_mutate_schema_and_checkpoint(tmp_path):
    agent=JointDQNAgent(TrainingConfig(seed=13))
    metadata=agent.metadata()
    state=np.zeros(30,dtype=np.float32);mask=[False]*97+[True]
    assert agent.select(state,mask,epsilon=1.)==97
    assert agent.select(state,mask,epsilon=0.)==97
    assert agent.metadata()==metadata
    path=tmp_path/'v1.pt';agent.save(path)
    other=JointDQNAgent(TrainingConfig(seed=13));other.load(path)
    assert agent.fingerprint()==other.fingerprint()
    with pytest.raises(ValueError):agent.select(state,mask,epsilon=1.1)


def test_pairing_rejects_exogenous_and_budget_mismatch():
    r=dict(scenario_seed=1,severity=20,physical={'exogenous_sha256':'a'},deadline_assignment_sha256='x')
    assert assert_paired([r,dict(r)])==1
    with pytest.raises(AssertionError):assert_paired([r,dict(r,deadline_assignment_sha256='y')])
    with pytest.raises(AssertionError):assert_paired([r,dict(r,physical={'exogenous_sha256':'b'})])


def test_actual_sequential_controllers_training_and_reset_pairing(tmp_path):
    run=subprocess.run([sys.executable,str(ROOT/'backend/tools/phase2c1_pilot.py'),
        '--stage','selftest','--output',str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads((tmp_path/'selftest.json').read_text())
    assert report['paired'] and report['repeat'] and report['controllers']==7


def test_watchdog_stops_own_small_worker_before_requested_memory_ceiling(tmp_path):
    # Deliberately small allocation tests stop behavior without approaching 3 GiB.
    report=tmp_path/'guard.json'
    cpus=','.join(map(str,sorted(os.sched_getaffinity(0))[:4]))
    worker='import time; payload=bytearray(64*1024*1024); time.sleep(5)'
    child=f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{worker!r}],start_new_session=True); time.sleep(5)'
    run=subprocess.run([sys.executable,str(ROOT/'backend/tools/phase2b_guard.py'),
        '--cpus',cpus,'--warning-gib','.01','--stop-gib','.05','--report',str(report),
        '--',sys.executable,'-c',child],capture_output=True,text=True)
    r=json.loads(report.read_text())
    assert run.returncode!=0 and r['stopped_for_memory'] and r['warning_triggered']
    assert r['stop_reason']=='aggregate_rss' and r['peak_aggregate_rss_bytes']<256*1024**2
    assert r['elapsed_seconds']<3  # nested-session worker/pipes also terminated
