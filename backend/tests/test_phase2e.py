"""Differential equivalence and independent protocol/timing validation; no learning."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from network_layer.deadline_scheduler import QueueView
from network_layer.feasibility_gated_fairness import ExperimentalFairAllocator
from network_layer.equivalent_g1 import OptimizedG1Allocator
from network_layer.queue_measurements import Batch
from network_layer.workload_feasibility import registration_stream_seeds
from tools.phase2e_study import OUT,summary_ns,verify

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_sources_and_independent_seed_streams():
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    used=set()
    for lo,hi in p['prior_index_blocks'].values():
        for index in range(lo,hi+1):used |= registration_stream_seeds(index*p['seed_stride'])
    new=set()
    for index in range(p['candidate_indices'][0],p['candidate_indices'][1]+1):
        streams=registration_stream_seeds(index*p['seed_stride'])
        assert not streams&used and not streams&new;new |= streams
    assert len(new)==24*16
    assert p['independent_policies']==['P','F','G1'] and p['placements_per_regime']==3


def test_nearest_rank_timing_and_strict_millisecond_exceedance():
    r=summary_ns([0,1000,1000000,1000001])
    assert r['n']==4 and r['over_1ms']==1
    assert r['median_us']==500.5 and r['p99_us']==1000.001
    assert summary_ns([])['n']==0


def fixtures(rng):
    queues={};pending={};windows={};ue=[]
    for uid in ('a','b','c'):
        budget=rng.choice([.002,.010]);rate=rng.choice([0.,.2,10.,57.362788865546214])
        batches=[]
        for i in range(rng.randrange(4)):
            accepted=rng.randrange(1,50);remaining=rng.randrange(1,accepted+1)
            batches.append(Batch(i,rng.choice([0.,.090,.099,.100]),accepted,remaining,budget,rng.choice([0,0,1])))
        pending[uid]=batches
        queues[uid]=QueueView(sum(b.remaining_bytes for b in batches),rate,budget,[(b.arrival_s,b.remaining_bytes) for b in batches])
        offered=rng.choice([0,20,100,10000]);served=rng.randrange(0,max(1,offered*2+1))
        windows[uid]=dict(full=rng.choice([True,True,False]),offered=offered,served=served,deficit=0.,queue_growth=0)
        mature=rng.choice([None,0,19,20,90]);failed=rng.randrange(mature+1) if mature is not None else None
        ue.append(NS(ue_id=uid,window=NS(full_window=rng.choice([True,False]),matured_offered_batches=mature,matured_failed_batches=failed)))
    cell=NS(base_station=NS(simulation_engine=NS(sim_step=100),queue_measurements=NS(pending=pending)))
    collector=NS(collect=lambda step:NS(ues=ue))
    return queues,pending,windows,cell,collector


def test_differential_integer_grants_for_partial_expired_zero_rate_cap_and_warmup():
    rng=random.Random(2202601)
    for _ in range(500):
        queues,pending,windows,cell,collector=fixtures(rng)
        original=ExperimentalFairAllocator(cell,collector,1.)
        optimized=OptimizedG1Allocator(cell,collector)
        original.window.view=optimized.window.view=lambda uid:windows[uid]
        quota=rng.randrange(50);cap=rng.choice([None,0,1,10,30]);ages={u:rng.randrange(6) for u in queues}
        before=[(u,b.batch_id,b.remaining_bytes) for u in pending for b in pending[u]]
        args=(queues,quota,.100,.001,ages,3,cap)
        assert original(*args,feasibility=True)==optimized(*args,feasibility=True)
        assert before==[(u,b.batch_id,b.remaining_bytes) for u in pending for b in pending[u]]


def test_telemetry_only_skipped_when_all_credits_are_proven_zero():
    rng=random.Random(12)
    queues,pending,windows,cell,collector=fixtures(rng)
    optimized=OptimizedG1Allocator(cell,collector)
    optimized.window.view=lambda uid:dict(full=True,offered=100,served=100,deficit=0.,queue_growth=0)
    collector.collect=lambda step:(_ for _ in ()).throw(AssertionError('unnecessary telemetry'))
    optimized(queues,10,.1,.001,{},feasibility=True)
    assert optimized.zero_credit_fast_paths==1 and optimized.telemetry_calls==0
    with pytest.raises(ValueError):optimized(queues,10,.1,.001,{},feasibility=False)


def test_actual_allocation_service_equivalence_and_complete_tick_timer():
    code=r'''
from tools.phase2c1_pilot import configure_environment
configure_environment()
from tools.phase2e_study import run,source_entries,C2,equivalent
from network_layer.joint_dqn_agent import JointDQNAgent
JointDQNAgent.__init__=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('neural model forbidden'))
e=next(e for e in source_entries() if e['regime']=='contested' and e['severity']==20)
a=run(e,C2,'G1','original',intervals=16)
b=run(e,C2,'G1','optimized',intervals=16)
equivalent(a,b)
for policy in ('P','F'):
 r=run(e,C2,policy,intervals=16)
 assert r['timing']['all_ticks']['n']==160 and r['timing']['nonterminal_ticks']['n']==159
 assert r['timing']['all_ticks']['p99_us']>0 and r['updates']==0
'''
    r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=dict(os.environ,PYTHONPATH='backend'),capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
