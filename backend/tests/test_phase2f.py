"""Fixed-weight reward arithmetic, missing-data honesty and read-only replays."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

from network_layer.joint_reward import RewardConfig,tick_reward,FailureLedger
from network_layer.queue_measurements import QueueMeasurements
from network_layer.reward_audit import saved_components,COMPONENTS
from tools.phase2f_audit import OUT,verify

ROOT=Path(__file__).resolve().parents[2]


def risk(v=1.,valid=True):return NS(values=(None,None,None,None,v,None),valid=(False,False,False,False,valid,False))


def test_fixed_component_arithmetic_and_interval_switch_once():
    totals=dict.fromkeys(COMPONENTS,0.)
    for i in range(10):
        reward,parts=tick_reward(counts={'new_known_failures':int(i==0)},terminal_charges=3 if i==9 else 0,
            risk=risk(),embb_bytes=0,dt=.001,reserve_prbs=18,switched=i==0,ue_count=8,config=RewardConfig())
        assert reward==pytest.approx(-sum(parts.values()))
        for k,v in parts.items():totals[k]+=v
    assert totals==pytest.approx(dict(failure_cost=.0125,worst_cost=.25,embb_cost=.1,reserve_cost=.02,switch_cost=.01,terminal_cost=.0375))


def test_all_saved_totals_and_explicit_unidentified_residual():
    counts={}
    for split in ('training','validation'):
        with gzip.open(ROOT/f'backend/evaluations/phase2c3/{split}.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
        counts[split]=len(rows)
        for r in rows:
            a=saved_components(r)
            assert a['total_cost']==pytest.approx(sum(a[k] for k in ('failure_cost','terminal_cost','reserve_cost','switch_cost','worst_embb_residual')),abs=1e-7)
            if r['known_failures']==0:assert a['worst_cost']==0 and a['embb_cost']==a['worst_embb_residual']
            else:assert a['worst_cost'] is None and a['embb_cost'] is None
            assert a['worst_lower']<=a['worst_upper']+1e-6 and a['embb_lower']<=a['embb_upper']+1e-6
    assert counts=={'training':72,'validation':432}


def test_terminal_only_charges_pending_not_already_failed():
    qm=QueueMeasurements();qm.set_budget('old',.002);qm.set_budget('fresh',.010)
    ledger=FailureLedger(qm,['old','fresh'])
    qm.arrival('old',0.,10,10);qm.arrival('fresh',.099,10,10)
    ledger.sample(.1);counts=ledger.consume();terminal=ledger.terminal(.1)
    assert counts['new_known_failures']==1 and ledger.total_failures==1
    assert terminal['unresolved_batches']==2 and terminal['terminal_unresolved_charges']==1
    ledger.close()


def test_frozen_minimum_replay_plan_and_no_training_reconstruction():
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    assert len(p['replays'])==18 and sum(r['role']=='grid' for r in p['replays'])==9
    assert len({(r['method'],r['network_seed'],r['episode'],r['scenario_id']) for r in p['replays']})==18
    assert {r['method'] for r in p['replays']}=={'A1','A2','A3','A4'}
    assert all('holdout' not in n for n in p['input_sha256'])
    assert p['reward_config']=={'failure':1.,'worst':.25,'embb':.1,'reserve':.02,'switch':.01,
        'embb_target_bps':96e6,'reference_batches_per_ue_s':1000.,'period_s':.01}


def test_recorded_ranking_best_return_agrees_with_aggregate_without_causality_claim():
    with (OUT/'candidate_ordering.csv').open() as f:rows=list(csv.DictReader(f))
    best_reward=min(rows,key=lambda r:int(r['reward_rank']))
    best_physical=min(rows,key=lambda r:int(r['aggregate_first_rank']))
    assert best_reward['candidate']==best_physical['candidate']=='A2-22026-6'


def test_actual_component_instrumentation_preserves_frozen_replay(tmp_path):
    code=r'''
from tools.phase2c1_pilot import configure_environment
configure_environment()
from functools import partial
import sys,json,gzip,csv
from pathlib import Path
import tools.phase2f_audit as audit
from tools.phase2c1_pilot import run_episode
from tools.phase2c2_design import resolve_catalog
from network_layer.matched_control import xAppMatchedPilot
from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
from network_layer.reward_audit import COMPONENTS
JointDQNAgent.learn=lambda *a:(_ for _ in ()).throw(AssertionError('training forbidden'))
JointDQNAgent.remember=JointDQNAgent.learn
audit.OUT=Path(sys.argv[1]);streams=audit.Streams()
p=json.loads((audit.C3/'protocol.json').read_text());p.update(network_seed=12026,training_scenario_seeds=[0,1,2],severities=[0,10,20])
a=JointDQNAgent(TrainingConfig(seed=12026,epsilon=1.,**p['training_config']));a.load(audit.C3/'checkpoints/A4_seed12026_ep6.pt')
e=next(e for e in audit.source_entries() if e['regime']=='contested' and e['severity']==20)
identity=dict(run_id='unit',method='A4',network_seed=12026,checkpoint_episode=6,scenario_id=e['id'],regime=e['regime'],severity=20,role='unit')
r=run_episode('A4',20,e['seed'],p,agent=a,train=False,intervals=16,catalog_override=resolve_catalog(audit.C2,e),
 app_class=partial(audit.audit_class(),identity=identity,streams=streams,prefix_steps=[100,160]))
b=run_episode('A4',20,e['seed'],p,agent=a,train=False,intervals=16,catalog_override=resolve_catalog(audit.C2,e),app_class=xAppMatchedPilot)
assert r['control_digest']==b['control_digest'] and r['return_value']==b['return_value']
assert r['physical']['per_ue']==b['physical']['per_ue']
assert r['audit_ticks']==150 and r['audit_intervals']==15
assert abs(sum(r['audited_component_totals'].values())+r['return_value'])<1e-8
streams.close()
with gzip.open(audit.OUT/'reward_intervals.csv.gz','rt') as f:rows=list(csv.DictReader(f))
assert len(rows)==15
assert abs(sum(float(v[k]) for v in rows for k in COMPONENTS)+r['return_value'])<1e-8
with gzip.open(audit.OUT/'observed_prefix_accounting.csv.gz','rt') as f:prefix=list(csv.DictReader(f))
assert prefix[-1]['actual_terminal_here']=='True'
assert not a.replay
'''
    r=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,env=dict(os.environ,PYTHONPATH='backend'),capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
