"""Nonlearning certificates, portable frozen splits and exact metric hierarchy."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
from network_layer.workload_feasibility import certificates, demand_bound, checkpoint_key, choose_original, registration_stream_seeds
from tools.phase2c2_design import freeze_stage, resolve_catalog, validate_protocol, sha

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'backend/evaluations/phase2c2/discovery_protocol.json'


def budgets():return {str(i):(.002 if i%2 else .010) for i in range(8)}


def test_constructive_contested_and_full_pool_impossibility_certificates():
    rates={u:100. for u in budgets()}
    assert certificates(500,rates,budgets())['feasible']
    contested=certificates(700,rates,budgets())
    assert contested['contested'] and not contested['overloaded']
    assert contested['baseline_on_bound']['zero_miss_impossible']
    assert certificates(1100,rates,budgets())['overloaded']
    # A fluid ratio below baseline does not establish a one-tick construction.
    c=certificates(650,rates,budgets())
    assert c['fluid_prbs']==52 and c['integer_one_tick_prbs']==56 and not c['feasible']


def test_strict_due_endpoints_and_finite_horizon_vs_longrun():
    assert demand_bound(10,{'u':10.},{'u':.002},3,1)['due_batches']['u']==0
    assert demand_bound(10,{'u':10.},{'u':.002},4,1)['due_batches']['u']==1
    assert not demand_bound(11,{'u':10.},{'u':.010},100,1)['zero_miss_impossible']
    assert demand_bound(11,{'u':10.},{'u':.010},200,1)['zero_miss_impossible']
    with pytest.raises(ValueError):certificates(650,{'u':0.},{'u':.002})


def score_rows(reserve=0,switches=0,failed=0,unresolved=0):
    return [dict(regime=g,severity=j,steps=3000,switches=switches,valid_action_checks=299,
        embb_served_offered_fraction=1.,physical=dict(mean_reserve_prbs=reserve,
        overall=dict(offered_batches=24000,known_failed_batches=failed,unresolved_not_due_batches=unresolved),
        per_ue={'u':dict(batch_outcomes=dict(offered=3000,known_failed=failed))}))
        for g in ('feasible','contested','overloaded') for j in (0,10,20)]


def test_prospective_key_discriminates_resources_switches_and_censoring():
    assert checkpoint_key(score_rows(0),9)<checkpoint_key(score_rows(9),3)
    assert checkpoint_key(score_rows(3,0),9)<checkpoint_key(score_rows(3,5),3)
    assert checkpoint_key(score_rows(0,0,1),3)>checkpoint_key(score_rows(18,10,0),9)
    assert checkpoint_key(score_rows(unresolved=1),3)>checkpoint_key(score_rows(unresolved=0),9)
    # Ignore sub-quantum accumulated float noise without ignoring a PRB tick.
    assert checkpoint_key(score_rows(3.),3)==checkpoint_key(score_rows(3.+1e-12),3)
    with pytest.raises(ValueError):checkpoint_key(score_rows()[:-1],3)


def test_protocol_refuses_learners_expanded_screen_and_overlapping_seeds():
    p=json.loads(PROTOCOL.read_text());validate_protocol(p)
    retired=copy.deepcopy(p);retired['version']='v1'
    with pytest.raises(ValueError):validate_protocol(retired)
    bad=copy.deepcopy(p);bad['methods']=['P']
    with pytest.raises(ValueError):validate_protocol(bad)
    bad=copy.deepcopy(p);bad['seed_blocks']['validation']=bad['seed_blocks']['training']
    with pytest.raises(ValueError):validate_protocol(bad)
    bad=copy.deepcopy(p);bad['seed_blocks']['training']=[1,1000]
    with pytest.raises(ValueError):validate_protocol(bad)


def test_freeze_is_exogenous_portable_and_scaling_preserves_sources(tmp_path):
    p=json.loads(PROTOCOL.read_text());p['seeds_per_regime_per_pool']=1
    p['seed_blocks']={name:[base,base+2] for name,base in [('training',901),('validation',911),('future_holdout',921)]}
    probes={}
    for pool,(base,_) in p['seed_blocks'].items():
        probes[pool]=[]
        for seed,rate in [(base,200.),(base+1,93.),(base+2,100.)]:
            rates={u:rate for u in budgets()}
            probes[pool].append(dict(seed=seed*p['seed_stride'],seed_index=seed,rates=rates,budgets=budgets(),certificate=certificates(650,rates,budgets())))
    (tmp_path/'probes.json').write_text(json.dumps(probes))
    source=ROOT/'backend/notebooks/Resilient_data/urllc_dl.csv';before=sha(source)
    freeze_stage(tmp_path,p)
    groups=[]
    for pool in p['seed_blocks']:
        manifest=json.loads((tmp_path/f'{pool}_manifest.json').read_text())
        assert len(manifest['entries'])==9
        groups.append({e['seed'] for e in manifest['entries']})
        overload=next(e for e in manifest['entries'] if e['regime']=='overloaded')
        assert overload['traffic_variant']=='phase12_scaled' and overload['certificate']['overloaded']
        catalog=resolve_catalog(tmp_path,overload)
        generated=Path(catalog['episodes'][0]['slices']['URLLC']['trace'])
        with source.open() as a,generated.open() as b:
            original=csv.DictReader(a);scaled=csv.DictReader(b)
            for row,copyrow in zip(original,scaled,strict=True):
                assert {k:v for k,v in row.items() if k!='length'}=={k:v for k,v in copyrow.items() if k!='length'}
        assert not any('@scenario/' in c['trace'] for c in catalog['episodes'][0]['slices'].values())
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert sha(source)==before


def test_original_seed_selection_ignores_any_performance_fields():
    records=[dict(seed=3,certificate={'feasible':True},proposed_failure=0.),
             dict(seed=1,certificate={'feasible':True},proposed_failure=1.)]
    assert choose_original(records,'feasible',1,set())[0]['seed']==1
    assert choose_original(records,'feasible',1,{1})[0]['seed']==3


def test_real_references_create_no_neural_models_and_observer_is_noninterfering(tmp_path):
    code=r'''
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'backend'))
from tools.phase2c1_pilot import configure_environment
configure_environment()
from network_layer.joint_dqn_agent import JointDQNAgent
from tools.phase2c2_design import run_reference
from tools.phase2c1_pilot import run_episode

def forbidden(*a,**kw):raise AssertionError('Neural model creation forbidden')
JointDQNAgent.__init__=forbidden
p=json.loads(Path('backend/evaluations/phase2c1/protocol.json').read_text())
rows=[run_reference(Path(sys.argv[1]),p,m,20,9901,intervals=8,capture=True) for m in ('B0','B1','B2')]
assert len({r['physical']['exogenous_sha256'] for r in rows})==1
assert rows[1]['physical']['quota_sha256']==rows[2]['physical']['quota_sha256']
base=run_episode('B0',20,9901,p,intervals=8)
assert base['control_digest']==rows[0]['control_digest']
assert base['physical']['exogenous_sha256']==rows[0]['physical']['exogenous_sha256']
assert all(r['updates']==0 and r['parameter_sha256'] is None for r in rows)
assert all(r['diagnostic']['static_rates_verified'] and r['diagnostic']['periodic_arrivals_verified'] for r in rows)
'''
    run=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_adjacent_legacy_seeds_overlap_but_stride_pool_streams_are_disjoint():
    assert len(registration_stream_seeds(3101)&registration_stream_seeds(3102))==14
    assert len(registration_stream_seeds(12001)&registration_stream_seeds(13001))==8
    p=json.loads(PROTOCOL.read_text());all_streams=set()
    for first,last in p['seed_blocks'].values():
        for i in range(first,last+1):
            current=registration_stream_seeds(i*p['seed_stride'])
            assert not (current&all_streams)
            all_streams |= current
    assert len(all_streams)==72*16
    p['seed_stride']=1
    with pytest.raises(ValueError):validate_protocol(p)
