"""Sequential reproducible training/validation/test pilot; no automatic budget growth."""

import argparse
import gc
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
DEFAULT_PROTOCOL = ROOT/'backend/evaluations/phase2c1/protocol.json'


def configure_environment():
    if len(os.sched_getaffinity(0)) > 4 or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Use the resource guard with <=4 CPUs and CUDA disabled')
    if int(os.environ.get('PHASE_GUARD_STOP_BYTES', '0')) not in range(1,int(2.5*1024**3)+1):
        raise RuntimeError('Phase 2C-1 requires a guard stop threshold <=2.5 GiB')
    if any(os.environ.get(v) != '1' for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise RuntimeError('Numerical thread safeguards must be 1')
    os.environ.update(RAN_TOPOLOGY_PRESET='simple',SIM_FREEZE_MOBILITY='1',
        SIM_STEP_TIME_DEFAULT='0.001',UE_SPAWN_RADIUS_M='400',UE_DEFAULT_MAX_COUNT='100',
        RES_DQN_TRANSITION_MODE='causal',RES_DQN_DECISION_PERIOD='10',
        PRB_UE_SCHEDULER='proportional_wc',PHASE0_MEASUREMENTS='1',PHASE0_MEASUREMENT_PATH='',
        JAM_RESERVED_PRBS='18',RES_DQN_TOTAL_RESILIENCY_PRB='18',DEADLINE_STARVATION_TICKS='3')
    # NumPy first avoids inheriting an MKL INTEL/libgomp import-order conflict.
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def validate_protocol(p):
    groups = [set(p[k]) for k in ('training_scenario_seeds','validation_scenario_seeds','test_scenario_seeds')]
    if any(not group for group in groups):
        raise ValueError('Every scenario split must be nonempty')
    if len(groups[1])>2 or len(groups[2])>3:
        raise ValueError('Pilot validation/test budget exceeds bounded scope')
    if set(p['training_methods']) != {'B3','B4','B5','P'} or set(p['test_methods']) != {'B0','B1','B2','B3','B4','B5','P'}:
        raise ValueError('The paired pilot requires all specified controllers')
    if p['training_methods'].index('B3')>p['training_methods'].index('B5'):
        raise ValueError('B3 validation must precede B5 rule selection')
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Scenario seed splits must be disjoint')
    if not 2 <= p['episode_intervals'] <= 400 or len(p['training_scenario_seeds']) > 3:
        raise ValueError('Pilot budget exceeds the authorized bounded protocol')
    if p['physical_reserve_prbs'] != 18 or p['severities'] != [0,10,20]:
        raise ValueError('Pilot requires the original physical pool and three severities')


def epsilon_schedule(protocol, step):
    n = len(protocol['training_scenario_seeds'])*len(protocol['severities'])*(protocol['episode_intervals']-1)
    e = protocol['exploration']
    f = min(1.,max(0.,step/max(1,n*e['decay_fraction'])))
    return e['start']+(e['end']-e['start'])*f


def make_agent(method, protocol):
    from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
    from network_layer.pilot_control import LegacyPilotAgent
    cfg = TrainingConfig(seed=protocol['network_seed'],epsilon=protocol['exploration']['start'],
                         **protocol['training_config'])
    return (JointDQNAgent if method=='P' else LegacyPilotAgent)(cfg)


def run_episode(method, severity, seed, protocol, *, agent=None, train=False, rule_entry=1.25, intervals=None):
    import numpy as np
    import torch
    import settings
    from network_layer.simulation_engine import SimulationEngine
    from network_layer.pilot_control import xAppPairedPilot
    # Same catalog identity and fresh episode index for every paired controller.
    catalog=json.loads((ROOT/f'backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json').read_text())
    catalog['episodes'][0]['duration_steps']=intervals or protocol['episode_intervals']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    before=agent.fingerprint() if agent else None
    with tempfile.TemporaryDirectory(prefix='phase2c1-') as tmp:
        path=Path(tmp)/'catalog.json';path.write_text(json.dumps(catalog))
        settings.PRB_GYM_CONFIG_PATH=str(path)
        engine=SimulationEngine();engine.reset_network()
        engine.network_setup(xapp_classes=[partial(xAppPairedPilot,method=method,agent=agent,
            scenario_seed=seed,train=train,rule_entry=rule_entry,
            rule_release_ratio=protocol['rule_release_ratio'],
            rule_release_decisions=protocol['rule_release_decisions'],
            exploration=partial(epsilon_schedule,protocol))])
        app=next(iter(engine.ric.xapp_list.values()))
        started=time.perf_counter()
        for _ in range(catalog['episodes'][0]['duration_steps']*10):
            engine.sim_step+=1;engine.step(.001)
            if app.finished:break
        elapsed=time.perf_counter()-started
        if not app.finished:raise RuntimeError('Episode horizon left a pending transition')
        report=app.report
        report.update(severity=severity,elapsed_seconds=elapsed,steps=engine.sim_step,
            rule_entry=rule_entry if method=='B5' else None,training=train,
            network_seed=protocol['network_seed'],
            epsilon_start=epsilon_schedule(protocol,max(0,agent.interactions-report['transitions'])) if train else 0.,
            epsilon_end=epsilon_schedule(protocol,agent.interactions) if train else 0.)
        if not train and agent and before!=agent.fingerprint():
            raise AssertionError('Evaluation modified network parameters')
        if torch.cuda.is_initialized():raise AssertionError('GPU unexpectedly initialized')
        report['deadline_assignment_sha256']=hashlib.sha256(json.dumps({
            uid:{k:v for k,v in row.items() if k in ('budget_s','budget_class','position')}
            for uid,row in report['physical']['assignments'].items()},sort_keys=True).encode()).hexdigest()
        app.close()
        for bs in engine.base_station_list.values():bs.queue_measurements.close()
        engine.reset_network()
        del app, bs
        gc.collect()
        return report


def score(rows):
    import statistics
    worst=[]
    for r in rows:
        worst.append(max(u['batch_outcomes']['failure_fraction_upper']
                         for u in r['physical']['per_ue'].values()))
    return (statistics.mean(r['physical']['overall']['failure_fraction_upper'] for r in rows),
            statistics.mean(worst),
            -statistics.mean(r['physical']['embb_throughput_mbps'] for r in rows))


def evaluate_validation(method,agent,protocol,rule_entry):
    return [run_episode(method,severity,seed,protocol,agent=agent,rule_entry=rule_entry)
            for seed in protocol['validation_scenario_seeds'] for severity in protocol['severities']]


def assert_paired(rows):
    pairs={}
    for row in rows:
        key=(row['scenario_seed'],row['severity'])
        fingerprint=(row['physical']['exogenous_sha256'],row['deadline_assignment_sha256'])
        if key in pairs and pairs[key]!=fingerprint:
            raise AssertionError(f'Exogenous mismatch in pair {key}')
        pairs[key]=fingerprint
    return len(pairs)


def train_stage(out,p):
    (out/'checkpoints').mkdir(exist_ok=True)
    history=[];validation=[];selected={};rule_records=[]
    rule_entry=p['rule_entry_candidates'][0]
    for method in p['training_methods']:
        if method=='B5':
            probe=make_agent('B3',p);probe.load(out/selected['B3']['checkpoint'])
            options=[]
            for entry in p['rule_entry_candidates']:
                rows=evaluate_validation('B5',probe,p,entry)
                options.append((score(rows),entry));rule_records.extend(rows)
            _,rule_entry=min(options)
            del probe
        agent=make_agent(method,p)
        candidates=[];episode=0
        for seed in p['training_scenario_seeds']:
            for severity in p['severities']:
                row=run_episode(method,severity,seed,p,agent=agent,train=True,rule_entry=rule_entry)
                episode+=1;row['episode']=episode;history.append(row)
                print(json.dumps(dict(stage='train',method=method,episode=episode,
                    transitions=agent.interactions,updates=agent.updates,
                    return_value=row['return_value'],failed=row['physical']['overall']['failure_fraction_lower']),allow_nan=False),flush=True)
                if episode in p['checkpoint_after_episodes']:
                    path=Path('checkpoints')/f'{method}_episode{episode}.pt'
                    agent.save(out/path,run_metadata=dict(protocol=p,method=method,episode=episode,
                        interactions=agent.interactions,rule_entry=rule_entry if method=='B5' else None))
                    rows=evaluate_validation(method,agent,p,rule_entry)
                    for r in rows:r['candidate_episode']=episode
                    validation.extend(rows);candidates.append((score(rows),episode,str(path)))
        best=min(candidates)
        selected[method]=dict(score=best[0],episode=best[1],checkpoint=best[2],
            rule_entry=rule_entry if method=='B5' else None,
            training_transitions=agent.interactions,updates=agent.updates)
        (out/'selection.json').write_text(json.dumps(dict(protocol=p,selected=selected,
            rule_entry=rule_entry,selection_complete=method==p['training_methods'][-1]),indent=2)+'\n')
        del agent
    assert_paired(history);assert_paired(validation+rule_records)
    (out/'training.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
    (out/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    (out/'rule_validation.json').write_text(json.dumps(rule_records,indent=2,allow_nan=False)+'\n')


def test_stage(out,p):
    selection=json.loads((out/'selection.json').read_text())
    if selection['protocol']!=p or not selection['selection_complete']:
        raise ValueError('Held-out evaluation requires the complete frozen matching selection')
    rows=[]
    for method in p['test_methods']:
        agent=None;entry=selection['rule_entry']
        if method in selection['selected']:
            agent=make_agent(method,p);agent.load(out/selection['selected'][method]['checkpoint'])
        for seed in p['test_scenario_seeds']:
            for severity in p['severities']:
                row=run_episode(method,severity,seed,p,agent=agent,rule_entry=entry)
                rows.append(row)
                print(json.dumps(dict(stage='heldout',method=method,seed=seed,severity=severity,
                    failed=row['physical']['overall']['failure_fraction_lower']),allow_nan=False),flush=True)
        del agent
    assert_paired(rows)
    (out/'heldout.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')


def selftest(out,p):
    """Tiny causal/paired/reset test, independent of final-test seed split."""
    rows=[]
    for method in ('B0','B1','B2','B3','B4','B5','P'):
        agent=make_agent(method,p) if method in ('B3','B4','B5','P') else None
        rows.append(run_episode(method,20,9091,p,agent=agent,intervals=8))
    assert_paired(rows)
    duplicate=run_episode('B0',20,9091,p,intervals=8)
    assert duplicate['control_digest']==rows[0]['control_digest']
    assert duplicate['physical']['exogenous_sha256']==rows[0]['physical']['exogenous_sha256']
    assert rows[0]['physical']['mean_reserve_prbs']==0
    assert rows[1]['physical']['quota_sha256']==rows[2]['physical']['quota_sha256']
    assert all(r['transitions']==7 and r['updates']==0 for r in rows)
    # Real gradient/replay validation for both action/state schemas.
    for method in ('B3','P'):
        agent=make_agent(method,p)
        r=run_episode(method,20,9091,p,agent=agent,train=True,intervals=40)
        assert r['updates']==8 and r['transitions']==39
        path=out/f'{method}_selftest.pt';agent.save(path)
        clone=make_agent(method,p);clone.load(path)
        assert clone.fingerprint()==agent.fingerprint()
    (out/'selftest.json').write_text(json.dumps(dict(paired=True,repeat=True,
        controllers=7,schemas_trained=['original15/49','joint30/98']),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['train','test','selftest'],required=True)
    a=parser.parse_args()
    p=json.loads(Path(a.protocol).read_text());validate_protocol(p)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    configure_environment()
    started=time.perf_counter()
    {'train':train_stage,'test':test_stage,'selftest':selftest}[a.stage](out,p)
    (out/f'{a.stage}_runtime.json').write_text(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,
        cpus=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        protocol_sha256=hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
