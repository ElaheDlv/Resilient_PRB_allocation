"""Bounded matched-objective pilot; reads training/validation pools exclusively."""
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backend'))
SOURCE = ROOT/'backend/evaluations/phase2c2'
OUT = ROOT/'backend/evaluations/phase2c3'
METHODS = ('A1','A2','A3','A4')
# Original eMBB trace hash already recorded by the frozen C2 provenance.
ORIGINAL_EMBB_SHA256 = 'f189e02c3de31fc1891835a2e99a2a5471ff735b26b8e688dd397440a0ca61f1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(pool):
    if pool not in ('training','validation'):
        raise ValueError('This pilot prohibits any other scenario pool')
    p = SOURCE/f'{pool}_manifest.json'
    data = json.loads(p.read_text())
    if data['pool'] != pool or len(data['entries']) != 18:
        raise ValueError('Expected the frozen 18-scenario pool')
    return data['entries']


def freeze():
    """Called before any learner or physical run; refuse to overwrite a protocol."""
    path = OUT/'protocol.json'
    if path.exists():
        raise FileExistsError('Protocol already frozen')
    training = manifest('training')
    # Latin-stratum blocks: every block contains all regimes and severities.
    strata = [('feasible',0),('contested',10),('overloaded',20),
              ('contested',20),('overloaded',0),('feasible',10),
              ('overloaded',10),('feasible',20),('contested',0)]
    order=[]
    for i,(regime,severity) in enumerate(strata):
        choices=sorted((e for e in training if e['regime']==regime and e['severity']==severity),key=lambda e:e['seed'])
        order.append(choices[(i//3)%2]['id'])
    paths=set()
    for pool in ('training','validation'):
        paths.add(SOURCE/f'{pool}_manifest.json')
        for e in manifest(pool):
            paths.add(SOURCE/e['catalog'])
            trace=e['trace']
            paths.add(SOURCE/trace[len('@scenario/'):] if trace.startswith('@scenario/') else ROOT/trace)
    p=dict(version='phase2c3-matched-v1',base_commit='64a861ec1cb5040e85c991ea20c8667df7ca178f',
        methods=list(METHODS),network_seeds=[12026,22026],episode_intervals=300,
        training_order=order,checkpoint_after_episodes=[3,6,9],
        transitions_per_learner=2691,training_config=dict(replay_capacity=2048,batch_size=32,
            gamma=.99,learning_rate=.0001,target_every=100),
        exploration=dict(start=1.,end=.05,decay_fraction=.8),
        rule_release_ratio=.5,rule_release_decisions=2,
        aggregate_selection='Unchanged workload_feasibility.checkpoint_key; balanced nine strata; earlier checkpoint final tie.',
        fairness_selection='Worst-UE-first: swap first two aggregate ranking terms; all remaining tie breaks unchanged. No SLA threshold.',
        tradeoff='Report both selections and every aggregate/worst checkpoint point; do not pick a preferred criterion after seeing outcomes.',
        budget_basis='Same nine episodes/2691 interactions per network as C1. Eight learners, 72 training and 432 validation episodes total. No automatic extension.',
        training_coverage='Nine of 18 frozen entries per learner, identical for both network seeds and all methods. Two placements represented per regime; nine strata once.',
        resources=dict(cpus=[6,7,11,14],warning_gib=2.,stop_gib=2.5,minimum_available_gib=4.,
            cuda=False,workers=1,dataloader_workers=0,library_threads=1,hard_memory_limit=False),
        input_sha256={str(f.relative_to(ROOT)):digest(f) for f in sorted(paths)})
    path.write_text(json.dumps(p,indent=2)+'\n')
    return p


def verify(p):
    if digest(ROOT/'backend/notebooks/Resilient_data/embb_dl.csv') != ORIGINAL_EMBB_SHA256:
        raise ValueError('Original eMBB trace differs from frozen C2 provenance')
    for name, expected in p['input_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f'Frozen input changed: {name}')
    if p['methods']!=list(METHODS) or p['checkpoint_after_episodes'] != [3,6,9]:
        raise ValueError('Unexpected experiment protocol')


def epsilon(p, interactions):
    e=p['exploration'];f=min(1.,interactions/(p['transitions_per_learner']*e['decay_fraction']))
    return e['start']+(e['end']-e['start'])*f


def run(entry,method,agent,p,*,train=False,intervals=None,after_service=None):
    from tools.phase2c1_pilot import run_episode
    from tools.phase2c2_design import resolve_catalog
    from network_layer.matched_control import xAppMatchedPilot
    # Reuse the original fresh-physics runner, with only an explicit app hook.
    # Its epsilon helper uses the count of three training seeds, i.e. 9 episodes.
    rp=dict(p,training_scenario_seeds=[0,1,2],severities=[0,10,20],network_seed=agent.config.seed)
    with open('/dev/null','w') as quiet, contextlib.redirect_stdout(quiet):
        row=run_episode(method,entry['severity'],entry['seed'],rp,agent=agent,train=train,
            intervals=intervals,catalog_override=resolve_catalog(SOURCE,entry),
            app_class=xAppMatchedPilot,after_service=after_service)
    row.update(scenario_id=entry['id'],regime=entry['regime'])
    return row


def selection_key(rows,episode,fair=False):
    from network_layer.workload_feasibility import checkpoint_key
    key=checkpoint_key(rows,episode)
    return (key[1],key[0],*key[2:]) if fair else key


def execute(out,p):
    from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
    from tools.phase2c1_pilot import assert_paired
    import torch
    import gc
    class LoggedAgent(JointDQNAgent):
        def learn(self):
            value=super().learn()
            if value is not None:
                updates.writerow(dict(method=method,network_seed=seed,episode=episode,
                    optimizer_updates=self.updates,transitions=self.interactions,**self.last_update))
            return value
    (out/'checkpoints').mkdir(exist_ok=True)
    training={e['id']:e for e in manifest('training')}
    validation=manifest('validation')
    selections=[];paired=[]
    fields=['method','network_seed','episode','optimizer_updates','transitions','loss','mean_abs_td','max_abs_q','max_abs_target']
    with (out/'updates.csv').open('w',newline='') as uf, (out/'training.jsonl').open('w') as tf, (out/'validation.jsonl').open('w') as vf:
        updates=csv.DictWriter(uf,fieldnames=fields,lineterminator="\n");updates.writeheader()
        for method in METHODS:
            for seed in p['network_seeds']:
                agent=LoggedAgent(TrainingConfig(seed=seed,epsilon=p['exploration']['start'],**p['training_config']))
                candidates=[]
                for episode,sid in enumerate(p['training_order'],1):
                    row=run(training[sid],method,agent,p,train=True)
                    row.update(episode=episode,total_transitions=agent.interactions)
                    tf.write(json.dumps(row,allow_nan=False)+'\n');tf.flush();uf.flush()
                    print(json.dumps(dict(stage='training',method=method,seed=seed,episode=episode,
                        transitions=agent.interactions,reward=row['return_value'],seconds=row['elapsed_seconds'])),flush=True)
                    if episode not in p['checkpoint_after_episodes']:continue
                    path=Path('checkpoints')/f'{method}_seed{seed}_ep{episode}.pt'
                    agent.save(out/path,run_metadata=dict(protocol_sha256=digest(out/'protocol.json'),
                        ablation=method,mask_version='matched-ablation-mask-v1',episode=episode,
                        interactions=agent.interactions,network_seed=seed))
                    rows=[]
                    fingerprint=agent.fingerprint();rng_state=agent.rng.getstate();n=agent.interactions;u=agent.updates
                    for entry in validation:
                        r=run(entry,method,agent,p)
                        r.update(candidate_episode=episode,total_transitions=agent.interactions)
                        vf.write(json.dumps(r,allow_nan=False)+'\n');vf.flush();rows.append(r)
                        paired.append({k:r[k] for k in ('scenario_seed','severity','deadline_assignment_sha256')} | {'physical':{'exogenous_sha256':r['physical']['exogenous_sha256']}})
                    if (fingerprint!=agent.fingerprint() or rng_state!=agent.rng.getstate() or n!=agent.interactions or u!=agent.updates):
                        raise AssertionError('Validation mutated learner state')
                    candidates.append(dict(episode=episode,checkpoint=str(path),
                        aggregate_key=[str(x) for x in selection_key(rows,episode)],
                        fairness_key=[str(x) for x in selection_key(rows,episode,True)]))
                    # Compare exact fractions, not their serialized strings.
                    from fractions import Fraction
                    chosen={kind:min(candidates,key=lambda c:tuple(Fraction(x) for x in c[kind+'_key']))
                            for kind in ('aggregate','fairness')}
                    print(json.dumps(dict(stage='validation',method=method,seed=seed,episode=episode,
                        aggregate_percent=float(selection_key(rows,episode)[0])*100,
                        worst_percent=float(selection_key(rows,episode)[1])*100)),flush=True)
                if agent.interactions!=p['transitions_per_learner']:
                    raise AssertionError('Interaction budget mismatch')
                selections.append(dict(method=method,network_seed=seed,candidates=candidates,selected=chosen))
                (out/'selection.json').write_text(json.dumps(selections,indent=2)+'\n')
                del agent;gc.collect()
        assert_paired(paired)
        if torch.cuda.is_initialized():raise AssertionError('CUDA initialized')



def audit_checkpoints(out,p):
    """Verify evaluation snapshots and metadata without running new scenarios."""
    import gzip
    from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
    expected={(m,seed,ep) for m in METHODS for seed in p['network_seeds']
              for ep in p['checkpoint_after_episodes']}
    fingerprints={}
    for method,seed,episode in sorted(expected):
        agent=JointDQNAgent(TrainingConfig(seed=seed,epsilon=p['exploration']['start'],**p['training_config']))
        path=out/'checkpoints'/f'{method}_seed{seed}_ep{episode}.pt'
        agent.load(path)
        meta=agent.checkpoint_run_metadata
        required=dict(protocol_sha256=digest(out/'protocol.json'),ablation=method,
            mask_version='matched-ablation-mask-v1',episode=episode,
            interactions=episode*(p['episode_intervals']-1),network_seed=seed)
        if meta!=required:
            raise AssertionError(f'Checkpoint experiment metadata mismatch: {path}')
        if agent.updates!=required['interactions']-p['training_config']['batch_size']+1:
            raise AssertionError('Checkpoint update budget mismatch')
        fingerprints[(method,seed,episode)]=agent.fingerprint()
        del agent
    path=out/'validation.jsonl'
    opener=path.open if path.exists() else lambda: gzip.open(str(path)+'.gz','rt')
    n=0
    with opener() as f:
        for line in f:
            r=json.loads(line);key=(r['method'],r['network_seed'],r['candidate_episode'])
            if fingerprints[key]!=r['parameter_sha256']:
                raise AssertionError('Evaluated parameters differ from saved checkpoint')
            n+=1
    if n!=432:raise AssertionError('Incomplete validation audit')
    report=dict(checkpoints_verified=len(fingerprints),validation_fingerprints_verified=n,
        fresh_scenarios_executed=0,pools_used=['training','validation'],
        training_resume_supported=False)
    (out/'checkpoint_audit.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run','audit'))
    a=parser.parse_args()
    if a.stage=='freeze':
        freeze();return
    from tools.phase2c1_pilot import configure_environment
    configure_environment()
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    if a.stage=='audit':
        audit_checkpoints(OUT,p);return
    if any((OUT/name).exists() for name in ('training.jsonl','training.jsonl.gz','selection.json')):
        raise FileExistsError('Refuse accidental rerun of the frozen pilot')
    t=time.perf_counter();execute(OUT,p)
    (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t),indent=2)+'\n')


if __name__=='__main__':main()
