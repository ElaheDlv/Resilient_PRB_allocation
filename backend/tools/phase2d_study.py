"""Sequential nonlearning diagnostics and fixed-parameter exploratory development."""
import argparse
import contextlib
import csv
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
OUT=ROOT/'backend/evaluations/phase2d'
C3=ROOT/'backend/evaluations/phase2c3'
C2=ROOT/'backend/evaluations/phase2c2'
SPECS=('P','F','OLD035','G0','G025','G05','G1')
FRACTIONS=dict(OLD035=.35,G0=0.,G025=.25,G05=.5,G1=1.)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validation_entries():
    data=json.loads((C2/'validation_manifest.json').read_text())
    assert data['pool']=='validation' and len(data['entries'])==18
    return data['entries']


def frozen_rows():
    with gzip.open(C3/'validation.jsonl.gz','rt') as f:return [json.loads(x) for x in f]


def freeze():
    path=OUT/'protocol.json'
    if path.exists():raise FileExistsError('Protocol already frozen')
    entries=validation_entries();selection=json.loads((C3/'selection.json').read_text())
    selected={(r['method'],r['network_seed']):r['selected'] for r in selection}
    replays=[]
    for seed in (12026,22026):
        for e in entries:
            replays.append(dict(method='A4',network_seed=seed,episode=selected['A4',seed]['aggregate']['episode'],scenario_id=e['id'],selection='aggregate'))
        for e in entries:
            if e['regime']=='contested' and e['severity']==20:
                replays.append(dict(method='A4',network_seed=seed,episode=selected['A4',seed]['fairness']['episode'],scenario_id=e['id'],selection='fairness'))
    for method,seed,sid in [('A1',12026,'validation_contested_seed53260288_j20'),('A2',22026,'validation_overloaded_seed53252096_j20')]:
        replays.append(dict(method=method,network_seed=seed,episode=selected[method,seed]['aggregate']['episode'],scenario_id=sid,selection='harm_comparator'))
    subset=[e['id'] for e in entries if e['regime']=='contested' or e['severity']==20]
    paths=[C2/'validation_manifest.json',C3/'protocol.json',C3/'selection.json',C3/'validation.jsonl.gz']
    for e in entries:
        paths.append(C2/e['catalog'])
        trace=e['trace'];paths.append(C2/trace[len('@scenario/'):] if trace.startswith('@scenario/') else ROOT/trace)
    paths.append(ROOT/'backend/notebooks/Resilient_data/embb_dl.csv')
    for r in replays:paths.append(C3/'checkpoints'/f"{r['method']}_seed{r['network_seed']}_ep{r['episode']}.pt")
    p=dict(version='phase2d-nonlearning-v1',base_commit='eafcea7',episode_intervals=300,
        replays=replays,prototype_scenarios=subset,prototype_specs=list(SPECS),fractions=FRACTIONS,
        reserve_controller='Unchanged ReactiveReserve for all seven nonlearning policies; physical pool18.',
        window_s=.1,minimum_matured=20,starvation_limit_ticks=3,
        budget='42 checkpoint replays + 70 nonlearning episodes. No DQN updates, tuning, holdout, adaptive parameters or automatic extension.',
        development_only=True,selection='All lambda results retained. No best-lambda optimization or final-test claims.',
        risk_saturation_epsilon=1e-12,shadow_scope='First serviced tick after each RIC command; same current queues/rates/quota and prior starvation ages; immediate timely completions only.',
        resources=dict(cpus=[6,7,11,14],warning_gib=2,stop_gib=2.5,minimum_available_gib=4,numerical_threads=1,workers=1,cuda=False,hard_memory_cap=False),
        input_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(paths))})
    path.write_text(json.dumps(p,indent=2)+'\n')


def verify(p):
    for n,d in p['input_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError(f'Changed frozen input: {n}')


class Streams:
    def __init__(self,out):
        self.stack=contextlib.ExitStack();self.out=out;self.writers={}
        self.decisions=self.stack.enter_context(gzip.open(out/'decisions.jsonl.gz','wt'))
        self.shadows=self.stack.enter_context(gzip.open(out/'shadows.jsonl.gz','wt'))
    def csv(self,name,row):
        if name not in self.writers:
            f=self.stack.enter_context((self.out/name).open('w',newline=''))
            w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();self.writers[name]=w
        self.writers[name].writerow(row)
    def close(self):self.stack.close()


def make_classes():
    from collections import defaultdict
    import math
    import settings
    from network_layer.deadline_scheduler import QueueView,deadline_allocation
    from network_layer.feasibility_gated_fairness import ExperimentalFairAllocator,ServiceWindow
    from network_layer.joint_dqn_schema import ACTIONS,encode_joint_state
    from network_layer.matched_control import xAppMatchedPilot
    from network_layer.pilot_control import xAppPairedPilot
    from network_layer.proportional_scheduler import proportional_allocation
    from tools.phase2c2_design import Observer

    class Recorder:
        def __init__(self,app,identity,streams):
            self.app,self.identity,self.streams=app,identity,streams
            self.window=ServiceWindow();self.previous={};self.stats=defaultdict(lambda:defaultdict(float))
            self.ages={};self.last_state=None;self.last_mode=None
            self.observer=Observer(capture=True)
            self.full_trace=(identity['scenario_id']=='validation_contested_seed53260288_j20' and identity.get('network_seed')==12026)
        def service(self,engine,dt):
            self.observer(engine,dt)
            cell=self.app.cell;bs=cell.base_station;qm=bs.queue_measurements
            jam=cell._effective_max_dl_prb < cell.max_dl_prb-18
            segment=('on' if jam else 'off') if self.identity['severity'] else 'no_jam'
            for uid,u in cell.connected_ue_list.items():
                if u.slice_type!='URLLC':continue
                t=qm.totals[uid];q=bs.get_dl_buf_bytes(uid)
                previous=self.previous.get(uid,(0,0))
                offered=t['offered_bytes']-previous[0];served=t['served_bytes']-previous[1]
                self.previous[uid]=(t['offered_bytes'],t['served_bytes'])
                self.window.observe(uid,bs._time_s,t['offered_bytes'],t['served_bytes'],q)
                w=self.window.view(uid);s=self.stats[uid,segment]
                deprived=w['full'] and w['served']<w['offered'] and q>0
                s['ticks']+=1;s['rate_deprived_ticks']+=deprived
                s['positive_service_deprived_ticks']+=bool(deprived and served>0)
                if self.full_trace:
                    self.streams.csv('harm_ticks.csv',dict(self.identity,step=engine.sim_step,ue=uid,
                        offered_bytes=offered,served_bytes=served,queued_bytes=q,
                        hol_s=bs.get_dl_head_wait_s(uid),grant=cell.prb_ue_allocation_dict[uid]['downlink'],
                        quota=cell.slice_dl_prb_quota['URLLC'],mode=cell.urllc_scheduler_mode.value,
                        deficit=w['deficit'],queue_growth=w['queue_growth']))
            self.ages=dict(cell._deadline_starvation_ticks)
            self.last_service_mode=cell.urllc_scheduler_mode.value
        def boundary(self):
            bs=self.app.cell.base_station;qm=bs.queue_measurements
            totals=[t for uid,t in qm.totals.items() if self.app.cell.connected_ue_list[uid].slice_type=='URLLC']
            self.streams.csv('interval_outcomes.csv',dict(self.identity,step=self.app.engine.sim_step,
                served_mode=self.last_service_mode,
                known_failed=self.app.ledger.total_failures,
                on_time_completed=sum(t['fully_admitted_on_time_batches'] for t in totals),
                offered_batches=sum(t['offered_batches'] for t in totals)))
        def decision(self,obs,previous,action):
            cell=self.app.cell;step=self.app.engine.sim_step
            from network_layer.joint_dqn_schema import encode_joint_state
            state=encode_joint_state(obs,previous)
            delta=max(abs(float(a)-float(b)) for a,b in zip(state,self.last_state)) if self.last_state is not None else None
            mode=action.urllc_scheduler_mode.value
            changed=self.last_mode is not None and mode!=self.last_mode
            row=dict(self.identity,step=step,mode_before=self.app.adapter.receipt.mode_before.value,
                mode_after=mode,switched=bool(self.app.adapter.receipt.mode_before!=action.urllc_scheduler_mode),
                delta_embb=action.delta_embb_prb,delta_urllc=action.delta_urllc_prb,
                quota_urllc=cell.slice_dl_prb_quota['URLLC'],reserve=self.app.adapter.receipt.activated_reserve_prbs,
                risk=list(obs['risk'].values),risk_valid=list(obs['risk'].valid),state=state.tolist(),
                state_linf_change=delta)
            self.streams.decisions.write(json.dumps(row,allow_nan=False)+'\n')
            self.last_state=state;self.last_mode=mode
            for u in obs['report'].ues:
                if u.slice_id!='URLLC':continue
                w=self.window.view(u.ue_id);m=u.window
                self.streams.csv('ue_windows.csv',dict(self.identity,step=step,ue=u.ue_id,
                    mode=mode,quota_urllc=cell.slice_dl_prb_quota['URLLC'],budget_s=u.budget_s,
                    queued_bytes=u.queued_bytes,hol_s=u.hol_wait_s,bytes_per_prb_tick=(u.capacity_bps_per_prb or 0)*.001/8,
                    offered_window=w['offered'],served_window=w['served'],deficit=w['deficit'],
                    queue_growth=w['queue_growth'],full_window=w['full'],
                    matured_offered=m.matured_offered_batches,matured_failed=m.matured_failed_batches,
                    pending_overdue=m.pending_overdue_batches))
        def allocation(self,cell,uids,grants,quota,dt,requirements):
            if not self.identity['replay'] or self.app.engine.sim_step%10!=1 or self.app.engine.sim_step<11:return
            bs=cell.base_station
            views={uid:QueueView(bs.get_dl_buf_bytes(uid),requirements.get(uid,{}).get('dl_throughput_per_prb',0)*dt/8,
                cell.connected_ue_list[uid].queue_budget_s,bs._dl_buf_queue.get(uid,())) for uid in uids}
            p=proportional_allocation({u:requirements.get(u,{}).get('backlog_prbs',0) for u in uids},quota,cell.prb_per_ue_cap)
            f=deadline_allocation(views,quota,bs._time_s,dt,self.ages,settings.DEADLINE_STARVATION_TICKS,cell.prb_per_ue_cap,feasibility=True)
            mode=cell.urllc_scheduler_mode.value
            assert grants==(p if mode=='PROPORTIONAL' else f),'Shadow changed actual scheduler semantics'
            def timely(g):
                count=0
                for uid in uids:
                    remaining=math.floor(g[uid]*views[uid].bytes_per_prb)
                    for b in bs.queue_measurements.pending.get(uid,()):
                        if remaining < b.remaining_bytes:break
                        remaining-=b.remaining_bytes
                        count+=int(b.overflow_bytes==0 and bs._time_s-b.arrival_s <= b.budget_s+1e-12)
                return count
            self.streams.shadows.write(json.dumps(dict(self.identity,step=self.app.engine.sim_step,
                mode=mode,quota=quota,p_timely=timely(p),f_timely=timely(f),p=p,f=f,actual=grants))+'\n')

    class DiagnosticMixin:
        def __init__(self,*args,identity,streams,spec=None,**kwargs):
            self.identity,self.streams,self.spec=identity,streams,spec
            super().__init__(*args,**kwargs)
        def start(self):
            super().start();self.recorder=Recorder(self,self.identity,self.streams)
            self.experimental=None
            if self.spec in FRACTIONS:
                self.experimental=ExperimentalFairAllocator(self.cell,self.collector,FRACTIONS[self.spec],legacy=self.spec=='OLD035')
                self.cell.experimental_urllc_allocator=self.experimental
            self.cell.allocation_diagnostic=self.recorder.allocation
            apply=self.adapter._apply_action
            def observed(action):
                obs=self.adapter._collect_obs();previous=self.adapter.prev_action
                apply(action);self.recorder.decision(obs,previous,action)
            self.adapter._apply_action=observed
        def observe_service(self,engine,dt):
            super().observe_service(engine,dt)
            self.recorder.service(engine,dt)
            if self.experimental is not None:self.experimental.observe_service(self.cell.base_station._time_s)
        def step(self):
            super().step()
            if self.engine.sim_step%10==0:
                self.recorder.boundary()
        def _finish(self):
            super()._finish()
            self.report['diagnostic']=self.recorder.observer.result()
            self.report['deprivation']={u+'|'+seg:dict(s) for (u,seg),s in self.recorder.stats.items()}
    class Matched(DiagnosticMixin,xAppMatchedPilot):pass
    class Reference(DiagnosticMixin,xAppPairedPilot):pass
    return Matched,Reference


def execute(p):
    import gc
    import torch
    from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
    from tools.phase2c1_pilot import run_episode
    from tools.phase2c2_design import resolve_catalog
    # Enforce this phase's nonlearning scope, including accidental callback use.
    def forbidden(*args,**kwargs):raise AssertionError('DQN training forbidden in Phase2D')
    JointDQNAgent.learn=forbidden;JointDQNAgent.remember=forbidden
    Matched,Reference=make_classes();streams=Streams(OUT)
    entries={e['id']:e for e in validation_entries()};c3p=json.loads((C3/'protocol.json').read_text())
    expected={(r['method'],r['network_seed'],r['candidate_episode'],r['scenario_id']):r for r in frozen_rows()}
    output=gzip.open(OUT/'runs.jsonl.gz','wt')
    pairs={};quota_pairs={};count=0
    try:
        for item in p['replays']:
            e=entries[item['scenario_id']]
            identity=dict(run_id=f"{item['method']}_{item['network_seed']}_ep{item['episode']}_{e['id']}_{item['selection']}",
                policy=item['method'],network_seed=item['network_seed'],episode=item['episode'],selection=item['selection'],
                scenario_id=e['id'],regime=e['regime'],severity=e['severity'],replay=True)
            agent=JointDQNAgent(TrainingConfig(seed=item['network_seed'],epsilon=c3p['exploration']['start'],**c3p['training_config']))
            agent.load(C3/'checkpoints'/f"{item['method']}_seed{item['network_seed']}_ep{item['episode']}.pt")
            assert agent.checkpoint_run_metadata['ablation']==item['method']
            before=agent.fingerprint();rng=agent.rng.getstate();updates=agent.updates
            rp=dict(c3p,network_seed=item['network_seed'],training_scenario_seeds=[0,1,2],severities=[0,10,20])
            with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
                row=run_episode(item['method'],e['severity'],e['seed'],rp,agent=agent,train=False,
                    catalog_override=resolve_catalog(C2,e),app_class=partial(Matched,identity=identity,streams=streams))
            old=expected[item['method'],item['network_seed'],item['episode'],e['id']]
            for key in ('control_digest','parameter_sha256','deadline_assignment_sha256'):
                assert row[key]==old[key],f'Replay mismatch: {key}'
            assert row['physical']['exogenous_sha256']==old['physical']['exogenous_sha256']
            assert row['physical']['overall']==old['physical']['overall']
            assert agent.fingerprint()==before and agent.rng.getstate()==rng and agent.updates==updates and not agent.replay
            row.update(identity);output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
            count+=1;print(json.dumps(dict(stage='replay',completed=count,run=identity['run_id'],seconds=row['elapsed_seconds'])),flush=True)
            del agent;gc.collect()
        for sid in p['prototype_scenarios']:
            f_outcome=None
            e=entries[sid]
            for spec in p['prototype_specs']:
                identity=dict(run_id=f'{spec}_{sid}',policy=spec,network_seed=0,episode=0,selection='development',
                    scenario_id=sid,regime=e['regime'],severity=e['severity'],replay=False)
                rp=dict(c3p,network_seed=0,training_scenario_seeds=[0,1,2],severities=[0,10,20])
                method='B1' if spec=='P' else 'B2'
                with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
                    row=run_episode(method,e['severity'],e['seed'],rp,agent=None,train=False,
                        catalog_override=resolve_catalog(C2,e),app_class=partial(Reference,identity=identity,streams=streams,spec=spec))
                assert row['updates']==0 and row['parameter_sha256'] is None
                pair=(row['physical']['exogenous_sha256'],row['deadline_assignment_sha256'])
                if sid in pairs:assert pair==pairs[sid]
                pairs[sid]=pair
                if sid in quota_pairs:assert row['physical']['quota_sha256']==quota_pairs[sid]
                quota_pairs[sid]=row['physical']['quota_sha256']
                if spec=='F':f_outcome=row['physical']['overall']
                if spec=='G0':assert row['physical']['overall']==f_outcome,'Lambda zero differs from F'
                row.update(identity);output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
                print(json.dumps(dict(stage='prototype',scenario=sid,policy=spec,
                    failure=row['physical']['overall']['failure_fraction_lower'],worst=row['physical']['worst_ue_failure_fraction'],seconds=row['elapsed_seconds'])),flush=True)
        assert not torch.cuda.is_initialized()
    finally:
        streams.close();output.close()
    (OUT/'validation_checks.json').write_text(json.dumps(dict(replayed_checkpoints_runs=count,
        prototype_runs=len(p['prototype_scenarios'])*len(p['prototype_specs']),
        paired_scenarios=len(pairs),all_reserve_trajectories_identical=True,
        all_replay_controls_and_outcomes_match=True,dqn_updates=0,cuda_initialized=False),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'))
    a=parser.parse_args()
    if a.stage=='freeze':freeze();return
    from tools.phase2c1_pilot import configure_environment
    configure_environment();p=json.loads((OUT/'protocol.json').read_text());verify(p)
    if (OUT/'runs.jsonl.gz').exists():raise FileExistsError('Refuse accidental repeated study')
    t=time.perf_counter();execute(p)
    (OUT/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-t),indent=2)+'\n')


if __name__=='__main__':main()
