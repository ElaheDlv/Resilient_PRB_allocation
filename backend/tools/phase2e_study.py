"""Frozen nonlearning G1 efficiency and independent exogenous validation study."""
import argparse
import contextlib
import copy
import cProfile
import csv
from functools import partial
import gzip
import hashlib
import json
import math
from pathlib import Path
import pstats
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
OUT=ROOT/'backend/evaluations/phase2e'
C2=ROOT/'backend/evaluations/phase2c2'
D=ROOT/'backend/evaluations/phase2d'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def freeze():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    c2=json.loads((C2/'discovery_protocol.json').read_text())
    sources=json.loads((C2/'validation_manifest.json').read_text())['entries']
    profiles=[next(e['id'] for e in sources if e['regime']==g and e['severity']==20) for g in ('feasible','contested','overloaded')]
    p=dict(version='phase2e-independent-v1',base_commit='fbafd09',candidate_indices=[15001,15024],seed_stride=4096,
        prior_index_blocks=c2['seed_blocks'],placements_per_regime=3,severities=[0,10,20],episode_intervals=300,probe_intervals=2,
        regimes=['feasible','contested','overloaded'],overload_fluid_target=c2['scaled_targets']['overloaded_fluid_prbs'],
        selection='First three unused original650-byte seeds satisfying existing exogenous certificates for feasible, then contested; first three remaining for overload using the existing C2 85.8-fluid-PRB length-scaling definition. No outcome-based replacement; no automatic extension if coverage is short.',
        profile_scenarios=profiles,equivalence_scenarios=json.loads((D/'protocol.json').read_text())['prototype_scenarios'],
        algorithm='Unchanged Phase2D ExperimentalFairAllocator fraction=1, legacy=False; same100ms service window,20 matured batches, credit/gate equations and fallback.',
        independent_policies=['P','F','G1'],independent_original_checks='Original G1 replay on all selected contested j20 entries; compare with optimized independent runs.',
        budget='24 exogenous20ms probes; 6 profiling episodes;20 existing-scenario equivalence episodes; at most81 independent P/F/G1 episodes and3 independent original checks. Max110 full3-second episodes. No learning or automatic growth.',
        interpretation='Report each placement and paired differences, not a tuned coefficient or a pooled-sample significance claim. No application-supported acceptable-loss SLA exists; quantify trade-offs explicitly. Necessary1ms timing benchmark: report all-tick and nonterminal p95/p99/max/exceedances, not a hard real-time guarantee.',
        timing_scope='Wall and process-CPU around entire engine.step: includes physics/traffic, all scheduling, reference RIC, telemetry/metrics and G1 post-service updates. Excludes setup/reset, post-timer audit hashing/CSV export. Also allocator+G1-update transaction timings. Profiling runs excluded from independent timing evidence.',
        resources=dict(cpus=[6,7,11,14],warning_gib=2,stop_gib=2.5,minimum_available_gib=4,threads=1,workers=1,cuda=False,hard_memory_cap=False))
    files=[ROOT/'backend/network_layer/feasibility_gated_fairness.py',ROOT/'backend/network_layer/deadline_scheduler.py',ROOT/'backend/network_layer/proportional_scheduler.py',
        C2/'discovery_protocol.json',C2/'validation_manifest.json',D/'protocol.json',
        ROOT/'backend/notebooks/Resilient_data/embb_dl.csv',ROOT/'backend/notebooks/Resilient_data/urllc_dl.csv']
    for j in (0,10,20):files.append(ROOT/f'backend/evaluations/phase1/main_scenarios/scenario_j{j}.json')
    for e in sources:
        if e['id'] in set(p['profile_scenarios']+p['equivalence_scenarios']):
            files.append(C2/e['catalog'])
            tr=e['trace'];files.append(C2/tr[len('@scenario/'):] if tr.startswith('@scenario/') else ROOT/tr)
    p['input_sha256']={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))}
    write(OUT/'protocol.json',p)


def verify(p):
    for n,d in p['input_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError(f'Frozen source/input changed: {n}')


def summary_ns(values):
    a=sorted(values);n=len(a)
    if not n:return dict(n=0,mean_us=None,median_us=None,p95_us=None,p99_us=None,max_us=None,over_1ms=0)
    return dict(n=n,mean_us=sum(a)/n/1000,median_us=(a[(n-1)//2]+a[n//2])/2000,
        p95_us=a[max(0,math.ceil(.95*n)-1)]/1000,p99_us=a[max(0,math.ceil(.99*n)-1)]/1000,
        max_us=a[-1]/1000,over_1ms=sum(x>1_000_000 for x in a))


def app_class():
    from collections import defaultdict
    from network_layer.pilot_control import xAppPairedPilot
    from network_layer.feasibility_gated_fairness import ExperimentalFairAllocator,ServiceWindow
    class MeasuredReference(xAppPairedPilot):
        def __init__(self,*args,variant,window_sink=None,tick_sink=None,identity=None,**kwargs):
            super().__init__(*args,**kwargs)
            self.variant,self.window_sink,self.tick_sink=variant,window_sink,tick_sink
            self.identity=identity or {}
        def start(self):
            super().start()
            self.experimental=None
            if self.variant=='original':self.experimental=ExperimentalFairAllocator(self.cell,self.collector,1.)
            elif self.variant=='optimized':
                from network_layer.equivalent_g1 import OptimizedG1Allocator
                self.experimental=OptimizedG1Allocator(self.cell,self.collector)
            elif self.variant!='none':raise ValueError('Unknown G1 implementation')
            if self.experimental is not None:self.cell.experimental_urllc_allocator=self.experimental
            self.window=self.experimental.window if self.experimental else ServiceWindow()
            self.wall=[];self.cpu=[];self.transaction=[];self.credit_updates=[]
            self.grants_hash=hashlib.sha256();self.service_hash=hashlib.sha256()
            self.deficits=defaultdict(lambda:dict(samples=0,valid_samples=0,sum_deficit=0.,max_deficit=0.,positive_samples=0,max_queue_growth=0))
            self._step=self.engine.step
            def measured(dt):
                cpu=time.process_time_ns();wall=time.perf_counter_ns()
                self._step(dt)
                elapsed=time.perf_counter_ns()-wall;used=time.process_time_ns()-cpu
                self.wall.append(elapsed);self.cpu.append(used)
                alloc=self.cell.last_scheduler_stats['URLLC']['elapsed_ns']
                transaction=alloc+self.last_credit_ns
                if transaction>elapsed:raise AssertionError('Transaction escaped complete tick timer')
                self.transaction.append(transaction);self.credit_updates.append(self.last_credit_ns)
                step=self.engine.sim_step;qm=self.cell.base_station.queue_measurements
                users=sorted(self.cell.connected_ue_list)
                grants=[(u,self.cell.prb_ue_allocation_dict[u]['downlink']) for u in users]
                self.grants_hash.update(json.dumps((step,grants),separators=(',',':')).encode())
                physical=[(u,qm.totals[u]['offered_bytes'],qm.totals[u]['served_bytes'],qm.totals[u]['overflow_bytes'],
                    self.cell.base_station.get_dl_buf_bytes(u),self.cell.base_station.get_dl_head_wait_s(u)) for u in users]
                self.service_hash.update(json.dumps((step,physical),separators=(',',':')).encode())
                if step%10==0:
                    for u in users:
                        if self.cell.connected_ue_list[u].slice_type!='URLLC':continue
                        w=self.window.view(u);d=self.deficits[u];d['samples']+=1
                        if w['full']:
                            d['valid_samples']+=1;d['sum_deficit']+=w['deficit'];d['max_deficit']=max(d['max_deficit'],w['deficit'])
                            d['positive_samples']+=w['deficit']>0;d['max_queue_growth']=max(d['max_queue_growth'],w['queue_growth'])
                        if self.window_sink:self.window_sink(dict(self.identity,step=step,ue=u,**w))
                if self.tick_sink:self.tick_sink(dict(self.identity,step=step,ric_boundary=step%10==0,
                    wall_ns=elapsed,cpu_ns=used,allocator_ns=alloc,credit_update_ns=self.last_credit_ns,transaction_ns=transaction))
                if self.finished:
                    self.report.update(allocation_trajectory_sha256=self.grants_hash.hexdigest(),
                        service_trajectory_sha256=self.service_hash.hexdigest(),deficits=dict(self.deficits),
                        timing=dict(all_ticks=summary_ns(self.wall),nonterminal_ticks=summary_ns(self.wall[:-1]),
                            cpu_ticks=summary_ns(self.cpu),scheduler_transaction=summary_ns(self.transaction),
                            post_service_credit_update=summary_ns(self.credit_updates),
                            ric_ticks=summary_ns(self.wall[9::10]),ordinary_ticks=summary_ns([v for i,v in enumerate(self.wall,1) if i%10])))
            self._measured_step=measured;self.engine.step=measured
        def observe_service(self,engine,dt):
            super().observe_service(engine,dt)
            now=self.cell.base_station._time_s
            if self.experimental:
                started=time.perf_counter_ns();self.experimental.observe_service(now)
                self.last_credit_ns=time.perf_counter_ns()-started
            else:
                self.last_credit_ns=0
                for uid,u in self.cell.connected_ue_list.items():
                    if u.slice_type!='URLLC':continue
                    t=self.cell.base_station.queue_measurements.totals[uid]
                    self.window.observe(uid,now,t['offered_bytes'],t['served_bytes'],self.cell.base_station.get_dl_buf_bytes(uid))
        def close(self):
            if hasattr(self,'_measured_step') and self.engine.step is self._measured_step:self.engine.step=self._step
            super().close()
    return MeasuredReference


def run(entry,root,policy,variant='none',*,intervals=300,window_sink=None,tick_sink=None):
    from tools.phase2c1_pilot import run_episode
    from tools.phase2c2_design import resolve_catalog
    p=json.loads((ROOT/'backend/evaluations/phase2c1/protocol.json').read_text());p['network_seed']=0
    identity=dict(scenario_id=entry['id'],policy=policy,variant=variant,regime=entry['regime'],severity=entry['severity'])
    with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
        row=run_episode('B1' if policy=='P' else 'B2',entry['severity'],entry['seed'],p,agent=None,train=False,
            intervals=intervals,catalog_override=resolve_catalog(root,entry),
            app_class=partial(app_class(),variant=variant,identity=identity,window_sink=window_sink,tick_sink=tick_sink))
    row.update(identity)
    assert row['updates']==0 and row['parameter_sha256'] is None
    return row


def source_entries():return json.loads((C2/'validation_manifest.json').read_text())['entries']


def discover(p):
    from network_layer.workload_feasibility import registration_stream_seeds,certificates
    from tools.phase2c2_design import run_reference
    from tools.phase12_study import scaled_trace
    first,last=p['candidate_indices'];prior=set()
    for lo,hi in p['prior_index_blocks'].values():
        for i in range(lo,hi+1):prior |= registration_stream_seeds(i*p['seed_stride'])
    used_streams=set();probes=[]
    pilot=json.loads((ROOT/'backend/evaluations/phase2c1/protocol.json').read_text())
    for i in range(first,last+1):
        seed=i*p['seed_stride'];streams=registration_stream_seeds(seed)
        assert not streams&prior and not streams&used_streams;used_streams |= streams
        # Screening runs P with zero reserve,20ms only. Retain exogenous fields,
        # never its deadline performance or a G1 outcome.
        r=run_reference(OUT,pilot,'B0',20,seed,intervals=p['probe_intervals'])
        d=r['diagnostic']['per_ue'];rates={u:v['capacity_min'] for u,v in d.items()};budgets={u:v['budget_s'] for u,v in d.items()}
        cert=certificates(650,rates,budgets)
        probes.append(dict(seed=seed,seed_index=i,rates=rates,budgets=budgets,certificate=cert,exogenous=d))
    write(OUT/'exogenous_probes.json',probes)
    used=set();selected={}
    for g in ('feasible','contested'):
        selected[g]=[(r,650) for r in probes if r['seed'] not in used and r['certificate'][g]][:p['placements_per_regime']]
        used |= {r['seed'] for r,_ in selected[g]}
    selected['overloaded']=[]
    for r in probes:
        if r['seed'] in used:continue
        n=math.floor(p['overload_fluid_target']/sum(1/v for v in r['rates'].values()))
        assert certificates(n,r['rates'],r['budgets'])['overloaded']
        selected['overloaded'].append((r,n));used.add(r['seed'])
        if len(selected['overloaded'])==p['placements_per_regime']:break
    (OUT/'configs').mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    entries=[]
    for g,rs in selected.items():
        for r,n in rs:
            tr='backend/notebooks/Resilient_data/urllc_dl.csv'
            if n!=650:
                generated=scaled_trace(OUT/'traces',r['seed'],g,n)
                tr='@scenario/'+str(generated.relative_to(OUT))
            for j in p['severities']:
                sid=f"independent_{g}_seed{r['seed']}_j{j}"
                catalog=json.loads((ROOT/f'backend/evaluations/phase1/main_scenarios/scenario_j{j}.json').read_text())
                catalog['episodes'][0]['slices']['URLLC']['trace']=tr
                cat=f'configs/{sid}.json';write(OUT/cat,catalog)
                entries.append(dict(id=sid,pool='phase2e_independent',regime=g,seed=r['seed'],severity=j,
                    batch_bytes=n,trace=tr,catalog=cat,certificate=certificates(n,r['rates'],r['budgets'])))
    write(OUT/'independent_manifest.json',dict(pool='phase2e_independent',entries=entries,selection_uses_only_exogenous_certificates=True))
    paths=[OUT/e['catalog'] for e in entries]+list((OUT/'traces').glob('*.csv'))
    write(OUT/'manifest_freeze.json',dict(manifest_sha256=sha(OUT/'independent_manifest.json'),
        inputs_sha256={str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f):sha(f) for f in sorted(set(paths))},
        placements={g:len(rs) for g,rs in selected.items()},candidate_streams_checked=len(used_streams),
        prior_streams_checked=len(prior),all_streams_disjoint=True,independent_g1_evaluations_before_freeze=0))


def profile(p,variant):
    entries={e['id']:e for e in source_entries()};rows=[]
    for sid in p['profile_scenarios']:
        profiler=cProfile.Profile();row=profiler.runcall(run,entries[sid],C2,'G1',variant)
        profiler.dump_stats(str(OUT/f"profile_{variant}_{entries[sid]['regime']}.prof"))
        stats=pstats.Stats(profiler)
        for (filename,line,func),(primitive,calls,own,cumulative,_) in stats.stats.items():
            if 'backend/' not in filename:continue
            rows.append(dict(variant=variant,regime=entries[sid]['regime'],file=filename.split('/backend/',1)[-1],line=line,function=func,
                primitive_calls=primitive,calls=calls,own_seconds=own,cumulative_seconds=cumulative))
    with (OUT/f'profile_{variant}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def equivalent(a,b):
    for key in ('allocation_trajectory_sha256','service_trajectory_sha256','control_digest','return_value','common_joint_monitor_return','deadline_assignment_sha256','deficits'):
        if a[key]!=b[key]:raise AssertionError(f'G1 optimization changed {key}')
    for key in ('per_ue','per_class','overall','recovery','embb_throughput_mbps','exogenous_sha256','quota_sha256','mean_reserve_prbs'):
        if a['physical'][key]!=b['physical'][key]:raise AssertionError(f'G1 optimization changed physical.{key}')


def equivalence(p):
    entries={e['id']:e for e in source_entries()};n=0
    with gzip.open(OUT/'equivalence_runs.jsonl.gz','wt') as f:
        for sid in p['equivalence_scenarios']:
            a=run(entries[sid],C2,'G1','original');b=run(entries[sid],C2,'G1','optimized')
            equivalent(a,b);n+=1
            f.write(json.dumps(a,allow_nan=False)+'\n'+json.dumps(b,allow_nan=False)+'\n');f.flush()
            print(json.dumps(dict(stage='equivalence',scenario=sid,pairs=n,exact=True)),flush=True)
    write(OUT/'equivalence.json',dict(exact=True,pairs=n,physical_ticks_per_implementation=n*3000,per_ue_allocations_checked=True))


def evaluate(p):
    from tools.phase2c1_pilot import assert_paired
    if not json.loads((OUT/'equivalence.json').read_text())['exact']:raise ValueError('No strict equivalence; do not substitute optimization')
    frozen=json.loads((OUT/'manifest_freeze.json').read_text())
    assert sha(OUT/'independent_manifest.json')==frozen['manifest_sha256']
    for n,d in frozen['inputs_sha256'].items():assert sha(ROOT/n)==d
    entries=json.loads((OUT/'independent_manifest.json').read_text())['entries'];rows=[];audit=[]
    with gzip.open(OUT/'independent_runs.jsonl.gz','wt') as output,gzip.open(OUT/'service_windows.csv.gz','wt',newline='') as wf,gzip.open(OUT/'tick_times.csv.gz','wt',newline='') as tf:
        writers={}
        def sink(handle,kind,row):
            if kind not in writers:
                writers[kind]=csv.DictWriter(handle,fieldnames=list(row),lineterminator='\n');writers[kind].writeheader()
            writers[kind].writerow(row)
        for e in entries:
            pair=[]
            for policy in ('P','F','G1'):
                variant='optimized' if policy=='G1' else 'none'
                r=run(e,OUT,policy,variant,window_sink=partial(sink,wf,'windows'),tick_sink=partial(sink,tf,'ticks'))
                for uid,resource in r['ue_resources'].items():
                    expected=e['certificate']['per_ue'][uid]['bytes_per_prb_tick']
                    assert abs(resource['capacity_bytes_per_prb']-expected)<1e-8,'Independent radio differs from calibration'
                pair.append(r);rows.append(r);output.write(json.dumps(r,allow_nan=False)+'\n');output.flush()
                print(json.dumps(dict(stage='independent',scenario=e['id'],policy=policy,
                    failed=r['physical']['overall']['failure_fraction_lower'],worst=r['physical']['worst_ue_failure_fraction'],tick_p99_us=r['timing']['all_ticks']['p99_us'])),flush=True)
            assert_paired(pair)
            assert len({r['physical']['quota_sha256'] for r in pair})==1
            if e['regime']=='contested' and e['severity']==20:
                old=run(e,OUT,'G1','original');equivalent(old,pair[-1]);audit.append(old)
    with gzip.open(OUT/'independent_original_checks.jsonl.gz','wt') as f:
        for r in audit:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(OUT/'validation_checks.json',dict(independent_episodes=len(rows),independent_scenarios=assert_paired(rows),
        independent_original_equivalence_checks=len(audit),all_quota_trajectories_identical=True,
        all_allocation_service_outcomes_equivalent=True,dqn_updates=0))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','discover','profile_original','profile_optimized','equivalence','evaluate'))
    parser.add_argument('--output',type=Path,default=OUT)
    a=parser.parse_args()
    OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='freeze':freeze();return
    from tools.phase2c1_pilot import configure_environment
    configure_environment()
    from network_layer.joint_dqn_agent import JointDQNAgent
    def forbidden(*args,**kwargs):raise AssertionError('No neural model creation/training in Phase2E study')
    JointDQNAgent.__init__=forbidden
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    marker={'discover':'manifest_freeze.json','equivalence':'equivalence.json','evaluate':'validation_checks.json'}.get(a.stage,a.stage+'.csv')
    if (OUT/marker).exists():raise FileExistsError('Refuse accidental frozen-stage rerun')
    t=time.perf_counter()
    if a.stage.startswith('profile_'):profile(p,a.stage.split('_',1)[1])
    else:dict(discover=discover,equivalence=equivalence,evaluate=evaluate)[a.stage](p)
    write(OUT/(a.stage+'_runtime.json'),dict(seconds=time.perf_counter()-t))


if __name__=='__main__':main()
