"""Bounded reference-only scenario discovery, freezing and characterization. No DQN."""

import argparse
from collections import defaultdict
import contextlib
import copy
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from network_layer.workload_feasibility import REGIMES, certificates, choose_original, checkpoint_key, registration_stream_seeds

PROTOCOL = ROOT/'backend/evaluations/phase2c2/discovery_protocol.json'
PILOT_PROTOCOL = ROOT/'backend/evaluations/phase2c1/protocol.json'



def validate_protocol(p):
    if p.get('version')!='phase2c2-design-v2-independent-streams':
        raise ValueError('Use final v2; the correlated v1 seed plan is retired')
    if p['methods']!=['B0','B1','B2'] or p['severities']!=[0,10,20]:
        raise ValueError('Only the three nonlearning references and severities are allowed')
    if set(p['seed_blocks'])!={'training','validation','future_holdout'}:
        raise ValueError('Three independent pools required')
    used=set()
    for first,last in p['seed_blocks'].values():
        if last<first or last-first+1>24:raise ValueError('Seed screening exceeds fixed cap')
        seeds=set(range(first,last+1))
        if used&seeds:raise ValueError('Overlapping discovery seed pools')
        used |= seeds
    streams=set()
    for first,last in p['seed_blocks'].values():
        for index in range(first,last+1):
            seed=index*p['seed_stride']
            if seed+101007>=2**32:raise ValueError('Seed mapping would wrap the legacy initializer')
            current=registration_stream_seeds(seed)
            if streams&current:raise ValueError('Reused per-UE registration streams across scenarios')
            streams |= current
    if p['seeds_per_regime_per_pool'] not in (1,2) or p['diagnostic_intervals']!=300 or p['calibration_intervals']!=2:
        raise ValueError('Only the bounded 20-ms/3-second diagnostic configuration is allowed')
    if p['physical_reserve_prbs']!=18:raise ValueError('Physical reserve must remain 18')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def resolve_catalog(root,entry):
    catalog=json.loads((root/entry['catalog']).read_text())
    for cfg in catalog['episodes'][0]['slices'].values():
        trace=cfg.get('trace','')
        if trace.startswith('@scenario/'):
            cfg['trace']=str((root/trace[len('@scenario/'):]).resolve())
    return catalog


class Observer:
    """Measurement-only, bounded per-UE/segment statistics and sparse frames."""

    def __init__(self,capture=False):
        self.stats={};self.previous={};self.frames=[];self.capture=capture
        self.segments=defaultdict(dict);self.seen_off=False;self.saw_jam=False
        self.first_pressure=False;self.caps=set();self.reserve_prb_ticks=0

    def __call__(self,engine,dt):
        from utils import estimate_throughput
        cell=next(iter(engine.cell_list.values()));bs=cell.base_station;qm=bs.queue_measurements
        jammed=cell._effective_max_dl_prb < cell.max_dl_prb-18
        self.saw_jam |= jammed
        if self.saw_jam and not jammed:self.seen_off=True
        segment=('on_after_off' if self.seen_off else 'on_initial') if jammed else ('off' if self.saw_jam else 'no_jam')
        occupancy=sum(cell.slice_dl_prb_quota.values())-sum(cell.base_dl_prb_quota.values())
        if not 0<=occupancy<=18:raise AssertionError('Reserve occupancy outside 18-PRB pool')
        self.reserve_prb_ticks+=occupancy
        self.caps.add(getattr(cell,'prb_per_ue_cap',None))
        snapshot=[];pressure=False
        for uid,u in sorted(cell.connected_ue_list.items()):
            if u.slice_type!='URLLC':continue
            total=qm.totals[uid]
            previous=self.previous.get(uid,dict(offered_bytes=0,offered_batches=0,served_bytes=0,overflow_bytes=0))
            delta={k:total[k]-previous[k] for k in previous}
            self.previous[uid]={k:total[k] for k in previous}
            m=u.downlink_mcs_data or {}
            capacity=estimate_throughput(m.get('modulation_order',0),m.get('target_code_rate',0),1)*dt/8
            if not math.isfinite(capacity) or capacity<=0:raise ValueError('Certificates need positive measured rates')
            grant=cell.prb_ue_allocation_dict[uid]['downlink'];queue=bs.get_dl_buf_bytes(uid)
            hol=bs.get_dl_head_wait_s(uid);late=queue>0 and hol>u.queue_budget_s+1e-12
            pressure |= late or total['overflowed_batches']>0 or total['fully_admitted_late_batches']>0
            stats=self.stats.setdefault(uid,dict(capacity_min=capacity,capacity_max=capacity,
                arrival_bytes_min=delta['offered_bytes'],arrival_bytes_max=delta['offered_bytes'],
                arrival_batches_min=delta['offered_batches'],arrival_batches_max=delta['offered_batches'],
                budget_s=u.queue_budget_s,budget_class=u.queue_budget_class,cqi=u.downlink_cqi,
                position=[u.position_x,u.position_y]))
            for name,value in [('capacity',capacity),('arrival_bytes',delta['offered_bytes']),('arrival_batches',delta['offered_batches'])]:
                stats[name+'_min']=min(stats[name+'_min'],value);stats[name+'_max']=max(stats[name+'_max'],value)
            s=self.segments[segment].setdefault(uid,dict(ticks=0,offered_bytes=0,served_bytes=0,
                granted_prbs=0,overflow_bytes=0,hol_violating_ticks=0,max_hol_s=0.,current_no_service_ticks=0,max_no_service_ticks=0))
            s['ticks']+=1;s['offered_bytes']+=delta['offered_bytes'];s['served_bytes']+=delta['served_bytes']
            s['overflow_bytes']+=delta['overflow_bytes'];s['granted_prbs']+=grant
            s['hol_violating_ticks']+=int(late);s['max_hol_s']=max(s['max_hol_s'],hol)
            s['current_no_service_ticks']=s['current_no_service_ticks']+1 if queue>0 and delta['served_bytes']==0 else 0
            s['max_no_service_ticks']=max(s['max_no_service_ticks'],s['current_no_service_ticks'])
            snapshot.append(dict(uid=uid,budget_s=u.queue_budget_s,capacity_bytes_per_prb_tick=capacity,
                offered_bytes=delta['offered_bytes'],served_bytes=delta['served_bytes'],grant=grant,
                queued_bytes=queue,hol_s=hol,late_hol=late))
        first=pressure and not self.first_pressure
        self.first_pressure |= pressure
        if self.capture and (engine.sim_step in {1,1249,1250,1260,2499,2500,3000} or first):
            self.frames.append(dict(step=engine.sim_step,segment=segment,first_pressure=first,
                quotas=dict(cell.slice_dl_prb_quota),baseline=dict(cell.base_dl_prb_quota),
                activated_reserve=occupancy,ues=snapshot))

    def result(self):
        if len(self.stats)!=8 or self.caps!={None}:
            raise AssertionError('Certificates assume eight URLLC UEs and no additional per-UE cap')
        for s in self.stats.values():
            if abs(s['capacity_max']-s['capacity_min'])>1e-8:
                raise AssertionError('Static-rate certificate invalid: channel varied')
            if s['arrival_bytes_min']!=s['arrival_bytes_max'] or s['arrival_batches_min']!=1 or s['arrival_batches_max']!=1:
                raise AssertionError('Periodic one-batch-per-tick assumption invalid')
        return dict(per_ue=self.stats,segments=dict(self.segments),frames=self.frames,
                    reserve_prb_ticks=self.reserve_prb_ticks,static_rates_verified=True,
                    periodic_arrivals_verified=True)


def run_reference(root,pilot,method,severity,seed,*,intervals,catalog=None,capture=False):
    if method not in ('B0','B1','B2'):raise ValueError('Learned controllers forbidden in scenario discovery')
    from tools.phase2c1_pilot import run_episode
    observer=Observer(capture)
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        row=run_episode(method,severity,seed,pilot,agent=None,train=False,intervals=intervals,
                        catalog_override=catalog,after_service=observer)
    if row['updates']!=0 or row['parameter_sha256'] is not None:
        raise AssertionError('Scenario study must not create/update a neural controller')
    row['diagnostic']=observer.result()
    return row


def probe_stage(root,p,pilot):
    results={}
    for pool,(first,last) in p['seed_blocks'].items():
        results[pool]=[]
        for index in range(first,last+1):
            seed=index*p['seed_stride']
            row=run_reference(root,pilot,'B0',20,seed,intervals=p['calibration_intervals'])
            # Deliberately retain only exogenous fields; probe performance discarded.
            observed=row['diagnostic']['per_ue']
            rates={u:s['capacity_min'] for u,s in observed.items()}
            budgets={u:s['budget_s'] for u,s in observed.items()}
            cert=certificates(650,rates,budgets)
            results[pool].append(dict(seed=seed,seed_index=index,registration_streams=sorted(registration_stream_seeds(seed)),exogenous=observed,rates=rates,budgets=budgets,certificate=cert))
            print(json.dumps(dict(stage='probe',pool=pool,seed=seed,
                fluid=cert['fluid_prbs'],integer=cert['integer_one_tick_prbs'],
                regimes=[g for g in REGIMES if cert[g]])),flush=True)
    write(root/'probes.json',results)


def freeze_stage(root,p):
    from tools.phase12_study import scaled_trace
    probes=json.loads((root/'probes.json').read_text())
    for pool,(first,last) in p['seed_blocks'].items():
        records=probes[pool]
        if {r.get('seed_index') for r in records}!=set(range(first,last+1)):
            raise ValueError('Probe indices do not match current protocol')
        if any(r['seed']!=r['seed_index']*p['seed_stride'] for r in records):
            raise ValueError('Probe seed mapping does not match current protocol')
    manifests={};coverage={}
    (root/'configs').mkdir(exist_ok=True);(root/'traces').mkdir(exist_ok=True)
    for pool,records in probes.items():
        used=set();selected={}
        coverage[pool]={g:sum(r['certificate'][g] for r in records) for g in REGIMES}
        for regime in REGIMES:
            selected[regime]=[(r,650,'original') for r in choose_original(records,regime,p['seeds_per_regime_per_pool'],used)]
            used.update(r['seed'] for r,_,_ in selected[regime])
        for regime in REGIMES:
            while len(selected[regime])<p['seeds_per_regime_per_pool']:
                record=next((r for r in sorted(records,key=lambda r:r['seed']) if r['seed'] not in used),None)
                if record is None:raise RuntimeError('Insufficient prespecified seeds; no automatic expansion')
                target=p['scaled_targets'][regime+'_fluid_prbs']
                size=math.floor(target/sum(1/b for b in record['rates'].values()))
                if not certificates(size,record['rates'],record['budgets'])[regime]:
                    raise RuntimeError('Rounded synthetic target fails prespecified certificate')
                selected[regime].append((record,size,'phase12_scaled'));used.add(record['seed'])
        entries=[]
        for regime in REGIMES:
            for record,size,variant in selected[regime]:
                seed=record['seed'];trace='backend/notebooks/Resilient_data/urllc_dl.csv'
                if variant!='original':
                    generated=scaled_trace(root/'traces',seed,regime,size)
                    trace='@scenario/'+str(generated.relative_to(root))
                for severity in p['severities']:
                    identity=f'{pool}_{regime}_seed{seed}_j{severity}'
                    catalog=json.loads((ROOT/f'backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json').read_text())
                    catalog['episodes'][0]['duration_steps']=p['diagnostic_intervals']
                    catalog['episodes'][0]['slices']['URLLC']['trace']=trace
                    relative=f'configs/{identity}.json';write(root/relative,catalog)
                    entries.append(dict(id=identity,pool=pool,regime=regime,seed=seed,severity=severity,
                        traffic_variant=variant,batch_bytes=size,catalog=relative,
                        seed_index=record.get('seed_index'),registration_streams=sorted(registration_stream_seeds(seed)),
                        trace=trace,trace_sha256=sha(root/trace[len('@scenario/'):]) if trace.startswith('@scenario/') else sha(ROOT/trace),
                        certificate=certificates(size,record['rates'],record['budgets'])))
        manifests[pool]=dict(schema='phase2c2-scenarios-v1',pool=pool,
            status='frozen_before_reference_characterization',model_evaluations=0,
            entries=entries)
        write(root/f'{pool}_manifest.json',manifests[pool])
    write(root/'freeze.json',dict(protocol=p,original_regime_candidate_counts=coverage,
        manifests_sha256={pool:sha(root/f'{pool}_manifest.json') for pool in manifests},
        membership_frozen=True,learned_model_evaluations=0,reference_characterization_complete=False))


def characterize_stage(root,p,pilot):
    frozen=json.loads((root/'freeze.json').read_text())
    if frozen['protocol']!=p:raise ValueError('Frozen discovery protocol mismatch')
    rows=[]
    # Stream each record to disk as well: interrupted diagnostics do not lose provenance.
    with (root/'reference_runs.jsonl').open('w') as stream:
        for pool in p['seed_blocks']:
            path=root/f'{pool}_manifest.json'
            if sha(path)!=frozen['manifests_sha256'][pool]:raise AssertionError('Manifest changed after freezing')
            entries=json.loads(path.read_text())['entries']
            for entry in entries:
                pair=[]
                for method in p['methods']:
                    row=run_reference(root,pilot,method,entry['severity'],entry['seed'],
                        intervals=p['diagnostic_intervals'],catalog=resolve_catalog(root,entry),capture=True)
                    row.update(scenario_id=entry['id'],pool=pool,regime=entry['regime'],
                               traffic_variant=entry['traffic_variant'],batch_bytes=entry['batch_bytes'])
                    for uid,s in row['diagnostic']['per_ue'].items():
                        if abs(s['capacity_min']-entry['certificate']['per_ue'][uid]['bytes_per_prb_tick'])>1e-8:
                            raise AssertionError('Selected rate differs from calibration')
                        if s['arrival_bytes_min']!=entry['batch_bytes']:raise AssertionError('Arrival size changed')
                    pair.append(row);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps(dict(stage='reference',scenario=entry['id'],method=method,
                        failures=row['physical']['overall']['failure_fraction_lower'],
                        worst=row['physical']['worst_ue_failure_fraction'],reserve=row['physical']['mean_reserve_prbs'])),flush=True)
                if len({(r['physical']['exogenous_sha256'],r['deadline_assignment_sha256']) for r in pair})!=1:
                    raise AssertionError('Reference scenarios not paired')
    write(root/'reference_runs.json',rows)


def old_audit(root):
    old={}
    for split in ('training','validation','heldout'):
        rows=json.loads((ROOT/f'backend/evaluations/phase2c1/{split}.json').read_text())
        chosen=[r for r in rows if r['method'] in ('P','B0')
                and (split!='validation' or r['candidate_episode']==3)]
        profiles=[]
        for r in chosen:
            physical=r['physical']
            profiles.append(dict(seed=r['scenario_seed'],severity=r['severity'],method=r['method'],
                fluid_prbs=sum(s['fluid_prbs'] for s in r['ue_resources'].values()),
                one_tick_integer_prbs=sum(math.ceil(650/s['capacity_bytes_per_prb']) for s in r['ue_resources'].values()),
                baseline_on={0:60,10:57,20:54}[r['severity']],baseline_off=60,
                overall=physical['overall'],per_class=physical['per_class'],per_ue=physical['per_ue'],
                resources=physical['resources'],assignments=physical['assignments'],
                ue_resources=r['ue_resources'],mean_reserve=physical['mean_reserve_prbs'],
                recovery=physical['recovery']))
        old[split]=profiles
    a=next(r for r in old['heldout'] if r['method']=='B0' and r['seed']==3101 and r['severity']==0)
    b=next(r for r in old['heldout'] if r['method']=='B0' and r['seed']==3102 and r['severity']==0)
    positions=lambda r:{tuple(u['position']) for u in r['assignments'].values()}
    old['seed_independence_audit']=dict(C1_shared_urllc_positions=len(positions(a)&positions(b)),
        C1_shared_registration_seeds=len(registration_stream_seeds(3101)&registration_stream_seeds(3102)),
        legacy_formula='scenario_seed+100000+slice_index*1000+UE_index for episode index zero',
        correction='Final v2 scenario seeds are 4096*index, with all derived registration seeds checked disjoint.')
    old['historical_harms']=json.loads((ROOT/'backend/evaluations/phase12/existing_candidate_harms.json').read_text())
    write(root/'prior_workload_audit.json',old)


def summary_stage(root,p,plots=False):
    import csv
    import statistics
    if (root/'reference_runs.json').exists():
        rows=json.loads((root/'reference_runs.json').read_text())
    else:
        with gzip.open(root/'reference_runs.json.gz','rt') as f:rows=json.load(f)
    expected=len(p['seed_blocks'])*len(REGIMES)*p['seeds_per_regime_per_pool']*len(p['severities'])*len(p['methods'])
    if len(rows)!=expected:raise AssertionError('Incomplete diagnostic grid')
    for r in rows:
        if r['diagnostic']['reserve_prb_ticks']!=round(r['physical']['mean_reserve_prbs']*r['steps']):
            raise AssertionError('Reserve metric differs from measured integer PRB-ticks')
    summaries=[];gates=[]
    for pool in p['seed_blocks']:
        for regime in REGIMES:
            for severity in p['severities']:
                for method in p['methods']:
                    rs=[r for r in rows if (r['pool'],r['regime'],r['severity'],r['method'])==(pool,regime,severity,method)]
                    mean=lambda f:statistics.mean(f(r) for r in rs)
                    windows=[w for r in rs for w in r['physical']['recovery']]
                    summaries.append(dict(pool=pool,regime=regime,severity=severity,method=method,n=len(rs),
                        failure_percent=100*mean(lambda r:r['physical']['overall']['failure_fraction_lower']),
                        failure_upper_percent=100*mean(lambda r:r['physical']['overall']['failure_fraction_upper']),
                        worst_percent=100*mean(lambda r:r['physical']['worst_ue_failure_fraction']),
                        tight_percent=100*mean(lambda r:r['physical']['per_class']['tight']['failure_fraction_lower']),
                        relaxed_percent=100*mean(lambda r:r['physical']['per_class']['relaxed']['failure_fraction_lower']),
                        embb_mbps=mean(lambda r:r['physical']['embb_throughput_mbps']),
                        reserve_prbs=mean(lambda r:r['physical']['mean_reserve_prbs']),
                        urllc_allocated_prbs=mean(lambda r:r['physical']['resources']['URLLC']['mean_allocated_prbs']),
                        urllc_unused_prbs=mean(lambda r:r['physical']['resources']['URLLC']['mean_unused_prbs']),
                        recoveries=sum(w['recovery_s'] is not None for w in windows),recovery_windows=len(windows)))
        entries=json.loads((root/f'{pool}_manifest.json').read_text())['entries']
        for e in entries:
            if e['severity']!=20:continue
            rs=[r for r in rows if r['scenario_id']==e['id']]
            rates=[r['physical']['overall']['failure_fraction_lower'] for r in rs]
            worst=[r['physical']['worst_ue_failure_fraction'] for r in rs]
            recovered=[all(w['recovery_s'] is not None for w in r['physical']['recovery']) for r in rs]
            if e['regime']=='feasible':passed=all(v==0 for v in rates)
            elif e['regime']=='contested':passed=(max(rates)-min(rates)>=.01 or max(worst)-min(worst)>=.01 or len(set(recovered))>1)
            else:passed=e['certificate']['full_pool_off_bound']['zero_miss_impossible'] and all(v>0 for v in rates)
            gates.append(dict(scenario_id=e['id'],pool=pool,regime=e['regime'],passed=passed,
                reference_failure_percent=[v*100 for v in rates],reference_worst_percent=[v*100 for v in worst],recovered=recovered))
    write(root/'reference_summary.json',summaries)
    with (root/'reference_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]),lineterminator='\n');writer.writeheader();writer.writerows(summaries)
    write(root/'coverage_gates.json',dict(reference_characterization_complete=True,ready_for_next_bounded_learning=all(g['passed'] for g in gates),
        gates=gates,model_holdout_status='reference-characterized, model-unseen; membership frozen before reference runs',
        learned_model_evaluations=0,optimizer_updates=0))
    keys={}
    for method in p['methods']:
        vals=[r for r in rows if r['pool']=='validation' and r['method']==method]
        keys[method]=[float(v) for v in checkpoint_key(vals,0)]
    write(root/'prospective_rule_reference_demo.json',dict(metric_order=['aggregate_known','mean_worst_ue','aggregate_upper','embb_deficit','reserve_fraction','switch_frequency','episode'],
        illustrative_reference_keys=keys,note='Demonstrates prospective scorer on reference data; no DQN checkpoint ranked or evaluated.'))
    per_ue=[]
    for r in rows:
        for uid,u in r['physical']['per_ue'].items():
            resource=r['ue_resources'][uid];live=r['live_accounting'][uid];outcome=u['batch_outcomes']
            per_ue.append(dict(pool=r['pool'],regime=r['regime'],seed=r['scenario_seed'],severity=r['severity'],method=r['method'],
                uid=uid,budget_class=u['budget_class'],budget_s=u['budget_s'],
                cqi=r['physical']['assignments'][uid]['cqi'],capacity_bytes_per_prb_tick=resource['capacity_bytes_per_prb'],
                offered_fluid_prbs=resource['fluid_prbs'],mean_granted_prbs=resource['prbs'],
                offered=outcome['offered'],known_failed=outcome['known_failed'],on_time=outcome['on_time'],
                unresolved_not_due=outcome['unresolved_not_due'],unfinished_overdue=outcome['unfinished_overdue'],
                overflowed=live['overflowed'],late_completed_without_overflow=live['fully_admitted_late'],
                served_bytes=u['served_bytes'],unfinished_bytes=u['unfinished_bytes'],
                failure_percent=100*outcome['failure_fraction_lower'],failure_upper_percent=100*outcome['failure_fraction_upper']))
    with (root/'per_ue_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_ue[0]),lineterminator='\n');writer.writeheader();writer.writerows(per_ue)
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        for i,method in enumerate(p['methods']):
            vals=[next(s for s in summaries if (s['pool'],s['regime'],s['severity'],s['method'])==('validation',g,20,method)) for g in REGIMES]
            for ax,key,title in zip(axes,['failure_percent','worst_percent'],['Known offered-batch failures','Worst-UE known failures']):
                ax.bar([x+(i-1)*.25 for x in range(3)],[v[key] for v in vals],width=.25,label=method)
                ax.set(xticks=range(3),xticklabels=REGIMES,ylabel='Percent',ylim=(0,105),title=title)
                ax.grid(axis='y',alpha=.2)
        axes[0].legend();fig.savefig(root/'validation_regimes.png',dpi=180);plt.close(fig)
    old_audit(root)
    save_trace_example(root,rows)




def save_trace_example(root,rows):
    """Post-hoc explanatory example, never used to select scenario membership."""
    references={(r['scenario_id'],r['method']):r for r in rows}
    comparisons=[]
    for r in rows:
        if (r['pool'],r['regime'],r['severity'],r['method'])!=('validation','contested',20,'B2'):
            continue
        p=references[(r['scenario_id'],'B1')]
        for uid,u in r['physical']['per_ue'].items():
            delta=u['batch_outcomes']['failure_fraction_lower']-p['physical']['per_ue'][uid]['batch_outcomes']['failure_fraction_lower']
            comparisons.append((delta,r['scenario_id'],uid))
    delta,scenario,uid=max(comparisons)
    cases=[]
    if delta>0:
        for method in ('B1','B2'):
            r=references[(scenario,method)];u=r['physical']['per_ue'][uid]
            frames=[]
            for f in r['diagnostic']['frames']:
                if f['step'] in (1249,1250,1260,2499,2500,3000) or f['first_pressure']:
                    frames.append(dict(step=f['step'],segment=f['segment'],quota=f['quotas']['URLLC'],
                        reserve=f['activated_reserve'],ue=next(x for x in f['ues'] if x['uid']==uid)))
            cases.append(dict(scenario_id=scenario,method=method,uid=uid,
                known_failure_percent=100*u['batch_outcomes']['failure_fraction_lower'],resource=r['ue_resources'][uid],
                outcomes=u['batch_outcomes'],recovery=r['physical']['recovery'],
                exogenous_sha256=r['physical']['exogenous_sha256'],quota_sha256=r['physical']['quota_sha256'],
                segments={k:v[uid] for k,v in r['diagnostic']['segments'].items()},frames=frames))
        if cases[0]['quota_sha256']!=cases[1]['quota_sha256'] or cases[0]['exogenous_sha256']!=cases[1]['exogenous_sha256']:
            raise AssertionError('Trace contrast must hold resource and exogenous inputs fixed')
    write(root/'trace_evidence.json',dict(selection='Largest per-UE B2-minus-B1 increase in validation contested severe cases; explanatory only',cases=cases))


def compatibility_stage(root,pilot):
    from tools.phase2c1_pilot import run_episode
    original=json.loads((ROOT/'backend/evaluations/phase2c1/heldout.json').read_text())
    checks=[]
    for old in original:
        if old['method'] not in ('B0','B1','B2') or old['scenario_seed']!=3101:
            continue
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
            current=run_episode(old['method'],old['severity'],3101,pilot)
        fields=['control_digest','deadline_assignment_sha256']
        for key in fields:
            if current[key]!=old[key]:raise AssertionError('Default legacy runner changed: '+key)
        for key in ('exogenous_sha256','quota_sha256','overall','per_class','per_ue','embb_throughput_mbps','mean_reserve_prbs'):
            if current['physical'][key]!=old['physical'][key]:raise AssertionError('Historical physical result changed: '+key)
        checks.append(dict(method=old['method'],seed=3101,severity=old['severity'],matches=True))
    if len(checks)!=9:raise AssertionError('Incomplete compatibility grid')
    write(root/'compatibility.json',dict(reference_episodes=checks,all_match=True,
        new_measurement_hooks_default_disabled=True,models_created=0,optimizer_updates=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',default=str(PROTOCOL));parser.add_argument('--output',required=True)
    parser.add_argument('--plots',action='store_true')
    parser.add_argument('--stage',choices=['probe','freeze','reference','summarize','compatibility'],required=True)
    a=parser.parse_args();p=json.loads(Path(a.protocol).read_text());validate_protocol(p);root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (len(os.sched_getaffinity(0))>4 or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or not 0<int(os.environ.get('PHASE_GUARD_STOP_BYTES','0'))<=int(2.5*1024**3)):
        raise RuntimeError('Use the CPU-only resource guard with a <=2.5-GiB stop threshold')
    if a.stage in ('probe','reference','compatibility'):
        from tools.phase2c1_pilot import configure_environment
        configure_environment()
    pilot=json.loads(PILOT_PROTOCOL.read_text())
    sources={name:sha(ROOT/f'backend/notebooks/Resilient_data/{name}_dl.csv') for name in ('embb','urllc')}
    started=time.perf_counter()
    if a.stage=='probe':probe_stage(root,p,pilot)
    elif a.stage=='freeze':freeze_stage(root,p)
    elif a.stage=='reference':characterize_stage(root,p,pilot)
    elif a.stage=='compatibility':compatibility_stage(root,pilot)
    else:summary_stage(root,p,plots=a.plots)
    if sources!={name:sha(ROOT/f'backend/notebooks/Resilient_data/{name}_dl.csv') for name in sources}:
        raise AssertionError('Original source datasets changed')
    write(root/f'{a.stage}_runtime.json',dict(elapsed_seconds=time.perf_counter()-started,
        protocol_sha256=sha(a.protocol),source_trace_sha256=sources,source_traces_unchanged=True,
        cpus=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        neural_models_created=0,optimizer_updates=0))


if __name__=='__main__':main()
