"""Measured, noncausal diagnostic tables and editable figures; no simulation."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics as st

OUT=Path(__file__).resolve().parents[1]/'evaluations/phase2d'
C3=OUT.parent/'phase2c3'
POLICIES=('P','F','OLD035','G0','G025','G05','G1')
COLORS=('#0072B2','#D55E00','#999999','#E69F00','#009E73','#CC79A7','#000000')


def jrows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def csvrows(path):
    opener=path.open if path.exists() else lambda: gzip.open(str(path)+'.gz','rt')
    with opener() as f:yield from csv.DictReader(f)


def csvwrite(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def values(r):
    p=r['physical'];o=p['overall'];u=p['resources']['URLLC']
    return dict(failure_pct=100*o['failure_fraction_lower'],worst_pct=100*p['worst_ue_failure_fraction'],
        upper_pct=100*o['failure_fraction_upper'],tight_pct=100*p['per_class']['tight']['failure_fraction_lower'],
        relaxed_pct=100*p['per_class']['relaxed']['failure_fraction_lower'],
        embb_mbps=p['embb_throughput_mbps'],embb_served_offered_pct=100*r['embb_served_offered_fraction'],
        reserve_prbs=p['mean_reserve_prbs'],unused_urllc_prbs=u['mean_unused_prbs'],
        scheduler_median_us=u['scheduler_median_us'],scheduler_p95_us=u['scheduler_p95_us'],scheduler_p99_us=u['scheduler_p99_us'],
        recovered=sum(not w['censored'] for w in p['recovery']),off_windows=len(p['recovery']))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'legend.fontsize':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    runs=list(jrows(OUT/'runs.jsonl.gz'))
    assert len(runs)==112,'Frozen study incomplete'
    # Publish segment totals only; consecutive deprivation is measured from the
    # separately sampled window stream, not inactive-segment accumulators.
    kept={'ticks','rate_deprived_ticks','positive_service_deprived_ticks'}
    cleaned=False
    for r in runs:
        for uid,counts in r['deprivation'].items():
            clean={k:v for k,v in counts.items() if k in kept}
            cleaned |= counts!=clean
            r['deprivation'][uid]=clean
    if cleaned:
        temp=OUT/'runs.clean.jsonl.gz'
        with gzip.open(temp,'wt') as f:
            for r in runs:f.write(json.dumps(r,allow_nan=False)+'\n')
        temp.replace(OUT/'runs.jsonl.gz')
    prototype=[r for r in runs if not r['replay']]
    replay=[r for r in runs if r['replay']]
    flat=[];ue=[];segments=[]
    for r in runs:
        identity={k:r[k] for k in ('run_id','policy','network_seed','episode','selection','scenario_id','regime','severity','replay')}
        if not r['replay']:flat.append(dict(identity,**values(r),elapsed_seconds=r['elapsed_seconds']))
        for uid,u in r['physical']['per_ue'].items():
            b=u['batch_outcomes'];d=r['diagnostic']['per_ue'][uid];res=r['ue_resources'][uid]
            ue.append(dict(identity,ue=uid,budget_class=u['budget_class'],budget_s=u['budget_s'],
                offered=b['offered'],known_failed=b['known_failed'],failure_pct=100*b['failure_fraction_lower'],
                unresolved_not_due=b['unresolved_not_due'],unfinished_overdue=b['unfinished_overdue'],
                late_completed=u['completed_over_budget_batches'],served_bytes=u['served_bytes'],
                unfinished_bytes=u['unfinished_bytes'],overflow_bytes=u['overflow_bytes'],
                bytes_per_prb_tick=d['capacity_min'],cqi=d['cqi'],mean_prbs=res['prbs'],
                fluid_demand_prbs=res['fluid_prbs']))
        for segment,users in r['diagnostic']['segments'].items():
            for uid,s in users.items():
                derived='on' if segment.startswith('on_') else ('no_jam' if segment=='no_jam' else 'off')
                depr=r['deprivation'].get(uid+'|'+derived,{})
                segments.append(dict(identity,ue=uid,segment=segment,ticks=s['ticks'],
                    offered_bytes=s['offered_bytes'],served_bytes=s['served_bytes'],
                    served_offered_ratio=s['served_bytes']/max(1,s['offered_bytes']),
                    mean_prbs=s['granted_prbs']/s['ticks'],max_hol_s=s['max_hol_s'],
                    max_no_service_ticks=s['max_no_service_ticks'],
                    rate_deprived_ticks_same_onoff_group=depr.get('rate_deprived_ticks',0),
                    positive_service_deprived_ticks_same_onoff_group=depr.get('positive_service_deprived_ticks',0)))
    csvwrite(OUT/'deprivation_counts.csv',(dict(run_id=r['run_id'],policy=r['policy'],selection=r['selection'],
        scenario_id=r['scenario_id'],ue=key.rsplit('|',1)[0],segment_group=key.rsplit('|',1)[1],**counts)
        for r in runs for key,counts in r['deprivation'].items()))
    csvwrite(OUT/'prototype_scenarios.csv',flat);csvwrite(OUT/'per_ue.csv',ue);csvwrite(OUT/'service_segments.csv',segments)
    summary=[]
    for g in ('feasible','contested','overloaded'):
        for j in ('all',0,10,20):
            for policy in POLICIES:
                rs=[r for r in flat if r['regime']==g and r['policy']==policy and (j=='all' or r['severity']==j)]
                if rs:summary.append(dict(regime=g,severity=j,policy=policy,runs=len(rs),
                    **{k:st.mean(r[k] for r in rs) for k in values(prototype[0])}))
    csvwrite(OUT/'prototype_summary.csv',summary)
    by={(r['scenario_id'],r['policy']):r for r in flat};paired=[]
    for r in flat:
        if r['policy'] in ('P','F'):continue
        for baseline in ('P','F'):
            b=by[r['scenario_id'],baseline]
            matched=[u for u in ue if not u['replay'] and u['scenario_id']==r['scenario_id'] and u['policy']==r['policy']]
            bu={u['ue']:u for u in ue if not u['replay'] and u['scenario_id']==r['scenario_id'] and u['policy']==baseline}
            harms=sum(u['failure_pct']>bu[u['ue']]['failure_pct']+1e-9 for u in matched)
            benefits=sum(u['failure_pct']<bu[u['ue']]['failure_pct']-1e-9 for u in matched)
            paired.append(dict(scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],policy=r['policy'],baseline=baseline,
                **{k:r[k]-b[k] for k in ('failure_pct','worst_pct','embb_mbps','reserve_prbs')},ues_harmed=harms,ues_benefited=benefits))
    csvwrite(OUT/'prototype_paired.csv',paired)
    # All recorded A4 candidates, not just replays, describe behavior over C3.
    recorded=[]
    for r in jrows(C3/'validation.jsonl.gz'):
        if r['method']!='A4':continue
        recorded.append(dict(network_seed=r['network_seed'],episode=r['candidate_episode'],
            scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],
            failure_pct=100*r['physical']['overall']['failure_fraction_lower'],
            worst_pct=100*r['physical']['worst_ue_failure_fraction'],
            f_mode_pct=100*r['mode_service_ticks'].get('FEASIBILITY_AWARE',0)/r['steps'],
            switches_per_decision=r['switches_per_decision'],reserve_prbs=r['physical']['mean_reserve_prbs']))
    csvwrite(OUT/'a4_recorded_candidates.csv',recorded)
    # Keep only replay decision aggregates; stream the much larger per-UE CSV.
    windows={};sampled_runs=defaultdict(lambda:dict(current=0,longest=0,observed=0,positive_deficit_samples=0))
    for r in csvrows(OUT/'ue_windows.csv'):
        if r['replay']!='True':continue
        key=(r['run_id'],int(r['step']))
        w=windows.setdefault(key,dict(max_deficit=0.,max_queue_growth=0.,min_service_ratio=1.,mean_deficit=0.,members=0))
        d=float(r['deficit']);a=float(r['offered_window']);b=float(r['served_window'])
        w['max_deficit']=max(w['max_deficit'],d);w['max_queue_growth']=max(w['max_queue_growth'],float(r['queue_growth']))
        w['min_service_ratio']=min(w['min_service_ratio'],b/max(1,a));w['mean_deficit']+=d;w['members']+=1
        if r['full_window']=='True':
            s=sampled_runs[r['run_id'],r['ue']];bad=d>0 and float(r['queued_bytes'])>0
            s['current']=s['current']+1 if bad else 0;s['longest']=max(s['longest'],s['current']);s['observed']+=1;s['positive_deficit_samples']+=bad
    csvwrite(OUT/'sampled_deprivation.csv',(dict(run_id=k[0],ue=k[1],longest_consecutive_10ms_samples=s['longest'],
        full_window_samples=s['observed'],deprived_samples=s['positive_deficit_samples']) for k,s in sampled_runs.items()))
    outcomes={(r['run_id'],int(r['step'])):r for r in csvrows(OUT/'interval_outcomes.csv') if r['replay']=='True'}
    decisions=[];last={};conditions=(0,1,6,7,8,9,14,15,16,17,18)
    for r in jrows(OUT/'decisions.jsonl.gz'):
        if not r['replay']:continue
        before=last.get(r['run_id']);state=r['state'];w=windows[r['run_id'],r['step']]
        change=max(abs(state[i]-before['state'][i]) for i in conditions) if before else None
        prior2=before.get('_previous_mode') if before else None
        reversal=bool(before and r['switched'] and before['switched'] and r['mode_after']==prior2)
        current=outcomes[r['run_id'],r['step']];nxt=outcomes.get((r['run_id'],r['step']+10))
        row=dict(run_id=r['run_id'],policy=r['policy'],network_seed=r['network_seed'],episode=r['episode'],selection=r['selection'],
            regime=r['regime'],severity=r['severity'],scenario_id=r['scenario_id'],step=r['step'],
            mode=r['mode_after'],switched=r['switched'],reversal_20ms=reversal,
            unchanged_capacity_reversal=bool(reversal and state[10]==before['state'][10]==before.get('_previous_capacity')),
            condition_linf_change=change,**{k:w[k] for k in ('max_deficit','max_queue_growth','min_service_ratio')},
            worst_risk=r['risk'][4],worst_valid=r['risk_valid'][4],
            saturated=bool(r['risk_valid'][4] and r['risk'][4]>=1-1e-12),
            urllc_quota=r['quota_urllc'],reserve=r['reserve'],
            following_known_failure_events=int(nxt['known_failed'])-int(current['known_failed']) if nxt else None,
            following_on_time_completions=int(nxt['on_time_completed'])-int(current['on_time_completed']) if nxt else None)
        row['_state']=state;decisions.append(row)
        r['_previous_mode']=before['mode_after'] if before else None
        r['_previous_capacity']=before['state'][10] if before else None
        last[r['run_id']]=r
    csvwrite(OUT/'decision_diagnostics.csv',({k:v for k,v in r.items() if k!='_state'} for r in decisions))
    feature_names=['embb_demand','urllc_demand','embb_quota','urllc_quota','embb_base','urllc_base','embb_queue',
        'urllc_queue','embb_throughput','urllc_throughput','effective_capacity','reserve','previous_delta_e','previous_delta_u',
        'HOL_pressure_transform','active_overdue','tight_failure','relaxed_failure','worst_failure','F_mode']
    feature_names += [f'valid_{i}' for i in range(6)]+['tight_exposure','relaxed_exposure','tight_known','relaxed_known']
    discrimination=[]
    for g in ('feasible','contested','overloaded'):
        for saturated in (False,True):
            ds=[r for r in decisions if r['policy']=='A4' and r['selection']=='aggregate' and r['regime']==g and r['worst_valid'] and r['saturated']==saturated]
            if not ds:continue
            for i,name in enumerate(feature_names):
                vs=[r['_state'][i] for r in ds]
                discrimination.append(dict(regime=g,worst_saturated=saturated,feature=name,index=i,n=len(vs),minimum=min(vs),maximum=max(vs),mean=st.mean(vs),std=st.pstdev(vs)))
            for name in ('max_deficit','max_queue_growth','min_service_ratio'):
                vs=[r[name] for r in ds]
                discrimination.append(dict(regime=g,worst_saturated=saturated,feature=name,index=-1,n=len(vs),minimum=min(vs),maximum=max(vs),mean=st.mean(vs),std=st.pstdev(vs)))
    csvwrite(OUT/'state_discrimination.csv',discrimination)
    switch_summary=[]
    for g in ('feasible','contested','overloaded'):
        for selected in ('aggregate','fairness'):
            for switched in (False,True):
                ds=[r for r in decisions if r['policy']=='A4' and r['regime']==g and r['selection']==selected and r['switched']==switched and r['following_known_failure_events'] is not None]
                if ds:switch_summary.append(dict(regime=g,selection=selected,switched=switched,n=len(ds),
                    mean_new_failure_events=st.mean(r['following_known_failure_events'] for r in ds),
                    mean_on_time_completions=st.mean(r['following_on_time_completions'] for r in ds),
                    median_condition_change=st.median(r['condition_linf_change'] for r in ds if r['condition_linf_change'] is not None),
                    reversals_20ms=sum(r['reversal_20ms'] for r in ds),unchanged_capacity_reversals=sum(r['unchanged_capacity_reversal'] for r in ds)))
    csvwrite(OUT/'switch_correlations.csv',switch_summary)
    strata=defaultdict(list)
    for r in decisions:
        if r['policy']=='A4':strata[(r['regime'],r['severity'],r['network_seed'],r['episode'],r['selection'],r['switched'])].append(r)
    csvwrite(OUT/'switch_strata.csv',(dict(regime=k[0],severity=k[1],network_seed=k[2],episode=k[3],selection=k[4],switched=k[5],
        n=len(rs),new_failure_events=st.mean(r['following_known_failure_events'] for r in rs),
        on_time_completions=st.mean(r['following_on_time_completions'] for r in rs)) for k,rs in strata.items()))
    choices=[]
    for g in ('feasible','contested','overloaded'):
        ds=[r for r in decisions if r['policy']=='A4' and r['selection']=='aggregate' and r['regime']==g]
        choices.append(dict(regime=g,decisions=len(ds),f_choices=sum(r['mode']=='FEASIBILITY_AWARE' for r in ds),
            switches=sum(r['switched'] for r in ds),
            switches_with_identical_condition_vector=sum(r['switched'] and r['condition_linf_change']==0 for r in ds),
            valid_worst_decisions=sum(r['worst_valid'] for r in ds),saturated_worst_decisions=sum(r['saturated'] for r in ds)))
    csvwrite(OUT/'a4_decision_summary.csv',choices)
    shadows=[];lookup={(r['run_id'],r['step']):r for r in decisions}
    for r in jrows(OUT/'shadows.jsonl.gz'):
        d=lookup[r['run_id'],r['step']-1]
        chosen=r['p_timely'] if r['mode']=='PROPORTIONAL' else r['f_timely']
        shadows.append(dict(run_id=r['run_id'],policy=r['policy'],selection=r['selection'],regime=r['regime'],severity=r['severity'],step=r['step'],
            mode=r['mode'],switched=d['switched'],p_timely=r['p_timely'],f_timely=r['f_timely'],
            chosen_timely=chosen,best_timely=max(r['p_timely'],r['f_timely']),gap=max(r['p_timely'],r['f_timely'])-chosen))
    csvwrite(OUT/'shadow_completions.csv',shadows)
    shadow_summary=[]
    for g in ('feasible','contested','overloaded'):
        ss=[r for r in shadows if r['policy']=='A4' and r['selection']=='aggregate' and r['regime']==g]
        shadow_summary.append(dict(regime=g,n=len(ss),p_strictly_better=sum(r['p_timely']>r['f_timely'] for r in ss),
            f_strictly_better=sum(r['f_timely']>r['p_timely'] for r in ss),
            chosen_strictly_worse=sum(r['gap']>0 for r in ss),switch_to_strictly_worse=sum(r['switched'] and r['gap']>0 for r in ss),
            missed_immediate_opportunities=sum(r['gap'] for r in ss)))
    csvwrite(OUT/'shadow_summary.csv',shadow_summary)
    # Exact math-invariant results on all ten frozen prototype scenarios.
    for sid in {r['scenario_id'] for r in flat}:
        p=next(r for r in prototype if r['scenario_id']==sid and r['policy']=='F')
        z=next(r for r in prototype if r['scenario_id']==sid and r['policy']=='G0')
        assert p['physical']['overall']==z['physical']['overall']
        assert p['physical']['per_ue']==z['physical']['per_ue']
    figs=OUT/'figures';figs.mkdir(exist_ok=True);captions=[]
    def save(fig,name,caption):
        svg=figs/(name+'.svg');fig.savefig(svg);fig.savefig(figs/(name+'.pdf'))
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(Path('/tmp')/('phase2d-'+name+'.png'),dpi=130);plt.close(fig)
        captions.append(f'**{name}.** {caption}')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for g,marker in zip(('feasible','contested','overloaded'),('o','s','^')):
        rs=[r for r in recorded if r['regime']==g]
        axes[0].scatter([r['switches_per_decision'] for r in rs],[r['failure_pct'] for r in rs],s=22,marker=marker,label=g,alpha=.7)
        axes[1].scatter([r['f_mode_pct'] for r in rs],[r['worst_pct'] for r in rs],s=22,marker=marker,label=g,alpha=.7)
    axes[0].set(xlabel='Mode switches / decision',ylabel='Known batch failures (%)');axes[1].set(xlabel='Time in F mode (%)',ylabel='Worst-UE batch failures (%)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    save(fig,'01_a4_recorded_behavior','All 108 recorded A4 candidate/scenario runs; learner seeds12026/22026, checkpoints3/6/9. Associations are not causal; workloads and reserve actions differ. Data: ../a4_recorded_candidates.csv.')
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),sharey=True)
    for ax,g in zip(axes,('feasible','contested','overloaded')):
        for switched,color in ((False,'#0072B2'),(True,'#D55E00')):
            vs=sorted(r['condition_linf_change'] for r in decisions if r['policy']=='A4' and r['selection']=='aggregate' and r['regime']==g and r['switched']==switched and r['condition_linf_change'] is not None)
            if vs:ax.plot(vs,[(i+1)/len(vs) for i in range(len(vs))],color=color,label='switch' if switched else 'stay')
        ax.set(title=g,xlabel='UE-condition feature change (L∞)');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Empirical cumulative fraction')
    save(fig,'02_switch_conditions','Aggregate-selected A4, both learner seeds. Feature change excludes mode, previous actions, quotas and exposure flags; includes workload/service/queue and deadline-risk inputs. This does not identify measurement noise. Data: ../decision_diagnostics.csv.')
    # Inspect the exact C3 94.667-pp harm case with 1-ms measured traces.
    harm=defaultdict(list)
    for r in csvrows(OUT/'harm_ticks.csv'):
        if r['selection'] not in ('aggregate','harm_comparator') or not r['ue'].endswith('_1'):continue
        if int(r['step'])%10==0:harm[r['policy']].append(r)
    fig,axes=plt.subplots(3,1,figsize=(8,6),sharex=True)
    for policy,color in (('A1','#0072B2'),('A4','#CC79A7')):
        rs=harm[policy];x=[int(r['step'])/1000 for r in rs]
        axes[0].plot(x,[float(r['queued_bytes'])/1000 for r in rs],label=policy,color=color)
        axes[1].plot(x,[float(r['deficit']) for r in rs],color=color)
        axes[2].plot(x,[float(r['quota']) for r in rs],color=color)
    for ax in axes:ax.axvline(1.25,color='gray',ls='--');ax.axvline(2.5,color='gray',ls='--');ax.grid(alpha=.2)
    axes[0].set_ylabel('UE1 backlog (kB)');axes[0].legend();axes[1].set_ylabel('100-ms deficit');axes[2].set(ylabel='URLLC quota (PRBs)',xlabel='Simulation time (s)')
    save(fig,'03_exact_harm_case','Placement53260288 j20, learner12026, aggregate-selected A1/A4 ep6. Original 94.667-pp degradation case. Curves sampled every10ms from stored 1-ms observations; dashed lines mark OFF/ON. Different learned quotas are visible, so scheduler-only causality is not inferred. Data: ../harm_ticks.csv.gz.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    ds=[r for r in decisions if r['policy']=='A4' and r['selection']=='aggregate' and r['regime']=='overloaded' and r['worst_valid']]
    # Vector scatter is thinned deterministically for readability; full numbers remain.
    ds=ds[::5]
    axes[0].scatter([r['worst_risk'] for r in ds],[r['max_deficit'] for r in ds],s=5,alpha=.25)
    axes[1].scatter([r['step']/1000 for r in ds],[r['max_deficit'] for r in ds],s=5,alpha=.25)
    axes[0].set(xlabel='Worst matured UE failure fraction',ylabel='Maximum measured service deficit');axes[1].set(xlabel='Simulation time (s)',ylabel='Maximum measured service deficit')
    save(fig,'04_risk_saturation','Overloaded aggregate-selected A4 replays, both learner seeds. Every fifth decision plotted; full conditional ranges/stds for all30 inputs and service deficits are in ../state_discrimination.csv. Saturation is a valid failure fraction reaching1, not an arbitrary encoder clip.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for policy,color in zip(POLICIES,COLORS):
        rs=[r for r in flat if r['regime']=='contested' and r['policy']==policy]
        axes[0].scatter([st.mean(r['failure_pct'] for r in rs)],[st.mean(r['worst_pct'] for r in rs)],color=color,label=policy,s=45)
        for j in (0,10,20):
            ss=[r for r in rs if r['severity']==j]
            axes[1].scatter([j+(.15*POLICIES.index(policy))],[st.mean(r['worst_pct'] for r in ss)],color=color,s=28)
    axes[0].set(xlabel='Aggregate known failures (%)',ylabel='Worst-UE known failures (%)');axes[1].set(xlabel='Jammer blocked PRBs (offset by policy)',ylabel='Worst-UE known failures (%)')
    axes[0].legend(ncol=2)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'05_fixed_parameter_tradeoff','Exploratory contested C2 validation: two placements x three severities, same reactive reserve trajectory. All seven policies are reported, with no lambda optimization or confidence interval. G0 equals F, so points overlap. Data: ../prototype_scenarios.csv and ../prototype_summary.csv.')
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),sharey=True)
    for ax,g in zip(axes,('feasible','contested','overloaded')):
        rs=[r for r in summary if r['regime']==g and r['severity']=='all']
        for policy,color in zip(POLICIES,COLORS):
            r=next(r for r in rs if r['policy']==policy)
            ax.scatter([r['scheduler_median_us']],[r['scheduler_p99_us']],label=policy,color=color,s=30)
        ax.set(title=g,xlabel='Mean per-run median (µs)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean per-run p99 (µs)');axes[2].legend(ncol=2,fontsize=7)
    save(fig,'06_scheduler_cost','Wall-clock allocation cost includes the experimental gate/telemetry query but excludes post-service credit/window updates, diagnostics and I/O. CPU affinity6,7,11,14, one numerical thread; no hard real-time guarantee. Feasible/overloaded use j20 only; contested uses j0/10/20. Data: ../prototype_summary.csv.')
    (figs/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
    results=dict(regime_summary=[r for r in summary if r['severity']=='all'],shadow=shadow_summary,
        switch_correlations=switch_summary,zero_fraction_matches_f_all_10_scenarios=True,
        raw_runs=len(runs),replays=len(replay),prototype_runs=len(prototype))
    (OUT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
