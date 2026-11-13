"""Independent paired outcomes and unprofiled end-to-end timing, without simulation."""
from collections import defaultdict
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from tools.phase2d_report import values
OUT=ROOT/'backend/evaluations/phase2e'
POLICIES=('P','F','G1')
COLORS=dict(P='#0072B2',F='#D55E00',G1='#009E73')
REGIMES=('feasible','contested','overloaded')


def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'legend.fontsize':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    manifest=json.loads((OUT/'independent_manifest.json').read_text())
    entries={e['id']:e for e in manifest['entries']}
    runs=read(OUT/'independent_runs.jsonl.gz')
    assert len(runs)==len(entries)*3
    assert {(r['scenario_id'],r['policy']) for r in runs}=={(e,p) for e in entries for p in POLICIES}
    flat=[];ue=[];recoveries=[]
    for r in runs:
        e=entries[r['scenario_id']];p=r['physical'];o=p['overall']
        identity=dict(scenario_id=e['id'],seed=e['seed'],regime=e['regime'],severity=e['severity'],policy=r['policy'])
        timing=r['timing']
        row=dict(identity,**values(r),
            known_failed=o['known_failed_batches'],offered=o['offered_batches'],
            unresolved_not_due=o['unresolved_not_due_batches'],unfinished_bytes=o['unfinished_bytes'],
            administrative_terminal_charges=(r.get('terminal') or {}).get('terminal_unresolved_charges',0),
            loop_seconds=r['elapsed_seconds'])
        for scope in ('all_ticks','nonterminal_ticks','cpu_ticks','scheduler_transaction','post_service_credit_update','ric_ticks','ordinary_ticks'):
            for key,value in timing[scope].items():row[scope+'_'+key]=value
        flat.append(row)
        for uid,u in p['per_ue'].items():
            b=u['batch_outcomes'];d=r['deficits'][uid];res=r['ue_resources'][uid];samples=d['valid_samples']
            ue.append(dict(identity,ue=uid,budget_class=u['budget_class'],budget_s=u['budget_s'],
                offered=b['offered'],known_failed=b['known_failed'],on_time=b['on_time'],
                failure_pct=100*b['failure_fraction_lower'],upper_pct=100*b['failure_fraction_upper'],
                unresolved_not_due=b['unresolved_not_due'],unfinished_overdue=b['unfinished_overdue'],
                completed=u['completed_batches'],late_completed=u['completed_over_budget_batches'],
                served_bytes=u['served_bytes'],unfinished_bytes=u['unfinished_bytes'],overflow_bytes=u['overflow_bytes'],
                mean_deficit=d['sum_deficit']/samples if samples else None,max_deficit=d['max_deficit'],
                positive_deficit_sample_fraction=d['positive_samples']/samples if samples else None,
                max_queue_growth=d['max_queue_growth'],mean_prbs=res['prbs'],
                capacity_bytes_per_prb=res['capacity_bytes_per_prb'],fluid_prb_demand=res['fluid_prbs'],
                served_offered_fraction=u['served_bytes']/max(1,b['offered']*e['batch_bytes'])))
        for i,w in enumerate(p['recovery']):recoveries.append(dict(identity,window=i,**w))
    write_csv(OUT/'scenarios.csv',flat);write_csv(OUT/'per_ue.csv',ue);write_csv(OUT/'recovery_windows.csv',recoveries)
    numeric=[k for k,v in flat[0].items() if isinstance(v,(int,float)) and k not in ('seed','severity')]
    summary=[]
    for g in REGIMES:
        for j in ('all',0,10,20):
            for policy in POLICIES:
                rs=[r for r in flat if r['regime']==g and r['policy']==policy and (j=='all' or r['severity']==j)]
                summary.append(dict(regime=g,severity=j,policy=policy,scenario_runs=len(rs),placements=len({r['seed'] for r in rs}),
                    **{k:st.mean(r[k] for r in rs) for k in numeric}))
    write_csv(OUT/'summary.csv',summary)
    paired=[];uepaired=[]
    lookup={(r['scenario_id'],r['policy']):r for r in flat}
    ulookup={(r['scenario_id'],r['policy'],r['ue']):r for r in ue}
    for r in flat:
        if r['policy']!='G1':continue
        for baseline in ('P','F'):
            b=lookup[r['scenario_id'],baseline]
            paired.append(dict(scenario_id=r['scenario_id'],seed=r['seed'],regime=r['regime'],severity=r['severity'],baseline=baseline,
                **{k:r[k]-b[k] for k in ('failure_pct','worst_pct','tight_pct','relaxed_pct','embb_mbps','reserve_prbs','recovered','all_ticks_p99_us','scheduler_transaction_p99_us')}))
            for u in [u for u in ue if u['scenario_id']==r['scenario_id'] and u['policy']=='G1']:
                bu=ulookup[u['scenario_id'],baseline,u['ue']]
                uepaired.append(dict(scenario_id=u['scenario_id'],seed=u['seed'],regime=u['regime'],severity=u['severity'],
                    baseline=baseline,ue=u['ue'],budget_class=u['budget_class'],
                    failure_delta_pp=u['failure_pct']-bu['failure_pct'],deficit_delta=u['mean_deficit']-bu['mean_deficit'],
                    service_fraction_delta=u['served_offered_fraction']-bu['served_offered_fraction']))
    write_csv(OUT/'paired_differences.csv',paired);write_csv(OUT/'paired_per_ue.csv',uepaired)
    consistency=[]
    for g in REGIMES:
        for j in (0,10,20):
            for baseline in ('P','F'):
                rs=[r for r in paired if r['regime']==g and r['severity']==j and r['baseline']==baseline]
                us=[r for r in uepaired if r['regime']==g and r['severity']==j and r['baseline']==baseline]
                consistency.append(dict(regime=g,severity=j,baseline=baseline,n_placements=len(rs),
                    worst_better=sum(r['worst_pct'] < -1e-9 for r in rs),worst_worse=sum(r['worst_pct']>1e-9 for r in rs),
                    worst_equal=sum(abs(r['worst_pct'])<=1e-9 for r in rs),
                    mean_worst_delta_pp=st.mean(r['worst_pct'] for r in rs),
                    min_worst_delta_pp=min(r['worst_pct'] for r in rs),max_worst_delta_pp=max(r['worst_pct'] for r in rs),
                    mean_aggregate_delta_pp=st.mean(r['failure_pct'] for r in rs),
                    max_aggregate_delta_pp=max(r['failure_pct'] for r in rs),
                    ue_harmed=sum(r['failure_delta_pp']>1e-9 for r in us),ue_benefited=sum(r['failure_delta_pp'] < -1e-9 for r in us),
                    ue_cases=len(us)))
    write_csv(OUT/'consistency.csv',consistency)
    profiles=[]
    for variant in ('original','optimized'):
        with (OUT/f'profile_{variant}.csv').open() as f:rows=list(csv.DictReader(f))
        for g in REGIMES:
            specs=[('telemetry','ue_telemetry.py','collect'),('cohort_window','ue_telemetry.py','_window'),
                ('FIFO_target','feasibility_gated_fairness.py','reachable_target'),
                ('service_window_update','feasibility_gated_fairness.py','observe'),
                ('post_service_credit','feasibility_gated_fairness.py','observe_service'),
                ('allocator_kernel','deadline_scheduler.py','deadline_allocation'),
                ('G1_credit_wrapper','feasibility_gated_fairness.py' if variant=='original' else 'equivalent_g1.py','__call__')]
            for name,filename,fn in specs:
                r=next(r for r in rows if r['regime']==g and r['function']==fn and r['file'].endswith(filename))
                profiles.append(dict(variant=variant,regime=g,component=name,calls=int(r['calls']),
                    own_seconds=float(r['own_seconds']),cumulative_seconds=float(r['cumulative_seconds'])))
    write_csv(OUT/'profile_components.csv',profiles)
    eq=read(OUT/'equivalence_runs.jsonl.gz');eqlookup={(r['scenario_id'],r['variant']):r for r in eq};speed=[]
    for sid in sorted({r['scenario_id'] for r in eq}):
        a,b=eqlookup[sid,'original'],eqlookup[sid,'optimized']
        speed.append(dict(scenario_id=sid,regime=a['regime'],severity=a['severity'],
            original_tick_p99_us=a['timing']['all_ticks']['p99_us'],optimized_tick_p99_us=b['timing']['all_ticks']['p99_us'],
            original_transaction_p99_us=a['timing']['scheduler_transaction']['p99_us'],optimized_transaction_p99_us=b['timing']['scheduler_transaction']['p99_us'],
            original_loop_seconds=a['elapsed_seconds'],optimized_loop_seconds=b['elapsed_seconds']))
    write_csv(OUT/'optimization_timings.csv',speed)
    figs=OUT/'figures';figs.mkdir(exist_ok=True);captions=[]
    def save(fig,name,caption):
        svg=figs/(name+'.svg');fig.savefig(svg);fig.savefig(figs/(name+'.pdf'))
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(Path('/tmp')/('phase2e-'+name+'.png'),dpi=130);plt.close(fig);captions.append(f'**{name}.** {caption}')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    comps=('telemetry','FIFO_target','service_window_update','allocator_kernel')
    for ax,g in zip(axes,REGIMES):
        for index,variant in enumerate(('original','optimized')):
            vals=[next(r['cumulative_seconds'] for r in profiles if r['variant']==variant and r['regime']==g and r['component']==c) for c in comps]
            ax.barh(np.arange(4)+index*.32,vals,height=.30,label=variant)
        ax.set_yticks(np.arange(4)+.16,comps);ax.set(title=g,xlabel='Profiled cumulative seconds');ax.legend(fontsize=7)
    save(fig,'01_profile_cost','One predeclared development j20 episode per regime and implementation. cProfile overhead inflates these times; nested cumulative times are not additive and are not deadline-budget measurements. Full function/call data: ../profile_components.csv and profile_*.prof.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for g,color in zip(REGIMES,('#0072B2','#D55E00','#009E73')):
        rs=[r for r in speed if r['regime']==g]
        for ax,key in zip(axes,('tick','transaction')):
            x=[r[f'original_{key}_p99_us'] for r in rs];y=[r[f'optimized_{key}_p99_us'] for r in rs]
            ax.scatter(x,y,label=g,color=color,s=32)
    for ax,title in zip(axes,('Entire simulator tick','Scheduler + G1 service update')):
        lim=max(*ax.get_xlim(),*ax.get_ylim());ax.plot([0,lim],[0,lim],color='gray',ls='--');ax.set(title=title,xlabel='Original p99 (µs)',ylabel='Optimized p99 (µs)');ax.grid(alpha=.2)
    axes[0].legend()
    save(fig,'02_equivalent_optimization','Ten paired development scenarios, separate unprofiled deterministic original/optimized runs with exact allocation/outcome equivalence. Points below diagonal are faster. Wall-clock variation and fixed original-first order limit timing inference. Data: ../optimization_timings.csv.')
    fig,axes=plt.subplots(3,3,figsize=(11,8),sharey=True)
    mapping=[]
    for i,g in enumerate(REGIMES):
        seeds=sorted({r['seed'] for r in flat if r['regime']==g})
        for index,seed in enumerate(seeds,1):mapping.append(dict(regime=g,placement_index=index,seed=seed))
        for col,j in enumerate((0,10,20)):
            ax=axes[i,col]
            for policy in POLICIES:
                rs=sorted((r for r in flat if r['regime']==g and r['severity']==j and r['policy']==policy),key=lambda r:r['seed'])
                ax.plot(range(1,len(rs)+1),[r['worst_pct'] for r in rs],marker='o',label=policy,color=COLORS[policy])
            ax.set(title=f'{g}, j={j}',xlabel='Independent placement index',xticks=range(1,len(seeds)+1));ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Worst-UE known failures (%)')
    axes[0,0].legend(ncol=3);fig.tight_layout()
    write_csv(figs/'placement_indices.csv',mapping)
    save(fig,'03_independent_placements','Every independent placement, regime and severity, with fixed P/F/G1 and identical reactive reserves. Three placements per stratum; no pooled-sample confidence interval. Placement indices map to actual seeds in placement_indices.csv. Data: ../scenarios.csv.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True,sharey=True)
    for ax,base in zip(axes,('P','F')):
        for g,color in zip(REGIMES,('#0072B2','#D55E00','#009E73')):
            rs=[r for r in paired if r['baseline']==base and r['regime']==g]
            ax.scatter([r['failure_pct'] for r in rs],[r['worst_pct'] for r in rs],color=color,label=g,s=30,alpha=.8)
        ax.axhline(0,color='gray',ls='--');ax.axvline(0,color='gray',ls='--');ax.set(title=f'G1 minus {base}',xlabel='Aggregate failure difference (pp)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Worst-UE failure difference (pp)');axes[1].legend()
    save(fig,'04_independent_tradeoffs','All27 paired independent scenarios per comparator. Lower-left improves both displayed metrics; this is not an all-UE or all-objective Pareto test. Overlapping zero points retained in numerical data. Data: ../paired_differences.csv.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),sharey=True)
    seeds=sorted({r['seed'] for r in ue if r['regime']=='contested'})
    for ax,base in zip(axes,('P','F')):
        mat=[]
        for seed in seeds:
            rs=sorted((r for r in uepaired if r['regime']=='contested' and r['severity']==20 and r['seed']==seed and r['baseline']==base),key=lambda r:r['ue'])
            mat.append([r['failure_delta_pp'] for r in rs])
        im=ax.imshow(mat,cmap='RdBu_r',vmin=-100,vmax=100,aspect='auto')
        ax.set(title=f'G1 minus {base}, contested j20',xlabel='URLLC UE index',xticks=range(8),yticks=range(len(seeds)),yticklabels=seeds)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Known failure difference (pp)',shrink=.85)
    save(fig,'05_individual_ue_harms','Each independent contested severe placement/UE; red means G1 harms that UE, blue improves it. Mean worst-UE improvement cannot establish individual protection. Data for every regime/class/severity: ../paired_per_ue.csv and ../per_ue.csv.')
    distributions=defaultdict(list)
    with gzip.open(OUT/'tick_times.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            distributions[r['regime'],r['policy'],'tick'].append(int(r['wall_ns'])/1000)
            distributions[r['regime'],r['policy'],'transaction'].append(int(r['transaction_ns'])/1000)
    fig,axes=plt.subplots(2,3,figsize=(11,6),sharey=True)
    for row,scope in enumerate(('transaction','tick')):
        for col,g in enumerate(REGIMES):
            ax=axes[row,col]
            for policy in POLICIES:
                vals=sorted(distributions[g,policy,scope]);stride=max(1,len(vals)//1500);indices=list(range(0,len(vals),stride))
                ax.plot([vals[i] for i in indices],[(i+1)/len(vals) for i in indices],label=policy,color=COLORS[policy])
            ax.axvline(1000,color='black',ls='--',lw=.9);ax.set_xscale('log');ax.set(title=f'{g}: {scope}',xlabel='Wall time (µs, log scale)');ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Empirical cumulative fraction')
    axes[0,0].legend()
    save(fig,'06_complete_tick_cost','Unprofiled independent execution. Transaction=URLLC allocator plus mandatory G1 post-service credit update; tick=entire engine.step including reference RIC/metrics and physics, excluding external audit/export. Baseline P/F service windows are measurement overhead in the full tick. Dashed line=1ms reference period, not real-time certification. Descriptive CDF across correlated ticks; no independent-sample inference. Data: ../tick_times.csv.gz and ../scenarios.csv.')
    (figs/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
    write_csv(OUT/'regime_means.csv',(r for r in summary if r['severity']=='all'))
    result=dict(regime_means=[r for r in summary if r['severity']=='all'],consistency=consistency,
        independent_runs=len(runs),independent_placements=len({e['seed'] for e in entries.values()}),
        original_checks=len(read(OUT/'independent_original_checks.jsonl.gz')))
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
