"""Generate matched validation tables and vector figures without additional simulation."""
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
import statistics as st

OUT=Path(__file__).resolve().parents[1]/'evaluations/phase2c3'
METHODS=('A1','A2','A3','A4')
REGIMES=('feasible','contested','overloaded')
COLORS=dict(zip(METHODS,('#0072B2','#D55E00','#009E73','#CC79A7')))


def read_rows(name):
    path=OUT/f'{name}.jsonl'
    if path.exists():
        with path.open() as f:return [json.loads(line) for line in f]
    with gzip.open(str(path)+'.gz','rt') as f:return [json.loads(line) for line in f]


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def metrics(r):
    p=r['physical'];o=p['overall'];recovery=p['recovery']
    return dict(failure_pct=100*o['failure_fraction_lower'],worst_ue_pct=100*p['worst_ue_failure_fraction'],
        tight_pct=100*p['per_class']['tight']['failure_fraction_lower'],
        relaxed_pct=100*p['per_class']['relaxed']['failure_fraction_lower'],
        upper_pct=100*o['failure_fraction_upper'],unresolved_not_due_batches=o['unresolved_not_due_batches'],
        unfinished_bytes=o['unfinished_bytes'],known_failed_batches=o['known_failed_batches'],
        offered_batches=o['offered_batches'],
        embb_mbps=p['embb_throughput_mbps'],embb_served_offered_pct=100*r['embb_served_offered_fraction'],
        reserve_prbs=p['mean_reserve_prbs'],switches_per_decision=r['switches_per_decision'],
        feasibility_mode_pct=100*r['mode_service_ticks'].get('FEASIBILITY_AWARE',0)/r['steps'],
        off_windows=len(recovery),recovered_windows=sum(not x['censored'] for x in recovery),
        elapsed_seconds=r['elapsed_seconds'])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':150,'savefig.bbox':'tight'})
    training=read_rows('training');validation=read_rows('validation')
    for rows in (training,validation):
        fingerprints={}
        for r in rows:
            key=(r['scenario_seed'],r['severity'])
            value=(r['physical']['exogenous_sha256'],r['deadline_assignment_sha256'])
            if key in fingerprints and fingerprints[key]!=value:
                raise AssertionError('Paired scenario mismatch')
            fingerprints[key]=value
    if len(training)!=72 or len(validation)!=432:
        raise ValueError('Incomplete frozen pilot')
    selection=json.loads((OUT/'selection.json').read_text())
    protocol=json.loads((OUT/'protocol.json').read_text())
    seeds=protocol['network_seeds']
    expected={(m,seed,ep) for m in METHODS for seed in seeds for ep in protocol['checkpoint_after_episodes']}
    observed=defaultdict(set)
    for r in validation:
        observed[(r['method'],r['network_seed'],r['candidate_episode'])].add(r['scenario_id'])
    if set(observed)!=expected or any(len(v)!=18 for v in observed.values()):
        raise AssertionError('Incomplete method/seed/checkpoint validation grid')
    if any(r['return_objective']!='joint-reward-v1' for r in training+validation):
        raise AssertionError('Mismatched reward objectives')
    figs=OUT/'figures';figs.mkdir(exist_ok=True)
    captions=[]
    def save(fig,name,caption):
        svg=figs/f'{name}.svg'
        fig.savefig(svg);fig.savefig(figs/f'{name}.pdf')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        if name in ('01_training_reward','07_aggregate_worst_tradeoff'):
            fig.savefig(Path('/tmp')/f'phase2c3-{name}.png')
        plt.close(fig)
        captions.append(f'**{name}.** {caption}')
    flat=[];per_ue=[];recoveries=[]
    for r in validation:
        identity={k:r[k] for k in ('method','network_seed','candidate_episode','total_transitions','scenario_id','regime','severity')}
        flat.append(dict(identity,**metrics(r)))
        for i,window in enumerate(r['physical']['recovery']):
            recoveries.append(dict(identity,window=i,**window))
        for uid,u in r['physical']['per_ue'].items():
            b=u['batch_outcomes'];resource=r['ue_resources'][uid]
            per_ue.append(dict(identity,ue=uid,budget_class=u['budget_class'],budget_s=u['budget_s'],
                offered=b['offered'],known_failed=b['known_failed'],on_time=b['on_time'],
                unresolved_not_due=b['unresolved_not_due'],unfinished_overdue=b['unfinished_overdue'],
                failure_pct=100*b['failure_fraction_lower'],upper_pct=100*b['failure_fraction_upper'],
                completed=u['completed_batches'],late_completed=u['completed_over_budget_batches'],
                served_bytes=u['served_bytes'],unfinished_bytes=u['unfinished_bytes'],overflow_bytes=u['overflow_bytes'],
                mean_prbs=resource['prbs'],bytes_per_prb_tick=resource['capacity_bytes_per_prb'],
                mean_fluid_demand_prbs=resource['fluid_prbs']))
    write_csv(OUT/'validation_scenarios.csv',flat);write_csv(OUT/'per_ue.csv',per_ue)
    write_csv(OUT/'recovery_windows.csv',recoveries)
    trainflat=[]
    for r in training:
        trainflat.append(dict(method=r['method'],network_seed=r['network_seed'],episode=r['episode'],
            transitions=r['total_transitions'],regime=r['regime'],severity=r['severity'],reward=r['return_value'],
            valid_action_fraction=r['mean_valid_action_fraction'],**metrics(r)))
    write_csv(figs/'training_data.csv',trainflat)
    groups=defaultdict(list)
    for r in flat:groups[(r['method'],r['network_seed'],r['candidate_episode'],r['total_transitions'])].append(r)
    curves=[]
    for (method,seed,episode,transitions),rs in sorted(groups.items()):
        curves.append(dict(method=method,network_seed=seed,episode=episode,transitions=transitions,
            **{k:st.mean(r[k] for r in rs) for k in metrics(validation[0])}))
    write_csv(figs/'validation_curves.csv',curves)
    # No pseudoreplicated confidence bands: each plotted band is the range of
    # the two initialization replicates after averaging the same 18 scenarios.
    def lines(ax,key):
        for method in METHODS:
            xs=sorted({r['transitions'] for r in curves if r['method']==method})
            ys=[[r[key] for r in curves if r['method']==method and r['transitions']==x] for x in xs]
            ax.plot(xs,[st.mean(v) for v in ys],marker='o',color=COLORS[method],label=method)
            ax.fill_between(xs,[min(v) for v in ys],[max(v) for v in ys],color=COLORS[method],alpha=.12)
        ax.set_xlabel('Training transitions per learner');ax.grid(alpha=.2);ax.legend(ncol=2)
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    marks={'feasible':'o','contested':'s','overloaded':'^'}
    for ax,method in zip(axes.flat,METHODS):
        for seed,style in zip(seeds,('-','--')):
            rs=[r for r in trainflat if r['method']==method and r['network_seed']==seed]
            ax.plot([r['transitions'] for r in rs],[r['reward'] for r in rs],style,color=COLORS[method],alpha=.6)
            for r in rs:
                ax.scatter(r['transitions'],r['reward'],marker=marks[r['regime']],s=30,
                    facecolors=COLORS[method] if r['severity']==20 else ('white' if r['severity']==0 else '#999999'),
                    edgecolors=COLORS[method],zorder=3)
        ax.set_title(method);ax.set_ylabel('Episode return (joint-reward-v1)');ax.set_xlabel('Training transitions');ax.grid(alpha=.2)
    handles=[Line2D([],[],marker=m,color='gray',linestyle='',label=g) for g,m in marks.items()]
    handles += [Line2D([],[],color='black',linestyle=s,label=f'NN seed {seed}') for seed,s in zip(seeds,('-','--'))]
    fig.legend(handles=handles,loc='upper center',ncol=5);fig.tight_layout(rect=(0,0,1,.92))
    save(fig,'01_training_reward','Matched reward for all methods. Markers distinguish regimes; white/gray/method-color fills denote j=0/10/20 PRBs. Lines distinguish NN seeds 12026/22026; connected points change scenario difficulty and are not a convergence curve. Data: training_data.csv.')
    for name,key,ylabel in [('02_validation_failure','failure_pct','Known failed offered batches (%)'),
                            ('03_validation_worst','worst_ue_pct','Worst-UE failed offered batches (%)'),
                            ('04_validation_reserve','reserve_prbs','Mean activated reserve (PRBs)')]:
        fig,ax=plt.subplots(figsize=(6,3.5));lines(ax,key);ax.set_ylabel(ylabel)
        save(fig,name,'Greedy fixed-validation mean over 18 paired scenarios (nine equally weighted strata), at checkpoints 3/6/9. Lines: mean of NN seeds 12026/22026; shaded band: their range, not a confidence interval. No untrained baseline was evaluated. Data: validation_curves.csv.')
    fig,axes=plt.subplots(1,2,figsize=(9,3.4));lines(axes[0],'feasibility_mode_pct');lines(axes[1],'switches_per_decision')
    axes[0].set_ylabel('Time in feasibility-aware mode (%)');axes[1].set_ylabel('Mode switches / decision')
    save(fig,'05_mode_selection','Fixed validation mode occupancy and switching for all matched controllers; A4 is the joint controller. Mode occupancy includes the initial 10-ms bootstrap. Two-seed means and ranges; data: validation_curves.csv.')
    updates=list(csv.DictReader((OUT/'updates.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    binned=[]
    for method in METHODS:
        for seed,style in zip(seeds,('-','--')):
            rs=[r for r in updates if r['method']==method and int(r['network_seed'])==seed]
            bins=[rs[i:i+100] for i in range(0,len(rs),100)]
            xs=[int(b[-1]['optimizer_updates']) for b in bins]
            for ax,key,label in zip(axes,('max_abs_q','loss'),('Mean minibatch max |Q|','Mean Huber TD loss')):
                vals=[st.mean(float(r[key]) for r in b) for b in bins]
                ax.plot(xs,vals,style,color=COLORS[method],label=f'{method}, {seed}')
                ax.set_xlabel('Optimizer updates');ax.set_ylabel(label);ax.grid(alpha=.2)
            for b in bins:
                binned.append(dict(method=method,network_seed=seed,update_end=int(b[-1]['optimizer_updates']),
                    loss=st.mean(float(r['loss']) for r in b),max_abs_q=st.mean(float(r['max_abs_q']) for r in b)))
    axes[1].legend(ncol=2,fontsize=6)
    save(fig,'06_learning_diagnostics','Nonoverlapping 100-update means (last bin may be shorter). Q magnitude is the minibatch maximum absolute selected-action Q before its optimizer step, not a convergence guarantee. Same Huber loss for every method. Data: learning_bins.csv and ../updates.csv.')
    write_csv(figs/'learning_bins.csv',binned)
    fig,ax=plt.subplots(figsize=(6,4))
    for method in METHODS:
        for seed,marker in zip(seeds,('o','s')):
            rs=[r for r in curves if r['method']==method and r['network_seed']==seed]
            ax.plot([r['failure_pct'] for r in rs],[r['worst_ue_pct'] for r in rs],':',color=COLORS[method],alpha=.5)
            for r in rs:
                ax.scatter(r['failure_pct'],r['worst_ue_pct'],color=COLORS[method],marker=marker)
                ax.annotate(str(r['episode']),(r['failure_pct'],r['worst_ue_pct']),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.set_xlabel('Mean aggregate known batch failure (%)');ax.set_ylabel('Mean worst-UE known batch failure (%)');ax.grid(alpha=.2)
    ax.legend(handles=[Line2D([],[],color=COLORS[m],label=m) for m in METHODS])
    save(fig,'07_aggregate_worst_tradeoff','All 24 method/NN-seed/checkpoint points, with episode numbers. Circles/squares denote NN seeds 12026/22026; dotted lines follow training. Both axes are minimized. No fairness SLA is assumed. Data: validation_curves.csv and ../selection.json.')
    (figs/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
    # Publish both prespecified selections, without rerunning or choosing a winner.
    selected=[];selected_ue=[];summaries=[];deltas=[]
    for criterion in ('aggregate','fairness'):
        for s in selection:
            episode=s['selected'][criterion]['episode']
            match=lambda r:r['method']==s['method'] and r['network_seed']==s['network_seed'] and r['candidate_episode']==episode
            selected.extend(dict(criterion=criterion,**r) for r in flat if match(r))
            selected_ue.extend(dict(criterion=criterion,**r) for r in per_ue if match(r))
        rows=[r for r in selected if r['criterion']==criterion]
        for method in METHODS:
            for regime in ('all',*REGIMES):
                for severity in ('all',0,10,20):
                    rs=[r for r in rows if r['method']==method and (regime=='all' or r['regime']==regime) and (severity=='all' or r['severity']==severity)]
                    if not rs:continue
                    summaries.append(dict(criterion=criterion,method=method,regime=regime,severity=severity,
                        nn_seeds=len(seeds),scenario_pairs=len(rs)//len(seeds),
                        **{k:st.mean(r[k] for r in rs) for k in metrics(validation[0])},
                        nn_failure_range_pp=max(st.mean(r['failure_pct'] for r in rs if r['network_seed']==seed) for seed in seeds)-min(st.mean(r['failure_pct'] for r in rs if r['network_seed']==seed) for seed in seeds)))
        bykey={(r['method'],r['network_seed'],r['scenario_id']):r for r in rows}
        for r in rows:
            if r['method']!='A4':continue
            for base in ('A1','A2','A3'):
                b=bykey[(base,r['network_seed'],r['scenario_id'])]
                deltas.append(dict(criterion=criterion,network_seed=r['network_seed'],scenario_id=r['scenario_id'],
                    regime=r['regime'],severity=r['severity'],comparison='A4-minus-'+base,
                    **{key:r[key]-b[key] for key in ('failure_pct','worst_ue_pct','embb_mbps','reserve_prbs','switches_per_decision','recovered_windows')}))
    write_csv(OUT/'selected_scenarios.csv',selected);write_csv(OUT/'selected_per_ue.csv',selected_ue)
    write_csv(OUT/'summary.csv',summaries);write_csv(OUT/'paired_differences.csv',deltas)
    # Selected per-UE paired harm counts; lower failure is better.
    ue_deltas=[]
    bykey={(r['criterion'],r['method'],r['network_seed'],r['scenario_id'],r['ue']):r for r in selected_ue}
    for r in selected_ue:
        if r['method']!='A4':continue
        for base in ('A1','A2','A3'):
            b=bykey[(r['criterion'],base,r['network_seed'],r['scenario_id'],r['ue'])]
            ue_deltas.append(dict(criterion=r['criterion'],network_seed=r['network_seed'],scenario_id=r['scenario_id'],
                regime=r['regime'],severity=r['severity'],ue=r['ue'],budget_class=r['budget_class'],
                comparison='A4-minus-'+base,failure_delta_pp=r['failure_pct']-b['failure_pct'],
                mean_prb_delta=r['mean_prbs']-b['mean_prbs']))
    write_csv(OUT/'paired_per_ue.csv',ue_deltas)
    allsummary=[r for r in summaries if r['regime']=='all' and r['severity']=='all']
    (OUT/'summary.json').write_text(json.dumps(allsummary,indent=2)+'\n')
    # Short candidate progression table keeps both NN initializations visible.
    write_csv(OUT/'candidate_scores.csv',curves)
    frontier=[]
    for r in curves:
        peers=[v for v in curves if v['network_seed']==r['network_seed']]
        dominated=any(v['failure_pct']<=r['failure_pct'] and v['worst_ue_pct']<=r['worst_ue_pct']
            and (v['failure_pct']<r['failure_pct'] or v['worst_ue_pct']<r['worst_ue_pct']) for v in peers)
        frontier.append(dict(r,aggregate_worst_nondominated=not dominated))
    write_csv(OUT/'pareto_frontier.csv',frontier)
    print(json.dumps(allsummary,indent=2))


if __name__=='__main__':main()
