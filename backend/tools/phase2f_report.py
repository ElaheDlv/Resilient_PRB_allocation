"""Read-only reward magnitudes, temporal diagnostics and objective-order plots."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from network_layer.reward_audit import COMPONENTS
OUT=ROOT/'backend/evaluations/phase2f'
REGIMES=('feasible','contested','overloaded')
COLORS=dict(zip(COMPONENTS,('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#444444')))


def csvread(path):
    opener=(lambda:gzip.open(path,'rt')) if str(path).endswith('.gz') else path.open
    with opener() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def distribution(values):
    a=sorted(values);n=len(a)
    return dict(n=n,mean=st.mean(a),median=st.median(a),p95=a[max(0,math.ceil(.95*n)-1)],maximum=a[-1],minimum=a[0],
        std=st.pstdev(a),zero_fraction=sum(v==0 for v in a)/n,total=sum(a))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);OUT=parser.parse_args().output.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    saved=csvread(OUT/'saved_episode_components.csv');intervals=csvread(OUT/'reward_intervals.csv.gz')
    prefixes=csvread(OUT/'observed_prefix_accounting.csv.gz');lag=csvread(OUT/'failure_timing_histogram.csv.gz')
    with gzip.open(OUT/'replay_summaries.jsonl.gz','rt') as f:runs=[json.loads(x) for x in f]
    assert len(saved)==504 and len(runs)==18 and len(intervals)==5382
    byrun={r['run_id']:r for r in runs}
    cases=[]
    for r in runs:
        parts=r['audited_component_totals'];total=sum(parts.values())
        cases.append(dict(run_id=r['run_id'],method=r['method'],network_seed=r['network_seed'],checkpoint_episode=r['checkpoint_episode'],
            scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],role=r['role'],
            total_cost=total,**parts,**{k+'_share':v/total if total else None for k,v in parts.items()}))
    write_csv(OUT/'replay_component_totals.csv',cases)
    saved_dist=[]
    for split in ('training','validation'):
        for g in REGIMES:
            rs=[r for r in saved if r['split']==split and r['regime']==g]
            for k in ('failure_cost','terminal_cost','reserve_cost','switch_cost','worst_embb_residual'):
                saved_dist.append(dict(split=split,regime=g,component=k,**distribution([float(r[k]) for r in rs])))
    write_csv(OUT/'saved_component_distributions.csv',saved_dist)
    # Checkpoint episode is training budget, never the physical horizon.
    terminal_summary=[]
    for split in ('training','validation'):
        for g in REGIMES:
            episodes=sorted({int(r['episode']) for r in saved if r['split']==split and r['regime']==g})
            for episode in episodes:
                rs=[r for r in saved if r['split']==split and r['regime']==g and int(r['episode'])==episode]
                costs=[float(r['terminal_cost']) for r in rs]
                shares=[float(r['terminal_cost'])/float(r['total_cost']) for r in rs if float(r['total_cost'])]
                terminal_summary.append(dict(split=split,regime=g,checkpoint_or_training_episode=episode,n=len(rs),
                    physical_horizon_s=3,mean_terminal_cost=st.mean(costs),max_terminal_cost=max(costs),
                    max_episode_cost_share=max(shares,default=0.)))
    write_csv(OUT/'terminal_checkpoint_summary.csv',terminal_summary)
    distributions=[]
    for role in ('grid','severe_replication','matched_comparison','checkpoint_comparison','learning_comparison'):
        for g in REGIMES:
            rs=[r for r in intervals if r['role']==role and r['regime']==g]
            if not rs:continue
            total=sum(-float(r['reward']) for r in rs)
            for k in COMPONENTS:
                d=distribution([float(r[k]) for r in rs]);distributions.append(dict(role=role,regime=g,component=k,**d,share_of_cost=d['total']/total if total else None))
    write_csv(OUT/'interval_component_distributions.csv',distributions)
    # Consecutive1ms temporal persistence; reset runs at episode boundaries.
    temporal={r['run_id']:dict(valid_ticks=0,saturated_ticks=0,high_ticks=0,historical_cost_without_new_failures_ticks=0,
        high_run=0,max_high_run=0,saturated_run=0,max_saturated_run=0,historical_only_run=0,max_historical_only_run=0,
        failures_with_pool_headroom_ticks=0,failures_with_low_quota_and_full_pool_fresh_feasibility_ticks=0,
        activated_prb_ticks=0,unused_activated_prb_ticks=0,unused_extra_with_no_failure_no_embb_deficit_ticks=0) for r in runs}
    entries={e['id']:e for e in json.loads((ROOT/'backend/evaluations/phase2c2/validation_manifest.json').read_text())['entries']}
    with gzip.open(OUT/'reward_ticks.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            s=temporal[r['run_id']];valid=r['worst_valid']=='True';h=float(r['worst_risk']) if valid else None
            high=valid and h>=.9;sat=valid and h>=1-1e-12;historic=float(r['worst_cost'])>0 and int(r['new_failures'])==0
            s['valid_ticks']+=valid;s['high_ticks']+=high;s['saturated_ticks']+=sat;s['historical_cost_without_new_failures_ticks']+=historic
            for name,yes in (('high',high),('saturated',sat),('historical_only',historic)):
                s[name+'_run']=s[name+'_run']+1 if yes else 0;s['max_'+name+'_run']=max(s['max_'+name+'_run'],s[name+'_run'])
            R=int(r['reserve_prbs']);unused=int(r['activated_extra_unused_in_observed_allocation'])
            s['activated_prb_ticks']+=R;s['unused_activated_prb_ticks']+=unused
            s['unused_extra_with_no_failure_no_embb_deficit_ticks']+=unused>0 and int(r['new_failures'])==0 and float(r['embb_cost'])==0
            failed=int(r['new_failures'])>0;s['failures_with_pool_headroom_ticks']+=failed and R<18
            cert=entries[r['scenario_id']]['certificate'];need=cert['integer_one_tick_prbs']
            s['failures_with_low_quota_and_full_pool_fresh_feasibility_ticks']+=failed and float(r['urllc_quota'])<need<=float(r['urllc_base'])+18
    temporal_rows=[]
    for run_id,s in temporal.items():
        r=byrun[run_id]
        temporal_rows.append(dict(run_id=run_id,method=r['method'],network_seed=r['network_seed'],checkpoint_episode=r['checkpoint_episode'],
            regime=r['regime'],severity=r['severity'],role=r['role'],
            **{k:v for k,v in s.items() if k not in ('high_run','saturated_run','historical_only_run')},
            saturation_fraction=s['saturated_ticks']/s['valid_ticks'] if s['valid_ticks'] else None,
            unused_activation_fraction=s['unused_activated_prb_ticks']/s['activated_prb_ticks'] if s['activated_prb_ticks'] else None))
    write_csv(OUT/'temporal_and_resource_diagnostics.csv',temporal_rows)
    persistence=[];strong_examples=[];history_examples=[];previous={};runlen=defaultdict(int);longest=defaultdict(int)
    for r in intervals:
        total=-float(r['reward']);known=float(r['new_failures']);w=float(r['worst_cost']);prior=previous.get(r['run_id'])
        strong=known==0 and w>0 and float(r['urllc_on_time'])>=float(r['urllc_offered_batches'])
        improvement=prior is not None and float(r['urllc_on_time'])>float(prior['urllc_on_time'])
        runlen[r['run_id']]=runlen[r['run_id']]+1 if strong else 0;longest[r['run_id']]=max(longest[r['run_id']],runlen[r['run_id']])
        derived=dict(r,total_cost=total,terminal_share=float(r['terminal_cost'])/total if total else None,
            switch_share=float(r['switch_cost'])/total if total else None,
            strong_current_service_with_historical_cost=strong,on_time_improved_from_prior=improvement)
        if strong:strong_examples.append(derived)
        if improvement and w>0:history_examples.append(derived)
        previous[r['run_id']]=r
        if float(r['terminal_cost']):persistence.append(derived)
    write_csv(OUT/'strong_history_carry_examples.csv',strong_examples)
    write_csv(OUT/'service_improvement_with_history_examples.csv',history_examples)
    write_csv(OUT/'actual_terminal_intervals.csv',persistence)
    write_csv(OUT/'successful_service_historical_runs.csv',(dict(run_id=k,longest_intervals=v,longest_ms=10*v) for k,v in longest.items()))
    failure_summary=[]
    for group in ('grid','severe_replication','matched_comparison','checkpoint_comparison','learning_comparison'):
        for budget in (.002,.010):
            rs=[r for r in lag if r['role']==group and float(r['budget_s'])==budget and r['reward_paid']=='True']
            if not rs:continue
            n=sum(int(r['failures']) for r in rs)
            failure_summary.append(dict(role=group,budget_s=budget,paid_failures=n,
                mean_lag_ms=sum(float(r['lag_ticks'])*int(r['failures']) for r in rs)/n,
                later_decision_interval_fraction=sum(int(r['failures']) for r in rs if int(r['issue_step_gap'])>0)/n,
                different_mode_fraction=sum(int(r['failures']) for r in rs if r['arrival_mode']!=r['failure_mode'])/n))
    write_csv(OUT/'failure_credit_timing.csv',failure_summary)
    switches=[]
    for g in REGIMES:
        for changed in (False,True):
            rs=[r for r in intervals if r['regime']==g and (r['switched']=='True')==changed]
            if not rs:continue
            switches.append(dict(regime=g,switched=changed,n=len(rs),
                **{k:st.mean(float(r[k]) for r in rs) for k in COMPONENTS},
                total_cost=st.mean(-float(r['reward']) for r in rs)))
    write_csv(OUT/'switch_cost_associations.csv',switches)
    # Figure exports: numeric tables retain all points and validity masks.
    figs=OUT/'figures';figs.mkdir(exist_ok=True);captions=[]
    def save(fig,name,caption):
        svg=figs/(name+'.svg');fig.savefig(svg);fig.savefig(figs/(name+'.pdf'))
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(Path('/tmp')/('phase2f-'+name+'.png'),dpi=130);plt.close(fig);captions.append(f'**{name}.** {caption}')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
    parts=('failure_cost','terminal_cost','reserve_cost','switch_cost','worst_embb_residual')
    for ax,split in zip(axes,('training','validation')):
        base=np.zeros(3)
        for part,color in zip(parts,('#0072B2','#444444','#CC79A7','#E69F00','#999999')):
            vals=[next(r['mean'] for r in saved_dist if r['split']==split and r['regime']==g and r['component']==part) for g in REGIMES]
            ax.bar(REGIMES,vals,bottom=base,label=part.replace('_cost',''),color=color);base+=vals
        ax.set(title=split,ylabel='Mean recorded episode cost');ax.tick_params(axis='x',rotation=15)
    axes[1].legend(fontsize=7)
    save(fig,'01_saved_episode_accounting','All72 training and432 validation episodes. Four costs are exactly identified; gray worst+eMBB residual is not split by estimation. No training trajectories are regenerated. Data: ../saved_episode_components.csv and ../saved_component_distributions.csv.')
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
    for ax,g in zip(axes,REGIMES):
        rs=[r for r in distributions if r['role']=='grid' and r['regime']==g]
        means=[next(r['mean'] for r in rs if r['component']==k) for k in COMPONENTS]
        p95=[next(r['p95'] for r in rs if r['component']==k) for k in COMPONENTS]
        ax.bar(range(6),means,color=[COLORS[k] for k in COMPONENTS]);ax.scatter(range(6),p95,color='black',s=15,label='p95')
        ax.set(title=g,xticks=range(6),xticklabels=[k.replace('_cost','') for k in COMPONENTS]);ax.tick_params(axis='x',rotation=55)
        ax.set_yscale('symlog',linthresh=1e-4);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Original cost /10ms interval (symlog)');axes[2].legend()
    save(fig,'02_interval_components','Primary nine-case A4 grid: one learner seed and one placement/regime, all severities,897 correlated intervals/regime. Bars=mean, points=p95; all six original components directly measured, with no new weights. Extras remain separately tabulated. Data: ../interval_component_distributions.csv and reward_intervals.csv.gz.')
    chosen=max(strong_examples,key=lambda r:float(r['worst_cost'])) if strong_examples else max(history_examples,key=lambda r:float(r['worst_cost']))
    chosen_id=chosen['run_id'];center=int(chosen['end_step']);example=[r for r in intervals if r['run_id']==chosen_id and center-150<=int(r['end_step'])<=center+150]
    fig,axes=plt.subplots(3,1,figsize=(8,6),sharex=True)
    t=[int(r['end_step'])/1000 for r in example]
    axes[0].plot(t,[float(r['urllc_on_time']) for r in example],label='On-time completions');axes[0].plot(t,[float(r['new_failures']) for r in example],label='New known failures');axes[0].legend()
    axes[1].plot(t,[float(r['worst_cost']) for r in example],color=COLORS['worst_cost'],label='Historical worst cost');axes[1].plot(t,[float(r['failure_cost']) for r in example],label='New failure cost');axes[1].legend()
    axes[2].plot(t,[float(r['worst_queue_end'])/1000 if r['worst_queue_end'] else 0 for r in example]);axes[2].set(ylabel='Selected worst queue (kB)',xlabel='Simulation time (s)')
    axes[0].set_ylabel('Batches /10ms');axes[1].set_ylabel('Original interval cost')
    for ax in axes:ax.axvline(center/1000,color='gray',ls='--');ax.grid(alpha=.2)
    save(fig,'03_service_recovery_and_history',f'Observed case {chosen_id}, centered at{center}ms. No hypothetical action outcome: improved current service coexists with retrospective cohort penalty. Worst UID can change/tie; data retains UID/validity. History is not itself an erroneous charge. Data: ../strong_history_carry_examples.csv and reward_intervals.csv.gz.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for g,color in zip(REGIMES,('#0072B2','#D55E00','#009E73')):
        for method,marker in (('A4','o'),('A3','s')):
            rs=[r for r in saved if r['split']=='validation' and r['regime']==g and r['method']==method and float(r['total_cost'])>0]
            axes[0].scatter([float(r['switch_cost']) for r in rs],[100*float(r['switch_cost'])/float(r['total_cost']) for r in rs],color=color,marker=marker,s=13,alpha=.55,label=f'{g}, {method}')
    axes[0].set(xlabel='Episode switching cost',ylabel='Switch cost / total cost (%)');axes[0].legend(fontsize=6,ncol=2)
    equivalents=[.0125,.25,.1,.02,.01]
    axes[1].bar(range(5),equivalents,color=('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00'))
    axes[1].set(xticks=range(5),xticklabels=['1 failure','worst=1','eMBB=0','reserve18','1 switch'],ylabel='10ms cost (fixed original weights)');axes[1].tick_params(axis='x',rotation=30)
    save(fig,'04_switch_magnitudes','Left: all recorded C3 validation A3/A4 episodes, not the single-seed grid alone. Right: exact algebraic exchange rates, not simulated counterfactual outcomes or a weight ablation. Correlations do not establish causal influence. Data: ../saved_episode_components.csv and MATHEMATICAL_AUDIT.md.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for g,color in zip(REGIMES,('#0072B2','#D55E00','#009E73')):
        rs=[r for r in intervals if r['role']=='grid' and r['regime']==g]
        axes[0].scatter([float(r['reserve_prbs'])/10 for r in rs],[float(r['new_failures']) for r in rs],s=9,color=color,alpha=.3,label=g)
        ds=[r for r in temporal_rows if r['role']=='grid' and r['regime']==g]
        axes[1].scatter([r['activated_prb_ticks']/2990 for r in ds],[100*r['unused_activation_fraction'] if r['unused_activation_fraction'] is not None else 0 for r in ds],color=color,s=35,label=g)
    axes[0].set(xlabel='Mean activated reserve /10ms (PRBs)',ylabel='New failure events /10ms');axes[1].set(xlabel='Mean activated reserve (PRBs)',ylabel='Activation unused by observed grants (%)')
    axes[0].legend();axes[1].legend()
    save(fig,'05_reserve_and_observed_service','Primary grid only. Failure/headroom associations do not show that another reserve command would save expired traffic. Unused means activation beyond minimum extra needed to make observed grants fit baseline quotas; it is accounting, not a reconstructed no-reserve rollout. Data: ../temporal_and_resource_diagnostics.csv and reward_ticks.csv.gz.')
    ranks=csvread(OUT/'candidate_ordering.csv');fig,ax=plt.subplots(figsize=(6,4))
    for method,color in zip(('A1','A2','A3','A4'),('#0072B2','#D55E00','#009E73','#CC79A7')):
        rs=[r for r in ranks if r['method']==method]
        ax.scatter([int(r['aggregate_first_rank']) for r in rs],[int(r['reward_rank']) for r in rs],label=method,color=color)
    ax.plot([1,24],[1,24],ls='--',color='gray');ax.set(xlabel='C2 aggregate-first rank (lower better)',ylabel='Original mean-return rank (lower better)');ax.legend();ax.grid(alpha=.2)
    save(fig,'06_reward_objective_order','All24 frozen candidate policies on the same18 validation entries. Six of276 pairwise orders disagree; best five agree and there is no mean six-criterion dominance reversal. This is descriptive, not an independent-sample inference or proof of globally correct action incentives. Data: ../candidate_ordering.csv and ordering_comparisons.csv.')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for g,color in zip(REGIMES,('#0072B2','#D55E00','#009E73')):
        for j,marker in ((0,'o'),(10,'s'),(20,'^')):
            rs=[r for r in prefixes if r['role']=='grid' and r['regime']==g and int(r['severity'])==j and r['closure_share_if_assessed_here']]
            axes[0].plot([float(r['physical_seconds']) for r in rs],[100*float(r['closure_share_if_assessed_here']) for r in rs],color=color,marker=marker,alpha=.7)
        end=[r for r in persistence if r['regime']==g]
        cs={r['run_id']:r for r in cases}
        axes[1].scatter([100*cs[r['run_id']]['terminal_cost_share'] for r in end],[100*r['terminal_share'] for r in end],color=color,label=g)
    axes[0].set(xlabel='Observed prefix (s)',ylabel='Administrative closure share if assessed (%)');axes[1].set(xlabel='Actual terminal / episode cost (%)',ylabel='Actual terminal / last interval cost (%)');axes[1].legend()
    save(fig,'07_terminal_accounting','Left: same unchanged physical prefix, hypothetical administrative accounting only; not different-horizon controllers, new outcomes or an ablation. Markers denote j0/10/20. Right: actual3s closure relative to whole episode and last10ms. Undefined zero-total shares omitted. Data: ../observed_prefix_accounting.csv.gz and actual_terminal_intervals.csv.')
    (figs/'CAPTIONS.md').write_text('\n\n'.join(captions)+'\n')
    report=dict(replays=len(runs),intervals=len(intervals),history_positive_without_new_failure_intervals=sum(float(r['new_failures'])==0 and float(r['worst_cost'])>0 for r in intervals),
        strong_current_service_historical_intervals=len(strong_examples),strong_example=chosen,
        longest_successful_service_history_run_ms=max(longest.values(),default=0)*10,
        temporal=temporal_rows,failure_timing=failure_summary,
        component_grid=[r for r in distributions if r['role']=='grid'])
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
