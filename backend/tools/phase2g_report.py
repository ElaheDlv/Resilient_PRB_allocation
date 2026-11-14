"""Read completed Phase 2G outputs; no model training or simulator evaluation."""
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'backend/evaluations/phase2g'
ARMS = ('R0','R1','R2')
SEEDS = (12026,22026)
REGIMES = ('feasible','contested','overloaded')
COMPONENTS = ('failure_cost','terminal_cost','fairness_cost','embb_cost','reserve_cost','switch_cost')
METRICS = ('known_failure_pct','worst_ue_failure_pct','upper_failure_pct','unresolved_pct','on_time_completion_pct',
           'tight_failure_pct','relaxed_failure_pct','embb_mbps','embb_offered_deficit_pct',
           'reserve_prbs','feasibility_mode_pct','switches','recovery_success_pct',
           'recovered_delay_ms','hol_violation_pct')


def read_jsonl(path):
    with gzip.open(path,'rt') as f:
        for line in f:
            yield json.loads(line)


def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT/name).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def average(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def metrics(r):
    p = r['physical'];o = p['overall'];rec = p['recovery']
    recovered = [x for x in rec if not x['censored']]
    result = dict(known_failure_pct=100*o['failure_fraction_lower'],
        worst_ue_failure_pct=100*p['worst_ue_failure_fraction'],
        upper_failure_pct=100*o['failure_fraction_upper'],
        on_time_completion_pct=100*(1-o['failure_fraction_upper']),
        unresolved_pct=100*o['unresolved_not_due_batches']/o['offered_batches'],
        tight_failure_pct=100*p['per_class']['tight']['failure_fraction_lower'],
        relaxed_failure_pct=100*p['per_class']['relaxed']['failure_fraction_lower'],
        embb_mbps=p['embb_throughput_mbps'],
        embb_offered_deficit_pct=100*max(0.,1-r['embb_served_offered_fraction']),
        reserve_prbs=p['mean_reserve_prbs'],
        feasibility_mode_pct=100*r['mode_service_ticks'].get('FEASIBILITY_AWARE',0)/r['steps'],
        switches=r['switches'], recovery_success_pct=100*len(recovered)/len(rec) if rec else None,
        recovered_delay_ms=average([x['recovery_s']*1000 for x in recovered]),
        hol_violation_pct=100*average([u['hol']['fraction'] for u in p['per_ue'].values()]))
    return result


def make_tables():
    rows = list(read_jsonl(OUT/'validation.jsonl.gz'))
    training = list(read_jsonl(OUT/'training.jsonl.gz'))
    assert len(rows)==324 and len(training)==54
    selections = json.loads((OUT/'selection.json').read_text())
    selected = {(r['reward_arm'],r['network_seed'],kind):r['selected'][kind]['episode']
                for r in selections for kind in ('aggregate','fairness')}
    points = []
    ue = []
    classes = []
    for r in rows:
        identity = dict(reward_arm=r['reward_arm'],network_seed=r['network_seed'],
            checkpoint_episode=r['candidate_episode'],interactions=r['total_transitions'],
            scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'])
        flags = {kind+'_selected':r['candidate_episode']==selected[r['reward_arm'],r['network_seed'],kind]
                 for kind in ('aggregate','fairness')}
        point = dict(identity,**flags,**metrics(r),offered_batches=r['physical']['overall']['offered_batches'],
            known_failed_batches=r['physical']['overall']['known_failed_batches'],
            unresolved_not_due_batches=r['physical']['overall']['unresolved_not_due_batches'],
            administrative_terminal_charges=r['terminal']['terminal_unresolved_charges'],
            recovery_windows=len(r['physical']['recovery']),
            recovered_windows=sum(not x['censored'] for x in r['physical']['recovery']),
            return_value=r['return_value'],reward_version=r['return_objective'],
            **r['reward_components'],original_reward_monitor=r['common_joint_monitor_return'],
            checkpoint_parameter_sha256=r['parameter_sha256'],control_digest=r['control_digest'])
        assert point['administrative_terminal_charges']==point['unresolved_not_due_batches']
        assert abs(sum(r['reward_components'].values())+r['return_value'])<1e-7
        points.append(point)
        for uid,u in r['physical']['per_ue'].items():
            b = u['batch_outcomes']
            ue.append(dict(identity,**flags,ue_id=uid,budget_s=u['budget_s'],budget_class=u['budget_class'],
                offered=b['offered'],on_time=b['on_time'],known_failed=b['known_failed'],
                unresolved_not_due=b['unresolved_not_due'],unfinished_overdue=b['unfinished_overdue'],
                known_failure_pct=100*b['failure_fraction_lower'],upper_failure_pct=100*b['failure_fraction_upper'],
                served_bytes=u['served_bytes'],served_over_budget_bytes=u['served_over_budget_bytes'],
                overflow_bytes=u['overflow_bytes'],unfinished_bytes=u['unfinished_bytes'],
                hol_violation_pct=100*u['hol']['fraction'],**r['ue_resources'][uid]))
        for deadline,u in r['physical']['per_class'].items():
            classes.append(dict(identity,**flags,deadline_class=deadline,**u))
    write_csv('validation_metrics.csv',points)
    write_csv('per_ue_validation.csv',ue)
    write_csv('per_class_validation.csv',classes)
    # Explicit paired differences at the same training interaction count.
    baseline_points = {(r['network_seed'],r['checkpoint_episode'],r['scenario_id']):r
                       for r in points if r['reward_arm']=='R0'}
    checkpoint_pairs = []
    for r in points:
        if r['reward_arm']=='R0':continue
        ref = baseline_points[r['network_seed'],r['checkpoint_episode'],r['scenario_id']]
        checkpoint_pairs.append(dict(reward_arm=r['reward_arm'],network_seed=r['network_seed'],
            checkpoint_episode=r['checkpoint_episode'],interactions=r['interactions'],
            scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],
            **{k+'_delta':r[k]-ref[k] if r[k] is not None and ref[k] is not None else None for k in METRICS}))
    assert len(checkpoint_pairs)==216
    write_csv('paired_checkpoint_differences.csv',checkpoint_pairs)
    curves = []
    numeric = (*METRICS,*COMPONENTS)
    for arm in ARMS:
        for seed in (*SEEDS,'mean'):
            for ep in (3,6,9):
                for regime in ('all',*REGIMES):
                    for severity in ('all',0,10,20):
                        group = [r for r in points if r['reward_arm']==arm and (seed=='mean' or r['network_seed']==seed)
                            and r['checkpoint_episode']==ep and (regime=='all' or r['regime']==regime)
                            and (severity=='all' or r['severity']==severity)]
                        curves.append(dict(reward_arm=arm,network_seed=seed,checkpoint_episode=ep,interactions=ep*299,
                            regime=regime,severity=severity,episodes=len(group),
                            **{k:average([r[k] for r in group]) for k in numeric}))
    write_csv('learning_curves.csv',curves)
    paired = []
    harms = []
    for kind in ('aggregate','fairness'):
        ps = [r for r in points if r[kind+'_selected']]
        us = [r for r in ue if r[kind+'_selected']]
        baseline = {(r['network_seed'],r['scenario_id']):r for r in ps if r['reward_arm']=='R0'}
        baseline_ue = {(r['network_seed'],r['scenario_id'],r['ue_id']):r for r in us if r['reward_arm']=='R0'}
        for r in ps:
            if r['reward_arm']=='R0':continue
            ref = baseline[r['network_seed'],r['scenario_id']]
            paired.append(dict(selection=kind,reward_arm=r['reward_arm'],network_seed=r['network_seed'],
                scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],
                candidate_episode=r['checkpoint_episode'],r0_episode=ref['checkpoint_episode'],
                **{k+'_delta':r[k]-ref[k] if r[k] is not None and ref[k] is not None else None for k in METRICS}))
        for r in us:
            if r['reward_arm']=='R0':continue
            ref = baseline_ue[r['network_seed'],r['scenario_id'],r['ue_id']]
            harms.append(dict(selection=kind,reward_arm=r['reward_arm'],network_seed=r['network_seed'],
                scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],ue_id=r['ue_id'],
                budget_class=r['budget_class'],candidate_episode=r['checkpoint_episode'],r0_episode=ref['checkpoint_episode'],
                r0_known_failure_pct=ref['known_failure_pct'],candidate_known_failure_pct=r['known_failure_pct'],
                known_failure_pp_delta=r['known_failure_pct']-ref['known_failure_pct'],
                unresolved_batch_delta=r['unresolved_not_due']-ref['unresolved_not_due'],
                served_byte_delta=r['served_bytes']-ref['served_bytes'],
                hol_violation_pp_delta=r['hol_violation_pct']-ref['hol_violation_pct']))
    write_csv('paired_selected_differences.csv',paired)
    write_csv('per_ue_paired_differences.csv',harms)
    summary = []
    for kind in ('aggregate','fairness'):
        for arm in ARMS:
            for seed in (*SEEDS,'mean'):
                for regime in ('all',*REGIMES):
                    for severity in ('all',0,10,20):
                        group = [r for r in points if r[kind+'_selected'] and r['reward_arm']==arm
                            and (seed=='mean' or r['network_seed']==seed) and (regime=='all' or r['regime']==regime)
                            and (severity=='all' or r['severity']==severity)]
                        summary.append(dict(selection=kind,reward_arm=arm,network_seed=seed,regime=regime,severity=severity,
                            episodes=len(group),recovery_windows=sum(r['recovery_windows'] for r in group),
                            recovered_windows=sum(r['recovered_windows'] for r in group),
                            **{k:average([r[k] for r in group]) for k in numeric}))
    write_csv('selected_summary.csv',summary)
    # Stream interval histories; never load the complete training/evaluation log.
    accum = {}
    for r in read_jsonl(OUT/'intervals.jsonl.gz'):
        key = (r['reward_arm'],r['network_seed'],r['training'],r['episode'],r['scenario_id'])
        if key not in accum:
            accum[key] = dict(reward_arm=r['reward_arm'],network_seed=r['network_seed'],training=r['training'],
                episode=r['episode'],scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],
                n=0,zero=0,total=0.,squared=0.,maximum=0.,changes=0,history_without_new_failure=0,
                event_above_point25=0,previous=None,reference_history_total=0.,event_total=0.,
                new_physical_failures=0,valid_risk_ticks=0,saturated_risk_ticks=0)
        a = accum[key];cost = r['reward_components']['fairness_cost']
        event = .25*(max(r['failures_by_ue'].values())/10)
        assert sum(r['failures_by_ue'].values())==r['new_failures']
        if r['reward_arm']=='R2':assert cost==event
        if r['reward_arm']=='R1':assert cost==0
        if r['reward_arm']=='R0':assert cost==r['reference_components']['fairness_cost']
        a['n']+=1;a['zero']+=cost==0;a['total']+=cost;a['squared']+=cost*cost;a['maximum']=max(a['maximum'],cost)
        a['changes']+=a['previous'] is not None and abs(cost-a['previous'])>1e-12
        a['previous']=cost
        a['history_without_new_failure']+=r['new_failures']==0 and r['reference_components']['fairness_cost']>0
        a['event_above_point25']+=event>.25+1e-12
        a['reference_history_total']+=r['reference_components']['fairness_cost'];a['event_total']+=event
        a['new_physical_failures']+=r['new_failures']
        a['valid_risk_ticks']+=r['valid_risk_ticks'];a['saturated_risk_ticks']+=r['saturated_risk_ticks']
    signal = []
    for a in accum.values():
        a.pop('previous');a['fairness_mean']=a['total']/a['n']
        a['fairness_std']=math.sqrt(max(0.,a['squared']/a['n']-a['fairness_mean']**2))
        a['fairness_zero_fraction']=a['zero']/a['n'];a['fairness_change_fraction']=a['changes']/max(1,a['n']-1)
        signal.append(a)
    assert len(signal)==378 and sum(a['n'] for a in signal)==113022
    write_csv('interval_signal_diagnostics.csv',signal)
    # Fixed100-update bins, no learned-model filtering or loss-based selection.
    bins = defaultdict(list)
    for r in read_jsonl(OUT/'updates.jsonl.gz'):
        bins[r['reward_arm'],r['network_seed'],(r['optimizer_updates']-1)//100].append(r)
    optimizer = []
    for (arm,seed,index),rs in bins.items():
        optimizer.append(dict(reward_arm=arm,network_seed=seed,bin_index=index,
            interactions=st.mean(r['transitions'] for r in rs),updates=len(rs),
            **{k:st.mean(r[k] for r in rs) for k in ('loss','mean_abs_td','max_abs_q','max_abs_target')}))
    assert sum(r['updates'] for r in optimizer)==15960
    write_csv('optimizer_curves.csv',optimizer)
    return points,ue,curves,summary,paired,harms,signal,optimizer


def figures(points,ue,curves,summary,optimizer):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = {'R0':'#0072B2','R1':'#D55E00','R2':'#009E73'}
    directory = OUT/'figures';directory.mkdir(exist_ok=True)
    captions = []
    def save(fig,name,caption):
        fig.tight_layout()
        svg = directory/(name+'.svg')
        fig.savefig(svg,bbox_inches='tight');fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig('/tmp/phase2g-'+name+'.png',dpi=100,bbox_inches='tight')
        plt.close(fig)
        captions.append(f'**{name}.** {caption}')
    for metric,name,label in [('known_failure_pct','01_aggregate_failures','Known failed offered batches (%)'),
        ('worst_ue_failure_pct','02_worst_ue_failures','Mean scenario-worst UE failure (%)'),
        ('reserve_prbs','03_reserve_use','Mean activated reserve (PRBs)'),
        ('embb_mbps','04_embb_throughput','eMBB throughput (Mbit/s)')]:
        fig,axes = plt.subplots(3,3,figsize=(11,8),sharex=True)
        for i,g in enumerate(REGIMES):
            for j,seed in enumerate((*SEEDS,'mean')):
                ax = axes[i,j]
                for arm in ARMS:
                    rs = sorted([r for r in curves if r['regime']==g and r['severity']=='all'
                        and r['network_seed']==seed and r['reward_arm']==arm],key=lambda r:r['interactions'])
                    ax.plot([r['interactions'] for r in rs],[r[metric] for r in rs],'-o',ms=3,label=arm,color=colors[arm])
                ax.set_title(f'{g}: '+('two-seed descriptive mean' if seed=='mean' else f'NN {seed}'))
                ax.grid(alpha=.18);ax.set_xticks([897,1794,2691])
                if 'failure_pct' in metric:ax.set_ylim(0,max(1.,ax.get_ylim()[1]))
                if j==0:ax.set_ylabel(label)
                if i==2:ax.set_xlabel('Paid training interactions')
        axes[0,0].legend()
        save(fig,name,'Identical fixed validation scenarios at each checkpoint, six episodes/regime/NN seed. '
             'Columns separate initialization seeds and their descriptive mean; rows are workload regimes. '
             'All three jamming severities included. No exploratory-training return is used. Data: learning_curves.csv; severity strata remain available there.')
    fig,axes = plt.subplots(2,3,figsize=(11,5.5),sharex=True)
    for i,(metric,label) in enumerate([('feasibility_mode_pct','F mode service ticks (%)'),('switches','Mode switches /3s episode')]):
        for j,seed in enumerate((*SEEDS,'mean')):
            for arm in ARMS:
                rs = sorted([r for r in curves if r['regime']=='all' and r['severity']=='all'
                    and r['network_seed']==seed and r['reward_arm']==arm],key=lambda r:r['interactions'])
                axes[i,j].plot([r['interactions'] for r in rs],[r[metric] for r in rs],'-o',ms=3,label=arm,color=colors[arm])
            axes[i,j].set_title(str(seed));axes[i,j].grid(alpha=.18)
            axes[i,j].set_xticks([897,1794,2691])
            if j==0:axes[i,j].set_ylabel(label)
            if i==1:axes[i,j].set_xlabel('Paid training interactions')
    axes[0,0].legend()
    save(fig,'05_modes_and_switching','Fixed validation18-scenario means. Mode occupancy includes the shared initial10ms P interval. '
         'Switches count mode changes, not reserve-command changes. Data: learning_curves.csv.')
    fig,axes = plt.subplots(3,3,figsize=(12,8))
    palette = ['#0072B2','#444444','#D55E00','#009E73','#CC79A7','#E69F00']
    for i,g in enumerate(REGIMES):
        for j,seed in enumerate((*SEEDS,'mean')):
            rs = [next(r for r in curves if r['regime']==g and r['severity']=='all'
                and r['network_seed']==seed and r['reward_arm']==arm and r['checkpoint_episode']==ep)
                for ep in (3,6,9) for arm in ARMS]
            base = np.zeros(9)
            for k,color in zip(COMPONENTS,palette):
                values = [r[k] for r in rs]
                axes[i,j].bar(range(9),values,bottom=base,label=k.replace('_cost',''),color=color)
                base += values
            axes[i,j].set_xticks(range(9),[f'{r["reward_arm"]}\ne{r["checkpoint_episode"]}' for r in rs],fontsize=6)
            axes[i,j].set_title(f'{g}: {seed}')
            if j==0:axes[i,j].set_ylabel('Mean original-arm cost /3s')
    axes[0,2].legend(fontsize=6)
    save(fig,'06_reward_components','Actual learned-objective costs for fixed validation evaluations. Fairness has the versioned '
         'R0/R1/R2 definition, so total heights are not a common physical performance measure. Data: learning_curves.csv and validation_metrics.csv.')
    fig,axes = plt.subplots(3,3,figsize=(11,7.5),sharex=True)
    for i,(metric,label) in enumerate([('loss','Huber TD loss'),('mean_abs_td','Mean absolute TD error'),('max_abs_q','Mean minibatch maximum |Q|')]):
        for j,seed in enumerate((*SEEDS,'mean')):
            for arm in ARMS:
                if seed!='mean':
                    rs = sorted([r for r in optimizer if r['reward_arm']==arm and r['network_seed']==seed],key=lambda r:r['bin_index'])
                else:
                    grouped = defaultdict(list)
                    for r in optimizer:
                        if r['reward_arm']==arm:grouped[r['bin_index']].append(r)
                    rs = [dict(interactions=st.mean(v['interactions'] for v in values),**{metric:st.mean(v[metric] for v in values)})
                          for _,values in sorted(grouped.items())]
                axes[i,j].plot([r['interactions'] for r in rs],[r[metric] for r in rs],label=arm,color=colors[arm])
            axes[i,j].set_title(str(seed));axes[i,j].grid(alpha=.18)
            if j==0:axes[i,j].set_ylabel(label)
            if i==2:axes[i,j].set_xlabel('Paid training interactions')
    axes[0,0].legend()
    save(fig,'07_td_loss_and_q','Original unchanged learner; nonoverlapping100-update bin means (last bin60), starting after31 replay warmup transitions. '
         'Different reward scales imply different TD/Q scales; lower loss is not a common physical performance ranking. Data: optimizer_curves.csv; raw updates.jsonl.gz.')
    fig,axes = plt.subplots(3,3,figsize=(11,8),sharey=True)
    for i,g in enumerate(REGIMES):
        for j,seed in enumerate((*SEEDS,'mean')):
            values = []
            for arm in ARMS:
                rs = [r for r in ue if r['aggregate_selected'] and r['regime']==g and r['reward_arm']==arm]
                if seed!='mean':
                    v = [r['known_failure_pct'] for r in rs if r['network_seed']==seed]
                else:
                    grouped = defaultdict(list)
                    for r in rs:grouped[r['scenario_id'],r['ue_id']].append(r['known_failure_pct'])
                    v = [st.mean(x) for x in grouped.values()]
                values.append(v)
            boxes = axes[i,j].boxplot(values,showfliers=False,patch_artist=True)
            for box,arm in zip(boxes['boxes'],ARMS):box.set_facecolor(colors[arm]);box.set_alpha(.35)
            for k,v in enumerate(values,1):
                offsets = np.linspace(-.16,.16,len(v))
                axes[i,j].scatter(k+offsets,v,s=7,alpha=.45,color=colors[ARMS[k-1]])
            axes[i,j].set_xticks([1,2,3],ARMS);axes[i,j].set_title(f'{g}: {seed}')
            if j==0:axes[i,j].set_ylabel('Individual UE known failures (%)')
    save(fig,'08_per_ue_distributions','Aggregate-first selected checkpoints. Each NN panel shows48 UE/scenario observations per arm/regime; '
         'mean panels average the two NN outcomes within each matched scenario/UE. These dependent observations are not independent replication. '
         'Boxes show quartiles/median, whiskers1.5IQR; points retain all observations. Exact per-UE paired differences and deadline classes are in the CSVs.')
    fig,axes = plt.subplots(3,3,figsize=(11,8),sharey=True)
    for i,g in enumerate(REGIMES):
        for j,severity in enumerate((0,10,20)):
            ax = axes[i,j]
            for k,seed in enumerate(SEEDS):
                values = [next(r for r in summary if r['selection']=='aggregate' and r['reward_arm']==arm
                    and r['network_seed']==seed and r['regime']==g and r['severity']==severity)['known_failure_pct'] for arm in ARMS]
                ax.plot(range(3),values,'-o',label=str(seed),ms=3)
            ax.set_xticks(range(3),ARMS);ax.set_title(f'{g}, j{severity}')
            if j==0:ax.set_ylabel('Known failed offered batches (%)')
    axes[0,0].legend()
    save(fig,'09_selected_severity_strata','Aggregate-first selected checkpoints, two placements per regime/severity/NN. '
         'Both seeds shown separately; no scenario replacement or tuning. Data: selected_summary.csv.')
    (directory/'CAPTIONS.md').write_text('# Phase 2G figures\n\n'+'\n\n'.join(captions)+'\n')


def verify_results():
    """Independent streamed accounting/selection checks; no simulator or learner."""
    import hashlib
    import sys
    sys.path.insert(0,str(ROOT/'backend'))
    from network_layer.workload_feasibility import checkpoint_key
    p = json.loads((OUT/'protocol.json').read_text())
    for name,expected in p['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    training = list(read_jsonl(OUT/'training.jsonl.gz'))
    validation = list(read_jsonl(OUT/'validation.jsonl.gz'))
    assert len(training)==54 and len(validation)==324
    cases = {}
    exogenous = defaultdict(set)
    for r in training+validation:
        ep = r['episode'] if r['training'] else r['candidate_episode']
        key = (r['reward_arm'],r['network_seed'],r['training'],ep,r['scenario_id'])
        assert key not in cases
        cases[key] = r
        if r['training']:
            assert r['scenario_id']==p['training_order'][ep-1]
            assert r['total_transitions']==299*ep
        else:
            assert r['scenario_id'] in p['validation_order']
        assert r['transitions']==299 and r['steps']==3000
        assert r['return_objective']==p['reward_versions'][r['reward_arm']]
        assert sum(r['paid_failures_by_ue'].values())==r['known_failures']-r['bootstrap']['new_known_failures']
        assert abs(r['reward_components']['failure_cost']-sum(r['paid_failures_by_ue'].values())/80)<1e-8
        assert abs(r['reward_components']['terminal_cost']-r['terminal']['terminal_unresolved_charges']/80)<1e-8
        assert r['terminal']['terminal_unresolved_charges']==r['physical']['overall']['unresolved_not_due_batches']
        for u in r['physical']['per_ue'].values():
            b = u['batch_outcomes']
            assert b['offered']==b['on_time']+b['known_failed']+b['unresolved_not_due']
        assert sum(u['batch_outcomes']['known_failed'] for u in r['physical']['per_ue'].values())==r['known_failures']
        exogenous[r['scenario_id']].add((r['physical']['exogenous_sha256'],r['deadline_assignment_sha256']))
    assert all(len(values)==1 for values in exogenous.values())
    sums = {k:dict(n=0,parts=dict.fromkeys(COMPONENTS,0.),failures=defaultdict(int)) for k in cases}
    total = 0
    for r in read_jsonl(OUT/'intervals.jsonl.gz'):
        key = (r['reward_arm'],r['network_seed'],r['training'],r['episode'],r['scenario_id'])
        a = sums[key]
        assert r['start_step']==10+10*a['n'] and r['end_step']==20+10*a['n']
        assert sum(r['failures_by_ue'].values())==r['new_failures']
        if r['reward_arm']=='R2':
            assert r['reward_components']['fairness_cost']==.25*(max(r['failures_by_ue'].values())/10)
        if r['reward_arm']=='R1':assert r['reward_components']['fairness_cost']==0
        if r['reward_arm']=='R0':assert r['reward_components']==r['reference_components']
        for k in COMPONENTS:a['parts'][k]+=r['reward_components'][k]
        for uid,n in r['failures_by_ue'].items():a['failures'][uid]+=n
        a['n']+=1;total+=1
    assert total==113022
    for key,a in sums.items():
        r = cases[key]
        assert a['n']==299
        assert all(abs(a['parts'][k]-r['reward_components'][k])<1e-7 for k in COMPONENTS)
        assert dict(a['failures'])==r['paid_failures_by_ue']
        assert abs(sum(a['parts'].values())+r['return_value'])<1e-7
    for seed in SEEDS:
        for arm in ARMS:
            rs = sorted([r for r in training if r['network_seed']==seed and r['reward_arm']==arm],key=lambda r:r['episode'])
            assert [r['episode'] for r in rs]==list(range(1,10))
            assert sum(r['updates'] for r in rs)==2660
            assert sum(r['transitions'] for r in rs)==2691
    selected = json.loads((OUT/'selection.json').read_text())
    assert len(selected)==6
    for choice in selected:
        candidates = {}
        for ep in (3,6,9):
            rs = [r for r in validation if r['reward_arm']==choice['reward_arm']
                  and r['network_seed']==choice['network_seed'] and r['candidate_episode']==ep]
            assert len(rs)==18 and {r['scenario_id'] for r in rs}==set(p['validation_order'])
            candidates[ep]=checkpoint_key(rs,ep)
        expected = min(candidates,key=candidates.get)
        fair = min(candidates,key=lambda ep:(candidates[ep][1],candidates[ep][0],*candidates[ep][2:]))
        assert choice['selected']['aggregate']['episode']==expected
        assert choice['selected']['fairness']['episode']==fair
    updates = defaultdict(int)
    for r in read_jsonl(OUT/'updates.jsonl.gz'):
        key = (r['reward_arm'],r['network_seed'])
        updates[key]+=1
        assert r['optimizer_updates']==updates[key] and r['transitions']==updates[key]+31
        assert all(math.isfinite(r[k]) for k in ('loss','mean_abs_td','max_abs_q','max_abs_target'))
    assert len(updates)==6 and set(updates.values())=={2660}
    checks = json.loads((OUT/'training_checks.json').read_text())
    assert all(r['initial_parameter_sha256']==p['initializations'][str(r['network_seed'])]['parameter_sha256'] for r in checks['learners'])
    assert len(checks['learners'])==6 and checks['cuda_initialized'] is False
    result = dict(passed=True,training_episodes=54,validation_episodes=324,learners=6,
        intervals=total,per_ue_partition_and_event_conservation=True,terminal_exclusion_verified=True,
        interval_boundaries_verified=True,all_six_budgets_exact=True,optimizer_updates=15960,
        physical_selection_recomputed=True,original_frozen_sources_and_inputs_unchanged=True,
        exogenous_pairing_verified=True,initial_fingerprints_matched=True,fresh_simulations=0)
    (OUT/'result_integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    if parser.parse_args().verify_only:
        verify_results()
        return
    values = make_tables()
    points,ue,curves,summary,paired,harms,signal,optimizer = values
    figures(points,ue,curves,summary,optimizer)
    diagnostics = dict(validation_rows=len(points),ue_rows=len(ue),selected_pairs=len(paired),ue_pairs=len(harms),
        interval_episode_groups=len(signal),optimizer_updates=sum(r['updates'] for r in optimizer),
        severe_harms_definition='No acceptance threshold inferred; report signed exact paired changes and largest increases.',
        primary_seed_summary=[r for r in summary if r['selection']=='aggregate' and r['regime']=='all' and r['severity']=='all'],
        largest_primary_ue_increases=sorted([r for r in harms if r['selection']=='aggregate'],key=lambda r:-r['known_failure_pp_delta'])[:20])
    (OUT/'analysis.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
