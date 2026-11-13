"""Independent pilot summaries and paired per-UE differences; no tuning/test feedback."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def mean(values):
    known=[v for v in values if v is not None]
    return statistics.mean(known) if known else None


def summarize(rows):
    recovery=[w for r in rows for w in r['physical']['recovery']]
    f=lambda r,k:r['physical']['overall'][k]
    return dict(independent_scenario_seeds=len({r['scenario_seed'] for r in rows}),
        failed_percent=100*mean([f(r,'failure_fraction_lower') for r in rows]),
        failed_upper_percent=100*mean([f(r,'failure_fraction_upper') for r in rows]),
        failed_seed_range_percent=[100*min(f(r,'failure_fraction_lower') for r in rows),
                                   100*max(f(r,'failure_fraction_lower') for r in rows)],
        worst_percent=100*mean([r['physical']['worst_ue_failure_fraction'] for r in rows]),
        worst_upper_percent=100*mean([max(u['batch_outcomes']['failure_fraction_upper']
                        for u in r['physical']['per_ue'].values()) for r in rows]),
        per_class_percent={name:100*mean([r['physical']['per_class'][name]['failure_fraction_lower']
                         for r in rows]) for name in ('tight','relaxed')},
        embb_mbps=mean([r['physical']['embb_throughput_mbps'] for r in rows]),
        embb_served_offered_fraction=mean([r['embb_served_offered_fraction'] for r in rows]),
        reserve_prbs=mean([r['physical']['mean_reserve_prbs'] for r in rows]),
        feasibility_service_fraction=mean([r['mode_service_ticks'].get('FEASIBILITY_AWARE',0)/r['steps'] for r in rows]),
        switches_per_decision=mean([r['switches_per_decision'] for r in rows]),
        valid_action_fraction=mean([r['mean_valid_action_fraction'] for r in rows]),
        recovery_success=sum(w['recovery_s'] is not None for w in recovery),
        recovery_windows=len(recovery),
        recovery_success_seconds=[w['recovery_s'] for w in recovery if w['recovery_s'] is not None],
        worst_risk_saturated_fraction=mean([r['worst_risk_saturated_fraction'] for r in rows]),
        fluid_load_ratio=mean([r['mean_exogenous_fluid_load_ratio'] for r in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args();src=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((src/'selection.json').read_text());protocol=selection['protocol']
    rows=json.loads((src/'heldout.json').read_text())
    training=json.loads((src/'training.json').read_text())
    validation=json.loads((src/'validation.json').read_text())
    expected={(m,s,j) for m in protocol['test_methods'] for s in protocol['test_scenario_seeds']
              for j in protocol['severities']}
    actual={(r['method'],r['scenario_seed'],r['severity']) for r in rows}
    if expected!=actual or len(rows)!=len(expected):raise AssertionError('Incomplete held-out grid')
    paired={}
    for r in rows:
        key=(r['scenario_seed'],r['severity'])
        fingerprints=(r['physical']['exogenous_sha256'],r['deadline_assignment_sha256'])
        if key in paired and paired[key]!=fingerprints:raise AssertionError('Unpaired held-out inputs')
        paired[key]=fingerprints
    groups=[]
    for method in protocol['test_methods']:
        for severity in protocol['severities']:
            rs=[r for r in rows if r['method']==method and r['severity']==severity]
            groups.append(dict(method=method,severity=severity,**summarize(rs)))
    (out/'pilot_summary.json').write_text(json.dumps(groups,indent=2)+'\n')
    fields=['method','severity','failed_percent','failed_upper_percent','worst_percent','embb_mbps',
            'reserve_prbs','feasibility_service_fraction','switches_per_decision',
            'recovery_success','recovery_windows','fluid_load_ratio']
    with (out/'pilot_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in groups)
    differences=[];per_ue=[]
    for reference in [m for m in protocol['test_methods'] if m!='P']:
        for seed,severity in sorted(paired):
            p_row=next(r for r in rows if (r['method'],r['scenario_seed'],r['severity'])==('P',seed,severity))
            b_row=next(r for r in rows if (r['method'],r['scenario_seed'],r['severity'])==(reference,seed,severity))
            p_summary=summarize([p_row]);b_summary=summarize([b_row])
            differences.append(dict(reference=reference,seed=seed,severity=severity,
                failed_difference_pp=p_summary['failed_percent']-b_summary['failed_percent'],
                worst_difference_pp=p_summary['worst_percent']-b_summary['worst_percent'],
                reserve_difference_prbs=p_summary['reserve_prbs']-b_summary['reserve_prbs'],
                embb_difference_mbps=p_summary['embb_mbps']-b_summary['embb_mbps']))
            for uid,p_ue in p_row['physical']['per_ue'].items():
                b_ue=b_row['physical']['per_ue'][uid]
                per_ue.append(dict(reference=reference,seed=seed,severity=severity,uid=uid,
                    budget_class=p_ue['budget_class'],cqi=p_row['physical']['assignments'][uid]['cqi'],
                    offered=p_ue['batch_outcomes']['offered'],
                    P_failure_percent=100*p_ue['batch_outcomes']['failure_fraction_lower'],
                    reference_failure_percent=100*b_ue['batch_outcomes']['failure_fraction_lower'],
                    difference_pp=100*(p_ue['batch_outcomes']['failure_fraction_lower']-b_ue['batch_outcomes']['failure_fraction_lower']),
                    P_mean_prbs=p_row['ue_resources'][uid]['prbs'],
                    reference_mean_prbs=b_row['ue_resources'][uid]['prbs'],
                    capacity_bytes_per_prb=p_row['ue_resources'][uid]['capacity_bytes_per_prb'],
                    offered_fluid_prbs=p_row['ue_resources'][uid]['fluid_prbs'],
                    P_unfinished_bytes=p_ue['unfinished_bytes'],reference_unfinished_bytes=b_ue['unfinished_bytes']))
    (out/'paired_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
    (out/'per_ue_differences.json').write_text(json.dumps(per_ue,indent=2)+'\n')
    diagnostics=[]
    for method in protocol['training_methods']:
        rs=[r for r in training if r['method']==method]
        candidates=[]
        for ep in protocol['checkpoint_after_episodes']:
            vals=[r for r in validation if r['method']==method and r['candidate_episode']==ep]
            candidates.append(dict(episode=ep,**summarize(vals)))
        returns=[r['return_value'] for r in rs]
        diagnostics.append(dict(method=method,episodes=len(rs),transitions=sum(r['transitions'] for r in rs),
            updates=sum(r['updates'] for r in rs),return_mean=mean(returns),return_population_sd=statistics.pstdev(returns),
            first_episode_losses=rs[0]['losses'],last_episode_losses=rs[-1]['losses'],
            first_q=rs[0]['q'],last_q=rs[-1]['q'],validation_checkpoints=candidates,
            training_loop_seconds=sum(r['elapsed_seconds'] for r in rs)))
    (out/'learning_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    if a.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
        for row in diagnostics:
            x=[r['episode'] for r in row['validation_checkpoints']]
            for ax,key,title in zip(axes,['failed_upper_percent','worst_upper_percent'],
                    ['Validation offered failure upper bound','Validation worst-UE upper bound']):
                ax.plot(x,[r[key] for r in row['validation_checkpoints']],marker='o',label=row['method'])
                ax.set(xlabel='Training episodes',ylabel='Percent',title=title,xticks=x)
                ax.grid(alpha=.25)
        if all(c['failed_upper_percent']==0 and c['worst_upper_percent']==0
               for d in diagnostics for c in d['validation_checkpoints']):
            for ax in axes:
                ax.set_ylim(0,1)
                ax.text(.5,.6,'All four curves overlap at 0%',transform=ax.transAxes,ha='center')
        axes[0].legend();fig.savefig(out/'validation_learning.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(heldout_runs=len(rows),paired_scenarios=len(paired),
        training_episodes=len(training),learning_diagnostics=diagnostics),indent=2))


if __name__=='__main__':main()
