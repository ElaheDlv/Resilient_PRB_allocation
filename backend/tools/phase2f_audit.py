"""Unchanged joint-reward audit: saved accounting and minimum frozen replays."""
import argparse
from collections import Counter,defaultdict
import contextlib
import csv
from dataclasses import asdict
from functools import partial
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics as st
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
OUT=ROOT/'backend/evaluations/phase2f'
C3=ROOT/'backend/evaluations/phase2c3'
C2=ROOT/'backend/evaluations/phase2c2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def jrows(path):
    with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def csvwrite(path,rows):
    rows=list(rows)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def source_entries():return json.loads((C2/'validation_manifest.json').read_text())['entries']


def freeze():
    from network_layer.joint_reward import RewardConfig
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    entries=source_entries();selection=json.loads((C3/'selection.json').read_text())
    selections={(r['method'],r['network_seed']):r['selected'] for r in selection}
    first={g:min(e['seed'] for e in entries if e['regime']==g) for g in ('feasible','contested','overloaded')}
    records=[]
    def add(method,seed,regime,severity,selection,role):
        e=next(e for e in entries if e['regime']==regime and e['severity']==severity and e['seed']==first[regime])
        episode=selections[method,seed][selection]['episode']
        records.append(dict(method=method,network_seed=seed,episode=episode,scenario_id=e['id'],role=role))
    for g in first:
        for j in (0,10,20):add('A4',12026,g,j,'aggregate','grid')
    for g in first:add('A4',22026,g,20,'aggregate','severe_replication')
    for m in ('A1','A2','A3'):add(m,12026,'contested',20,'aggregate','matched_comparison')
    for s in (12026,22026):add('A4',s,'contested',20,'fairness','checkpoint_comparison')
    add('A2',22026,'overloaded',20,'aggregate','learning_comparison')
    assert len(records)==18
    files=[C3/'training.jsonl.gz',C3/'validation.jsonl.gz',C3/'protocol.json',C3/'selection.json',C2/'validation_manifest.json',
        ROOT/'backend/network_layer/joint_reward.py',ROOT/'backend/network_layer/pilot_control.py',
        ROOT/'backend/network_layer/xApps/xapp_joint_deadline_dqn.py',ROOT/'backend/network_layer/resilient_interval.py']
    for r in records:
        e=next(e for e in entries if e['id']==r['scenario_id']);files.append(C2/e['catalog'])
        trace=e['trace'];files.append(C2/trace[len('@scenario/'):] if trace.startswith('@scenario/') else ROOT/trace)
        files.append(C3/'checkpoints'/f"{r['method']}_seed{r['network_seed']}_ep{r['episode']}.pt")
    files.append(ROOT/'backend/notebooks/Resilient_data/embb_dl.csv')
    p=dict(version='phase2f-readonly-v1',base_commit='2c31c96',reward_config=asdict(RewardConfig()),
        replays=records,prefix_accounting_steps=[100,500,1000,2000,3000],
        scope='All72 training and432 validation episode totals;18 frozen validation replays only. No training/action-history reconstruction, weights/reward changes, new scenarios or future holdout.',
        sampling='Nine A4 grid cases: one NN seed, one placement/regime, all severities. Three severe second-seed replications plus six declared matched/checkpoint/learning comparisons. Stratify grid distributions; extras are diagnostic, not extra independent replications.',
        prefix_scope='Read-only terminal administrative count at observed prefixes; not different-horizon policy outcomes or training. Saved physical horizon is always3s, checkpoint episodes3/6/9 are training budgets.',
        resources=dict(cpus=[6,7,11,14],warning_gib=2,stop_gib=2.5,minimum_available_gib=4,threads=1,workers=1,cuda=False,hard_memory_cap=False),
        input_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files))})
    write(OUT/'protocol.json',p)


def verify(p):
    from network_layer.joint_reward import RewardConfig
    assert p['reward_config']==asdict(RewardConfig())
    for n,d in p['input_sha256'].items():
        if sha(ROOT/n)!=d:raise ValueError(f'Frozen input/reward changed: {n}')


def saved(p):
    from network_layer.reward_audit import saved_components
    from network_layer.workload_feasibility import checkpoint_key
    output=[];validation=[]
    for split in ('training','validation'):
        rows=jrows(C3/(split+'.jsonl.gz'))
        for r in rows:
            components=saved_components(r)
            output.append(dict(split=split,method=r['method'],network_seed=r['network_seed'],
                episode=r.get('candidate_episode',r.get('episode')),scenario_id=r['scenario_id'],regime=r['regime'],severity=r['severity'],
                known_failure_pct=100*r['physical']['overall']['failure_fraction_lower'],
                worst_failure_pct=100*r['physical']['worst_ue_failure_fraction'],**components))
        if split=='validation':validation=rows
    csvwrite(OUT/'saved_episode_components.csv',output)
    groups=defaultdict(list)
    for r in validation:groups[r['method'],r['network_seed'],r['candidate_episode']].append(r)
    candidates=[];keys={}
    for key,rs in groups.items():
        score=checkpoint_key(rs,key[2]);keys[key]=score
        c=dict(method=key[0],network_seed=key[1],episode=key[2],candidate='-'.join(map(str,key)),
            mean_return=st.mean(r['return_value'] for r in rs),
            **{name:float(v) for name,v in zip(('known_failure','worst_failure','upper_failure','embb_deficit','reserve','switching'),score[:6])})
        cs=[saved_components(r) for r in rs]
        for part in ('failure_cost','terminal_cost','reserve_cost','switch_cost','worst_embb_residual'):
            c[part]=st.mean(v[part] for v in cs)
        candidates.append(c)
    for rank,c in enumerate(sorted(candidates,key=lambda r:-r['mean_return']),1):c['reward_rank']=rank
    for rank,c in enumerate(sorted(candidates,key=lambda r:keys[r['method'],r['network_seed'],r['episode']]),1):c['aggregate_first_rank']=rank
    csvwrite(OUT/'candidate_ordering.csv',candidates)
    comparisons=[];axes=('known_failure','worst_failure','upper_failure','embb_deficit','reserve','switching')
    for a,b in itertools.combinations(candidates,2):
        ka=(a['method'],a['network_seed'],a['episode']);kb=(b['method'],b['network_seed'],b['episode'])
        preference=(keys[ka]>keys[kb])-(keys[ka]<keys[kb])
        reward_preference=(b['mean_return']>a['mean_return'])-(b['mean_return']<a['mean_return'])
        dominates_a=all(a[k]<=b[k] for k in axes) and any(a[k]<b[k] for k in axes)
        dominates_b=all(b[k]<=a[k] for k in axes) and any(b[k]<a[k] for k in axes)
        comparisons.append(dict(candidate_a=a['candidate'],candidate_b=b['candidate'],
            ordering_disagrees=preference*reward_preference<0,
            a_dominates_b=dominates_a,b_dominates_a=dominates_b,
            dominance_disagrees=(dominates_a and reward_preference>0) or (dominates_b and reward_preference<0),
            **{k:a[k]-b[k] for k in ('mean_return',*axes,'failure_cost','terminal_cost','reserve_cost','switch_cost','worst_embb_residual')}))
    csvwrite(OUT/'ordering_comparisons.csv',comparisons)
    write(OUT/'saved_checks.json',dict(training_episodes=72,validation_episodes=432,candidates=len(candidates),
        pair_comparisons=len(comparisons),ordering_disagreements=sum(r['ordering_disagrees'] for r in comparisons),
        dominance_disagreements=sum(r['dominance_disagrees'] for r in comparisons),
        missing_training_action_rng_replay_history=True,training_interval_components_reconstructable=False))


class Streams:
    def __init__(self):self.stack=contextlib.ExitStack();self.writers={}
    def row(self,name,row):
        if name not in self.writers:
            f=self.stack.enter_context(gzip.open(OUT/name,'wt',newline=''))
            w=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');w.writeheader();self.writers[name]=w
        self.writers[name].writerow(row)
    def close(self):self.stack.close()


def audit_class():
    from network_layer.matched_control import xAppMatchedPilot
    from network_layer.joint_reward import tick_reward
    from network_layer.reward_audit import COMPONENTS
    from network_layer.feasibility_gated_fairness import ServiceWindow
    class Audited(xAppMatchedPilot):
        def __init__(self,*args,identity,streams,prefix_steps,**kwargs):
            super().__init__(*args,**kwargs)
            self.identity,self.streams,self.prefix_steps=identity,streams,prefix_steps
        def start(self):
            super().start()
            self.cumulative=dict.fromkeys(COMPONENTS,0.)
            self.interval_parts=dict.fromkeys(COMPONENTS,0.)
            self.interval_measurements=defaultdict(float);self.interval_ue=defaultdict(lambda:defaultdict(float))
            self.failure_hist=Counter();self.failures_by_step=defaultdict(list)
            self.service_window=ServiceWindow();self.previous={};self.current_service={}
            self.tick_rows=0;self.intervals=0
            self.actions={0:dict(mode=self.initial_mode.value,action_idx=None)}
            apply=self.adapter._apply_action
            def observed_apply(action):
                apply(action)
                self.actions[self.engine.sim_step]=dict(mode=action.urllc_scheduler_mode.value,action_idx=self.adapter.actions.index(action))
            self.adapter._apply_action=observed_apply
            fail=self.ledger._fail
            def observed_fail(uid,record):
                fresh=not record['failed'];fail(uid,record)
                if fresh:
                    step=self.engine.sim_step;arrival_step=round(record['arrival']/.001)
                    arrival_issue=max(0,((arrival_step-1)//10)*10);current_issue=max(self.actions)
                    before=self.actions[arrival_issue];current=self.actions[current_issue]
                    age_steps=round((self.cell.base_station._time_s-record['arrival'])/.001)
                    later_interval=arrival_issue!=current_issue;mode_changed=before['mode']!=current['mode']
                    self.failures_by_step[step].append((uid,later_interval,mode_changed,arrival_issue==0,age_steps))
                    self.failure_hist[(record['budget'],age_steps,before['mode'],current['mode'],current_issue-arrival_issue,step>10)]+=1
            self.ledger._fail=observed_fail
            compute=self.adapter._compute_reward
            def observed_reward(obs,action,previous):
                outcome=dict(self.adapter.tick_outcome);outcome['counts']=dict(outcome['counts'])
                switched=self.adapter.switched
                expected,parts=tick_reward(**outcome,risk=obs['risk'],dt=.001,switched=switched,
                    ue_count=len(self.ledger.ue_ids),config=self.reward_config)
                result=compute(obs,action,previous)
                if result[0]!=expected:raise AssertionError('Instrumentation disagrees with actual reward')
                self.record_tick(obs,action,parts,outcome,switched)
                return result
            self.adapter._compute_reward=observed_reward
        def observe_service(self,engine,dt):
            super().observe_service(engine,dt)
            qm=self.cell.base_station.queue_measurements;now=self.cell.base_station._time_s
            current={}
            for uid,u in self.cell.connected_ue_list.items():
                if u.slice_type!='URLLC':continue
                t=qm.totals[uid];prior=self.previous.get(uid,(0,0,0,0))
                cumulative=(t['offered_bytes'],t['served_bytes'],t['fully_admitted_on_time_batches'],t['offered_batches'])
                delta=[v-old for v,old in zip(cumulative,prior)];self.previous[uid]=cumulative
                current[uid]=dict(offered_bytes=delta[0],served_bytes=delta[1],on_time=delta[2],offered_batches=delta[3])
                self.service_window.observe(uid,now,cumulative[0],cumulative[1],self.cell.base_station.get_dl_buf_bytes(uid))
            self.current_service=current
        def record_tick(self,obs,action,parts,outcome,switched):
            step=self.engine.sim_step;now=self.cell.base_station._time_s;pending=self.interval.pending
            for k,v in parts.items():self.cumulative[k]+=v;self.interval_parts[k]+=v
            events=self.failures_by_step.pop(step,[])
            if len(events)!=outcome['counts']['new_known_failures']:raise AssertionError('Event ledger attribution lost failures')
            self.interval_measurements['cross_decision_interval_failures']+=sum(e[1] for e in events)
            self.interval_measurements['cross_mode_failures']+=sum(e[2] for e in events)
            self.interval_measurements['bootstrap_arrival_failures']+=sum(e[3] for e in events)
            self.interval_measurements['failure_lag_ticks_sum']+=sum(e[4] for e in events)
            for uid,d in self.current_service.items():
                for k,v in d.items():self.interval_ue[uid][k]+=v
            for uid,*_ in events:self.interval_ue[uid]['new_failures']+=1
            users=[u for u in obs['report'].ues if u.slice_id=='URLLC']
            def fraction(u):
                n=u.window.matured_offered_batches
                return u.window.matured_failed_batches/n if u.window.full_window and n is not None and n>=20 else -1.
            worst=max(users,key=lambda u:(fraction(u),u.ue_id))
            valid=obs['risk'].valid[4]
            stats=self.cell.last_scheduler_stats;baseline=self.cell.base_dl_prb_quota
            used_extra=sum(max(0,stats[s]['allocated']-baseline[s]) for s in ('eMBB','URLLC'))
            R=outcome['reserve_prbs'];unused_extra=R-used_extra
            if unused_extra<0:raise AssertionError('Observed extra use exceeds activated reserve')
            c=self.current_service
            d=dict(self.identity,step=step,interval_start=max(self.actions),action_idx=pending['action_idx'],mode=action.urllc_scheduler_mode.value,
                delta_e=action.delta_embb_prb,delta_u=action.delta_urllc_prb,switched=bool(switched),
                new_failures=outcome['counts']['new_known_failures'],terminal_charges=outcome['terminal_charges'],
                worst_risk=obs['risk'].values[4] if obs['risk'].valid[4] else None,worst_valid=valid,worst_uid=worst.ue_id if valid else '',
                worst_hol=worst.hol_wait_s if valid else None,worst_queue=worst.queued_bytes if valid else None,
                worst_deficit=self.service_window.view(worst.ue_id)['deficit'] if valid else None,
                embb_bytes=outcome['embb_bytes'],reserve_prbs=R,urllc_quota=self.cell.slice_dl_prb_quota['URLLC'],urllc_base=baseline['URLLC'],
                used_extra_lower_bound=used_extra,activated_extra_unused_in_observed_allocation=unused_extra,
                urllc_offered_bytes=sum(v['offered_bytes'] for v in c.values()),urllc_served_bytes=sum(v['served_bytes'] for v in c.values()),
                urllc_on_time=sum(v['on_time'] for v in c.values()),urllc_offered_batches=sum(v['offered_batches'] for v in c.values()),**parts)
            self.streams.row('reward_ticks.csv.gz',d);self.tick_rows+=1
            for name in ('new_failures','terminal_charges','embb_bytes','reserve_prbs','used_extra_lower_bound','activated_extra_unused_in_observed_allocation',
                         'urllc_offered_bytes','urllc_served_bytes','urllc_on_time','urllc_offered_batches'):
                self.interval_measurements[name]+=d[name]
            self.interval_measurements['worst_valid_ticks']+=obs['risk'].valid[4]
            self.interval_measurements['worst_saturated_ticks']+=bool(obs['risk'].valid[4] and obs['risk'].values[4]>=1-1e-12)
            if step%10==0:
                failed_cohorts=([b for b in self.collector.records.get(worst.ue_id,{}).values() if b.overdue(now) and b.failed(now)] if valid else [])
                arrival_other_mode=sum(self.actions[max(0,((round(b.arrival_s/.001)-1)//10)*10)]['mode']!=action.urllc_scheduler_mode.value for b in failed_cohorts)
                row=dict(self.identity,start_step=max(self.actions),end_step=step,action_idx=pending['action_idx'],mode=action.urllc_scheduler_mode.value,
                    switched=bool(self.adapter.receipt.mode_before!=self.adapter.receipt.mode_after),
                    delta_e=action.delta_embb_prb,delta_u=action.delta_urllc_prb,
                    quota_u_end=self.cell.slice_dl_prb_quota['URLLC'],baseline_u_end=baseline['URLLC'],
                    worst_risk_end=d['worst_risk'],worst_uid_end=d['worst_uid'],worst_hol_end=d['worst_hol'],worst_queue_end=d['worst_queue'],
                    worst_deficit_end=d['worst_deficit'],worst_failed_cohort_count=len(failed_cohorts),
                    worst_failed_cohorts_arrived_in_other_mode=arrival_other_mode,
                    max_failed_cohort_age_s=max((now-b.arrival_s for b in failed_cohorts),default=0.),
                    reward=-sum(self.interval_parts.values()),**dict(self.interval_measurements),**self.interval_parts)
                self.streams.row('reward_intervals.csv.gz',row);self.intervals+=1
                for u in users:
                    values=self.interval_ue[u.ue_id];w=self.service_window.view(u.ue_id)
                    self.streams.row('ue_intervals.csv.gz',dict(self.identity,end_step=step,ue=u.ue_id,budget=u.budget_s,budget_class=u.budget_class,
                        mode=action.urllc_scheduler_mode.value,queue=u.queued_bytes,hol=u.hol_wait_s,matured_failure_fraction=fraction(u) if fraction(u)>=0 else None,matured_valid=fraction(u)>=0,
                        window_deficit=w['deficit'],window_queue_growth=w['queue_growth'],
                        **{k:values.get(k,0) for k in ('offered_bytes','served_bytes','on_time','offered_batches','new_failures')}))
                self.interval_parts=dict.fromkeys(COMPONENTS,0.);self.interval_measurements=defaultdict(float);self.interval_ue=defaultdict(lambda:defaultdict(float))
            if step in self.prefix_steps:
                charges=sum(not record['failed'] for record in self.ledger.live.values())
                administrative_cost=charges/(len(self.ledger.ue_ids)*1000*.01)
                without_terminal=sum(v for k,v in self.cumulative.items() if k!='terminal_cost')
                self.streams.row('observed_prefix_accounting.csv.gz',dict(self.identity,step=step,physical_seconds=now,
                    not_yet_failed_live_batches=charges,hypothetical_prefix_closure_cost=administrative_cost,
                    observed_nonterminal_cost=without_terminal,
                    closure_share_if_assessed_here=administrative_cost/(without_terminal+administrative_cost) if without_terminal+administrative_cost else None,
                    prefix_cost_if_closed_here=without_terminal+administrative_cost,actual_terminal_here=self.ledger.closed))
        def _finish(self):
            super()._finish()
            if abs(sum(self.cumulative.values())+self.report['return_value'])>1e-8:raise AssertionError('Component sum differs from actual return')
            self.report.update(audited_component_totals=dict(self.cumulative),audit_ticks=self.tick_rows,audit_intervals=self.intervals)
            for (budget,lag,arrival_mode,failure_mode,issue_gap,paid),count in self.failure_hist.items():
                self.streams.row('failure_timing_histogram.csv.gz',dict(self.identity,budget_s=budget,lag_ticks=lag,arrival_mode=arrival_mode,
                    failure_mode=failure_mode,issue_step_gap=issue_gap,reward_paid=paid,failures=count))
    return Audited


def replay(p):
    from network_layer.joint_dqn_agent import JointDQNAgent,TrainingConfig
    from tools.phase2c1_pilot import run_episode
    from tools.phase2c2_design import resolve_catalog
    from network_layer.reward_audit import saved_components
    def forbidden(*args,**kwargs):raise AssertionError('Training forbidden in reward audit')
    JointDQNAgent.learn=forbidden;JointDQNAgent.remember=forbidden
    entries={e['id']:e for e in source_entries()};expected={(r['method'],r['network_seed'],r['candidate_episode'],r['scenario_id']):r for r in jrows(C3/'validation.jsonl.gz')}
    c3p=json.loads((C3/'protocol.json').read_text());streams=Streams();records=[]
    try:
        for item in p['replays']:
            e=entries[item['scenario_id']]
            identity=dict(run_id=f"{item['method']}_{item['network_seed']}_ep{item['episode']}_{e['id']}",method=item['method'],network_seed=item['network_seed'],
                checkpoint_episode=item['episode'],scenario_id=e['id'],regime=e['regime'],severity=e['severity'],role=item['role'])
            agent=JointDQNAgent(TrainingConfig(seed=item['network_seed'],epsilon=c3p['exploration']['start'],**c3p['training_config']))
            agent.load(C3/'checkpoints'/f"{item['method']}_seed{item['network_seed']}_ep{item['episode']}.pt")
            assert agent.checkpoint_run_metadata['ablation']==item['method']
            before=agent.fingerprint();rng=agent.rng.getstate();updates=agent.updates
            rp=dict(c3p,network_seed=item['network_seed'],training_scenario_seeds=[0,1,2],severities=[0,10,20])
            with open('/dev/null','w') as quiet,contextlib.redirect_stdout(quiet):
                row=run_episode(item['method'],e['severity'],e['seed'],rp,agent=agent,train=False,
                    catalog_override=resolve_catalog(C2,e),app_class=partial(audit_class(),identity=identity,streams=streams,prefix_steps=p['prefix_accounting_steps']))
            original=expected[item['method'],item['network_seed'],item['episode'],e['id']]
            for key in ('control_digest','parameter_sha256','return_value','common_joint_monitor_return','deadline_assignment_sha256'):
                assert row[key]==original[key],f'Frozen replay changed {key}'
            for key in ('per_ue','per_class','overall','recovery','exogenous_sha256','quota_sha256','mean_reserve_prbs'):
                assert row['physical'][key]==original['physical'][key],f'Frozen replay changed physical.{key}'
            assert agent.fingerprint()==before and agent.rng.getstate()==rng and agent.updates==updates and not agent.replay
            comp=saved_components(original)
            for k in ('failure_cost','terminal_cost','reserve_cost','switch_cost'):
                assert abs(comp[k]-row['audited_component_totals'][k])<1e-7,(k,comp[k],row['audited_component_totals'][k])
            row.update(identity);records.append(row)
            print(json.dumps(dict(stage='replay',completed=len(records),run=identity['run_id'],components=row['audited_component_totals'])),flush=True)
            del agent
        with gzip.open(OUT/'replay_summaries.jsonl.gz','wt') as f:
            for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
        write(OUT/'replay_checks.json',dict(replays=len(records),paid_ticks=sum(r['audit_ticks'] for r in records),intervals=sum(r['audit_intervals'] for r in records),
            all_original_controls_returns_physical_outcomes_match=True,all_components_sum_to_actual_rewards=True,
            all_saved_identifiable_totals_match_replays=True,dqn_updates=0))
    finally:streams.close()


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','saved','replay'));parser.add_argument('--output',type=Path,default=OUT)
    a=parser.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='freeze':freeze();return
    p=json.loads((OUT/'protocol.json').read_text());verify(p)
    marker='saved_checks.json' if a.stage=='saved' else 'replay_checks.json'
    if (OUT/marker).exists():raise FileExistsError('Refuse repeated completed audit stage')
    if a.stage=='replay':
        from tools.phase2c1_pilot import configure_environment
        configure_environment()
    start=time.perf_counter();dict(saved=saved,replay=replay)[a.stage](p)
    write(OUT/(a.stage+'_runtime.json'),dict(seconds=time.perf_counter()-start))


if __name__=='__main__':main()
