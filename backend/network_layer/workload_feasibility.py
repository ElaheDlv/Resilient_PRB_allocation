"""Exogenous capacity certificates and prospective validation selection; no learner."""

from fractions import Fraction
import math

DT = 0.001
REGIMES = ('feasible', 'contested', 'overloaded')



def registration_stream_seeds(scenario_seed):
    """Mirror the unchanged legacy episode-index-0 registration seed formula."""
    return {(scenario_seed+100000+slice_index*1000+ue_index) % (2**32)
            for slice_index in (0,1) for ue_index in range(8)}


def demand_bound(batch_bytes, rates, budgets, ticks, quota):
    """Conservative necessary bound, excluding equality-at-end deadline arrivals."""
    if batch_bytes < 1 or ticks < 1 or quota < 0 or set(rates) != set(budgets):
        raise ValueError('Invalid capacity-bound inputs')
    if any(not math.isfinite(b) or b <= 0 for b in rates.values()):
        raise ValueError('Certificates require positive finite radio rates')
    if any(not math.isfinite(d) or d <= 0 for d in budgets.values()):
        raise ValueError('Certificates require positive finite budgets')
    due = {u:max(0,math.ceil(ticks-budgets[u]/DT-1e-9)-1) for u in rates}
    work = sum(due[u]*batch_bytes/rates[u] for u in rates)
    return dict(ticks=ticks, quota=quota, due_batches=due, required_prb_ticks=work,
                available_prb_ticks=quota*ticks, excess_prb_ticks=work-quota*ticks,
                zero_miss_impossible=work > quota*ticks+1e-8,
                endpoint='strictly due; equality-at-end arrivals conservatively excluded')


def certificates(batch_bytes, rates, budgets):
    if not rates:
        raise ValueError('No UE rates')
    baseline_on = demand_bound(batch_bytes,rates,budgets,1249,54)
    optimistic_off = demand_bound(batch_bytes,rates,budgets,1250,78)
    fluid = sum(batch_bytes/b for b in rates.values())
    integer = sum(math.ceil(batch_bytes/b) for b in rates.values())
    return dict(batch_bytes=batch_bytes,fluid_prbs=fluid,integer_one_tick_prbs=integer,
                per_ue={u:dict(bytes_per_prb_tick=b,fluid_prbs=batch_bytes/b,
                    integer_prbs=math.ceil(batch_bytes/b),budget_s=budgets[u]) for u,b in rates.items()},
                baseline_on_bound=baseline_on,full_pool_off_bound=optimistic_off,
                feasible=integer<=54,
                contested=(baseline_on['zero_miss_impossible'] and integer<=72 and fluid<60),
                overloaded=optimistic_off['zero_miss_impossible'])


def choose_original(probes, regime, count, used):
    """Selection sees exogenous certificates only, never controller/model outcomes."""
    return [p for p in sorted(probes,key=lambda p:p['seed'])
            if p['seed'] not in used and p['certificate'][regime]][:count]


def checkpoint_key(rows, episode):
    """Prospective balanced metric hierarchy, not a training reward or old reranking."""
    expected={(g,j) for g in REGIMES for j in (0,10,20)}
    groups={}
    for r in rows:
        key=(r['regime'],r['severity'])
        groups.setdefault(key,[]).append(r)
    if set(groups)!=expected:
        raise ValueError('Validation requires every regime/severity stratum')
    terms=[]
    for rs in groups.values():
        group=[]
        for r in rs:
            overall=r['physical']['overall']
            n=overall['offered_batches']
            if n <= 0:raise ValueError('Missing offered exposure')
            failed=Fraction(overall['known_failed_batches'],n)
            worst=max(Fraction(u['batch_outcomes']['known_failed'],u['batch_outcomes']['offered'])
                      for u in r['physical']['per_ue'].values())
            upper=Fraction(overall['known_failed_batches']+overall['unresolved_not_due_batches'],n)
            deficit=Fraction(str(round(max(0.,1-r['embb_served_offered_fraction']),12)))
            ticks=r['steps']
            occupancy=Fraction(round(r['physical']['mean_reserve_prbs']*ticks),18*ticks)
            switching=Fraction(r['switches'],max(1,r['valid_action_checks']))
            group.append((failed,worst,upper,deficit,occupancy,switching))
        terms.append(tuple(sum(x[i] for x in group)/len(group) for i in range(6)))
    return (*tuple(sum(x[i] for x in terms)/len(terms) for i in range(6)),episode)
