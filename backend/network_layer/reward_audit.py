"""Read-only reward accounting; no alternative reward or controller is defined."""
from dataclasses import asdict
import math
from .joint_reward import RewardConfig

COMPONENTS=('failure_cost','worst_cost','embb_cost','reserve_cost','switch_cost','terminal_cost')


def saved_components(row,config=RewardConfig(),dt=.001,window_s=.100):
    """Exactly identified totals plus explicit bounds for unlogged components.

    C3 starts with zero activated reserve and excludes its first10 service ticks
    from learned reward. Resource PRB-ticks are integer despite recorded float
    averaging. Worst risk is exactly zero if no physical failure ever occurred.
    Never split the otherwise unidentified worst/eMBB residual by estimation.
    """
    if row['return_objective']!='joint-reward-v1':raise ValueError('Mixed reward objectives')
    n=len(row['physical']['per_ue']);nref=n*config.reference_batches_per_ue_s*config.period_s
    paid=row['transitions']*round(config.period_s/dt)
    bootstrap=row['steps']-paid
    if bootstrap!=10:raise ValueError('Audit requires C3 ten-tick zero-reserve initialization')
    total=-row['return_value']
    failed=row['known_failures']-row['bootstrap']['new_known_failures']
    terminal=row['terminal']['terminal_unresolved_charges']
    if terminal!=row['physical']['overall']['unresolved_not_due_batches']:
        raise AssertionError('Administrative closure does not match unresolved-not-due partition')
    prb_ticks=row['physical']['mean_reserve_prbs']*row['steps']
    if abs(prb_ticks-round(prb_ticks))>1e-5:raise AssertionError('Noninteger PRB-time accounting')
    result=dict(total_cost=total,failure_cost=config.failure*failed/nref,
        terminal_cost=config.failure*terminal/nref,
        reserve_cost=config.reserve*(dt/config.period_s)*round(prb_ticks)/18,
        switch_cost=config.switch*row['switches'],paid_ticks=paid,bootstrap_ticks=bootstrap,
        rewarded_known_failures=failed,administrative_charges=terminal,reserve_prb_ticks=round(prb_ticks))
    residual=total-sum(result[k] for k in ('failure_cost','terminal_cost','reserve_cost','switch_cost'))
    if residual < -1e-7:raise AssertionError('Negative unexplained reward residual')
    residual=max(0.,residual)
    emax=config.embb*paid*dt/config.period_s
    # Full-window validity cannot begin before100ms; exposure can delay it more.
    possible_valid=max(0,row['steps']-max(bootstrap,math.ceil(window_s/dt)-1))
    wmax=config.worst*possible_valid*dt/config.period_s
    duration=row['steps']*dt
    served=round(row['physical']['embb_throughput_mbps']*1e6*duration/8)
    target_bytes=dt*config.embb_target_bps/8
    # Paid service is <=whole-episode service: a conservative Jensen bound only.
    emin=config.embb*(dt/config.period_s)*max(0,paid-served/target_bytes)
    result.update(worst_embb_residual=residual,
        worst_lower=max(0.,residual-emax),worst_upper=min(wmax,residual),
        embb_lower=max(emin,residual-wmax,0.),embb_upper=min(emax,residual),
        worst_cost=None,embb_cost=None,
        component_identification='Four exact totals; worst+eMBB residual/bounds')
    if row['known_failures']==0:
        result.update(worst_cost=0.,embb_cost=residual,worst_lower=0.,worst_upper=0.,embb_lower=residual,embb_upper=residual,
            component_identification='Six totals identified: no physical failures implies worst risk zero')
    if result['worst_lower']>result['worst_upper']+1e-6 or result['embb_lower']>result['embb_upper']+1e-6:
        raise AssertionError('Inconsistent component bounds')
    return result
