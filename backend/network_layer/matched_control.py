"""Matched Phase 2C-3 ablations: only the feasible action subset changes.

A3 restricts reserve increments to the existing causal heuristic's current
command. The selected 98-catalog action is then committed atomically by the
unchanged bridge. The next-state mask recomputes that command at the next
observation, rather than retaining the previous reserve increment.
"""
import numpy as np

from .joint_dqn_schema import ACTIONS, encode_joint_state, joint_action_mask
from .joint_control import URLLCSchedulerMode as Mode
from .pilot_control import PilotAdapter, xAppPairedPilot
from .resilient_interval import CausalInterval
from .xApps.xapp_resilient_dqn_allocator import ACTION_LIST

METHODS = ('A1', 'A2', 'A3', 'A4')
MASK_VERSION = 'matched-ablation-mask-v1'


def restricted_mask(method, physical_mask, reactive_pair=None):
    """Intersect experiment restrictions with physical feasibility; never relax it."""
    if method not in METHODS or len(physical_mask) != 98:
        raise ValueError('Expected a matched method and the 98-action physical mask')
    if method == 'A3' and reactive_pair not in ACTION_LIST:
        raise ValueError('A3 requires a catalog reserve command')
    result = np.asarray([bool(valid) and (
        (method == 'A1' and a.urllc_scheduler_mode is Mode.PROPORTIONAL) or
        (method == 'A2' and a.urllc_scheduler_mode is Mode.FEASIBILITY_AWARE) or
        (method == 'A3' and (a.delta_embb_prb, a.delta_urllc_prb) == reactive_pair) or
        method == 'A4') for a, valid in zip(ACTIONS, physical_mask)], dtype=bool)
    if not result.any():
        raise ValueError('Ablation mask has no physically feasible action')
    return result


class MatchedAdapter(PilotAdapter):
    def _action_mask(self):
        physical = joint_action_mask(self.app.bridge)
        pair = None
        if self.app.ablation == 'A3':
            # Both scheduling modes share the same resource feasibility. Use OR
            # to preserve correctness if a mode is unavailable in a future catalog.
            reserve_mask = [bool(physical[i] or physical[i+1]) for i in range(0,98,2)]
            index = self.app.reactive.select(self.app.physical._collect_obs(), reserve_mask)
            pair = ACTION_LIST[index]
        return restricted_mask(self.app.ablation, physical, pair)


class xAppMatchedPilot(xAppPairedPilot):
    def __init__(self, ric=None, *, method, **kwargs):
        if method not in METHODS:
            raise ValueError('Unknown matched ablation')
        # The parent P branch uses the joint encoder, reward, actions and learner.
        # Keep this internal dispatch distinct from the public ablation label.
        super().__init__(ric, method='P', **kwargs)
        self.ablation = method
        self.initial_mode = Mode.FEASIBILITY_AWARE if method == 'A2' else Mode.PROPORTIONAL

    def start(self):
        super().start()
        self.adapter = MatchedAdapter(self)
        self.interval = CausalInterval(self.adapter, 10, encode_joint_state)

    def _finish(self):
        super()._finish()
        self.report.update(method=self.ablation, mask_version=MASK_VERSION)
