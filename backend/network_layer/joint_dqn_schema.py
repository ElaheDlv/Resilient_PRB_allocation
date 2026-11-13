"""Versioned joint action/state schema, separate from all legacy checkpoints."""

from dataclasses import asdict
import math
from numbers import Integral

import numpy as np
import settings
from .joint_control import JointRICAction, URLLCSchedulerMode
from .prb_accounting import validate_quotas
from .xApps.xapp_resilient_dqn_allocator import build_state

LEVELS = (-9, -3, -1, 0, 1, 3, 9)
MODES = (URLLCSchedulerMode.PROPORTIONAL, URLLCSchedulerMode.FEASIBILITY_AWARE)
ACTIONS = tuple(JointRICAction(e, u, mode) for e in LEVELS for u in LEVELS for mode in MODES)
STATE_VERSION = 'joint-state-v1-30'
ACTION_VERSION = 'joint-action-v1-98'
STATE_DIM = 30
LEGACY_RETAINED = tuple(i for i in range(15) if i != 10)


def decode_action(index):
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(ACTIONS):
        raise ValueError('Joint action index must be an integer in [0,98)')
    return ACTIONS[index]


def encode_action(action):
    if not isinstance(action, JointRICAction):
        raise TypeError('Expected JointRICAction')
    try:
        return ACTIONS.index(action)
    except ValueError as exc:
        raise ValueError('Action increments are outside the joint catalog') from exc


def joint_action_mask(bridge):
    """Same feasibility contract as atomic execution, without consuming a boundary."""
    bridge.check_physics()
    result = []
    cell = bridge.cell
    for action in ACTIONS:
        quotas = dict(cell.slice_dl_prb_quota)
        quotas[settings.NETWORK_SLICE_EMBB_NAME] += action.delta_embb_prb
        quotas[settings.NETWORK_SLICE_URLLC_NAME] += action.delta_urllc_prb
        try:
            bridge._validate_mode(action.urllc_scheduler_mode)
            validate_quotas(cell, quotas, bridge.pool)
        except (AssertionError, ValueError):
            result.append(False)
        else:
            result.append(True)
    if not any(result):
        raise AssertionError('No feasible joint action')
    return result


def encode_joint_state(obs, previous_action):
    """20 primary features + 6 validity bits + 2 exposures + 2 known bits."""
    risk = obs['risk']
    legacy = build_state(obs['legacy'], (previous_action.delta_embb_prb,
                                          previous_action.delta_urllc_prb))
    values = []
    for i, (value, valid) in enumerate(zip(risk.values, risk.valid)):
        if valid:
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid finite risk input')
            values.append(value / (1+value) if i == 0 else value)
        else:
            values.append(0.0)  # explicitly paired with validity=0
    exposures = []
    known = []
    for name, count in zip(('tight', 'relaxed'), risk.matured_exposures):
        members = sum(u.slice_id == 'URLLC' and u.budget_class == name
                      for u in obs['report'].ues)
        exposures.append(min(1.0, count/max(1, members*obs['minimum_matured']))
                         if count is not None else 0.0)
        known.append(float(count is not None))
    state = np.asarray([*legacy[list(LEGACY_RETAINED)], *values,
                        *map(float, risk.valid), *exposures, *known], dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError('Joint state must contain 30 finite inputs')
    return state


def state_metadata(obs):
    return dict(risk=asdict(obs['risk']), primary_features=20, input_dimension=STATE_DIM,
                state_version=STATE_VERSION, legacy_retained_indices=LEGACY_RETAINED)
