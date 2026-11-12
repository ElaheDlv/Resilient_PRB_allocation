"""Opt-in validation features; historical controller/scheduler remain defaults."""

import os

RES_DQN_TRANSITION_MODE = os.getenv("RES_DQN_TRANSITION_MODE", "legacy")
if RES_DQN_TRANSITION_MODE not in {"legacy", "causal"}:
    raise ValueError("RES_DQN_TRANSITION_MODE must be legacy or causal")
PRB_UE_SCHEDULER = os.getenv("PRB_UE_SCHEDULER", "legacy")
if PRB_UE_SCHEDULER not in {"legacy", "proportional_wc"}:
    raise ValueError("PRB_UE_SCHEDULER must be legacy or proportional_wc")
PHASE0_MEASUREMENTS = os.getenv("PHASE0_MEASUREMENTS", "0").lower() in {"1", "true"}
PHASE0_MEASUREMENT_PATH = os.getenv("PHASE0_MEASUREMENT_PATH", "")
