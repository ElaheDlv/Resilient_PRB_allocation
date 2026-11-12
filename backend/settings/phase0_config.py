"""Opt-in validation features; historical controller/scheduler remain defaults."""

import os

RES_DQN_TRANSITION_MODE = os.getenv("RES_DQN_TRANSITION_MODE", "legacy")
if RES_DQN_TRANSITION_MODE not in {"legacy", "causal"}:
    raise ValueError("RES_DQN_TRANSITION_MODE must be legacy or causal")
PRB_UE_SCHEDULER = os.getenv("PRB_UE_SCHEDULER", "legacy")
if PRB_UE_SCHEDULER not in {"legacy", "proportional_wc", "deadline_slack"}:
    raise ValueError("PRB_UE_SCHEDULER must be legacy, proportional_wc or deadline_slack")
DEADLINE_STARVATION_TICKS = int(os.getenv("DEADLINE_STARVATION_TICKS", "5"))
if DEADLINE_STARVATION_TICKS < 1:
    raise ValueError("DEADLINE_STARVATION_TICKS must be positive")
PHASE0_MEASUREMENTS = os.getenv("PHASE0_MEASUREMENTS", "0").lower() in {"1", "true"}
PHASE0_MEASUREMENT_PATH = os.getenv("PHASE0_MEASUREMENT_PATH", "")
