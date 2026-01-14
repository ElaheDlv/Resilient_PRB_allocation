"""Configuration for validating resilient PRB allocation policies.

This keeps the validator disabled by default and exposes a small set of
environment-variable knobs to select the policy to replay and the dashboard
port.
"""

import os

# Toggle validation xApp
RES_VALIDATE_ENABLE = os.getenv("RES_VALIDATE_ENABLE", "0") in ("1", "true", "True")

# Policy selection: qrdqn | dqn | random | idle
RES_VALIDATE_POLICY = (os.getenv("RES_VALIDATE_POLICY", "random") or "random").lower()

# Optional RNG seed to make validation reproducible
try:
    _seed_raw = os.getenv("RES_VALIDATE_SEED", "")
    RES_VALIDATE_SEED = None if _seed_raw == "" else int(_seed_raw)
except Exception:
    RES_VALIDATE_SEED = None

# Decision cadence (simulation steps between actions)
try:
    RES_VALIDATE_DECISION_PERIOD = max(
        1,
        int(
            os.getenv(
                "RES_VALIDATE_DECISION_PERIOD",
                os.getenv("RES_QRDQN_DECISION_PERIOD", os.getenv("RES_DQN_DECISION_PERIOD", "1")),
            )
        ),
    )
except Exception:
    RES_VALIDATE_DECISION_PERIOD = 1

# Optional checkpoints to load for the QR-DQN or DQN policies
RES_VALIDATE_QRDQN_MODEL = os.getenv("RES_VALIDATE_QRDQN_MODEL", "")
RES_VALIDATE_DQN_MODEL = os.getenv("RES_VALIDATE_DQN_MODEL", "")

# Dashboard controls (reuses QR-DQN defaults)
RES_VALIDATE_DASH_ENABLE = os.getenv(
    "RES_VALIDATE_DASH_ENABLE",
    os.getenv("RES_QRDQN_DASH_ENABLE", "0"),
) in ("1", "true", "True")
try:
    RES_VALIDATE_DASH_PORT = int(os.getenv("RES_VALIDATE_DASH_PORT", os.getenv("RES_QRDQN_DASH_PORT", "5060")))
except Exception:
    RES_VALIDATE_DASH_PORT = 5060

# TensorBoard logging
RES_VALIDATE_TB_ENABLE = os.getenv("RES_VALIDATE_TB_ENABLE", "0") in ("1", "true", "True")
RES_VALIDATE_TB_DIR = os.getenv("RES_VALIDATE_TB_DIR", "")
