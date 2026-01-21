"""Configuration for the resilient DQN xApp (non-episodic trainer).

All knobs can be overridden via environment variables. Defaults keep the
xApp disabled until explicitly enabled.
"""

import os

# Toggle and training mode
RES_DQN_ENABLE = os.getenv("RES_DQN_ENABLE", "0") in ("1", "true", "True")
RES_DQN_TRAIN = os.getenv("RES_DQN_TRAIN", "1") in ("1", "true", "True")

# Decision cadence (simulation steps between actions)
try:
    RES_DQN_DECISION_PERIOD = max(1, int(os.getenv("RES_DQN_DECISION_PERIOD", "1")))
except Exception:
    RES_DQN_DECISION_PERIOD = 1

# Action space (resiliency PRBs per slice)
_levels_raw = os.getenv("RES_DQN_LEVELS", "")
if _levels_raw:
    try:
        RES_DQN_LEVELS = [int(float(x.strip())) for x in _levels_raw.split(",") if x.strip()]
    except Exception:
        RES_DQN_LEVELS = None
else:
    RES_DQN_LEVELS = [-9, -3, -1, 0, 1, 3, 9]
try:
    RES_DQN_TOTAL_RESILIENCY_PRB = int(os.getenv("RES_DQN_TOTAL_RESILIENCY_PRB", "18"))
except Exception:
    RES_DQN_TOTAL_RESILIENCY_PRB = 18
try:
    RES_DQN_MAX_DL_PRB = int(os.getenv("RES_DQN_MAX_DL_PRB", "218"))
except Exception:
    RES_DQN_MAX_DL_PRB = 218

# State normalization anchors
try:
    RES_DQN_EMBB_BASE_PRB = float(os.getenv("RES_DQN_EMBB_BASE_PRB", str(RES_DQN_TOTAL_RESILIENCY_PRB)))
except Exception:
    RES_DQN_EMBB_BASE_PRB = float(RES_DQN_TOTAL_RESILIENCY_PRB)
try:
    RES_DQN_URLLC_BASE_PRB = float(os.getenv("RES_DQN_URLLC_BASE_PRB", str(RES_DQN_TOTAL_RESILIENCY_PRB)))
except Exception:
    RES_DQN_URLLC_BASE_PRB = float(RES_DQN_TOTAL_RESILIENCY_PRB)
try:
    RES_DQN_EMBB_QUEUE_MAX = float(os.getenv("RES_DQN_EMBB_QUEUE_MAX", "1200000"))
    RES_DQN_URLLC_QUEUE_MAX = float(os.getenv("RES_DQN_URLLC_QUEUE_MAX", "5200000"))
except Exception:
    RES_DQN_EMBB_QUEUE_MAX, RES_DQN_URLLC_QUEUE_MAX = 1200000.0, 5200000.0
try:
    RES_DQN_EMBB_THR_MAX = float(os.getenv("RES_DQN_EMBB_THR_MAX", "9.6e7"))
    RES_DQN_URLLC_THR_MAX = float(os.getenv("RES_DQN_URLLC_THR_MAX", "4.16e7"))
except Exception:
    RES_DQN_EMBB_THR_MAX, RES_DQN_URLLC_THR_MAX = 9.6e7, 4.16e7
try:
    RES_DQN_URLLC_LATENCY_DEADLINE = float(os.getenv("RES_DQN_URLLC_LATENCY_DEADLINE", "0.001"))
except Exception:
    RES_DQN_URLLC_LATENCY_DEADLINE = 0.01

# DQN hyperparameters
try:
    RES_DQN_EPS_START = float(os.getenv("RES_DQN_EPS_START", "1.0"))
    RES_DQN_EPS_END = float(os.getenv("RES_DQN_EPS_END", "0.05"))
    RES_DQN_EPS_DECAY = int(os.getenv("RES_DQN_EPS_DECAY", "20000"))
except Exception:
    RES_DQN_EPS_START, RES_DQN_EPS_END, RES_DQN_EPS_DECAY = 1.0, 0.05, 20000
try:
    RES_DQN_GAMMA = float(os.getenv("RES_DQN_GAMMA", "0.99"))
    RES_DQN_LR = float(os.getenv("RES_DQN_LR", "1e-4"))
    RES_DQN_BATCH = int(os.getenv("RES_DQN_BATCH", "64"))
    RES_DQN_BUFFER = int(os.getenv("RES_DQN_BUFFER", "100000"))
    RES_DQN_TARGET_UPDATE = int(os.getenv("RES_DQN_TARGET_UPDATE", "2000"))
except Exception:
    RES_DQN_GAMMA, RES_DQN_LR, RES_DQN_BATCH, RES_DQN_BUFFER, RES_DQN_TARGET_UPDATE = (
        0.99,
        1e-4,
        64,
        100000,
        2000,
    )

# Reward weights
try:
    RES_DQN_W_E = float(os.getenv("RES_DQN_W_E", "1.0"))
    RES_DQN_W_U = float(os.getenv("RES_DQN_W_U", "1.0"))
    RES_DQN_LAMBDA_LAT = float(os.getenv("RES_DQN_LAMBDA_LAT", "2.0"))
    RES_DQN_MU_RESERVE = float(os.getenv("RES_DQN_MU_RESERVE", "0.5"))
except Exception:
    RES_DQN_W_E, RES_DQN_W_U, RES_DQN_LAMBDA_LAT, RES_DQN_MU_RESERVE = 1.0, 1.0, 2.0, 0.5

# Telemetry / dashboard
RES_DQN_DASH_ENABLE = os.getenv("RES_DQN_DASH_ENABLE", "0") in ("1", "true", "True")
try:
    RES_DQN_DASH_PORT = int(os.getenv("RES_DQN_DASH_PORT", "5060"))
except Exception:
    RES_DQN_DASH_PORT = 5060

# TensorBoard logging
RES_DQN_TB_ENABLE = os.getenv("RES_DQN_TB_ENABLE", "0") in ("1", "true", "True")
RES_DQN_TB_DIR = os.getenv("RES_DQN_TB_DIR", "backend/tb_logs/res_dqn")

# Checkpointing
RES_DQN_MODEL_DIR = os.getenv("RES_DQN_MODEL_DIR", "backend/checkpoints/res_dqn")
try:
    RES_DQN_SAVE_INTERVAL = int(os.getenv("RES_DQN_SAVE_INTERVAL", "0"))
except Exception:
    RES_DQN_SAVE_INTERVAL = 0

# TensorBoard smoothing (window over recent rewards)
try:
    RES_DQN_TB_SMOOTH_WIN = int(os.getenv("RES_DQN_TB_SMOOTH_WIN", "200"))

    # Optional in-process eval during training (0 disables)
    RES_DQN_EVAL_INTERVAL = int(os.getenv("RES_DQN_EVAL_INTERVAL", "0"))
    RES_DQN_EVAL_EPISODES = int(os.getenv("RES_DQN_EVAL_EPISODES", "1"))
except Exception:
    RES_DQN_TB_SMOOTH_WIN = 200
