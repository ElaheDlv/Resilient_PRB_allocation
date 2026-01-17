"""Configuration for the resilient QR-DQN xApp (non-episodic trainer).

All knobs can be overridden via environment variables. Defaults keep the
xApp disabled until explicitly enabled.
"""

import os

# Toggle and training mode
RES_QRDQN_ENABLE = os.getenv("RES_QRDQN_ENABLE", "0") in ("1", "true", "True")
RES_QRDQN_TRAIN = os.getenv("RES_QRDQN_TRAIN", "1") in ("1", "true", "True")

# Decision cadence (simulation steps between actions)
try:
    RES_QRDQN_DECISION_PERIOD = max(1, int(os.getenv("RES_QRDQN_DECISION_PERIOD", "1")))
except Exception:
    RES_QRDQN_DECISION_PERIOD = 1

# Action space (resiliency PRBs per slice)
_levels_raw = os.getenv("RES_QRDQN_LEVELS", "")
if _levels_raw:
    try:
        RES_QRDQN_LEVELS = [int(float(x.strip())) for x in _levels_raw.split(",") if x.strip()]
    except Exception:
        RES_QRDQN_LEVELS = None
else:
    RES_QRDQN_LEVELS = [-9, -3, -1, 0, 1, 3, 9]
try:
    RES_QRDQN_TOTAL_RESILIENCY_PRB = int(os.getenv("RES_QRDQN_TOTAL_RESILIENCY_PRB", "18"))
except Exception:
    RES_QRDQN_TOTAL_RESILIENCY_PRB = 18
try:
    RES_QRDQN_MAX_DL_PRB = int(os.getenv("RES_QRDQN_MAX_DL_PRB", "218"))
except Exception:
    RES_QRDQN_MAX_DL_PRB = 218

# State normalization anchors
try:
    RES_QRDQN_EMBB_BASE_PRB = float(os.getenv("RES_QRDQN_EMBB_BASE_PRB", str(RES_QRDQN_TOTAL_RESILIENCY_PRB)))
except Exception:
    RES_QRDQN_EMBB_BASE_PRB = float(RES_QRDQN_TOTAL_RESILIENCY_PRB)
try:
    RES_QRDQN_URLLC_BASE_PRB = float(os.getenv("RES_QRDQN_URLLC_BASE_PRB", str(RES_QRDQN_TOTAL_RESILIENCY_PRB)))
except Exception:
    RES_QRDQN_URLLC_BASE_PRB = float(RES_QRDQN_TOTAL_RESILIENCY_PRB)
try:
    RES_QRDQN_EMBB_QUEUE_MAX = float(os.getenv("RES_QRDQN_EMBB_QUEUE_MAX", "1200000"))
    RES_QRDQN_URLLC_QUEUE_MAX = float(os.getenv("RES_QRDQN_URLLC_QUEUE_MAX", "5200000"))
except Exception:
    RES_QRDQN_EMBB_QUEUE_MAX, RES_QRDQN_URLLC_QUEUE_MAX = 1200000.0, 5200000.0
try:
    RES_QRDQN_EMBB_THR_MAX = float(os.getenv("RES_QRDQN_EMBB_THR_MAX", "9.6e7"))
    RES_QRDQN_URLLC_THR_MAX = float(os.getenv("RES_QRDQN_URLLC_THR_MAX", "4.16e7"))
except Exception:
    RES_QRDQN_EMBB_THR_MAX, RES_QRDQN_URLLC_THR_MAX = 9.6e7, 4.16e7
try:
    RES_QRDQN_URLLC_LATENCY_DEADLINE = float(os.getenv("RES_QRDQN_URLLC_LATENCY_DEADLINE", "0.001"))
except Exception:
    RES_QRDQN_URLLC_LATENCY_DEADLINE = 0.01

# QR-DQN hyperparameters
try:
    RES_QRDQN_EPS_START = float(os.getenv("RES_QRDQN_EPS_START", "1.0"))
    RES_QRDQN_EPS_END = float(os.getenv("RES_QRDQN_EPS_END", "0.05"))
    RES_QRDQN_EPS_DECAY = int(os.getenv("RES_QRDQN_EPS_DECAY", "20000"))
except Exception:
    RES_QRDQN_EPS_START, RES_QRDQN_EPS_END, RES_QRDQN_EPS_DECAY = 1.0, 0.05, 20000
try:
    RES_QRDQN_GAMMA = float(os.getenv("RES_QRDQN_GAMMA", "0.99"))
    RES_QRDQN_LR = float(os.getenv("RES_QRDQN_LR", "1e-3"))
    RES_QRDQN_BATCH = int(os.getenv("RES_QRDQN_BATCH", "64"))
    RES_QRDQN_BUFFER = int(os.getenv("RES_QRDQN_BUFFER", "100000"))
    RES_QRDQN_TARGET_UPDATE = int(os.getenv("RES_QRDQN_TARGET_UPDATE", "2000"))
    RES_QRDQN_ATOMS = int(os.getenv("RES_QRDQN_ATOMS", "51"))
except Exception:
    RES_QRDQN_GAMMA, RES_QRDQN_LR, RES_QRDQN_BATCH, RES_QRDQN_BUFFER, RES_QRDQN_TARGET_UPDATE, RES_QRDQN_ATOMS = (
        0.99,
        1e-3,
        64,
        100000,
        2000,
        51,
    )

# Reward weights
try:
    RES_QRDQN_W_E = float(os.getenv("RES_QRDQN_W_E", "1.0"))
    RES_QRDQN_W_U = float(os.getenv("RES_QRDQN_W_U", "1.5"))
    RES_QRDQN_LAMBDA_LAT = float(os.getenv("RES_QRDQN_LAMBDA_LAT", "2.0"))
    RES_QRDQN_MU_RESERVE = float(os.getenv("RES_QRDQN_MU_RESERVE", "0.5"))
except Exception:
    RES_QRDQN_W_E, RES_QRDQN_W_U, RES_QRDQN_LAMBDA_LAT, RES_QRDQN_MU_RESERVE = 1.0, 1.5, 2.0, 0.5

# Telemetry / dashboard
RES_QRDQN_DASH_ENABLE = os.getenv("RES_QRDQN_DASH_ENABLE", "0") in ("1", "true", "True")
try:
    RES_QRDQN_DASH_PORT = int(os.getenv("RES_QRDQN_DASH_PORT", "5060"))
except Exception:
    RES_QRDQN_DASH_PORT = 5060

# TensorBoard logging
RES_QRDQN_TB_ENABLE = os.getenv("RES_QRDQN_TB_ENABLE", "0") in ("1", "true", "True")
RES_QRDQN_TB_DIR = os.getenv("RES_QRDQN_TB_DIR", "backend/tb_logs/res_qrdqn")

# Checkpointing
RES_QRDQN_MODEL_DIR = os.getenv("RES_QRDQN_MODEL_DIR", "backend/checkpoints/res_qrdqn")
try:
    RES_QRDQN_SAVE_INTERVAL = int(os.getenv("RES_QRDQN_SAVE_INTERVAL", "0"))
except Exception:
    RES_QRDQN_SAVE_INTERVAL = 0

# TensorBoard smoothing (window over recent rewards)
try:
    RES_QRDQN_TB_SMOOTH_WIN = int(os.getenv("RES_QRDQN_TB_SMOOTH_WIN", "200"))

    # Optional in-process eval during training (0 disables)
    RES_QRDQN_EVAL_INTERVAL = int(os.getenv("RES_QRDQN_EVAL_INTERVAL", "0"))
    RES_QRDQN_EVAL_EPISODES = int(os.getenv("RES_QRDQN_EVAL_EPISODES", "1"))
except Exception:
    RES_QRDQN_TB_SMOOTH_WIN = 200
