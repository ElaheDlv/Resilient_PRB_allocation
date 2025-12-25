# ---------------------------
# Resilient Gym Allocator Configuration
# ---------------------------
import os

# Enable/disable the xApp and select agent behavior
RES_GYM_ENABLE = os.getenv("RES_GYM_ENABLE", "0") in ("1", "true", "True")
RES_GYM_TRAIN = os.getenv("RES_GYM_TRAIN", "1") in ("1", "true", "True")
RES_GYM_EVAL_ONLY = os.getenv("RES_GYM_EVAL_ONLY", "0") in ("1", "true", "True")
RES_GYM_AGENT = os.getenv("RES_GYM_AGENT", "dqn").strip().lower()
RES_GYM_CONFIG_PATH = os.getenv("RES_GYM_CONFIG_PATH", "").strip()
RES_GYM_LOOP = os.getenv("RES_GYM_LOOP", "0") in ("1", "true", "True")
RES_GYM_SHUFFLE_EPISODES = os.getenv("RES_GYM_SHUFFLE_EPISODES", "0") in ("1", "true", "True")

# Decision cadence and action discretization
try:
    RES_GYM_DECISION_PERIOD = max(1, int(os.getenv("RES_GYM_DECISION_PERIOD", "1")))
except Exception:
    RES_GYM_DECISION_PERIOD = 1
_levels_raw = os.getenv("RES_GYM_ACTION_LEVELS")
if _levels_raw:
    try:
        RES_GYM_ACTION_LEVELS = [int(float(x.strip())) for x in _levels_raw.split(",") if x.strip()]
    except Exception:
        RES_GYM_ACTION_LEVELS = None
else:
    RES_GYM_ACTION_LEVELS = None

# Core RL hyperparameters
try:
    RES_GYM_GAMMA = float(os.getenv("RES_GYM_GAMMA", "0.99"))
except Exception:
    RES_GYM_GAMMA = 0.99
try:
    RES_GYM_W_EMBB = float(os.getenv("RES_GYM_W_EMBB", "1.0"))
    RES_GYM_W_URLLC = float(os.getenv("RES_GYM_W_URLLC", "1.5"))
except Exception:
    RES_GYM_W_EMBB, RES_GYM_W_URLLC = 1.0, 1.5
try:
    RES_GYM_EPS_START = float(os.getenv("RES_GYM_EPS_START", "1.0"))
    RES_GYM_EPS_END = float(os.getenv("RES_GYM_EPS_END", "0.05"))
    RES_GYM_EPS_DECAY_STEPS = int(os.getenv("RES_GYM_EPS_DECAY_STEPS", "20000"))
except Exception:
    RES_GYM_EPS_START, RES_GYM_EPS_END, RES_GYM_EPS_DECAY_STEPS = 1.0, 0.05, 20000
try:
    RES_GYM_LR = float(os.getenv("RES_GYM_LR", "1e-3"))
except Exception:
    RES_GYM_LR = 1e-3
try:
    RES_GYM_BATCH = int(os.getenv("RES_GYM_BATCH", "64"))
except Exception:
    RES_GYM_BATCH = 64
try:
    RES_GYM_BUFFER = int(os.getenv("RES_GYM_BUFFER", "100000"))
except Exception:
    RES_GYM_BUFFER = 100000
try:
    RES_GYM_TARGET_UPDATE = int(os.getenv("RES_GYM_TARGET_UPDATE", "2000"))
except Exception:
    RES_GYM_TARGET_UPDATE = 2000
try:
    RES_GYM_QR_ATOMS = int(os.getenv("RES_GYM_QR_ATOMS", "51"))
except Exception:
    RES_GYM_QR_ATOMS = 51

# PPO-specific knobs
try:
    RES_GYM_PPO_CLIP = float(os.getenv("RES_GYM_PPO_CLIP", "0.2"))
    RES_GYM_PPO_EPOCHS = int(os.getenv("RES_GYM_PPO_EPOCHS", "4"))
    RES_GYM_PPO_ROLLOUT = int(os.getenv("RES_GYM_PPO_ROLLOUT", "256"))
    RES_GYM_PPO_ENT_COEF = float(os.getenv("RES_GYM_PPO_ENT_COEF", "0.0"))
    RES_GYM_PPO_VF_COEF = float(os.getenv("RES_GYM_PPO_VF_COEF", "0.5"))
    RES_GYM_PPO_GAE_LAMBDA = float(os.getenv("RES_GYM_PPO_GAE_LAMBDA", "0.95"))
except Exception:
    RES_GYM_PPO_CLIP, RES_GYM_PPO_EPOCHS, RES_GYM_PPO_ROLLOUT = 0.2, 4, 256
    RES_GYM_PPO_ENT_COEF, RES_GYM_PPO_VF_COEF, RES_GYM_PPO_GAE_LAMBDA = 0.0, 0.5, 0.95

# Stateless proportional-fair heuristic weights
try:
    RES_GYM_PF_LAT_W = float(os.getenv("RES_GYM_PF_LAT_W", "1.0"))
    RES_GYM_PF_BUF_W = float(os.getenv("RES_GYM_PF_BUF_W", "0.1"))
except Exception:
    RES_GYM_PF_LAT_W, RES_GYM_PF_BUF_W = 1.0, 0.1

# Episode/trace helpers
try:
    RES_GYM_REGISTER_RETRIES = max(1, int(os.getenv("RES_GYM_REGISTER_RETRIES", "20")))
except Exception:
    RES_GYM_REGISTER_RETRIES = 20
PRB_RESILIENT_NORM_BUF_BYTES = float(os.getenv("PRB_RESILIENT_NORM_BUF_BYTES", "200000"))

# Telemetry and dashboards
RES_GYM_DEBUG_LOG = os.getenv("RES_GYM_DEBUG_LOG", "").strip()
RES_GYM_TB_LOGDIR = os.getenv("RES_GYM_TB_LOGDIR", "").strip()
RES_GYM_DASH_ENABLE = os.getenv("RES_GYM_DASH_ENABLE", "0") in ("1", "true", "True")
try:
    RES_GYM_DASH_PORT = int(os.getenv("RES_GYM_DASH_PORT", "5055"))
except Exception:
    RES_GYM_DASH_PORT = 5055
