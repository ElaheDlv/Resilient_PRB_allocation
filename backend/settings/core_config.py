import os
from . import slice_config as _slice_cfg
from .ue_config import UE_DEFAULT_MAX_COUNT

import random

NETWORK_SLICE_EMBB_NAME = _slice_cfg.NETWORK_SLICE_EMBB_NAME
NETWORK_SLICE_URLLC_NAME = _slice_cfg.NETWORK_SLICE_URLLC_NAME
CORE_UE_SUBSCRIPTION_DATA = {}

# Optional explicit counts for the simple preset
_preset = os.getenv("RAN_TOPOLOGY_PRESET", "default")
e_cnt = os.getenv("UE_SIMPLE_COUNT_EMBB")
u_cnt = os.getenv("UE_SIMPLE_COUNT_URLLC")
try:
    e_cnt = int(e_cnt) if e_cnt is not None else None
    u_cnt = int(u_cnt) if u_cnt is not None else None
except ValueError:
    e_cnt = u_cnt = None

if _preset == "simple" and any(v is not None for v in (e_cnt, u_cnt)):
    # Deterministic distribution: first u_cnt IMSIs are URLLC, next e_cnt eMBB
    total = UE_DEFAULT_MAX_COUNT
    u = max(0, min(u_cnt or 0, total))
    e = max(0, min(e_cnt or 0, max(0, total - u)))
    # Fill remainder with eMBB
    rem = max(0, total - (u + e))
    e += rem

    idx = 0
    for _ in range(u):
        CORE_UE_SUBSCRIPTION_DATA[f"IMSI_{idx}"] = [NETWORK_SLICE_URLLC_NAME]
        idx += 1
    for _ in range(e):
        CORE_UE_SUBSCRIPTION_DATA[f"IMSI_{idx}"] = [NETWORK_SLICE_EMBB_NAME]
        idx += 1
else:
    # Default randomized distribution
    for i in range(UE_DEFAULT_MAX_COUNT):
        UE_IMSI = f"IMSI_{i}"

        # general UEs default to eMBB
        CORE_UE_SUBSCRIPTION_DATA[UE_IMSI] = [NETWORK_SLICE_EMBB_NAME]

        # 50% chance they also subscribe to URLLC
        if random.random() < 0.5:
            CORE_UE_SUBSCRIPTION_DATA[UE_IMSI].append(NETWORK_SLICE_URLLC_NAME)
