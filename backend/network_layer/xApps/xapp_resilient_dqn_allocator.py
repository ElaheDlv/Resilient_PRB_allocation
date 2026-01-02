"""Resilient PRB allocator using a DQN agent with a fixed action/state space.

Action space (resiliency PRB split across eMBB/URLLC):
RESILIENCY_LEVELS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
TOTAL_RESILIENCY_PRB = 18
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import settings
from .xapp_base import xAppBase
from utils import load_raw_packet_csv

SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None

try:
    import dash
    from dash import Dash, dcc, html, Input, Output
    import plotly.graph_objs as go

    DASH_AVAILABLE = True
except Exception:
    dash = None
    Dash = None
    dcc = None
    html = None
    Input = None
    Output = None
    go = None
    DASH_AVAILABLE = False


def _safe_div(a: float, b: float) -> float:
    b = float(b)
    if abs(b) < 1e-9:
        return 0.0
    return float(a) / b


# --------------------------- Action/state helpers --------------------------- #
RESILIENCY_LEVELS = (
    getattr(settings, "RES_DQN_LEVELS", None) or [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
)
TOTAL_RESILIENCY_PRB = int(getattr(settings, "RES_DQN_TOTAL_RESILIENCY_PRB", 18))
DEFAULT_MAX_DL_PRB = int(getattr(settings, "RES_DQN_MAX_DL_PRB", 218))


def build_action_list() -> List[Tuple[int, int]]:
    actions = []
    for r_e in RESILIENCY_LEVELS:
        for r_u in RESILIENCY_LEVELS:
            if r_e + r_u <= TOTAL_RESILIENCY_PRB:
                actions.append((r_e, r_u))
    return actions


ACTION_LIST = build_action_list()
NUM_ACTIONS = len(ACTION_LIST)

EMBB_BASE_PRB = float(getattr(settings, "RES_DQN_EMBB_BASE_PRB", TOTAL_RESILIENCY_PRB))
URLLC_BASE_PRB = float(getattr(settings, "RES_DQN_URLLC_BASE_PRB", TOTAL_RESILIENCY_PRB))
EMBB_QUEUE_MAX = float(getattr(settings, "RES_DQN_EMBB_QUEUE_MAX", 200000.0))
URLLC_QUEUE_MAX = float(getattr(settings, "RES_DQN_URLLC_QUEUE_MAX", 200000.0))
EMBB_THR_MAX = float(getattr(settings, "RES_DQN_EMBB_THR_MAX", 1.8e8))  # bps
URLLC_THR_MAX = float(getattr(settings, "RES_DQN_URLLC_THR_MAX", 7.5e7))  # bps
URLLC_LATENCY_DEADLINE = float(getattr(settings, "RES_DQN_URLLC_LATENCY_DEADLINE", 0.001))


def build_state(obs: Dict, prev_action: Tuple[int, int]) -> np.ndarray:
    """
    obs: dict with simulator measurements
    prev_action: (r_e_prev, r_u_prev)
    """
    denom_prb = max(1.0, float(obs.get("max_dl_prb", DEFAULT_MAX_DL_PRB)))
    state = [
        _safe_div(obs.get("embb_demand_prb", 0.0), denom_prb),
        _safe_div(obs.get("urllc_demand_prb", 0.0), denom_prb),
        _safe_div(obs.get("embb_queue", 0.0), obs.get("embb_queue_max", EMBB_QUEUE_MAX)),
        _safe_div(obs.get("urllc_queue", 0.0), obs.get("urllc_queue_max", URLLC_QUEUE_MAX)),
        _safe_div(obs.get("embb_throughput", 0.0), obs.get("embb_throughput_max", EMBB_THR_MAX)),
        _safe_div(obs.get("urllc_throughput", 0.0), obs.get("urllc_throughput_max", URLLC_THR_MAX)),
        _safe_div(obs.get("urllc_latency", 0.0), obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE)),
        float(obs.get("effective_prb_ratio", 0.0)),
        _safe_div(prev_action[0], TOTAL_RESILIENCY_PRB),
        _safe_div(prev_action[1], TOTAL_RESILIENCY_PRB),
    ]
    return np.array(state, dtype=np.float32)


def latency_excess(latency: float, deadline: float) -> float:
    """Return positive slack beyond deadline (ratio-1, clamped at 0)."""
    if deadline <= 0:
        return 0.0
    return max(0.0, float(latency) / float(deadline) - 1.0)


def compute_reward(obs: Dict, action: Tuple[int, int], prev_action: Tuple[int, int], weights: Dict) -> float:
    """
    obs: simulator outputs after applying the action
    action: (r_e, r_u)
    prev_action: (r_e_prev, r_u_prev)
    weights: dict of reward weights
    """

    r_e, r_u = action

    reward_embb = weights["w_e"] * np.log(1 + obs.get("embb_throughput", 0.0))
    reward_urllc = weights["w_u"] * np.log(1 + obs.get("urllc_throughput", 0.0))

    excess = latency_excess(
        obs.get("urllc_latency", 0.0),
        obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE),
    )
    penalty_latency = weights["lambda_latency"] * excess

    penalty_reserve = weights["mu"] * (r_e + r_u) / TOTAL_RESILIENCY_PRB

    reward = reward_embb + reward_urllc - penalty_latency - penalty_reserve
    return float(reward)


# --------------------------- Replay buffer + model --------------------------- #
class _ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = []
        self.cap = capacity

    def push(self, s, a, r, ns, d):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buf))
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buf[i] for i in idx])
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


class _DQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------- Environment --------------------------- #
class ResilientDQNEnv:
    """Lightweight environment that exposes the requested state and action space.

    All KPI names and aggregates mirror simulator attributes (dl_total_prb_demand,
    dl_buffer_bytes, served_downlink_bitrate, downlink_latency, slice_dl_prb_quota).
    """

    def __init__(self, ric):
        self.ric = ric
        self.sim_engine = getattr(ric, "simulation_engine", None)
        self.decision_period = max(1, int(getattr(settings, "RES_DQN_DECISION_PERIOD", 1)))
        self.actions = ACTION_LIST
        self.prev_action: Tuple[int, int] = (0, 0)
        self._base_quota: Dict[str, Dict[str, int]] = {}
        self._snapshot_base_quotas()
        self._on_clear = None

        # Episode/catalog handling mirrors xapp_plot_dashboard_traces
        self._episodes = self._load_episode_specs()
        self._episode_idx = -1
        self._current_episode = None
        self._steps_in_episode = 0
        self._episode_step_limit = 0
        self._managed_ues: set[str] = set()
        self._trace_cache: Dict[Tuple[str, str, float], list] = {}

    def register_clear_hook(self, fn):
        self._on_clear = fn

    def reset(self) -> np.ndarray:
        self.prev_action = (0, 0)
        self._restore_base_quotas()
        self._steps_in_episode = 0

        self._episode_idx += 1
        if self._episode_idx >= len(self._episodes):
            return build_state(self._blank_obs(), self.prev_action)

        self._current_episode = self._episodes[self._episode_idx]
        self._episode_step_limit = self._calc_episode_step_limit(self._current_episode)
        self._deploy_episode(self._current_episode)

        obs = self._collect_obs()
        return build_state(obs, self.prev_action)

    def ready(self) -> bool:
        """Return True when managed UEs are attached (prevents stepping on empty sims)."""
        if not self._managed_ues or self.sim_engine is None:
            return False
        for imsi in self._managed_ues:
            ue = self.sim_engine.ue_list.get(imsi)
            if ue is None or getattr(ue, "current_cell", None) is None:
                return False
        return True

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= len(self.actions):
            action_idx = 0
        prev_act = self.prev_action
        cur_action = self.actions[action_idx]
        self.prev_action = cur_action
        self._apply_action(cur_action)
        obs = self._collect_obs()
        state = build_state(obs, prev_act)
        reward = self._compute_reward(obs, cur_action, prev_act)
        self._steps_in_episode += 1
        done = (self._steps_in_episode >= self._episode_step_limit) or self._traces_consumed()
        info = {
            "episode_id": self._current_episode.get("id") if self._current_episode else None,
            "obs": obs,
            "action": cur_action,
            "prev_action": prev_act,
        }
        return state, reward, done, info

    # ---------- Helpers ---------- #
    def _snapshot_base_quotas(self):
        if not self.ric:
            return
        for cell in self.ric.cell_list.values():
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            self._base_quota[cell.cell_id] = {SL_E: int(quotas.get(SL_E, 0)), SL_U: int(quotas.get(SL_U, 0))}

    def _restore_base_quotas(self):
        for cell in self.ric.cell_list.values():
            base = self._base_quota.get(cell.cell_id)
            if not base:
                continue
            quotas = {SL_E: int(base.get(SL_E, 0)), SL_U: int(base.get(SL_U, 0))}
            cell.slice_dl_prb_quota = quotas

    def _load_episode_specs(self) -> List[Dict]:
        """Load/normalise the episode catalog described in PRB_GYM_CONFIG_PATH."""
        path = getattr(settings, "PRB_GYM_CONFIG_PATH", "").strip()
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except Exception as exc:
            print(f"ResilientDQNEnv: failed to load episode config {path}: {exc}")
            return []
        entries = raw.get("episodes", raw) if isinstance(raw, dict) else raw
        specs = []
        for idx, entry in enumerate(entries):
            spec = self._normalize_episode(entry, idx)
            if spec:
                specs.append(spec)
        return specs

    def _normalize_episode(self, entry, idx: int) -> Optional[Dict]:
        if not isinstance(entry, dict):
            return None
        slices = entry.get("slices") or {}
        if SL_E not in slices or SL_U not in slices:
            return None
        return {
            "id": entry.get("id") or f"episode_{idx}",
            "duration_steps": int(entry.get("duration_steps", 0) or 0),
            "freeze_mobility": bool(entry.get("freeze_mobility", True)),
            "initial_prb": entry.get("initial_prb", {}),
            "slices": slices,
        }

    def _deploy_episode(self, spec: Dict):
        self._clear_managed_ues()
        self._apply_initial_prb(spec.get("initial_prb", {}))
        for sl in (SL_E, SL_U):
            cfg = spec["slices"].get(sl, {})
            for i in range(int(cfg.get("ue_count", 0))):
                imsi = f"DQNIMSI_{spec['id']}_{sl}_{i}"
                self._register_episode_ue(imsi, sl, cfg)

    def _apply_initial_prb(self, prb_map):
        if not self.sim_engine:
            return
        for cell in self.sim_engine.cell_list.values():
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            for sl, v in prb_map.items():
                quotas[sl] = int(v)
            cell.slice_dl_prb_quota = quotas
        self._snapshot_base_quotas()

    def _clear_managed_ues(self):
        if self.sim_engine:
            for imsi in list(self.sim_engine.ue_list.keys()):
                try:
                    self.sim_engine.deregister_ue(imsi)
                except Exception:
                    continue
        self._managed_ues.clear()
        if callable(self._on_clear):
            self._on_clear()

    def _register_episode_ue(self, imsi, slice_name, cfg):
        if not self.sim_engine:
            return
        max_attempts = 3
        attached = False
        for _ in range(max_attempts):
            ok = self.sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
            ue = self.sim_engine.ue_list.get(imsi)
            if ok and ue:
                attached = True
                break
            time.sleep(0.1)
        if not attached:
            print(f"[TRACE] Failed to register {imsi} after {max_attempts} attempts; skipping.")
            return
        ue = self.sim_engine.ue_list[imsi]
        if self._current_episode and self._current_episode.get("freeze_mobility", True):
            ue.speed_mps = 0
            try:
                ue.set_target(ue.position_x, ue.position_y)
            except Exception:
                pass
        trace = cfg.get("trace")
        if trace:
            self._attach_trace(ue, trace, cfg)
        self._managed_ues.add(imsi)

    def _attach_trace(self, ue, path, cfg):
        bin_s = float(cfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)))
        key = (path, cfg.get("ue_ip", ""), bin_s)
        if key not in self._trace_cache:
            try:
                self._trace_cache[key] = load_raw_packet_csv(
                    path,
                    ue_ip=cfg.get("ue_ip"),
                    bin_s=bin_s,
                    overhead_sub_bytes=int(getattr(settings, "TRACE_OVERHEAD_BYTES", 0)),
                )
            except Exception as exc:
                print(f"[TRACE] Failed to load trace {path}: {exc}")
                self._trace_cache[key] = []
        samples = self._trace_cache.get(key) or []
        if not samples:
            return
        speed = float(cfg.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        bs = getattr(ue, "current_bs", None)
        try:
            if bs and hasattr(bs, "attach_dl_trace"):
                bs.attach_dl_trace(ue.ue_imsi, samples, speed)
            else:
                ue.attach_trace(samples, speed)
        except Exception as exc:
            print(f"[TRACE] Failed to attach trace {path} to {ue.ue_imsi}: {exc}")

    def _calc_episode_step_limit(self, spec) -> int:
        """Derive episode length from config; fall back to trace duration / sim_step when unset."""
        cfg_steps = int(spec.get("duration_steps", 0) or 0)
        if cfg_steps > 0:
            return cfg_steps
        sim_dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0) or 1.0)
        est_steps = 0
        for sl in (SL_E, SL_U):
            scfg = spec.get("slices", {}).get(sl, {})
            path = scfg.get("trace")
            if not path:
                continue
            bin_s = float(scfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)))
            key = (path, scfg.get("ue_ip", ""), bin_s)
            if key not in self._trace_cache:
                try:
                    self._trace_cache[key] = load_raw_packet_csv(
                        path,
                        ue_ip=scfg.get("ue_ip"),
                        bin_s=bin_s,
                        overhead_sub_bytes=int(getattr(settings, "TRACE_OVERHEAD_BYTES", 0)),
                    )
                except Exception:
                    self._trace_cache[key] = []
            samples = self._trace_cache.get(key) or []
            if samples:
                duration = float(samples[-1][0])
                est_steps = max(est_steps, math.ceil(duration / sim_dt))
        return max(1, est_steps)

    def _apply_action(self, action: Tuple[int, int]):
        """Apply resiliency PRBs on top of the snapshot baseline quotas."""
        for cell in self.ric.cell_list.values():
            base = self._base_quota.get(cell.cell_id, {})
            quotas = {
                SL_E: int(base.get(SL_E, 0)) + int(action[0]),
                SL_U: int(base.get(SL_U, 0)) + int(action[1]),
            }
            max_prb = max(1, int(getattr(cell, "max_dl_prb", 1)))
            total = sum(quotas.values())
            if total > max_prb:
                print(f"[DQN ENV] Adjusting quotas {quotas} to fit max_prb {max_prb}")
                scale = max_prb / float(total)
                quotas[SL_E] = int(quotas[SL_E] * scale)
                quotas[SL_U] = int(quotas[SL_U] * scale)
            quotas[SL_E] = max(0, int(quotas[SL_E]))
            quotas[SL_U] = max(0, int(quotas[SL_U]))
            cell.slice_dl_prb_quota = quotas

    def _collect_obs(self) -> Dict:
        """Aggregate simulator KPIs into the obs dict expected by build_state()."""
        if not self.ric or not self.ric.cell_list:
            return self._blank_obs()
        cell = next(iter(self.ric.cell_list.values()))
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        expected_prb = max(1.0, float(sum(quotas.values()) or 1.0))
        alloc_total = float(getattr(cell, "allocated_dl_prb", 0.0) or 0.0)
        effective_ratio = _safe_div(alloc_total, expected_prb)

        embb = self._aggregate_slice(cell, SL_E)
        urllc = self._aggregate_slice(cell, SL_U)
        return {
            "embb_demand_prb": embb["demand_prb"],
            "urllc_demand_prb": urllc["demand_prb"],
            "max_dl_prb": float(getattr(cell, "max_dl_prb", DEFAULT_MAX_DL_PRB)),
            "embb_queue": embb["queue"],
            "embb_queue_max": EMBB_QUEUE_MAX,
            "urllc_queue": urllc["queue"],
            "urllc_queue_max": URLLC_QUEUE_MAX,
            "embb_throughput": embb["throughput"],
            "embb_throughput_max": EMBB_THR_MAX,
            "urllc_throughput": urllc["throughput"],
            "urllc_throughput_max": URLLC_THR_MAX,
            "urllc_latency": urllc["latency"],
            "urllc_latency_deadline": URLLC_LATENCY_DEADLINE,
            "effective_prb_ratio": effective_ratio,
        }

    def _traces_consumed(self) -> bool:
        """Return True if every managed UE has replayed its DL trace to completion."""
        cell = self._target_cell()
        if cell is None or not self._managed_ues:
            return False
        bs = getattr(cell, "base_station", None)
        replay = getattr(bs, "_dl_replay", None) if bs else None
        if not replay:
            return False
        for imsi in self._managed_ues:
            state = replay.get(imsi)
            if not state:
                continue
            samples = state.get("samples") or []
            idx = int(state.get("idx", 0))
            if samples and idx < len(samples):
                return False
        return True

    def _target_cell(self):
        return next(iter(self.sim_engine.cell_list.values())) if self.sim_engine else None

    def _aggregate_slice(self, cell, slice_name: str) -> Dict[str, float]:
        demand_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        demand = 0.0
        queue = 0.0
        throughput = 0.0
        latencies = []
        for ue in getattr(cell, "connected_ue_list", {}).values():
            if getattr(ue, "slice_type", None) != slice_name:
                continue
            imsi = getattr(ue, "ue_imsi", None)
            if imsi:
                demand += float(demand_map.get(imsi, 0.0) or 0.0)
            throughput += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0)
            queue += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            latencies.append(float(getattr(ue, "downlink_latency", 0.0) or 0.0))
        latency = float(np.mean(latencies)) if latencies else 0.0
        return {
            "demand_prb": demand,
            "queue": queue,
            "throughput": throughput,
            "latency": latency,
        }

    def _compute_reward(self, obs: Dict, action: Tuple[int, int], prev_action: Tuple[int, int]) -> float:
        weights = {
            "w_e": float(getattr(settings, "RES_DQN_W_E", 1.0)),
            "w_u": float(getattr(settings, "RES_DQN_W_U", 1.5)),
            "lambda_latency": float(getattr(settings, "RES_DQN_LAMBDA_LAT", 2.0)),
            "mu": float(getattr(settings, "RES_DQN_MU_RESERVE", 0.5)),
        }
        return compute_reward(obs, action, prev_action, weights)

    def _blank_obs(self) -> Dict:
        return {
            "embb_demand_prb": 0.0,
            "urllc_demand_prb": 0.0,
            "embb_queue": 0.0,
            "embb_queue_max": EMBB_QUEUE_MAX,
            "urllc_queue": 0.0,
            "urllc_queue_max": URLLC_QUEUE_MAX,
            "embb_throughput": 0.0,
            "embb_throughput_max": EMBB_THR_MAX,
            "urllc_throughput": 0.0,
            "urllc_throughput_max": URLLC_THR_MAX,
            "urllc_latency": 0.0,
            "urllc_latency_deadline": URLLC_LATENCY_DEADLINE,
            "effective_prb_ratio": 0.0,
        }


# --------------------------- xApp --------------------------- #
class xAppResilientDQNAllocator(xAppBase):
    """DQN-based resiliency allocator with the requested action/state space."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = bool(getattr(settings, "RES_DQN_ENABLE", False))
        self.train = bool(getattr(settings, "RES_DQN_TRAIN", True))
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.gamma = float(getattr(settings, "RES_DQN_GAMMA", 0.99))
        self.lr = float(getattr(settings, "RES_DQN_LR", 1e-3))
        self.batch = int(getattr(settings, "RES_DQN_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "RES_DQN_BUFFER", 100000))
        self.target_update = int(getattr(settings, "RES_DQN_TARGET_UPDATE", 2000))
        self.eps_start = float(getattr(settings, "RES_DQN_EPS_START", 1.0))
        self.eps_end = float(getattr(settings, "RES_DQN_EPS_END", 0.05))
        self.eps_decay = int(getattr(settings, "RES_DQN_EPS_DECAY", 20000))
        self._eps = self.eps_start
        self._global_step = 0
        self._decision_period = max(1, int(getattr(settings, "RES_DQN_DECISION_PERIOD", 1)))
        self._env: Optional[ResilientDQNEnv] = None
        self._policy: Optional[_DQN] = None
        self._target: Optional[_DQN] = None
        self._optimizer = None
        self._replay: Optional[_ReplayBuffer] = None
        self._last_state: Optional[np.ndarray] = None
        self._episode_done = False
        self.dash_enable = bool(getattr(settings, "RES_DQN_DASH_ENABLE", False))
        self.dash_port = int(getattr(settings, "RES_DQN_DASH_PORT", 5060))
        self._dash_app = None
        self._dash_thread = None
        self._lock = threading.Lock()
        # History buffers for quick plotting
        self._t = deque(maxlen=400)
        self._hist_reward = deque(maxlen=400)
        self._hist_action = deque(maxlen=400)
        self._hist_embb_thr = deque(maxlen=400)
        self._hist_urllc_thr = deque(maxlen=400)
        self._hist_latency = deque(maxlen=400)
        self._hist_deadline = deque(maxlen=400)
        self._hist_queue_e = deque(maxlen=400)
        self._hist_queue_u = deque(maxlen=400)
        self._hist_reserve = deque(maxlen=400)
        self._hist_prb_ratio = deque(maxlen=400)

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        if not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False
            return
        self._env = ResilientDQNEnv(self.ric)
        self._init_agent()
        self._last_state = self._env.reset()
        if self.dash_enable:
            self._start_dashboard()
        print(
            f"{self.xapp_id}: enabled train={self.train} actions={NUM_ACTIONS} state_dim={len(self._last_state)}"
        )

    def step(self):
        if not self.enabled or self._env is None:
            return
        self._global_step += 1
        # Only act every decision_period steps
        if self._global_step % self._decision_period != 0:
            return

        if self._last_state is None or self._episode_done:
            self._last_state = self._env.reset()
            self._episode_done = False
            return

        if self.train:
            frac = max(0.0, (self.eps_decay - self._global_step) / float(max(1, self.eps_decay)))
            self._eps = self.eps_end + (self.eps_start - self.eps_end) * frac
        else:
            self._eps = 0.0

        if hasattr(self._env, "ready") and not self._env.ready():
            return

        action_idx = self._select_action(self._last_state)
        next_state, reward, done, info = self._env.step(action_idx)
        self._record_metrics(reward, info)
        if self.train and self._replay is not None:
            self._replay.push(self._last_state, action_idx, reward, next_state, float(done))
            self._learn()
        self._last_state = next_state
        self._episode_done = bool(done)

    # ---------- Telemetry / dashboard ---------- #
    def _record_metrics(self, reward: float, info: Optional[Dict]):
        if info is None:
            return
        obs = info.get("obs") if isinstance(info, dict) else {}
        action = info.get("action") if isinstance(info, dict) else (0, 0)
        with self._lock:
            t = self._global_step
            self._t.append(t)
            self._hist_reward.append(float(reward))
            self._hist_action.append(float(sum(action) if action else 0))
            self._hist_embb_thr.append(float(obs.get("embb_throughput", 0.0) or 0.0))
            self._hist_urllc_thr.append(float(obs.get("urllc_throughput", 0.0) or 0.0))
            self._hist_latency.append(float(obs.get("urllc_latency", 0.0) or 0.0))
            self._hist_deadline.append(float(obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE)))
            self._hist_queue_e.append(float(obs.get("embb_queue", 0.0) or 0.0))
            self._hist_queue_u.append(float(obs.get("urllc_queue", 0.0) or 0.0))
            self._hist_reserve.append(float(sum(action) if action else 0) / max(1.0, TOTAL_RESILIENCY_PRB))
            self._hist_prb_ratio.append(float(obs.get("effective_prb_ratio", 0.0) or 0.0))

    def _start_dashboard(self):
        if not self.dash_enable or not DASH_AVAILABLE:
            if self.dash_enable and not DASH_AVAILABLE:
                print(f"{self.xapp_id}: dash not available; dashboard disabled.")
            return
        if self._dash_thread and self._dash_thread.is_alive():
            return

        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.H3("Resilient DQN Dashboard"),
                dcc.Graph(id="thr-graph"),
                dcc.Graph(id="lat-graph"),
                dcc.Graph(id="queue-graph"),
                dcc.Graph(id="reward-graph"),
                dcc.Interval(id="tick", interval=1000, n_intervals=0),
            ]
        )

        @app.callback(
            Output("thr-graph", "figure"),
            Output("lat-graph", "figure"),
            Output("queue-graph", "figure"),
            Output("reward-graph", "figure"),
            Input("tick", "n_intervals"),
        )
        def _update(_):
            with self._lock:
                t = list(self._t)
                thr_e = list(self._hist_embb_thr)
                thr_u = list(self._hist_urllc_thr)
                lat = list(self._hist_latency)
                deadline = list(self._hist_deadline)
                q_e = list(self._hist_queue_e)
                q_u = list(self._hist_queue_u)
                reward = list(self._hist_reward)
                reserve = list(self._hist_reserve)
                prb_ratio = list(self._hist_prb_ratio)

            thr_fig = go.Figure()
            thr_fig.add_trace(go.Scatter(x=t, y=thr_e, mode="lines", name="eMBB throughput (bps)"))
            thr_fig.add_trace(go.Scatter(x=t, y=thr_u, mode="lines", name="URLLC throughput (bps)"))
            thr_fig.update_layout(title="Throughput", xaxis_title="step")

            lat_fig = go.Figure()
            lat_fig.add_trace(go.Scatter(x=t, y=lat, mode="lines", name="URLLC latency (s)"))
            lat_fig.add_trace(go.Scatter(x=t, y=deadline, mode="lines", name="deadline", line=dict(dash="dash")))
            lat_fig.update_layout(title="Latency", xaxis_title="step")

            q_fig = go.Figure()
            q_fig.add_trace(go.Scatter(x=t, y=q_e, mode="lines", name="eMBB queue (bytes)"))
            q_fig.add_trace(go.Scatter(x=t, y=q_u, mode="lines", name="URLLC queue (bytes)"))
            q_fig.update_layout(title="Queue backlog", xaxis_title="step")

            r_fig = go.Figure()
            r_fig.add_trace(go.Scatter(x=t, y=reward, mode="lines", name="reward"))
            r_fig.add_trace(
                go.Scatter(x=t, y=reserve, mode="lines", name="reserve fraction", yaxis="y2")
            )
            r_fig.add_trace(
                go.Scatter(x=t, y=prb_ratio, mode="lines", name="effective PRB ratio", yaxis="y2")
            )
            r_fig.update_layout(
                title="Reward / Reserve",
                xaxis_title="step",
                yaxis=dict(title="reward"),
                yaxis2=dict(title="fractions", overlaying="y", side="right"),
            )
            return thr_fig, lat_fig, q_fig, r_fig

        def _run():
            app.run(host="0.0.0.0", port=self.dash_port, debug=False, use_reloader=False)

        self._dash_app = app
        self._dash_thread = threading.Thread(target=_run, daemon=True)
        self._dash_thread.start()
        print(f"{self.xapp_id}: dashboard running on port {self.dash_port}")

    # ---------- Agent helpers ---------- #
    def _init_agent(self):
        in_dim = len(build_state(self._env._blank_obs(), (0, 0)))  # type: ignore[arg-type]
        out_dim = NUM_ACTIONS
        self._policy = _DQN(in_dim, out_dim).to(self.device)
        self._target = _DQN(in_dim, out_dim).to(self.device)
        self._target.load_state_dict(self._policy.state_dict())
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self.lr)
        self._replay = _ReplayBuffer(capacity=self.buffer_cap)

    def _select_action(self, state: np.ndarray) -> int:
        if self.train and random.random() < self._eps:
            return int(random.randrange(NUM_ACTIONS))
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self._policy(s)
            idx = int(torch.argmax(q, dim=1).item())
        return idx

    def _learn(self):
        if self._replay is None or len(self._replay) < self.batch:
            return
        states, actions, rewards, next_states, dones = self._replay.sample(self.batch)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_pred = self._policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._target(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(q_pred, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        if self._global_step % self.target_update == 0:
            self._target.load_state_dict(self._policy.state_dict())
