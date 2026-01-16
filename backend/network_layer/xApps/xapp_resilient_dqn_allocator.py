"""Resilient PRB allocator using a DQN agent with a fixed action/state space.

Action space (resiliency PRB split across eMBB/URLLC):
RESILIENCY_LEVELS = [-9, -3, -1, 0, 1, 3, 9]
TOTAL_RESILIENCY_PRB = 18
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import os
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

import settings
from .xapp_base import xAppBase
from utils import load_raw_packet_csv, estimate_throughput

SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None
    SummaryWriter = None

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


def _as_bool(val, default: bool = False) -> bool:
    """Robust bool parser for env/config inputs."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    try:
        return bool(int(val))
    except Exception:
        return default


# --------------------------- Action/state helpers --------------------------- #
RESILIENCY_LEVELS = getattr(settings, "RES_DQN_LEVELS", None) or [-9, -3, -1, 0, 1, 3, 9]
# Prefer jammer-reserved PRBs as the resiliency pool; fall back to explicit config.
_reserve_from_jammer = int(getattr(settings, "JAM_RESERVED_PRBS", 0) or 0)
_reserve_from_cfg = int(getattr(settings, "RES_DQN_TOTAL_RESILIENCY_PRB", 18))
TOTAL_RESILIENCY_PRB = max(_reserve_from_jammer, _reserve_from_cfg)
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
EMBB_THR_MAX = float(getattr(settings, "RES_DQN_EMBB_THR_MAX", 9.6e7))  # bps
URLLC_THR_MAX = float(getattr(settings, "RES_DQN_URLLC_THR_MAX", 4.16e7))  # bps
URLLC_LATENCY_DEADLINE = float(getattr(settings, "RES_DQN_URLLC_LATENCY_DEADLINE", 0.001))


def build_state(obs: Dict, prev_action: Tuple[int, int]) -> np.ndarray:
    """
    obs: dict with simulator measurements
    prev_action: (r_e_prev, r_u_prev)
    """
    denom_prb = max(1.0, float(obs.get("max_dl_prb", DEFAULT_MAX_DL_PRB)))
    # lat_ratio = _safe_div(obs.get("urllc_latency", 0.0), obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE))
    # lat_log = min(5.0, math.log1p(max(0.0, lat_ratio)))  # cap extreme spikes
    # Normalized latency excess
    latency = float(obs.get("urllc_latency", 0.0))
    deadline = float(obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE))
    excess = max(0.0, latency / deadline - 1.0)
    lat_excess_norm = excess / (1.0 + excess)
    
    state = [
        min(1.0, _safe_div(obs.get("embb_demand_prb", 0.0), denom_prb)),
        min(1.0, _safe_div(obs.get("urllc_demand_prb", 0.0), denom_prb)),
        min(1.0, _safe_div(obs.get("embb_prb", 0.0), denom_prb)),
        min(1.0, _safe_div(obs.get("urllc_prb", 0.0), denom_prb)),
        min(1.0, _safe_div(obs.get("embb_queue", 0.0), obs.get("embb_queue_max", EMBB_QUEUE_MAX))),
        min(1.0, _safe_div(obs.get("urllc_queue", 0.0), obs.get("urllc_queue_max", URLLC_QUEUE_MAX))),
        _safe_div(obs.get("embb_throughput", 0.0), obs.get("embb_throughput_max", EMBB_THR_MAX)),
        _safe_div(obs.get("urllc_throughput", 0.0), obs.get("urllc_throughput_max", URLLC_THR_MAX)),
        #lat_log,
        lat_excess_norm,  # new version
        float(obs.get("effective_prb_ratio", 0.0)),
        min(1.0, _safe_div(obs.get("reserve_used_prb", 0.0), TOTAL_RESILIENCY_PRB)),
        _safe_div(prev_action[0], TOTAL_RESILIENCY_PRB),
        _safe_div(prev_action[1], TOTAL_RESILIENCY_PRB),
    ]
    return np.array(state, dtype=np.float32)


def latency_excess(latency: float, deadline: float) -> float:
    """Return positive slack beyond deadline (ratio-1, clamped at 0)."""
    if deadline <= 0:
        return 0.0
    return max(0.0, float(latency) / float(deadline) - 1.0)


def compute_reward(
    obs: Dict, action: Tuple[int, int], prev_action: Tuple[int, int], weights: Dict, return_parts: bool = False
):
    """
    obs: simulator outputs after applying the action
    action: (r_e, r_u)
    prev_action: (r_e_prev, r_u_prev)
    weights: dict of reward weights
    """

    #r_e, r_u = action
    base_ach_embb = float(obs.get("embb_base_achievable_throughput", 0.0) or 0.0)
    embb_max_throughput = float(obs.get("embb_throughput_max", EMBB_THR_MAX))
    target = min(base_ach_embb, embb_max_throughput)
    reward_embb = max(0.0, weights["w_e"] *(1/1e6) * (target - obs.get("embb_throughput", 0.0)))
    penalty_latency = max(0.0, weights["w_u"] * (1000)* (obs.get("urllc_latency", 0.0) - URLLC_LATENCY_DEADLINE))
    penalty_reserve = weights["mu"] * _safe_div(obs.get("reserve_used_prb", 0.0), TOTAL_RESILIENCY_PRB)
    
    #print("reward_embb:", reward_embb, " penalty_latency:", penalty_latency, " penalty_reserve:", penalty_reserve)


    r0 = 0  # Base reward to keep things positive

    reward = r0 - reward_embb - penalty_latency - penalty_reserve
    #print("Total reward:", reward)
    
    if return_parts:
        parts = {
            "reward_base": float(r0),
            "reward_embb": float(reward_embb),
            "penalty_latency": float(penalty_latency),
            "penalty_reserve": float(penalty_reserve),
        }
        return float(reward), parts
    return float(reward)


# --------------------------- Replay buffer + model --------------------------- #
class _ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = []
        self.cap = capacity

    def push(self, s, a, r, ns, d, next_mask=None):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        if next_mask is None:
            next_mask = [True] * NUM_ACTIONS
        self.buf.append((s, a, r, ns, d, next_mask))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buf))
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        states, actions, rewards, next_states, dones, next_masks = zip(*[self.buf[i] for i in idx])
        # Collapse to contiguous numpy arrays before torch conversion (avoids slow list-of-arrays warning)
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        next_masks = np.asarray(next_masks, dtype=np.bool_)
        return (
            torch.from_numpy(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(next_states),
            torch.tensor(dones, dtype=torch.float32),
            torch.from_numpy(next_masks),
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

    def __init__(self, ric, spawn_seed: Optional[int] = None):
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
        self._spawn_seed_base = spawn_seed

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
        reward, reward_parts = self._compute_reward(obs, cur_action, prev_act)
        self._steps_in_episode += 1
        done = (self._steps_in_episode >= self._episode_step_limit) or self._traces_consumed()
        info = {
            "episode_id": self._current_episode.get("id") if self._current_episode else None,
            "obs": obs,
            "action": cur_action,
            "prev_action": prev_act,
            "action_idx": action_idx,
            "reward_parts": reward_parts,
        }
        return state, reward, done, info

    def _action_mask(self) -> List[bool]:
        """Mask actions that overflow current (jammed) PRB capacity."""
        cell = self._target_cell()
        if cell is None:
            return [True] * len(self.actions)
        # Jammer reports effective PRBs excluding reserved+jam; add reserved back to get usable capacity for reserve pool.
        eff_max = float(getattr(cell, "_effective_max_dl_prb", getattr(cell, "max_dl_prb", 0)) or 0.0)
        max_prb_phys = eff_max + float(_reserve_from_jammer)
        base_dl_prb_quota = dict(getattr(cell, "base_dl_prb_quota", {}) or {})
        base_e = float(base_dl_prb_quota.get(SL_E, 0) or 0.0)
        base_u = float(base_dl_prb_quota.get(SL_U, 0) or 0.0)
        if base_e == 0 and base_u == 0:
            base = self._base_quota.get(cell.cell_id, {})
            base_e = float(base.get(SL_E, 0) or 0.0)
            base_u = float(base.get(SL_U, 0) or 0.0)
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        cur_e = float(quotas.get(SL_E, 0) or 0)
        cur_u = float(quotas.get(SL_U, 0) or 0)
        base_total = base_e + base_u
        cur_total = cur_e + cur_u
        max_prb = min(max_prb_phys, base_total + TOTAL_RESILIENCY_PRB)
        pool_used = max(0.0, min(TOTAL_RESILIENCY_PRB, cur_total - base_total))

        mask = []
        for de, du in self.actions:
            new_e = cur_e + de
            new_u = cur_u + du
            if new_e < base_e or new_u < base_u:
                #print("Action would reduce below base quota:", de, du)
                #print(" base_e:", base_e, " base_u:", base_u, " cur_e:", cur_e, " cur_u:", cur_u,"eff_max:", eff_max)
                mask.append(False)
                continue
            if new_e + new_u > max_prb + 1e-9:
                mask.append(False)
                continue
            # Do not return more to the pool than currently borrowed
            pool_return = max(0.0, -float(de)) + max(0.0, -float(du))
            if pool_return > pool_used + 1e-9:
                mask.append(False)
                continue
            mask.append(True)
        return mask

    # ---------- Helpers ---------- #
    def _snapshot_base_quotas(self):
        if not self.ric:
            return
        for cell in self.ric.cell_list.values():
            quotas = dict(getattr(cell, "base_dl_prb_quota", getattr(cell, "slice_dl_prb_quota", {})) or {})
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
                self._register_episode_ue(imsi, sl, cfg, ue_idx=i)

    def _apply_initial_prb(self, prb_map):
        if not self.sim_engine:
            return
        for cell in self.sim_engine.cell_list.values():
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            for sl, v in prb_map.items():
                quotas[sl] = int(v)
            cell.slice_dl_prb_quota = quotas
            # Reset jammer/base baseline so new episodes start from fresh quotas.
            cell.base_dl_prb_quota = dict(quotas)
            cell._jam_base_quota = dict(quotas)
            cell._jam_was_active = False
            cell._effective_max_dl_prb = getattr(cell, "max_dl_prb", cell._effective_max_dl_prb)
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

    def _register_episode_ue(self, imsi, slice_name, cfg, ue_idx: int = 0):
        if not self.sim_engine:
            return
        max_attempts = 3
        attached = False
        seed = None
        if self._spawn_seed_base is not None and self._current_episode is not None:
            sl_offset = 0 if slice_name == SL_E else 1
            seed = int(self._spawn_seed_base) + (self._episode_idx + 1) * 100000 + sl_offset * 1000 + int(ue_idx)
            seed = int(seed % (2**32))
        for _ in range(max_attempts):
            if seed is None:
                ok = self.sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
            else:
                rng_state = random.getstate()
                random.seed(seed)
                try:
                    ok = self.sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
                finally:
                    random.setstate(rng_state)
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
            base_dl_prb_quota = dict(getattr(cell, "base_dl_prb_quota", {}) or {})
            base_e = float(base_dl_prb_quota.get(SL_E, 0) or 0.0)
            base_u = float(base_dl_prb_quota.get(SL_U, 0) or 0.0)
            if base_e == 0 and base_u == 0:
                base = self._base_quota.get(cell.cell_id, {})
                base_e = float(base.get(SL_E, 0) or 0.0)
                base_u = float(base.get(SL_U, 0) or 0.0)
            base_total = base_e + base_u
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            cur_e = float(quotas.get(SL_E, 0) or 0)
            cur_u = float(quotas.get(SL_U, 0) or 0)
            cur_total = cur_e + cur_u
            eff_max = float(getattr(cell, "_effective_max_dl_prb", getattr(cell, "max_dl_prb", 0)) or 0.0)
            max_prb_phys = eff_max + float(_reserve_from_jammer)
            # Reserve pool is fixed; jamming affects baseline but not available reserve.
            capacity = min(max_prb_phys, base_total + TOTAL_RESILIENCY_PRB)

            de, du = action
            # Apply deltas relative to current quotas but never below baseline
            # new_e = cur_e + float(de)
            # new_u = cur_u + float(du)
            #print(f"Applying action to cell {cell.cell_id}: eMBB PRB={quotas[SL_E]}, URLLC PRB={quotas[SL_U]}")

            new_e = max(base_e, cur_e + float(de))
            new_u = max(base_u, cur_u + float(du))
            #print(base_e, base_u)
            #print(new_e, new_u)
            # print(" base_e:", base_e, " base_u:", base_u, " cur_e:", cur_e, " cur_u:", cur_u)
            # if base_e > cur_e or base_u > cur_u:
            #     print("Warning: current quota below base quota; adjusting upwards.")
                #print(" base_e:", base_e, " base_u:", base_u, " cur_e:", cur_e, " cur_u:", cur_u)
                # new_e = base_e
                # new_u = 
                
                
            # delta_e = new_e - cur_e
            # delta_u = new_u - cur_u
            # delta_sum = delta_e + delta_u
            # avail_extra = max(0.0, capacity - cur_total)
            # if delta_sum > avail_extra and delta_sum > 0:
            #     print("Warning: requested quota increase exceeds available capacity; scaling down.")
            #     scale = avail_extra / delta_sum
            #     delta_e *= scale
            #     delta_u *= scale
            # new_e = cur_e + delta_e
            # new_u = cur_u + delta_u
            # total = new_e + new_u
            # if total > capacity and total > 0:
            #     scale = capacity / total
            #     new_e *= scale
            #     new_u *= scale
            quotas = {SL_E: int(round(new_e)), SL_U: int(round(new_u))}
            #print(f"Applying action to cell {cell.cell_id}: eMBB PRB={quotas[SL_E]}, URLLC PRB={quotas[SL_U]}")
            #print(f"  (Requested deltas: eMBB={de}, URLLC={du})")
            cell.slice_dl_prb_quota = quotas

    def _collect_obs(self) -> Dict:
        """Aggregate simulator KPIs into the obs dict expected by build_state()."""
        if not self.ric or not self.ric.cell_list:
            return self._blank_obs()
        cell = next(iter(self.ric.cell_list.values()))
        max_dl_prb = float(getattr(cell, "max_dl_prb", DEFAULT_MAX_DL_PRB))
        effective_max_prb = float(getattr(cell, "_effective_max_dl_prb", max_dl_prb) or max_dl_prb)
        phys_max = max_dl_prb - getattr(settings, "JAM_RESERVED_PRBS", 0)
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        base = dict(getattr(cell, "base_dl_prb_quota", {}) or {})
        base_total = float(sum(base.values()) or 0.0)
        reserve_used = max(0.0, float(sum(quotas.values()) or 0.0) - base_total)
        #print("Reserve used PRBs:", reserve_used)
        effective_prb_ratio = min(1.0, _safe_div(effective_max_prb, phys_max))

        embb = self._aggregate_slice(cell, SL_E, effective_prb_ratio)
        urllc = self._aggregate_slice(cell, SL_U, effective_prb_ratio)
        return {
            "embb_demand_prb": embb["demand_prb"],
            "urllc_demand_prb": urllc["demand_prb"],
            "embb_prb": float(quotas.get(SL_E, 0.0) or 0.0),
            "urllc_prb": float(quotas.get(SL_U, 0.0) or 0.0),
            "embb_queue": embb["queue"],
            "embb_queue_max": EMBB_QUEUE_MAX,
            "urllc_queue": urllc["queue"],
            "urllc_queue_max": URLLC_QUEUE_MAX,
            "embb_throughput": embb["throughput"],
            "embb_throughput_max": EMBB_THR_MAX,
            "embb_base_achievable_throughput": embb.get("base_achievable_throughput", 0.0),
            "urllc_throughput": urllc["throughput"],
            "urllc_throughput_max": URLLC_THR_MAX,
            "urllc_base_achievable_throughput": urllc.get("base_achievable_throughput", 0.0),
            "urllc_latency": urllc["latency"],
            "urllc_latency_deadline": URLLC_LATENCY_DEADLINE,
            "effective_prb_ratio": effective_prb_ratio,
            "effective_max_prb": effective_max_prb,
            "reserve_used_prb": reserve_used,
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

    def _aggregate_slice(self, cell, slice_name: str, effective_prb_ratio: float) -> Dict[str, float]:
        demand_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        demand = 0.0
        queue = 0.0
        throughput = 0.0
        #achievable_throughput = 0.0
        base_achievable_throughput = 0.0
        latencies = []
        ues = [
            ue
            for ue in getattr(cell, "connected_ue_list", {}).values()
            if getattr(ue, "slice_type", None) == slice_name
        ]
        ue_count = max(1, len(ues))  # avoid divide-by-zero below
        quota_map = dict(getattr(cell, "base_dl_prb_quota", {}) or self._base_quota.get(cell.cell_id, {}) or {})
        base_quota = float(quota_map.get(slice_name, 0.0) or 0.0)

        base_prb_per_ue = base_quota / ue_count
        for ue in ues:
            imsi = getattr(ue, "ue_imsi", None)
            if imsi:
                demand += float(demand_map.get(imsi, 0.0) or 0.0)
            throughput += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0)
            queue += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            latencies.append(float(getattr(ue, "downlink_latency", 0.0) or 0.0))
            mcs = getattr(ue, "downlink_mcs_data", None) or {}
            mod_order = float(mcs.get("modulation_order", 0.0) or 0.0)
            code_rate = float(mcs.get("target_code_rate", 0.0) or 0.0)
            if mod_order > 0 and code_rate > 0 and base_prb_per_ue > 0:
                base_achievable_throughput += float(estimate_throughput(mod_order, code_rate, base_prb_per_ue))
        latency = float(np.mean(latencies)) if latencies else 0.0
        return {
            "demand_prb": demand,
            "queue": queue,
            "throughput": throughput,
            "base_achievable_throughput": base_achievable_throughput,
            "latency": latency,
        }
        
    def _compute_reward(
        self, obs: Dict, action: Tuple[int, int], prev_action: Tuple[int, int]
    ) -> Tuple[float, Dict[str, float]]:
        weights = {
            "w_e": float(getattr(settings, "RES_DQN_W_E", 1.0)),
            "w_u": float(getattr(settings, "RES_DQN_W_U", 1.0)),
            "lambda_latency": float(getattr(settings, "RES_DQN_LAMBDA_LAT", 2.0)),
            "mu": float(getattr(settings, "RES_DQN_MU_RESERVE", 0.5)),
        }
        return compute_reward(obs, action, prev_action, weights, return_parts=True)

    def _blank_obs(self) -> Dict:
        return {
            "embb_demand_prb": 0.0,
            "urllc_demand_prb": 0.0,
            "embb_prb": 0.0,
            "urllc_prb": 0.0,
            "embb_queue": 0.0,
            "embb_queue_max": EMBB_QUEUE_MAX,
            "urllc_queue": 0.0,
            "urllc_queue_max": URLLC_QUEUE_MAX,
            "embb_throughput": 0.0,
            "embb_throughput_max": EMBB_THR_MAX,
            "embb_base_achievable_throughput": 0.0,
            "urllc_throughput": 0.0,
            "urllc_throughput_max": URLLC_THR_MAX,
            "urllc_base_achievable_throughput": 0.0,
            "urllc_latency": 0.0,
            "urllc_latency_deadline": URLLC_LATENCY_DEADLINE,
            "effective_prb_ratio": 0.0,
            "reserve_used_prb": 0.0,
            "max_dl_prb": DEFAULT_MAX_DL_PRB,
            "jammed_prb": 0.0,
            "jammed_ratio": 0.0,
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
        self.lr = float(getattr(settings, "RES_DQN_LR", 1e-4))
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
        self._decision_steps = 0  # counts RL decisions (not simulator ticks)
        self._last_state: Optional[np.ndarray] = None
        self._episode_done = False
        self._episode_return = 0.0
        self._episode_len = 0
        self._episode_counter = 0
        self._episode_reward_parts = {
            "reward_base": 0.0,
            "reward_embb": 0.0,
            "penalty_latency": 0.0,
            "penalty_reserve": 0.0,
        }
        self._run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tb_enable = bool(getattr(settings, "RES_DQN_TB_ENABLE", False))
        self.tb_dir = (getattr(settings, "RES_DQN_TB_DIR", "") or "").strip()
        self.tb_smooth_win = int(getattr(settings, "RES_DQN_TB_SMOOTH_WIN", 200) or 0)
        self._tb_writer = None
        self.model_dir = (getattr(settings, "RES_DQN_MODEL_DIR", "") or "").strip()
        self.save_interval = int(getattr(settings, "RES_DQN_SAVE_INTERVAL", 0) or 0)
        self._last_save_step = -1
        self.dash_enable = bool(getattr(settings, "RES_DQN_DASH_ENABLE", False))
        self.dash_port = int(getattr(settings, "RES_DQN_DASH_PORT", 5060))
        self._dash_app = None
        self._dash_thread = None
        self._lock = threading.Lock()
        self._last_info = {}
        # History buffers for quick plotting
        self._t = deque(maxlen=400)
        self._hist_reward = deque(maxlen=400)
        self._hist_reward_base = deque(maxlen=400)
        self._hist_reward_embb = deque(maxlen=400)
        self._hist_penalty_latency = deque(maxlen=400)
        self._hist_penalty_reserve = deque(maxlen=400)
        self._hist_action = deque(maxlen=400)
        self._hist_embb_thr = deque(maxlen=400)
        self._hist_urllc_thr = deque(maxlen=400)
        self._hist_latency = deque(maxlen=400)
        self._hist_deadline = deque(maxlen=400)
        self._hist_queue_e = deque(maxlen=400)
        self._hist_queue_u = deque(maxlen=400)
        self._hist_reserve = deque(maxlen=400)
        self._hist_reserve_used = deque(maxlen=400)
        self._hist_prb_ratio = deque(maxlen=400)
        self._hist_action_idx = deque(maxlen=400)
        self._hist_episode_return = deque(maxlen=200)
        self._hist_episode_idx = deque(maxlen=200)
        # Per-slice and per-UE histories (lightweight, capped)
        self._slice_prb = defaultdict(lambda: deque(maxlen=400))
        self._slice_tx = defaultdict(lambda: deque(maxlen=400))
        self._slice_sat = defaultdict(lambda: deque(maxlen=400))
        self._slice_sat_raw = defaultdict(lambda: deque(maxlen=400))
        self._ue_tx = defaultdict(lambda: deque(maxlen=400))
        self._ue_prb_demand = defaultdict(lambda: deque(maxlen=400))
        self._ue_prb_sat = defaultdict(lambda: deque(maxlen=400))
        self._ue_prb_sat_raw = defaultdict(lambda: deque(maxlen=400))
        self._ue_sinr = defaultdict(lambda: deque(maxlen=400))
        self._ue_cqi = defaultdict(lambda: deque(maxlen=400))
        self._cell_max_prb = deque(maxlen=400)
        self._cell_eff_prb = deque(maxlen=400)
        # Time scaling
        self._sim_dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0))

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        if not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False
            return
        self._env = ResilientDQNEnv(self.ric)
        self._sim_dt = self._detect_sim_dt()
        # Clear plots when env wipes managed UEs
        self._env.register_clear_hook(self._clear_plot_caches)
        self._init_agent()
        self._last_state = self._env.reset()
        if self.dash_enable:
            self._start_dashboard()
        self._init_tb()
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

        # Only count epsilon decay on actual RL decisions (not sim ticks)
        self._decision_steps += 1

        if self._last_state is None or self._episode_done:
            self._reset_episode_trackers()
            self._last_state = self._env.reset()
            self._episode_done = False
            return

        if self.train:
            # Decay based on decision steps so eps only changes when we act
            decisions_elapsed = max(0, self._decision_steps - 1)
            frac = max(0.0, (self.eps_decay - decisions_elapsed) / float(max(1, self.eps_decay)))
            self._eps = self.eps_end + (self.eps_start - self.eps_end) * frac
        else:
            self._eps = 0.0

        if hasattr(self._env, "ready") and not self._env.ready():
            return

        action_idx = self._select_action(self._last_state)
        next_state, reward, done, info = self._env.step(action_idx)
        self._episode_return += float(reward)
        self._episode_len += 1
        parts = info.get("reward_parts", {}) if isinstance(info, dict) else {}
        for k in self._episode_reward_parts:
            self._episode_reward_parts[k] += float(parts.get(k, 0.0))
        self._record_metrics(reward, info)
        if self.train and self._replay is not None:
            next_mask = getattr(self._env, "_action_mask", lambda: [True] * NUM_ACTIONS)()
            self._replay.push(self._last_state, action_idx, reward, next_state, float(done), next_mask)
            self._learn()
        self._tb_log(self._last_state, action_idx, reward, info, done=bool(done))
        self._last_state = next_state
        self._episode_done = bool(done)
        if done:
            with self._lock:
                self._hist_episode_idx.append(self._episode_counter)
                self._hist_episode_return.append(self._episode_return)
            self._episode_counter += 1
            self._reset_episode_trackers()

    # ---------- Telemetry / dashboard ---------- #
    def _record_metrics(self, reward: float, info: Optional[Dict]):
        if info is None:
            return
        obs = info.get("obs") if isinstance(info, dict) else {}
        action = info.get("action") if isinstance(info, dict) else (0, 0)
        action_idx = info.get("action_idx") if isinstance(info, dict) else None
        with self._lock:
            t = self._global_step * self._sim_dt
            self._last_info = info or {}
            self._t.append(t)
            self._hist_reward.append(float(reward))
            parts = info.get("reward_parts", {}) if isinstance(info, dict) else {}
            self._hist_reward_base.append(float(parts.get("reward_base", 0.0)))
            self._hist_reward_embb.append(float(parts.get("reward_embb", 0.0)))
            self._hist_penalty_latency.append(float(parts.get("penalty_latency", 0.0)))
            self._hist_penalty_reserve.append(float(parts.get("penalty_reserve", 0.0)))
            self._hist_action.append(float(sum(action) if action else 0))
            if action_idx is not None:
                self._hist_action_idx.append(int(action_idx))
            self._hist_embb_thr.append(float(obs.get("embb_throughput", 0.0) or 0.0))
            self._hist_urllc_thr.append(float(obs.get("urllc_throughput", 0.0) or 0.0))
            self._hist_latency.append(float(obs.get("urllc_latency", 0.0) or 0.0))
            self._hist_deadline.append(float(obs.get("urllc_latency_deadline", URLLC_LATENCY_DEADLINE)))
            self._hist_queue_e.append(float(obs.get("embb_queue", 0.0) or 0.0))
            self._hist_queue_u.append(float(obs.get("urllc_queue", 0.0) or 0.0))
            self._hist_reserve.append(float(sum(action) if action else 0) / max(1.0, TOTAL_RESILIENCY_PRB))
            self._hist_reserve_used.append(float(obs.get("reserve_used_prb", 0.0) or 0.0))
            self._hist_prb_ratio.append(float(obs.get("effective_prb_ratio", 0.0) or 0.0))
            # Slice metrics from current cell
            cell = getattr(self._env, "_target_cell", lambda: None)()
            if cell:
                quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
                alloc = dict(getattr(cell, "allocated_dl_prb_by_slice", {}) or {})
                demand_map = getattr(cell, "dl_total_prb_demand", {}) or {}
                alloc_map = getattr(cell, "prb_ue_allocation_dict", {}) or {}
                for sl in (SL_E, SL_U):
                    self._slice_prb[sl].append(float(quotas.get(sl, 0.0) or 0.0))
                    self._slice_tx[sl].append(float((self._hist_embb_thr if sl == SL_E else self._hist_urllc_thr)[-1]))
                    dem = 0.0
                    for ue in getattr(cell, "connected_ue_list", {}).values():
                        if getattr(ue, "slice_type", None) != sl:
                            continue
                        imsi = getattr(ue, "ue_imsi", None)
                        if imsi:
                            dem += float(demand_map.get(imsi, 0.0) or 0.0)
                    allo = float(alloc.get(sl, 0.0) or 0.0)
                    raw_ratio = 1.0 if dem <= 0 else (allo / dem)
                    sat = 1.0 if dem <= 0 else min(1.0, raw_ratio)
                    self._slice_sat[sl].append(sat)
                    self._slice_sat_raw[sl].append(raw_ratio)
                # Per-UE throughput/demand/sat/SINR/CQI
                managed = set(getattr(self._env, "_managed_ues", set()) or [])
                for imsi, ue in getattr(self.ric, "ue_list", {}).items():
                    if managed and imsi not in managed:
                        continue
                    bps = getattr(ue, "served_downlink_bitrate", None)
                    if bps is None:
                        bps = getattr(ue, "downlink_bitrate", 0.0)
                    self._ue_tx[imsi].append(float(bps) / 1e6)
                    demand = float(demand_map.get(imsi, 0.0) or 0.0)
                    granted = float(alloc_map.get(imsi, {}).get("downlink", 0.0) or 0.0)
                    self._ue_prb_demand[imsi].append(demand)
                    raw_ratio = 1.0 if demand <= 0 else (granted / demand)
                    sat = 1.0 if demand <= 0 else min(1.0, raw_ratio)
                    self._ue_prb_sat[imsi].append(sat)
                    self._ue_prb_sat_raw[imsi].append(raw_ratio)
                    sinr = getattr(ue, "downlink_sinr", None)
                    if sinr is not None:
                        self._ue_sinr[imsi].append(float(sinr))
                    cqi = getattr(ue, "downlink_cqi", None)
                    if cqi is not None:
                        self._ue_cqi[imsi].append(float(cqi))
                self._cell_max_prb.append(int(getattr(cell, "max_dl_prb", 0) or 0))
                eff = getattr(cell, "_effective_max_dl_prb", getattr(cell, "max_dl_prb", 0))
                self._cell_eff_prb.append(int(eff or 0))

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
                html.Pre(id="snapshot", style={"whiteSpace": "pre-wrap"}),
                dcc.Graph(id="action-graph"),
                dcc.Graph(id="prb-graph"),
                dcc.Graph(id="throughput-graph"),
                dcc.Graph(id="lat-graph"),
                dcc.Graph(id="reward-graph"),
                dcc.Graph(id="reward-trend-graph"),
                dcc.Graph(id="episode-return-graph"),
                dcc.Graph(id="reserve-graph"),
                dcc.Graph(id="ratio-graph"),
                dcc.Graph(id="ue-throughput-graph"),
                dcc.Graph(id="ue-demand-graph"),
                dcc.Graph(id="ue-sat-graph"),
                dcc.Graph(id="slice-sat-graph"),
                dcc.Graph(id="slice-sat-raw-graph"),
                dcc.Graph(id="sinr-graph"),
                dcc.Graph(id="cqi-graph"),
                dcc.Graph(id="supply-graph"),
                dcc.Interval(id="tick", interval=1000, n_intervals=0),
            ]
        )

        @app.callback(
            Output("snapshot", "children"),
            Output("action-graph", "figure"),
            Output("prb-graph", "figure"),
            Output("throughput-graph", "figure"),
            Output("lat-graph", "figure"),
            Output("reward-graph", "figure"),
            Output("reward-trend-graph", "figure"),
            Output("episode-return-graph", "figure"),
            Output("reserve-graph", "figure"),
            Output("ratio-graph", "figure"),
            Output("ue-throughput-graph", "figure"),
            Output("ue-demand-graph", "figure"),
            Output("ue-sat-graph", "figure"),
            Output("slice-sat-graph", "figure"),
            Output("slice-sat-raw-graph", "figure"),
            Output("sinr-graph", "figure"),
            Output("cqi-graph", "figure"),
            Output("supply-graph", "figure"),
            Input("tick", "n_intervals"),
        )
        def _update(_):
            with self._lock:
                snap = dict(self._last_info)
                t = list(self._t)
                thr_e = list(self._hist_embb_thr)
                thr_u = list(self._hist_urllc_thr)
                lat = list(self._hist_latency)
                deadline = list(self._hist_deadline)
                reward = list(self._hist_reward)
                reward_base = list(self._hist_reward_base)
                reward_embb = list(self._hist_reward_embb)
                penalty_latency = list(self._hist_penalty_latency)
                penalty_reserve = list(self._hist_penalty_reserve)
                reserve = list(self._hist_reserve)
                reserve_used = list(self._hist_reserve_used)
                prb_ratio = list(self._hist_prb_ratio)
                action_idx = list(self._hist_action_idx)
                episode_returns = list(self._hist_episode_return)
                episode_idx = list(self._hist_episode_idx)
                slice_prb = {k: list(v) for k, v in self._slice_prb.items()}
                slice_sat = {k: list(v) for k, v in self._slice_sat.items()}
                slice_sat_raw = {k: list(v) for k, v in self._slice_sat_raw.items()}
                ue_tx = {k: list(v) for k, v in self._ue_tx.items()}
                ue_demand = {k: list(v) for k, v in self._ue_prb_demand.items()}
                ue_sat = {k: list(v) for k, v in self._ue_prb_sat.items()}
                ue_sinr = {k: list(v) for k, v in self._ue_sinr.items()}
                ue_cqi = {k: list(v) for k, v in self._ue_cqi.items()}
                cell_max = list(self._cell_max_prb)
                cell_eff = list(self._cell_eff_prb)

            action_fig = go.Figure()
            if action_idx:
                action_fig.add_trace(go.Scatter(x=t[-len(action_idx):], y=action_idx, mode="lines", name="action idx"))
            action_fig.update_layout(title="Actions", xaxis_title="time (s)")

            prb_fig = go.Figure()
            for sl, series in slice_prb.items():
                prb_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} PRB"))
            prb_fig.update_layout(title="Slice PRB quotas", xaxis_title="time (s)")

            thr_fig = go.Figure()
            thr_fig.add_trace(go.Scatter(x=t, y=[v / 1e6 for v in thr_e], mode="lines", name="eMBB throughput (Mbps)"))
            thr_fig.add_trace(go.Scatter(x=t, y=[v / 1e6 for v in thr_u], mode="lines", name="URLLC throughput (Mbps)"))
            thr_fig.update_layout(title="Throughput", xaxis_title="time (s)", yaxis_title="Mbps")

            lat_fig = go.Figure()
            lat_fig.add_trace(go.Scatter(x=t, y=lat, mode="lines", name="URLLC latency (s)"))
            lat_fig.add_trace(go.Scatter(x=t, y=deadline, mode="lines", name="deadline", line=dict(dash="dash")))
            lat_fig.update_layout(title="Latency", xaxis_title="time (s)")

            reward_fig = go.Figure()
            reward_fig.add_trace(go.Scatter(x=t, y=reward, mode="lines", name="reward"))
            if reward_base:
                reward_fig.add_trace(
                    go.Scatter(
                        x=t[-len(reward_base):],
                        y=reward_base,
                        mode="lines",
                        name="reward base",
                        line=dict(dash="dash"),
                    )
                )
            if reward_embb:
                reward_fig.add_trace(
                    go.Scatter(
                        x=t[-len(reward_embb):],
                        y=[-v for v in reward_embb],
                        mode="lines",
                        name="embb penalty (neg)",
                    )
                )
            if penalty_latency:
                reward_fig.add_trace(
                    go.Scatter(
                        x=t[-len(penalty_latency):],
                        y=[-v for v in penalty_latency],
                        mode="lines",
                        name="latency penalty (neg)",
                    )
                )
            if penalty_reserve:
                reward_fig.add_trace(
                    go.Scatter(
                        x=t[-len(penalty_reserve):],
                        y=[-v for v in penalty_reserve],
                        mode="lines",
                        name="reserve penalty (neg)",
                    )
                )
            reward_fig.update_layout(title="Reward", xaxis_title="time (s)", yaxis_title="reward")

            # Smoothed view to highlight trend direction.
            reward_trend_fig = go.Figure()
            if reward:
                reward_arr = np.asarray(reward, dtype=float)
                cum_avg = np.cumsum(reward_arr) / np.arange(1, len(reward_arr) + 1)
                reward_trend_fig.add_trace(go.Scatter(x=t, y=cum_avg, mode="lines", name="cumulative avg"))
                window = min(len(reward_arr), min(100, max(5, len(reward_arr) // 5)))
                if window > 1:
                    kernel = np.ones(window) / float(window)
                    rolling = np.convolve(reward_arr, kernel, mode="valid")
                    reward_trend_fig.add_trace(
                        go.Scatter(x=t[window - 1 :], y=rolling, mode="lines", name=f"{window}-step avg")
                    )
            reward_trend_fig.update_layout(title="Reward trend", xaxis_title="time (s)", yaxis_title="avg reward")

            episode_fig = go.Figure()
            if episode_idx and episode_returns:
                episode_fig.add_trace(
                    go.Scatter(x=episode_idx, y=episode_returns, mode="lines+markers", name="episode return")
                )
                window_ep = min(len(episode_returns), min(50, max(2, len(episode_returns) // 5)))
                if window_ep > 1:
                    kernel = np.ones(window_ep) / float(window_ep)
                    rolling_ep = np.convolve(np.asarray(episode_returns, dtype=float), kernel, mode="valid")
                    episode_fig.add_trace(
                        go.Scatter(
                            x=episode_idx[window_ep - 1 :],
                            y=rolling_ep,
                            mode="lines",
                            name=f"{window_ep}-ep avg",
                        )
                    )
            episode_fig.update_layout(title="Episode returns", xaxis_title="episode", yaxis_title="return")

            reserve_fig = go.Figure()
            reserve_fig.add_trace(go.Scatter(x=t, y=reserve, mode="lines", name="reserve fraction"))
            if reserve_used:
                reserve_fig.add_trace(
                    go.Scatter(x=t[-len(reserve_used):], y=reserve_used, mode="lines", name="reserve used (PRB)")
                )
            reserve_fig.update_layout(title="Reserve usage", xaxis_title="time (s)", yaxis_title="fraction / PRB")

            ratio_fig = go.Figure()
            ratio_fig.add_trace(go.Scatter(x=t, y=prb_ratio, mode="lines", name="effective PRB ratio"))
            ratio_fig.update_layout(title="Effective PRB ratio", xaxis_title="time (s)", yaxis_title="ratio")
            ue_thr_fig = go.Figure()
            for imsi, series in ue_tx.items():
                ue_thr_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} tx (Mbps)"))
            ue_thr_fig.update_layout(title="Per-UE throughput", xaxis_title="time (s)", yaxis_title="Mbps", legend=dict(orientation="h"))

            ue_dem_fig = go.Figure()
            for imsi, series in ue_demand.items():
                ue_dem_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} prb demand"))
            ue_dem_fig.update_layout(title="Per-UE PRB demand", xaxis_title="time (s)", yaxis_title="PRB", legend=dict(orientation="h"))

            ue_sat_fig = go.Figure()
            for imsi, series in ue_sat.items():
                ue_sat_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} sat"))
            ue_sat_fig.update_layout(title="Per-UE satisfaction", xaxis_title="time (s)", yaxis_title="sat", legend=dict(orientation="h"))

            slice_sat_fig = go.Figure()
            for sl, series in slice_sat.items():
                slice_sat_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} sat capped"))
            slice_sat_fig.update_layout(title="Slice satisfaction (capped)", xaxis_title="time (s)", yaxis_title="sat", legend=dict(orientation="h"))

            slice_sat_raw_fig = go.Figure()
            for sl, series in slice_sat_raw.items():
                slice_sat_raw_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} sat raw"))
            slice_sat_raw_fig.update_layout(title="Slice satisfaction (raw)", xaxis_title="time (s)", yaxis_title="sat raw", legend=dict(orientation="h"))

            sinr_fig = go.Figure()
            for imsi, series in ue_sinr.items():
                sinr_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} SINR (dB)"))
            sinr_fig.update_layout(title="Per-UE SINR", xaxis_title="time (s)", yaxis_title="dB", legend=dict(orientation="h"))

            cqi_fig = go.Figure()
            for imsi, series in ue_cqi.items():
                cqi_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} CQI"))
            cqi_fig.update_layout(title="Per-UE CQI", xaxis_title="time (s)", yaxis_title="CQI", legend=dict(orientation="h"))

            supply_fig = go.Figure()
            supply_fig.add_trace(go.Scatter(x=t[-len(cell_max):], y=cell_max, mode="lines", name="cell max PRB"))
            supply_fig.add_trace(go.Scatter(x=t[-len(cell_eff):], y=cell_eff, mode="lines", name="effective PRB"))
            supply_fig.update_layout(
                title="PRB supply (max vs effective)",
                xaxis_title="time (s)",
                yaxis_title="PRB",
                legend=dict(orientation="h"),
            )

            snap_text = self._format_snapshot(snap)
            return (
                snap_text,
                action_fig,
                prb_fig,
                thr_fig,
                lat_fig,
                reward_fig,
                reward_trend_fig,
                episode_fig,
                reserve_fig,
                ratio_fig,
                ue_thr_fig,
                ue_dem_fig,
                ue_sat_fig,
                slice_sat_fig,
                slice_sat_raw_fig,
                sinr_fig,
                cqi_fig,
                supply_fig,
            )

        def _run():
            app.run(host="0.0.0.0", port=self.dash_port, debug=False, use_reloader=False)

        self._dash_app = app
        self._dash_thread = threading.Thread(target=_run, daemon=True)
        self._dash_thread.start()
        print(f"{self.xapp_id}: dashboard running on port {self.dash_port}")

    def _clear_plot_caches(self):
        with self._lock:
            for store in (
                self._ue_tx,
                self._ue_prb_demand,
                self._ue_prb_sat,
                self._ue_prb_sat_raw,
                self._ue_sinr,
                self._ue_cqi,
            ):
                store.clear()

    def _detect_sim_dt(self) -> float:
        dt = float(getattr(settings, "SIM_STEP_TIME_DEFAULT", 1.0))
        eng = getattr(self.ric, "simulation_engine", None)
        for attr in ("time_step", "sim_step_time", "step_time", "dt"):
            try:
                val = getattr(eng, attr, None)
                if val is not None:
                    return float(val)
            except Exception:
                continue
        return dt

    def _format_snapshot(self, snap: Dict) -> str:
        if not snap:
            return "No data yet"
        action = snap.get("action") or (0, 0)
        action_idx = snap.get("action_idx")
        obs = snap.get("obs") or {}
        cell = getattr(self._env, "_target_cell", lambda: None)()
        lines = []
        if action_idx is not None:
            lines.append(f"Action: {action_idx}: Δe={action[0]}, Δu={action[1]}")
        else:
            lines.append(f"Action: Δe={action[0]}, Δu={action[1]}")
        lines.append(f"Time: step={self._global_step}, sim={self._global_step * self._sim_dt:.3f}s")
        ep_id = snap.get("episode_id")
        ep_idx = getattr(self._env, "_episode_idx", None)
        if ep_id or ep_idx is not None:
            lines.append(f"Episode: {ep_id if ep_id is not None else 'unknown'} (idx={ep_idx if ep_idx is not None else '?'})")
        parts = snap.get("reward_parts") or {}
        if parts:
            total = (
                float(parts.get("reward_base", 0.0))
                - float(parts.get("reward_embb", 0.0))
                - float(parts.get("penalty_latency", 0.0))
                - float(parts.get("penalty_reserve", 0.0))
            )
            lines.append(
                "Reward parts: "
                f"total={total:.3f}, base={float(parts.get('reward_base', 0.0)):.3f}, "
                f"embb_pen={float(parts.get('reward_embb', 0.0)):.3f}, "
                f"lat_pen=-{float(parts.get('penalty_latency', 0.0)):.3f}, "
                f"reserve_pen={float(parts.get('penalty_reserve', 0.0)):.3f}"
            )
        reserve_used = float(obs.get("reserve_used_prb", 0.0) or 0.0)
        lines.append(
            f"Reserve pool: total={TOTAL_RESILIENCY_PRB}, used={reserve_used:.1f}, "
            f"action_delta={float(sum(action)):.1f}"
        )
        # Slice metrics
        cell_ref = cell or (list(getattr(self.ric, "cell_list", {}).values()) or [None])[0]
        if cell_ref:
            quotas = dict(getattr(cell_ref, "slice_dl_prb_quota", {}) or {})
            alloc = dict(getattr(cell_ref, "allocated_dl_prb_by_slice", {}) or {})
            req = getattr(cell_ref, "dl_total_prb_demand", {}) or {}
            connected = getattr(cell_ref, "connected_ue_list", {}) or {}
            for sl in (SL_E, SL_U):
                tx = float(obs.get("embb_throughput" if sl == SL_E else "urllc_throughput", 0.0) or 0.0) / 1e6
                lat = float(obs.get("urllc_latency", 0.0) or 0.0) if sl == SL_U else float(
                    obs.get("embb_latency", 0.0) or 0.0
                )
                ues = sum(1 for ue in connected.values() if getattr(ue, "slice_type", None) == sl)
                prb = float(quotas.get(sl, 0.0) or 0.0)
                # Satisfaction (capped and raw) for snapshot
                dem = 0.0
                for ue in connected.values():
                    if getattr(ue, "slice_type", None) != sl:
                        continue
                    imsi = getattr(ue, "ue_imsi", None)
                    if imsi:
                        dem += float(req.get(imsi, 0.0) or 0.0)
                allo = float(alloc.get(sl, 0.0) or 0.0)
                raw_sat = 1.0 if dem <= 0 else (allo / dem)
                cap_sat = 1.0 if dem <= 0 else min(1.0, raw_sat)
                lines.append(f"{sl}: PRB={prb:.0f}, Tx={tx:.3f} Mbps, Latency={lat:.3f}, UEs={ues}, Sat={cap_sat:.2f} (raw {raw_sat:.2f})")
            total_ues = len(getattr(self.ric, "ue_list", {}) or {})
            ue_max = int(getattr(settings, "UE_DEFAULT_MAX_COUNT", 0) or 0)
            lines.append(f"UEs total={total_ues}, UE_max={ue_max}")
            # Jamming / effective PRB
            eff = getattr(cell_ref, "_effective_max_dl_prb", getattr(cell_ref, "max_dl_prb", 0))
            max_dl = getattr(cell_ref, "max_dl_prb", 0)
        if eff != max_dl:
            lines.append(f"Jamming: effective PRB={eff}/{max_dl}")
        state_block = self._format_state_block(obs, self._last_state, snap.get("prev_action"))
        if state_block:
            lines.append(state_block)
        return "\n".join(lines)

    def _log_state_debug(self, obs: Optional[Dict], state: Optional[np.ndarray], info: Optional[Dict]):
        """Console print of raw obs vs normalized state for debugging."""
        if obs is None or state is None:
            return
        try:
            action = (info.get("action") if isinstance(info, dict) else None) or (0, 0)
            prev_action = (info.get("prev_action") if isinstance(info, dict) else None) or (0, 0)
            action_idx = info.get("action_idx") if isinstance(info, dict) else None
            raw_line = (
                "raw: "
                f"e_demand={obs.get('embb_demand_prb', 0.0):.3f}, "
                f"u_demand={obs.get('urllc_demand_prb', 0.0):.3f}, "
                f"e_prb={obs.get('embb_prb', 0.0):.3f}, "
                f"u_prb={obs.get('urllc_prb', 0.0):.3f}, "
                f"e_queue={obs.get('embb_queue', 0.0):.1f}/{obs.get('embb_queue_max', EMBB_QUEUE_MAX):.1f}, "
                f"u_queue={obs.get('urllc_queue', 0.0):.1f}/{obs.get('urllc_queue_max', URLLC_QUEUE_MAX):.1f}, "
                f"e_thr={obs.get('embb_throughput', 0.0):.3f}/{obs.get('embb_throughput_max', EMBB_THR_MAX):.3f}, "
                f"u_thr={obs.get('urllc_throughput', 0.0):.3f}/{obs.get('urllc_throughput_max', URLLC_THR_MAX):.3f}, "
                f"u_lat={obs.get('urllc_latency', 0.0):.6f}/{obs.get('urllc_latency_deadline', URLLC_LATENCY_DEADLINE):.6f}, "
                f"eff_prb_ratio={obs.get('effective_prb_ratio', 0.0):.3f}, "
                f"reserve_used={obs.get('reserve_used_prb', 0.0):.1f}, "
                f"prev_action=({prev_action[0]:.1f},{prev_action[1]:.1f})"
            )
            norm_names = [
                "e_demand_n",
                "u_demand_n",
                "e_prb_n",
                "u_prb_n",
                "e_queue_n",
                "u_queue_n",
                "e_thr_n",
                "u_thr_n",
                "u_lat_n",
                "eff_ratio_n",
                "reserve_used_n",
                "prev_e_n",
                "prev_u_n",
            ]
            norm_vals = state.tolist() if hasattr(state, "tolist") else list(state)
            norm_pairs = ", ".join(f"{k}={float(v):.3f}" for k, v in zip(norm_names, norm_vals))
            prefix = f"[STATE][DQN] step={self._global_step}"
            if action_idx is not None:
                prefix += f" action_idx={action_idx}"
            prefix += f" Δe={action[0]}, Δu={action[1]}"
            print("\n".join([prefix, raw_line, f"norm: {norm_pairs}"]))
        except Exception:
            pass

    def _format_state_block(self, obs: Optional[Dict], state: Optional[np.ndarray], prev_action: Optional[Tuple[int, int]]):
        if obs is None or state is None:
            return ""
        try:
            prev = prev_action or (0, 0)
            raw = (
                f"raw: "
                f"e_demand={obs.get('embb_demand_prb', 0.0):.3f}, "
                f"u_demand={obs.get('urllc_demand_prb', 0.0):.3f}, "
                f"e_prb={obs.get('embb_prb', 0.0):.3f}, "
                f"u_prb={obs.get('urllc_prb', 0.0):.3f}, "
                f"e_queue={obs.get('embb_queue', 0.0):.1f}/{obs.get('embb_queue_max', EMBB_QUEUE_MAX):.1f}, "
                f"u_queue={obs.get('urllc_queue', 0.0):.1f}/{obs.get('urllc_queue_max', URLLC_QUEUE_MAX):.1f}, "
                f"e_thr={obs.get('embb_throughput', 0.0):.3f}/{obs.get('embb_throughput_max', EMBB_THR_MAX):.3f}, "
                f"u_thr={obs.get('urllc_throughput', 0.0):.3f}/{obs.get('urllc_throughput_max', URLLC_THR_MAX):.3f}, "
                f"u_lat={obs.get('urllc_latency', 0.0):.6f}/{obs.get('urllc_latency_deadline', URLLC_LATENCY_DEADLINE):.6f}, "
                f"eff_ratio={obs.get('effective_prb_ratio', 0.0):.3f}, "
                f"reserve_used={obs.get('reserve_used_prb', 0.0):.1f}, "
                f"prev_action=({prev[0]:.1f},{prev[1]:.1f})"
            )
            norm_names = [
                "e_dem_n",
                "u_dem_n",
                "e_prb_n",
                "u_prb_n",
                "e_q_n",
                "u_q_n",
                "e_thr_n",
                "u_thr_n",
                "u_lat_n",
                "eff_n",
                "reserve_used_n",
                "prev_e_n",
                "prev_u_n",
            ]
            norm_vals = state.tolist() if hasattr(state, "tolist") else list(state)
            norm = "norm: " + ", ".join(f"{k}={float(v):.3f}" for k, v in zip(norm_names, norm_vals))
            return "\n".join([raw, norm])
        except Exception:
            return ""

    # ---------- Agent helpers ---------- #
    def _init_tb(self):
        if not self.tb_enable or SummaryWriter is None or not self.tb_dir:
            return
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = (
                f"{ts}_resdqn_lr{self.lr}_gamma{self.gamma}_eps{self.eps_start}-{self.eps_end}"
                f"_batch{self.batch}_buffer{self.buffer_cap}_total{TOTAL_RESILIENCY_PRB}"
            )
            run_dir = os.path.join(self.tb_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=run_dir)
            print(f"{self.xapp_id}: tensorboard logging -> {run_dir}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to init tensorboard logger {self.tb_dir}: {exc}")
            self._tb_writer = None

    def _tb_log(
        self, state: Optional[np.ndarray], action_idx: Optional[int], reward: float, info: Optional[Dict], done: bool = False
    ):
        if self._tb_writer is None:
            return
        step = self._global_step
        try:
            if state is not None:
                self._tb_writer.add_scalar("reward", reward, step)
                self._tb_writer.add_scalar("reward_cumulative", self._episode_return, step)
                # Smoothed reward to make trend visible in TensorBoard
                if self.tb_smooth_win > 0:
                    with self._lock:
                        recent = list(self._hist_reward)[-self.tb_smooth_win :]
                    if recent:
                        self._tb_writer.add_scalar("reward_smooth", float(np.mean(recent)), step)
                self._tb_writer.add_scalar("epsilon", float(self._eps), step)
                if action_idx is not None:
                    self._tb_writer.add_scalar("action_idx", action_idx, step)
            obs = info.get("obs") if isinstance(info, dict) else None
            if obs:
                self._tb_writer.add_scalar(f"{SL_E}/throughput_bps", float(obs.get("embb_throughput", 0.0) or 0.0), step)
                self._tb_writer.add_scalar(f"{SL_U}/throughput_bps", float(obs.get("urllc_throughput", 0.0) or 0.0), step)
                self._tb_writer.add_scalar(f"{SL_E}/queue_bytes", float(obs.get("embb_queue", 0.0) or 0.0), step)
                self._tb_writer.add_scalar(f"{SL_U}/queue_bytes", float(obs.get("urllc_queue", 0.0) or 0.0), step)
                self._tb_writer.add_scalar(f"{SL_U}/latency_s", float(obs.get("urllc_latency", 0.0) or 0.0), step)
                self._tb_writer.add_scalar("pool/effective_prb_ratio", float(obs.get("effective_prb_ratio", 0.0) or 0.0), step)
                # Capacity visibility
                cell = getattr(self._env, "_target_cell", lambda: None)()
                cell_eff = float(getattr(cell, "_effective_max_dl_prb", 0.0) or 0.0) if cell else 0.0
                cell_max = float(getattr(cell, "max_dl_prb", 0.0) or 0.0) if cell else 0.0
                base_total = 0.0
                reserve_used = 0.0
                if cell:
                    base_total = float(sum((self._base_quota.get(getattr(cell, "cell_id", ""), {}) or {}).values()))
                    cur_quota = getattr(cell, "slice_dl_prb_quota", {}) or {}
                    cur_total = float(sum(cur_quota.values()) or 0.0)
                    reserve_used = max(0.0, min(TOTAL_RESILIENCY_PRB, cur_total - base_total))
                self._tb_writer.add_scalar("capacity/effective_prb", cell_eff, step)
                self._tb_writer.add_scalar("capacity/physical_max_prb", cell_max, step)
                self._tb_writer.add_scalar("capacity/base_total_prb", base_total, step)
                self._tb_writer.add_scalar("capacity/reserve_pool_prb", TOTAL_RESILIENCY_PRB, step)
                self._tb_writer.add_scalar("capacity/reserve_used_prb", reserve_used, step)
                if cell:
                    req = getattr(cell, "dl_total_prb_demand", {}) or {}
                    alloc = getattr(cell, "allocated_dl_prb_by_slice", {}) or {}
                    for sl in (SL_E, SL_U):
                        dem = 0.0
                        for ue in getattr(cell, "connected_ue_list", {}).values():
                            if getattr(ue, "slice_type", None) != sl:
                                continue
                            imsi = getattr(ue, "ue_imsi", None)
                            if imsi:
                                dem += float(req.get(imsi, 0.0) or 0.0)
                        allo = float(alloc.get(sl, 0.0) or 0.0)
                        self._tb_writer.add_scalar(f"{sl}/prb_alloc", allo, step)
                        sat = 1.0 if dem <= 0 else min(1.0, allo / dem)
                        self._tb_writer.add_scalar(f"{sl}/satisfaction_capped", sat, step)
                        self._tb_writer.add_scalar(f"{sl}/satisfaction_raw", (1.0 if dem <= 0 else (allo / dem)), step)
        except Exception:
            pass
        if done:
            try:
                ep_idx = self._episode_counter
                self._tb_writer.add_scalar("episode_return", self._episode_return, ep_idx)
                self._tb_writer.add_scalar("episode_return_vs_step", self._episode_return, step)
                self._tb_writer.add_scalar("episode_length", self._episode_len, ep_idx)
                for name, val in self._episode_reward_parts.items():
                    self._tb_writer.add_scalar(f"episode_reward_parts/{name}", val, ep_idx)
            except Exception:
                pass
        try:
            self._tb_writer.flush()
        except Exception:
            pass

    def _checkpoint_path(self, step: Optional[int] = None) -> str:
        if not self.model_dir:
            return ""
        base = os.path.join(self.model_dir, f"{self.xapp_id}_{self._run_tag}")
        suffix = "_final" if step is None else f"_step{int(step)}"
        return f"{base}{suffix}.pt"

    def _save_model(self, path: str):
        if not TORCH_AVAILABLE or self._policy is None or not path:
            return
        directory = os.path.dirname(path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception:
                pass
        try:
            torch.save(self._policy.state_dict(), path)
        except Exception:
            pass

    def _maybe_save_checkpoint(self, force: bool = False):
        if not TORCH_AVAILABLE or self._policy is None:
            return
        if not self.train:
            return
        if force:
            final_path = self._checkpoint_path(None)
            self._save_model(final_path)
            if self._decision_steps > 0:
                step_path = self._checkpoint_path(self._decision_steps)
                self._save_model(step_path)
            return
        if self.save_interval <= 0:
            return
        if self._decision_steps <= 0 or self._decision_steps == self._last_save_step:
            return
        if (self._decision_steps % self.save_interval) != 0:
            return
        final_path = self._checkpoint_path(None)
        self._save_model(final_path)
        step_path = self._checkpoint_path(self._decision_steps)
        self._save_model(step_path)
        self._last_save_step = self._decision_steps

    def _reset_episode_trackers(self):
        self._episode_return = 0.0
        self._episode_len = 0
        for k in self._episode_reward_parts:
            self._episode_reward_parts[k] = 0.0

    def _init_agent(self):
        in_dim = len(build_state(self._env._blank_obs(), (0, 0)))  # type: ignore[arg-type]
        out_dim = NUM_ACTIONS
        self._policy = _DQN(in_dim, out_dim).to(self.device)
        self._target = _DQN(in_dim, out_dim).to(self.device)
        self._target.load_state_dict(self._policy.state_dict())
        self._optimizer = optim.Adam(self._policy.parameters(), lr=self.lr)
        self._replay = _ReplayBuffer(capacity=self.buffer_cap)

    def _select_action(self, state: np.ndarray) -> int:
        mask = getattr(self._env, "_action_mask", lambda: [True] * NUM_ACTIONS)()
        valid_idx = [i for i, ok in enumerate(mask) if ok]
        if not valid_idx:
            return 0
        if self.train and random.random() < self._eps:
            return int(random.choice(valid_idx))
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self._policy(s)
            q_np = q.cpu().numpy().flatten()
            for i, ok in enumerate(mask):
                if not ok:
                    q_np[i] = -1e9
            idx = int(np.argmax(q_np))
        return idx

    def _learn(self):
        if self._replay is None or len(self._replay) < self.batch:
            return
        states, actions, rewards, next_states, dones, next_masks = self._replay.sample(self.batch)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        q_pred = self._policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_all = self._target(next_states)
            next_q_all = next_q_all.masked_fill(~next_masks, -1e9)
            next_q = next_q_all.max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(q_pred, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar("loss", float(loss.item()), self._global_step)
            except Exception:
                pass

        if self._decision_steps > 0 and self._decision_steps % self.target_update == 0:
            self._target.load_state_dict(self._policy.state_dict())

        self._maybe_save_checkpoint()

    def __del__(self):
        try:
            self._maybe_save_checkpoint(force=True)
            if self._tb_writer is not None:
                self._tb_writer.flush()
        except Exception:
            pass
