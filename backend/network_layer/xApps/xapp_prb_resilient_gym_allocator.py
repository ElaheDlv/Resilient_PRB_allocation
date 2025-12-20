"""Episodic (Gym-like) trainer for the resilient KPI state.

Highlights
- Uses the KPI-first state from xapp_prb_resilient_allocator (buffers,
  mean latency, throughput, per-slice PRB quota/allocation, total/free PRB pool).
- Supports configurable discrete action space (PRB deltas between slices).
- Agents: vanilla DQN, Quantile DQN (QR-DQN), PPO baseline, plus a stateless
  proportional-fair heuristic (no training) for quick comparison.
- Optional jamming events per episode to shrink available PRBs.

Enable via settings:
    RES_GYM_ENABLE=1
    RES_GYM_AGENT=dqn|qrdqn
    RES_GYM_CONFIG_PATH=path/to/episodes.json
    RES_GYM_TRAIN=1 (set 0 to run eval only)
Other knobs have sensible defaults; see constructor for names.
"""

from __future__ import annotations

import json
import os
import random
import threading
from collections import deque
from itertools import product
from typing import Dict, List, Optional, Tuple
import csv

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
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    F = None
    SummaryWriter = None

try:
    from flask import Flask, jsonify, render_template_string

    FLASK_AVAILABLE = True
except Exception:
    Flask = None
    jsonify = None
    render_template_string = None
    FLASK_AVAILABLE = False


# --------------------------- Replay buffer --------------------------- #
class _ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
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


# --------------------------- Models --------------------------- #
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


class _QRDQN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_atoms: int = 51):
        super().__init__()
        self.n_atoms = n_atoms
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim * n_atoms),
        )

    def forward(self, x):
        # Return [batch, actions, atoms]
        q = self.net(x)
        return q.view(q.size(0), self.out_dim, self.n_atoms)


class _ActorCritic(nn.Module):
    """Shared torso with separate policy/value heads for PPO."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, out_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        feat = self.body(x)
        return self.policy(feat), self.value(feat).squeeze(-1)


# --------------------------- Environment --------------------------- #
class ResilientPRBGymEnv:
    """Gym-like environment built on KPI-first resilient state."""

    def __init__(self, ric):
        self.ric = ric
        self.sim_engine = getattr(ric, "simulation_engine", None)
        self.decision_period = max(1, int(getattr(settings, "RES_GYM_DECISION_PERIOD", 1)))
        levels = getattr(settings, "RES_GYM_ACTION_LEVELS", None)
        if levels is None:
            levels = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
        self._action_levels = [int(x) for x in levels]
        self._action_combos = list(product(self._action_levels, repeat=2))  # (d_embb, d_urllc)
        self.actions = self._action_combos  # back-compat for len()
        self.gamma = float(getattr(settings, "RES_GYM_GAMMA", 0.99))
        self.w_embb = float(getattr(settings, "RES_GYM_W_EMBB", 1.0))
        self.w_urllc = float(getattr(settings, "RES_GYM_W_URLLC", 1.5))
        self.norm_buf_bytes = float(getattr(settings, "PRB_RESILIENT_NORM_BUF_BYTES", 2e5))
        self.latency_targets = self._init_latency_targets()

        self._episodes = self._load_episode_specs()
        self._episode_idx = -1
        self._episode_loop = bool(getattr(settings, "RES_GYM_LOOP", False))
        self._current_episode: Optional[Dict] = None
        self._steps_in_episode = 0
        self._episode_step_limit = 0
        self._base_quotas: Dict[str, Dict[str, int]] = {}
        self._jam_events: List[Dict] = []
        self._catalog_done = False
        self._state_dim = 12  # KPI vector length
        self._managed_ues: set[str] = set()
        self._trace_cache: Dict[Tuple[str, str, float], List[Tuple[float, float, float]]] = {}
        self._last_kpis: Dict = {}

    # ---------- Episode control ---------- #
    def reset(self) -> np.ndarray:
        if self._catalog_done or self.sim_engine is None:
            return np.zeros(self._state_dim, dtype=np.float32)
        if not self._episodes:
            print("ResilientGymEnv: no episode catalog configured.")
            self._catalog_done = True
            return np.zeros(self._state_dim, dtype=np.float32)

        self._episode_idx += 1
        if self._episode_idx >= len(self._episodes):
            if self._episode_loop:
                self._episode_idx = 0
            else:
                self._catalog_done = True
                return np.zeros(self._state_dim, dtype=np.float32)

        ep = self._episodes[self._episode_idx]
        self._current_episode = ep
        self._steps_in_episode = 0
        self._episode_step_limit = int(ep["duration_steps"])
        self._jam_events = ep.get("jamming", [])
        self._base_quotas = {}
        self._deploy_episode(ep)
        return self._observe()

    def step(self, action_idx: Optional[int]) -> Tuple[np.ndarray, float, bool, Dict]:
        self._steps_in_episode += 1
        if action_idx is not None:
            self._apply_action(action_idx)
        self._apply_jamming()
        state = self._observe()
        reward = self._compute_reward(state)
        done = self._steps_in_episode >= self._episode_step_limit
        info = {"episode_id": self._current_episode.get("id") if self._current_episode else None}
        return state, reward, done, info

    # ---------- Jamming: shrink slice quotas to simulate PRB loss ---------- #
    def _apply_jamming(self):
        if not self._jam_events:
            return
        step = self._steps_in_episode
        for cell in self.ric.cell_list.values():
            base = self._base_quotas.get(cell.cell_id, {})
            factor = 1.0
            for ev in self._jam_events:
                try:
                    start = int(ev.get("start_step", 0))
                    end = int(ev.get("end_step", 0))
                    f = float(ev.get("factor", 1.0))
                except Exception:
                    continue
                if start <= step <= end:
                    factor = min(factor, max(0.0, f))
            quotas = {s: int(v * factor) for s, v in base.items()}
            total = sum(quotas.values())
            if total > cell.max_dl_prb and cell.max_dl_prb > 0:
                scale = cell.max_dl_prb / float(total)
                for sl in quotas:
                    quotas[sl] = int(quotas[sl] * scale)
            cell.slice_dl_prb_quota = quotas

    def _deploy_episode(self, spec: Dict):
        """Apply quotas, clear old UEs, and spawn scripted UEs + traces per episode."""
        self._clear_managed_ues()
        if bool(spec.get("freeze_mobility", True)):
            setattr(settings, "SIM_FREEZE_MOBILITY", True)
        self._apply_initial_prb(spec.get("initial_prb", {}))
        slices = spec.get("slices") or {}
        for slice_name in (SL_E, SL_U):
            cfg = slices.get(slice_name)
            if cfg is None:
                continue
            for idx in range(int(cfg.get("ue_count", 0))):
                imsi = f"RGIMSI_{spec['id']}_{slice_name}_{idx}"
                self._register_episode_ue(imsi, slice_name, cfg)

    def _clear_managed_ues(self):
        if self.sim_engine is None:
            return
        for imsi in list(self.sim_engine.ue_list.keys()):
            try:
                self.sim_engine.deregister_ue(imsi)
            except Exception as exc:
                print(f"ResilientGymEnv: failed to deregister {imsi}: {exc}")
        self._managed_ues.clear()

    def _apply_initial_prb(self, prb_map: Dict[str, int]):
        if self.sim_engine is None:
            return
        for cell in self.sim_engine.cell_list.values():
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            for sl, val in prb_map.items():
                quotas[sl] = max(0, int(val))
            total = sum(quotas.values())
            max_prb = max(1, int(getattr(cell, "max_dl_prb", 1)))
            if total > max_prb:
                scale = max_prb / float(total)
                for sl in quotas:
                    quotas[sl] = int(quotas[sl] * scale)
            cell.slice_dl_prb_quota = quotas
            self._base_quotas[cell.cell_id] = quotas.copy()

    def _register_episode_ue(self, imsi: str, slice_name: str, slice_cfg: Dict):
        if self.sim_engine is None:
            return
        max_attempts = max(1, int(getattr(settings, "RES_GYM_REGISTER_RETRIES", 20)))
        ok = False
        for _ in range(max_attempts):
            ok = self.sim_engine.register_ue(imsi, [slice_name], register_slice=slice_name)
            if ok:
                break
        if not ok:
            print(f"ResilientGymEnv: unable to register {imsi} after {max_attempts} attempts; skipping.")
            return
        ue = self.sim_engine.ue_list.get(imsi)
        if ue is None:
            return
        if self._current_episode.get("freeze_mobility", True):
            ue.speed_mps = 0
            try:
                ue.set_target(ue.position_x, ue.position_y)
            except Exception:
                pass
        trace_path = slice_cfg.get("trace")
        if trace_path:
            self._attach_trace(ue, trace_path, slice_cfg)
        self._managed_ues.add(imsi)

    def _attach_trace(self, ue, path: str, slice_cfg: Dict):
        cache_key = (path, slice_cfg.get("ue_ip") or "", float(slice_cfg.get("trace_bin", 1.0)))
        if cache_key not in self._trace_cache:
            try:
                samples = load_raw_packet_csv(
                    path,
                    ue_ip=slice_cfg.get("ue_ip"),
                    bin_s=slice_cfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)),
                    overhead_sub_bytes=int(getattr(settings, "TRACE_OVERHEAD_BYTES", 0)),
                )
            except Exception as exc:
                print(f"ResilientGymEnv: failed to load trace {path}: {exc}")
                samples = []
            self._trace_cache[cache_key] = samples
        samples = self._trace_cache[cache_key]
        if not samples:
            return
        speed = float(slice_cfg.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        bs = ue.current_bs
        try:
            if bs and hasattr(bs, "attach_dl_trace"):
                bs.attach_dl_trace(ue.ue_imsi, samples, speed)
            else:
                ue.attach_trace(samples, speed)
        except Exception as exc:
            print(f"ResilientGymEnv: failed to attach trace {path} to {ue.ue_imsi}: {exc}")

    # ---------- Observation / reward ---------- #
    def _observe(self) -> np.ndarray:
        # Aggregate first cell (single-cell assumption for now)
        if not self.ric.cell_list:
            return np.zeros(self._state_dim, dtype=np.float32)
        cell = next(iter(self.ric.cell_list.values()))
        kpis = self._collect_kpis(cell)
        self._last_kpis = kpis
        embb = kpis["slices"][SL_E]
        urllc = kpis["slices"][SL_U]
        pool = kpis["pool"]
        state = np.asarray(
            [
                embb["buffer_bytes"],
                embb["tx_mbps"],
                embb["latency_mean_s"],
                embb["prb_quota"],
                embb["prb_alloc"],
                urllc["buffer_bytes"],
                urllc["tx_mbps"],
                urllc["latency_mean_s"],
                urllc["prb_quota"],
                urllc["prb_alloc"],
                pool["max_prb"],
                pool["free_prb"],
            ],
            dtype=np.float32,
        )
        return state

    def _collect_kpis(self, cell) -> Dict[str, Dict]:
        max_prb = max(0.0, float(getattr(cell, "max_dl_prb", 0.0) or 0.0))
        alloc_dl = float(getattr(cell, "allocated_dl_prb", 0.0) or 0.0)
        free_prb = max(0.0, max_prb - alloc_dl)
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        alloc_by_slice = dict(getattr(cell, "allocated_dl_prb_by_slice", {}) or {})
        req_map = getattr(cell, "dl_total_prb_demand", {}) or {}
        kpis = {SL_E: self._blank_slice(quotas.get(SL_E, 0.0), SL_E),
                SL_U: self._blank_slice(quotas.get(SL_U, 0.0), SL_U)}
        for ue in getattr(cell, "connected_ue_list", {}).values():
            sl = getattr(ue, "slice_type", None)
            if sl not in kpis:
                continue
            data = kpis[sl]
            data["tx_mbps"] += float(getattr(ue, "served_downlink_bitrate", 0.0) or 0.0) / 1e6
            data["buffer_bytes"] += float(getattr(ue, "dl_buffer_bytes", 0.0) or 0.0)
            lat = float(getattr(ue, "downlink_latency", 0.0) or 0.0)
            data["latencies"].append(lat)
            imsi = getattr(ue, "ue_imsi", None)
            if imsi:
                data["prb_demand"] += float(req_map.get(imsi, 0.0) or 0.0)
        for sl, data in kpis.items():
            lats = data["latencies"]
            data["latency_mean_s"] = float(np.mean(lats)) if lats else 0.0
            data["prb_alloc"] = float(alloc_by_slice.get(sl, 0.0) or 0.0)
            data.pop("latencies", None)
        return {
            "slices": kpis,
            "pool": {"max_prb": max_prb, "allocated_prb": alloc_dl, "free_prb": free_prb},
        }

    def _compute_reward(self, state: np.ndarray) -> float:
        dec = self._split_state(state)
        embb = dec[SL_E]
        urllc = dec[SL_U]
        embb_buf_mb = embb["buf"] / 1e6
        url_buf_mb = urllc["buf"] / 1e6
        score_embb = embb["prb_alloc"] - embb_buf_mb
        score_urllc = -(urllc["lat_mean"] + url_buf_mb)
        return self.w_embb * score_embb + self.w_urllc * score_urllc

    def _split_state(self, v: np.ndarray) -> Dict[str, Dict[str, float]]:
        v = v.tolist()
        return {
            SL_E: {
                "buf": v[0],
                "tx": v[1],
                "lat_mean": v[2],
                "prb_quota": v[3],
                "prb_alloc": v[4],
            },
            SL_U: {
                "buf": v[5],
                "tx": v[6],
                "lat_mean": v[7],
                "prb_quota": v[8],
                "prb_alloc": v[9],
            },
            "pool": {"max": v[10], "free": v[11]},
        }

    # ---------- Actions ---------- #
    def _apply_action(self, action_idx: int):
        if action_idx < 0 or action_idx >= len(self._action_combos):
            return
        combo = self._action_combos[action_idx]
        for cell in self.ric.cell_list.values():
            quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
            for sl in (SL_E, SL_U):
                quotas.setdefault(sl, 0)
            max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
            if max_prb <= 0:
                continue
            # Remove PRBs for negative moves first
            for sl, delta in zip((SL_E, SL_U), combo):
                if delta < 0:
                    reduce = min(abs(delta), quotas[sl])
                    quotas[sl] -= reduce
            # Apply positive moves
            for sl, delta in zip((SL_E, SL_U), combo):
                if delta > 0:
                    quotas[sl] += delta
            total = sum(quotas.values())
            if total > max_prb:
                scale = max_prb / float(max(1, total))
                for sl in quotas:
                    quotas[sl] = int(quotas[sl] * scale)
            cell.slice_dl_prb_quota = {k: int(max(0, v)) for k, v in quotas.items()}

    # ---------- Helpers ---------- #
    def _load_episode_specs(self) -> List[Dict]:
        path = getattr(settings, "RES_GYM_CONFIG_PATH", "").strip()
        if not path:
            return []
        try:
            with open(path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
        except Exception as exc:
            print(f"ResilientGymEnv: failed to load episode config {path}: {exc}")
            return []
        entries = raw.get("episodes", raw) if isinstance(raw, dict) else raw
        specs = []
        for idx, entry in enumerate(entries):
            spec = self._normalize_episode(entry, idx)
            if spec:
                specs.append(spec)
        if bool(getattr(settings, "RES_GYM_SHUFFLE_EPISODES", False)) and specs:
            random.shuffle(specs)
        return specs

    def _normalize_episode(self, entry, idx: int) -> Optional[Dict]:
        if not isinstance(entry, dict):
            return None
        duration = int(entry.get("duration_steps", 0))
        if duration <= 0:
            print(f"ResilientGymEnv: episode {idx} missing duration_steps; skipping.")
            return None
        init_prb = entry.get("initial_prb", {})
        slices = entry.get("slices") or {}
        e_cfg = self._normalize_slice_cfg(slices.get(SL_E) or slices.get("eMBB"))
        u_cfg = self._normalize_slice_cfg(slices.get(SL_U) or slices.get("URLLC"))
        jam = entry.get("jamming", [])
        return {
            "id": entry.get("id") or f"episode_{idx}",
            "duration_steps": duration,
            "freeze_mobility": bool(entry.get("freeze_mobility", True)),
            "initial_prb": {
                SL_E: int(init_prb.get(SL_E, init_prb.get("eMBB", 0))),
                SL_U: int(init_prb.get(SL_U, init_prb.get("URLLC", 0))),
            },
            "slices": {
                SL_E: e_cfg,
                SL_U: u_cfg,
            },
            "jamming": jam if isinstance(jam, list) else [],
        }

    def _normalize_slice_cfg(self, cfg) -> Optional[Dict]:
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            return None
        count = int(cfg.get("ue_count", 0))
        if count <= 0:
            return None
        trace_speed = float(cfg.get("trace_speedup", getattr(settings, "TRACE_SPEEDUP", 1.0)))
        trace_bin = float(cfg.get("trace_bin", getattr(settings, "TRACE_BIN", 1.0)))
        if getattr(settings, "TRACE_BIN_OVERRIDE", False):
            trace_bin = float(getattr(settings, "TRACE_BIN", trace_bin))
        return {
            "ue_count": count,
            "trace": cfg.get("trace"),
            "ue_ip": cfg.get("ue_ip"),
            "trace_speedup": trace_speed,
            "trace_bin": trace_bin,
        }

    def _blank_slice(self, prb: float, name: str) -> Dict[str, float]:
        return {
            "prb_quota": float(prb),
            "prb_alloc": 0.0,
            "tx_mbps": 0.0,
            "prb_demand": 0.0,
            "buffer_bytes": 0.0,
            "latencies": [],
            "latency_mean_s": 0.0,
            "latency_target": self.latency_targets.get(name, 0.1),
        }

    def _init_latency_targets(self) -> Dict[str, float]:
        slices_cfg = getattr(settings, "NETWORK_SLICES", {})

        def _target(name: str, default: float) -> float:
            try:
                return float(slices_cfg.get(name, {}).get("latency_dl", default) or default)
            except Exception:
                return default

        return {SL_E: _target(SL_E, 0.1), SL_U: _target(SL_U, 0.001)}


# --------------------------- xApp wrapper --------------------------- #
class xAppResilientGymAllocator(xAppBase):
    """xApp that trains/evaluates DQN or QR-DQN on the resilient KPI state."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = bool(getattr(settings, "RES_GYM_ENABLE", False))
        self.train = bool(getattr(settings, "RES_GYM_TRAIN", True))
        self.agent_type = (getattr(settings, "RES_GYM_AGENT", "dqn") or "dqn").lower()
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.eps_start = float(getattr(settings, "RES_GYM_EPS_START", 1.0))
        self.eps_end = float(getattr(settings, "RES_GYM_EPS_END", 0.05))
        self.eps_decay = int(getattr(settings, "RES_GYM_EPS_DECAY_STEPS", 20000))
        self.lr = float(getattr(settings, "RES_GYM_LR", 1e-3))
        self.gamma = float(getattr(settings, "RES_GYM_GAMMA", 0.99))
        self.batch = int(getattr(settings, "RES_GYM_BATCH", 64))
        self.buffer_cap = int(getattr(settings, "RES_GYM_BUFFER", 100000))
        self.target_update = int(getattr(settings, "RES_GYM_TARGET_UPDATE", 2000))
        self.qr_atoms = int(getattr(settings, "RES_GYM_QR_ATOMS", 51))
        # PPO-specific knobs
        self.ppo_clip = float(getattr(settings, "RES_GYM_PPO_CLIP", 0.2))
        self.ppo_epochs = int(getattr(settings, "RES_GYM_PPO_EPOCHS", 4))
        self.ppo_rollout = int(getattr(settings, "RES_GYM_PPO_ROLLOUT", 256))
        self.ppo_ent_coef = float(getattr(settings, "RES_GYM_PPO_ENT_COEF", 0.0))
        self.ppo_vf_coef = float(getattr(settings, "RES_GYM_PPO_VF_COEF", 0.5))
        self.ppo_gae_lambda = float(getattr(settings, "RES_GYM_PPO_GAE_LAMBDA", 0.95))
        # PF heuristic knobs
        self.pf_lat_w = float(getattr(settings, "RES_GYM_PF_LAT_W", 1.0))
        self.pf_buf_w = float(getattr(settings, "RES_GYM_PF_BUF_W", 0.1))
        # Debug logging (CSV) to inspect state/action/reward per step
        self.debug_log_path = (getattr(settings, "RES_GYM_DEBUG_LOG", "") or "").strip()
        self._dbg_fp = None
        self._dbg_csv = None
        self._dbg_step = 0
        self.tb_logdir = (getattr(settings, "RES_GYM_TB_LOGDIR", "") or "").strip()
        self._tb_writer = None
        # Live dashboard
        self.dash_enable = bool(getattr(settings, "RES_GYM_DASH_ENABLE", False))
        self.dash_port = int(getattr(settings, "RES_GYM_DASH_PORT", 5055))
        self._dash_app = None
        self._dash_thread = None
        self._dash_snapshot = {}
        self._dash_lock = threading.Lock()
        self._global_step = 0
        self._eps = self.eps_start
        self._last_state: Optional[np.ndarray] = None
        self._env: Optional[ResilientPRBGymEnv] = None
        self._policy = None
        self._target = None
        self._optimizer = None
        self._replay = None
        # PPO rollout storage
        self._ppo_storage = {"s": [], "a": [], "logp": [], "v": [], "r": [], "d": []}

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        if not TORCH_AVAILABLE and self.agent_type in ("dqn", "qrdqn", "ppo"):
            print(f"{self.xapp_id}: torch not available; disabling.")
            self.enabled = False
            return
        self._env = ResilientPRBGymEnv(self.ric)
        self._init_agent()
        self._last_state = self._env.reset()
        # Publish initial snapshot so dashboard shows data before first step
        self._update_snapshot(self._last_state, None, reward=0.0, done=False)
        self._init_debug_logger()
        self._init_tb_logger()
        self._start_dashboard()
        print(
            f"{self.xapp_id}: enabled agent={self.agent_type} train={self.train} "
            f"actions={len(self._env.actions)} state_dim=12"
        )

    def step(self):
        if not self.enabled or self._env is None:
            return
        self._global_step += 1
        if self.agent_type == "ppo":
            self._step_ppo()
            return

        # DQN / QR-DQN path (epsilon decay only for value-based agents)
        if self.agent_type in ("dqn", "qrdqn"):
            frac = max(0.0, (self.eps_decay - self._global_step) / float(max(1, self.eps_decay)))
            self._eps = self.eps_end + (self.eps_start - self.eps_end) * frac

        if self._env._steps_in_episode % self._env.decision_period != 0:
            return
        if self._last_state is None:
            self._last_state = self._env.reset()
            return

        action_idx = self._select_action(self._last_state)
        next_state, reward, done, _ = self._env.step(action_idx)
        self._log_debug(self._last_state, action_idx, reward, done)
        self._update_snapshot(self._last_state, action_idx, reward, done)
        if self.train and self._replay is not None:
            self._replay.push(self._last_state, action_idx, reward, next_state, float(done))
            self._learn()
        self._tb_log(self._last_state, action_idx, reward, done)
        self._last_state = None if done else next_state
        if done:
            self._last_state = self._env.reset()

    # ---------- Agent helpers ---------- #
    def _init_agent(self):
        env = self._env
        if env is None:
            return
        in_dim = env._state_dim
        out_dim = len(env.actions)
        if self.agent_type == "qrdqn":
            self._policy = _QRDQN(in_dim, out_dim, n_atoms=self.qr_atoms).to(self.device)
            self._target = _QRDQN(in_dim, out_dim, n_atoms=self.qr_atoms).to(self.device)
        elif self.agent_type == "dqn":
            self._policy = _DQN(in_dim, out_dim).to(self.device)
            self._target = _DQN(in_dim, out_dim).to(self.device)
        elif self.agent_type == "ppo":
            self._policy = _ActorCritic(in_dim, out_dim).to(self.device)
            self._target = None
        else:  # proportional fair heuristic: no nets
            self._policy = None
            self._target = None
        if self.agent_type in ("dqn", "qrdqn"):
            self._target.load_state_dict(self._policy.state_dict())
        if self.agent_type in ("dqn", "qrdqn", "ppo"):
            self._optimizer = optim.Adam(self._policy.parameters(), lr=self.lr)
        if self.agent_type in ("dqn", "qrdqn"):
            self._replay = _ReplayBuffer(capacity=self.buffer_cap)

    def _select_action(self, state: np.ndarray) -> int:
        if self.agent_type == "pf":
            return self._pf_action(state)
        if self.agent_type == "ppo":
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.tensor(self._action_mask(state), device=self.device)
            # Mask invalid actions by setting logits to -inf
            logits, value = self._policy(s)
            logits = logits.masked_fill(~mask, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            if self.train:
                action = dist.sample()
            else:
                action = torch.argmax(dist.probs, dim=1)
            logp = dist.log_prob(action)
            self._stash_ppo_step(state, int(action.item()), float(logp.item()), float(value.item()))
            return int(action.item())

        valid = self._action_mask(state)
        valid_idx = [i for i, ok in enumerate(valid) if ok]
        if not valid_idx:
            return 0
            if self.train and random.random() < self._eps:
                return random.choice(valid_idx)
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.agent_type == "qrdqn":
                q = self._policy(s)  # [1, actions, atoms]
                q_mean = q.mean(dim=2)
                q_mean = q_mean.cpu().numpy().flatten()
            else:
                q = self._policy(s)
                q_mean = q.cpu().numpy().flatten()
        # Mask invalid by setting to -inf
        for i, ok in enumerate(valid):
            if not ok:
                q_mean[i] = -1e9
        return int(np.argmax(q_mean))

    def _learn(self):
        if self._replay is None or len(self._replay) < self.batch:
            return
        states, actions, rewards, next_states, dones = self._replay.sample(self.batch)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        if self.agent_type == "qrdqn":
            self._learn_qrdqn(states, actions, rewards, next_states, dones)
        else:
            self._learn_dqn(states, actions, rewards, next_states, dones)

        if self._global_step % self.target_update == 0:
            self._target.load_state_dict(self._policy.state_dict())

    # ---------- PPO branch ---------- #
    def _stash_ppo_step(self, state, action, logp, value):
        self._ppo_storage["s"].append(state)
        self._ppo_storage["a"].append(action)
        self._ppo_storage["logp"].append(logp)
        self._ppo_storage["v"].append(value)

    def _step_ppo(self):
        if self._env._steps_in_episode % self._env.decision_period != 0:
            return
        if self._last_state is None:
            self._last_state = self._env.reset()
            return
        action_idx = self._select_action(self._last_state)
        next_state, reward, done, _ = self._env.step(action_idx)
        self._log_debug(self._last_state, action_idx, reward, done)
        # Stash reward/done for last transition
        self._ppo_storage["r"].append(reward)
        self._ppo_storage["d"].append(float(done))
        self._tb_log(self._last_state, action_idx, reward, done)
        self._update_snapshot(self._last_state, action_idx, reward, done)

        if self.train and len(self._ppo_storage["r"]) >= self.ppo_rollout:
            self._ppo_update(next_state if not done else None)

        self._last_state = None if done else next_state
        if done:
            # Flush remaining partial rollout
            if self.train and self._ppo_storage["r"]:
                self._ppo_update(None)
            self._last_state = self._env.reset()

    def _ppo_update(self, next_state: Optional[np.ndarray]):
        device = self.device
        states = torch.tensor(np.asarray(self._ppo_storage["s"]), dtype=torch.float32, device=device)
        actions = torch.tensor(self._ppo_storage["a"], dtype=torch.long, device=device)
        old_logp = torch.tensor(self._ppo_storage["logp"], dtype=torch.float32, device=device)
        values = torch.tensor(self._ppo_storage["v"], dtype=torch.float32, device=device)
        rewards = torch.tensor(self._ppo_storage["r"], dtype=torch.float32, device=device)
        dones = torch.tensor(self._ppo_storage["d"], dtype=torch.float32, device=device)

        # Bootstrap
        with torch.no_grad():
            next_val = 0.0
            if next_state is not None:
                ns = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                _, nv = self._policy(ns)
                next_val = nv.squeeze(0)
            adv, returns = self._gae(rewards, values, dones, next_val)

        for _ in range(self.ppo_epochs):
            logits, v_pred = self._policy(states)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * adv
            policy_loss = -(torch.min(ratio * adv, clipped)).mean()
            value_loss = F.mse_loss(v_pred, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.ppo_vf_coef * value_loss - self.ppo_ent_coef * entropy
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        # Clear storage
        for k in self._ppo_storage:
            self._ppo_storage[k] = []

    def _gae(self, rewards, values, dones, next_value):
        gae = 0.0
        adv = []
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.ppo_gae_lambda * mask * gae
            adv.insert(0, gae)
            next_value = values[t]
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # ---------- Debug logging ---------- #
    def _init_debug_logger(self):
        if not self.debug_log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True) if os.path.dirname(self.debug_log_path) else None
            self._dbg_fp = open(self.debug_log_path, "w", newline="")
            self._dbg_csv = csv.DictWriter(
                self._dbg_fp,
                fieldnames=[
                    "step",
                    "episode_idx",
                    "action_idx",
                    "reward",
                    "done",
                    "state",
                ],
            )
            self._dbg_csv.writeheader()
            self._dbg_step = 0
            print(f"{self.xapp_id}: debug logging -> {self.debug_log_path}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to init debug logger {self.debug_log_path}: {exc}")
            self._dbg_fp = None
            self._dbg_csv = None

    def _log_debug(self, state: Optional[np.ndarray], action_idx: Optional[int], reward: float, done: bool):
        if self._dbg_csv is None or state is None:
            return
        try:
            self._dbg_step += 1
            self._dbg_csv.writerow(
                {
                    "step": self._dbg_step,
                    "episode_idx": self._env._episode_idx if self._env else -1,
                    "action_idx": action_idx,
                    "reward": reward,
                    "done": done,
                    "state": ";".join(f"{float(x):.4f}" for x in state.tolist()),
                }
            )
            if self._dbg_fp:
                self._dbg_fp.flush()
        except Exception:
            pass

    def _init_tb_logger(self):
        if not self.tb_logdir or SummaryWriter is None:
            return
        try:
            self._tb_writer = SummaryWriter(log_dir=self.tb_logdir)
            print(f"{self.xapp_id}: tensorboard logging -> {self.tb_logdir}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to init tensorboard logger {self.tb_logdir}: {exc}")
            self._tb_writer = None

    def _tb_log(self, state: Optional[np.ndarray], action_idx: Optional[int], reward: float, done: bool):
        if self._tb_writer is None or state is None:
            return
        step = self._global_step
        try:
            self._tb_writer.add_scalar("reward", reward, step)
            self._tb_writer.add_scalar("done", int(done), step)
            if action_idx is not None:
                self._tb_writer.add_scalar("action_idx", action_idx, step)
            # Log key state features separately for readability
            s = state.tolist()
            self._tb_writer.add_scalar("embb/buffer_bytes", s[0], step)
            self._tb_writer.add_scalar("embb/tx_mbps", s[1], step)
            self._tb_writer.add_scalar("embb/lat_mean_s", s[2], step)
            self._tb_writer.add_scalar("embb/prb_quota", s[3], step)
            self._tb_writer.add_scalar("embb/prb_alloc", s[4], step)
            self._tb_writer.add_scalar("urllc/buffer_bytes", s[5], step)
            self._tb_writer.add_scalar("urllc/tx_mbps", s[6], step)
            self._tb_writer.add_scalar("urllc/lat_mean_s", s[7], step)
            self._tb_writer.add_scalar("urllc/prb_quota", s[8], step)
            self._tb_writer.add_scalar("urllc/prb_alloc", s[9], step)
            self._tb_writer.add_scalar("pool/max_prb", s[10], step)
            self._tb_writer.add_scalar("pool/free_prb", s[11], step)
            sat = self._compute_satisfaction()
            if sat:
                for sl, val in sat.items():
                    self._tb_writer.add_scalar(f"satisfaction/{sl}", val, step)
            self._tb_writer.flush()
        except Exception:
            pass

    # ---------- Live dashboard (Flask) ---------- #
    def _start_dashboard(self):
        if not self.dash_enable:
            return
        if not FLASK_AVAILABLE:
            print(f"{self.xapp_id}: Flask not available; dashboard disabled.")
            return
        try:
            app = Flask(f"{self.xapp_id}_dash")

            @app.route("/snapshot")
            def snapshot():
                with self._dash_lock:
                    data = dict(self._dash_snapshot)
                return jsonify(data)

            @app.route("/")
            def index():
                return render_template_string(
                    """
<!doctype html>
<html>
<head><title>Resilient Gym Dashboard</title></head>
<body style="font-family:Arial, sans-serif; padding:16px;">
  <h3>Resilient Gym Snapshot</h3>
  <pre id="data">Loading...</pre>
  <script>
    async function refresh() {
      try {
        const res = await fetch('/snapshot');
        const json = await res.json();
        document.getElementById('data').textContent = JSON.stringify(json, null, 2);
      } catch (e) {
        document.getElementById('data').textContent = 'Error fetching snapshot';
      }
    }
    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
                    """
                )

            def _run():
                app.run(host="0.0.0.0", port=self.dash_port, debug=False, use_reloader=False)

            self._dash_app = app
            self._dash_thread = threading.Thread(target=_run, daemon=True)
            self._dash_thread.start()
            print(f"{self.xapp_id}: dashboard running on port {self.dash_port}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to start dashboard: {exc}")

    def _update_snapshot(self, state: Optional[np.ndarray], action_idx: Optional[int], reward: float, done: bool):
        if not self.dash_enable or state is None:
            return
        snap = {
            "step": self._global_step,
            "episode_idx": self._env._episode_idx if self._env else -1,
            "action_idx": action_idx,
            "reward": reward,
            "done": done,
            "state": state.tolist(),
        }
        sat = self._compute_satisfaction()
        if sat:
            snap["satisfaction"] = sat
        with self._dash_lock:
            self._dash_snapshot = snap

    def _compute_satisfaction(self) -> Optional[Dict[str, float]]:
        """Compute PRB satisfaction per slice: alloc / demand (clamped to [0,1])."""
        kpis = getattr(self._env, "_last_kpis", None) if hasattr(self, "_env") else None
        if not kpis:
            return None
        sat = {}
        slices = kpis.get("slices", {})
        for sl, data in slices.items():
            demand = float(data.get("prb_demand", 0.0) or 0.0)
            alloc = float(data.get("prb_alloc", 0.0) or 0.0)
            if demand <= 0:
                sat[sl] = 1.0  # no demand -> fully satisfied
            else:
                sat[sl] = max(0.0, min(1.0, alloc / demand))
        return sat

    # ---------- Proportional-fair heuristic (stateless) ---------- #
    def _pf_action(self, state: np.ndarray) -> int:
        """Pick action that maximizes log-sum utility with latency/buffer penalties."""
        max_prb = max(1e-6, float(state[10]))
        buf_e, buf_u = float(state[0]), float(state[5])
        lat_u = float(state[7])
        alloc_e, alloc_u = float(state[4]), float(state[9])
        best_idx = 0
        best_score = -1e9
        valid = self._action_mask(state)
        for idx, combo in enumerate(self._env.actions):
            if not valid[idx]:
                continue
            de, du = combo
            new_e = max(0.0, alloc_e + de)
            new_u = max(0.0, alloc_u + du)
            total = new_e + new_u
            if total > max_prb:
                scale = max_prb / float(max(1e-6, total))
                new_e *= scale
                new_u *= scale
            util = np.log1p(max(new_e, 0.0)) + np.log1p(max(new_u, 0.0))
            util -= self.pf_lat_w * lat_u
            util -= self.pf_buf_w * ((buf_e + buf_u) / max(1.0, self._env.norm_buf_bytes))
            if util > best_score:
                best_score = util
                best_idx = idx
        return best_idx

    def _action_mask(self, state: np.ndarray) -> List[bool]:
        """Mask actions that would drive quotas <0 or overflow total PRB budget."""
        qe = float(state[3])
        qu = float(state[8])
        max_prb = float(state[10])
        mask = []
        for de, du in self._action_combos:
            new_e = qe + de
            new_u = qu + du
            if new_e < 0 or new_u < 0:
                mask.append(False)
                continue
            if new_e + new_u > max_prb + 1e-9:
                mask.append(False)
                continue
            mask.append(True)
        return mask

    def _learn_dqn(self, states, actions, rewards, next_states, dones):
        q_pred = self._policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self._target(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(q_pred, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _learn_qrdqn(self, states, actions, rewards, next_states, dones):
        n_atoms = self._policy.n_atoms
        with torch.no_grad():
            next_q = self._policy(next_states).mean(dim=2)
            next_actions = torch.argmax(next_q, dim=1)
            next_dist = self._target(next_states)[range(len(next_states)), next_actions]
            target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_dist
        dist = self._policy(states)[range(len(states)), actions]
        # Quantile Huber loss
        tau = torch.linspace(0.0, 1.0, steps=n_atoms + 1, device=self.device)[1:]
        diff = target.unsqueeze(1) - dist.unsqueeze(2)
        huber = torch.where(diff.abs() < 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
        loss = (torch.abs(tau.unsqueeze(0) - (diff.detach() < 0).float()) * huber).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
