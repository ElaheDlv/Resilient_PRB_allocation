"""Validation harness to replay different resiliency policies with the shared dashboard.

Supported modes:
 - QR-DQN: load a trained QR-DQN checkpoint and act greedily.
 - DQN: load a trained DQN checkpoint and act greedily.
 - random: sample any valid action uniformly.
 - idle: never borrow/return resiliency PRBs (keeps baseline quotas).

The dashboard mirrors the allocator xApps so KPI plots stay consistent.
"""

from __future__ import annotations

import random
import threading
from collections import defaultdict, deque
from typing import Dict, Optional
import os
from datetime import datetime

import numpy as np

import settings
from .xapp_base import xAppBase
from .xapp_resilient_dqn_allocator import (
    ACTION_LIST,
    NUM_ACTIONS,
    SL_E,
    SL_U,
    TOTAL_RESILIENCY_PRB,
    EMBB_THR_MAX,
    EMBB_QUEUE_MAX,
    URLLC_THR_MAX,
    URLLC_QUEUE_MAX,
    URLLC_LATENCY_DEADLINE,
    build_state,
    ResilientDQNEnv,
    _DQN,
)
from .xapp_resilient_qrdqn_allocator import _QRDQN

try:
    import torch
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None
except Exception:
    torch = None  # type: ignore
    SummaryWriter = None

TORCH_AVAILABLE = torch is not None

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


class xAppResilientPolicyValidator(xAppBase):
    """Headless validator that replays a chosen policy with the shared plots."""

    def __init__(self, ric=None):
        super().__init__(ric=ric)
        self.enabled = bool(getattr(settings, "RES_VALIDATE_ENABLE", False))
        self.policy_mode = (getattr(settings, "RES_VALIDATE_POLICY", "random") or "random").lower()
        if self.policy_mode not in ("qrdqn", "dqn", "random", "idle"):
            self.policy_mode = "random"
        seed_raw = getattr(settings, "RES_VALIDATE_SEED", None)
        try:
            self.seed = int(seed_raw) if seed_raw not in (None, "") else None
        except Exception:
            print(f"{self.xapp_id}: invalid RES_VALIDATE_SEED={seed_raw}; ignoring.")
            self.seed = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._decision_period = max(
            1,
            int(
                getattr(
                    settings,
                    "RES_VALIDATE_DECISION_PERIOD",
                    getattr(settings, "RES_QRDQN_DECISION_PERIOD", getattr(settings, "RES_DQN_DECISION_PERIOD", 1)),
                )
            ),
        )
        self.qrdqn_checkpoint = (getattr(settings, "RES_VALIDATE_QRDQN_MODEL", "") or "").strip()
        self.dqn_checkpoint = (getattr(settings, "RES_VALIDATE_DQN_MODEL", "") or "").strip()
        self.dash_enable = bool(
            getattr(settings, "RES_VALIDATE_DASH_ENABLE", getattr(settings, "RES_QRDQN_DASH_ENABLE", False))
        )
        self.dash_port = int(getattr(settings, "RES_VALIDATE_DASH_PORT", getattr(settings, "RES_QRDQN_DASH_PORT", 5060)))
        self.tb_enable = bool(getattr(settings, "RES_VALIDATE_TB_ENABLE", False))
        self.tb_dir = (getattr(settings, "RES_VALIDATE_TB_DIR", "") or "").strip()
        self._env: Optional[ResilientDQNEnv] = None
        self._policy = None
        self._last_state = None
        self._episode_done = False
        self._global_step = 0
        self._idle_idx = next((i for i, a in enumerate(ACTION_LIST) if a == (0, 0)), 0)
        self._dash_app = None
        self._dash_thread = None
        self._lock = threading.Lock()
        self._last_info: Dict = {}
        # History buffers for dashboard plots
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
        self._hist_action_idx = deque(maxlen=400)
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
        self._tb_writer = None
        self._episode_return = 0.0
        self._episode_len = 0
        self._episode_counter = 0

    def _init_tb(self):
        if not self.tb_enable:
            return
        if SummaryWriter is None:
            print(f"{self.xapp_id}: tensorboard not available; disable RES_VALIDATE_TB_ENABLE to silence.")
            return
        model_path = self.qrdqn_checkpoint if self.policy_mode == "qrdqn" else self.dqn_checkpoint
        model_tag = os.path.splitext(os.path.basename(model_path))[0] if model_path else self.policy_mode
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jam_start = getattr(settings, "JAM_CONST_START_OFFSET", None)
        jam_prbs = getattr(settings, "JAM_RESERVED_PRBS", None)
        jam_const_prb = getattr(settings, "JAM_CONST_PRBS", None)
        jam_period = getattr(settings, "JAM_CONST_PERIOD_STEPS", None)
        jam_duration = getattr(settings, "JAM_CONST_DURATION_STEPS", None)
        jam_parts = [
            f"res{jam_prbs}" if jam_prbs is not None else "resNA",
            f"off{jam_start}" if jam_start is not None else "offNA",
            f"const{jam_const_prb}" if jam_const_prb is not None else "constNA",
            f"per{jam_period}" if jam_period is not None else "perNA",
            f"dur{jam_duration}" if jam_duration is not None else "durNA",
        ]
        jam_tag = "_".join(jam_parts)
        base_dir = self.tb_dir or "runs/res_validator"
        logdir = os.path.join(base_dir, f"{model_tag}__{ts}__{jam_tag}")
        try:
            self._tb_writer = SummaryWriter(logdir)
            print(f"{self.xapp_id}: tensorboard logging to {logdir}")
        except Exception as exc:
            print(f"{self.xapp_id}: failed to init tensorboard at {logdir}: {exc}")
            self._tb_writer = None

    def _apply_seed(self):
        if self.seed is None:
            return
        random.seed(self.seed)
        try:
            np.random.seed(self.seed)
        except Exception:
            pass
        if TORCH_AVAILABLE:
            try:
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
            except Exception:
                pass
        print(f"{self.xapp_id}: applied validation seed {self.seed}")

    def start(self):
        if not self.enabled:
            print(f"{self.xapp_id}: disabled")
            return
        self._apply_seed()
        self._env = ResilientDQNEnv(self.ric, spawn_seed=self.seed)
        self._env.register_clear_hook(self._clear_plot_caches)
        self._init_policy()
        self._last_state = self._env.reset()
        if self.dash_enable:
            self._start_dashboard()
        self._init_tb()
        print(
            f"{self.xapp_id}: mode={self.policy_mode} actions={NUM_ACTIONS} state_dim={len(self._last_state)}"
        )

    def step(self):
        if not self.enabled or self._env is None:
            return
        self._global_step += 1
        if self._global_step % self._decision_period != 0:
            return
        if self._last_state is None or self._episode_done:
            self._last_state = self._env.reset()
            self._episode_done = False
            return
        if hasattr(self._env, "ready") and not self._env.ready():
            return

        action_idx = self._select_action(self._last_state)
        next_state, reward, done, info = self._env.step(action_idx)
        self._episode_return += float(reward)
        self._episode_len += 1
        self._record_metrics(reward, info)
        self._last_state = next_state
        self._episode_done = bool(done)
        if done:
            self._tb_log_episode()
            self._episode_counter += 1
            self._episode_return = 0.0
            self._episode_len = 0

    # ---------- Policy selection ---------- #
    def _init_policy(self):
        if self.policy_mode in ("qrdqn", "dqn") and not TORCH_AVAILABLE:
            print(f"{self.xapp_id}: torch not available; falling back to random policy.")
            self.policy_mode = "random"
            return
        in_dim = len(build_state(self._env._blank_obs(), (0, 0)))  # type: ignore[arg-type]
        if self.policy_mode == "qrdqn":
            atoms = int(getattr(settings, "RES_QRDQN_ATOMS", 51))
            self._policy = _QRDQN(in_dim, NUM_ACTIONS, n_atoms=atoms).to(self.device)
            self._policy.eval()
            if not self._load_checkpoint(self._policy, self.qrdqn_checkpoint) and not self.qrdqn_checkpoint:
                print(f"{self.xapp_id}: no QR-DQN checkpoint provided; using random weights.")
        elif self.policy_mode == "dqn":
            if _DQN is None:
                print(f"{self.xapp_id}: DQN model unavailable; falling back to random policy.")
                self.policy_mode = "random"
                return
            self._policy = _DQN(in_dim, NUM_ACTIONS).to(self.device)
            self._policy.eval()
            if not self._load_checkpoint(self._policy, self.dqn_checkpoint) and not self.dqn_checkpoint:
                print(f"{self.xapp_id}: no DQN checkpoint provided; using random weights.")

    def _load_checkpoint(self, model, path: str) -> bool:
        if not path:
            return False
        try:
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            print(f"{self.xapp_id}: loaded checkpoint {path}")
            return True
        except Exception as exc:
            msg = f"{self.xapp_id}: failed to load checkpoint {path}: {exc}"
            print(msg)
            raise RuntimeError(msg) from exc

    def _select_action(self, state) -> int:
        mask = getattr(self._env, "_action_mask", lambda: [True] * NUM_ACTIONS)()
        valid_idx = [i for i, ok in enumerate(mask) if ok]
        if not valid_idx:
            return self._idle_idx
        if self.policy_mode == "idle":
            return self._idle_idx if mask[self._idle_idx] else valid_idx[0]
        if self.policy_mode == "random" or self._policy is None:
            return int(random.choice(valid_idx))
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.policy_mode == "dqn":
                q = self._policy(s)  # type: ignore[call-arg, operator]
            else:
                q = self._policy(s).mean(dim=2)  # type: ignore[call-arg]
            q_np = q.cpu().numpy().flatten()
            for i, ok in enumerate(mask):
                if not ok:
                    q_np[i] = -1e9
            idx = int(np.argmax(q_np))
        return idx if mask[idx] else int(random.choice(valid_idx))

    # ---------- Telemetry / dashboard ---------- #
    def _record_metrics(self, reward: float, info: Optional[Dict]):
        if info is None:
            return
        obs = info.get("obs") if isinstance(info, dict) else {}
        action = info.get("action") if isinstance(info, dict) else (0, 0)
        action_idx = info.get("action_idx") if isinstance(info, dict) else None
        with self._lock:
            t = self._global_step
            self._last_info = info or {}
            self._t.append(t)
            self._hist_reward.append(float(reward))
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
        # Snapshot state vector and slice satisfaction for TensorBoard
        prev_action = info.get("prev_action", (0, 0)) if isinstance(info, dict) else (0, 0)
        try:
            state_vec = build_state(obs, prev_action)
        except Exception:
            state_vec = None
        sat_map = {}
        for sl in (SL_E, SL_U):
            if self._slice_sat.get(sl):
                sat_map.setdefault(sl, {})["capped"] = float(self._slice_sat[sl][-1])
            if self._slice_sat_raw.get(sl):
                sat_map.setdefault(sl, {})["raw"] = float(self._slice_sat_raw[sl][-1])
        self._tb_log(t, reward, obs, action_idx, action, state_vec, sat_map)

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
                html.H3("Resilient Policy Validator"),
                html.Pre(id="snapshot", style={"whiteSpace": "pre-wrap"}),
                dcc.Graph(id="action-graph"),
                dcc.Graph(id="prb-graph"),
                dcc.Graph(id="throughput-graph"),
                dcc.Graph(id="lat-graph"),
                dcc.Graph(id="reward-graph"),
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
                reserve = list(self._hist_reserve)
                prb_ratio = list(self._hist_prb_ratio)
                action_idx = list(self._hist_action_idx)
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
            action_fig.update_layout(title="Actions", xaxis_title="step")

            prb_fig = go.Figure()
            for sl, series in slice_prb.items():
                prb_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} PRB"))
            prb_fig.update_layout(title="Slice PRB quotas", xaxis_title="step")

            thr_fig = go.Figure()
            thr_fig.add_trace(go.Scatter(x=t, y=[v / 1e6 for v in thr_e], mode="lines", name="eMBB throughput (Mbps)"))
            thr_fig.add_trace(go.Scatter(x=t, y=[v / 1e6 for v in thr_u], mode="lines", name="URLLC throughput (Mbps)"))
            thr_fig.update_layout(title="Throughput", xaxis_title="step", yaxis_title="Mbps")

            lat_fig = go.Figure()
            lat_fig.add_trace(go.Scatter(x=t, y=lat, mode="lines", name="URLLC latency (s)"))
            lat_fig.add_trace(go.Scatter(x=t, y=deadline, mode="lines", name="deadline", line=dict(dash="dash")))
            lat_fig.update_layout(title="Latency", xaxis_title="step")

            reward_fig = go.Figure()
            reward_fig.add_trace(go.Scatter(x=t, y=reward, mode="lines", name="reward"))
            reward_fig.update_layout(title="Reward", xaxis_title="step", yaxis_title="reward")

            reserve_fig = go.Figure()
            reserve_fig.add_trace(go.Scatter(x=t, y=reserve, mode="lines", name="reserve fraction"))
            reserve_fig.update_layout(title="Reserve fraction", xaxis_title="step", yaxis_title="fraction")

            ratio_fig = go.Figure()
            ratio_fig.add_trace(go.Scatter(x=t, y=prb_ratio, mode="lines", name="effective PRB ratio"))
            ratio_fig.update_layout(title="Effective PRB ratio", xaxis_title="step", yaxis_title="ratio")
            ue_thr_fig = go.Figure()
            for imsi, series in ue_tx.items():
                ue_thr_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} tx (Mbps)"))
            ue_thr_fig.update_layout(title="Per-UE throughput", xaxis_title="step", yaxis_title="Mbps", legend=dict(orientation="h"))

            ue_dem_fig = go.Figure()
            for imsi, series in ue_demand.items():
                ue_dem_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} prb demand"))
            ue_dem_fig.update_layout(title="Per-UE PRB demand", xaxis_title="step", yaxis_title="PRB", legend=dict(orientation="h"))

            ue_sat_fig = go.Figure()
            for imsi, series in ue_sat.items():
                ue_sat_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} sat"))
            ue_sat_fig.update_layout(title="Per-UE satisfaction", xaxis_title="step", yaxis_title="sat", legend=dict(orientation="h"))

            slice_sat_fig = go.Figure()
            for sl, series in slice_sat.items():
                slice_sat_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} sat capped"))
            slice_sat_fig.update_layout(title="Slice satisfaction (capped)", xaxis_title="step", yaxis_title="sat", legend=dict(orientation="h"))

            slice_sat_raw_fig = go.Figure()
            for sl, series in slice_sat_raw.items():
                slice_sat_raw_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{sl} sat raw"))
            slice_sat_raw_fig.update_layout(title="Slice satisfaction (raw)", xaxis_title="step", yaxis_title="sat raw", legend=dict(orientation="h"))

            sinr_fig = go.Figure()
            for imsi, series in ue_sinr.items():
                sinr_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} SINR (dB)"))
            sinr_fig.update_layout(title="Per-UE SINR", xaxis_title="step", yaxis_title="dB", legend=dict(orientation="h"))

            cqi_fig = go.Figure()
            for imsi, series in ue_cqi.items():
                cqi_fig.add_trace(go.Scatter(x=t[-len(series):], y=series, mode="lines", name=f"{imsi} CQI"))
            cqi_fig.update_layout(title="Per-UE CQI", xaxis_title="step", yaxis_title="CQI", legend=dict(orientation="h"))

            supply_fig = go.Figure()
            supply_fig.add_trace(go.Scatter(x=t[-len(cell_max):], y=cell_max, mode="lines", name="cell max PRB"))
            supply_fig.add_trace(go.Scatter(x=t[-len(cell_eff):], y=cell_eff, mode="lines", name="effective PRB"))
            supply_fig.update_layout(
                title="PRB supply (max vs effective)",
                xaxis_title="step",
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
                lines.append(
                    f"{sl}: PRB={prb:.0f}, Tx={tx:.3f} Mbps, Latency={lat:.3f}, "
                    f"UEs={ues}, Sat={cap_sat:.2f} (raw {raw_sat:.2f})"
                )
            total_ues = len(getattr(self.ric, "ue_list", {}) or {})
            ue_max = int(getattr(settings, "UE_DEFAULT_MAX_COUNT", 0) or 0)
            lines.append(f"UEs total={total_ues}, UE_max={ue_max}")
            # Jamming / effective PRB
            eff = getattr(cell_ref, "_effective_max_dl_prb", getattr(cell_ref, "max_dl_prb", 0))
            max_dl = getattr(cell_ref, "max_dl_prb", 0)
        if eff != max_dl:
            lines.append(f"Jamming: effective PRB={eff}/{max_dl}")
        return "\n".join(lines)

    def _tb_log(self, t, reward, obs, action_idx, action, state_vec=None, sat_map=None):
        if self._tb_writer is None:
            return
        try:
            self._tb_writer.add_scalar("reward/total", float(reward), global_step=t)
            self._tb_writer.add_scalar("action/index", float(action_idx) if action_idx is not None else -1, global_step=t)
            self._tb_writer.add_scalar("action/sum", float(sum(action) if action else 0.0), global_step=t)
            self._tb_writer.add_scalar(
                "throughput/embb_mbps", float(obs.get("embb_throughput", 0.0) or 0.0) / 1e6, global_step=t
            )
            self._tb_writer.add_scalar(
                "throughput/urllc_mbps", float(obs.get("urllc_throughput", 0.0) or 0.0) / 1e6, global_step=t
            )
            self._tb_writer.add_scalar("latency/urllc", float(obs.get("urllc_latency", 0.0) or 0.0), global_step=t)
            self._tb_writer.add_scalar("queues/embb_bytes", float(obs.get("embb_queue", 0.0) or 0.0), global_step=t)
            self._tb_writer.add_scalar("queues/urllc_bytes", float(obs.get("urllc_queue", 0.0) or 0.0), global_step=t)
            self._tb_writer.add_scalar(
                "prb/reserve_fraction",
                float(sum(action) if action else 0.0) / max(1.0, TOTAL_RESILIENCY_PRB),
                global_step=t,
            )
            self._tb_writer.add_scalar("prb/effective_ratio", float(obs.get("effective_prb_ratio", 0.0) or 0.0), global_step=t)
            # Log raw obs components useful for state reconstruction
            obs_keys = [
                "embb_demand_prb",
                "urllc_demand_prb",
                "embb_throughput",
                "urllc_throughput",
                "embb_queue",
                "urllc_queue",
                "embb_base_achievable_throughput",
                "urllc_base_achievable_throughput",
                "reserve_used_prb",
                "effective_max_prb",
            ]
            for k in obs_keys:
                if k in obs:
                    self._tb_writer.add_scalar(f"obs/{k}", float(obs.get(k, 0.0) or 0.0), global_step=t)
            # Log normalized state vector components
            if state_vec is not None:
                state_names = [
                    "embb_demand_norm",
                    "urllc_demand_norm",
                    "embb_queue_norm",
                    "urllc_queue_norm",
                    "embb_thr_norm",
                    "urllc_thr_norm",
                    "urllc_lat_excess_norm",
                    "effective_prb_ratio",
                    "reserve_used_norm",
                    "prev_action_e_norm",
                    "prev_action_u_norm",
                ]
                for idx, val in enumerate(state_vec):
                    name = state_names[idx] if idx < len(state_names) else f"state_{idx}"
                    self._tb_writer.add_scalar(f"state/{name}", float(val), global_step=t)
            # Slice satisfaction (capped and raw)
            if sat_map:
                for sl, parts in sat_map.items():
                    if "capped" in parts:
                        self._tb_writer.add_scalar(f"satisfaction/{sl}_capped", parts["capped"], global_step=t)
                    if "raw" in parts:
                        self._tb_writer.add_scalar(f"satisfaction/{sl}_raw", parts["raw"], global_step=t)
        except Exception:
            pass

    def _tb_log_episode(self):
        if self._tb_writer is None:
            return
        try:
            self._tb_writer.add_scalar("episode/return", float(self._episode_return), global_step=self._episode_counter)
            self._tb_writer.add_scalar("episode/length", float(self._episode_len), global_step=self._episode_counter)
        except Exception:
            pass
