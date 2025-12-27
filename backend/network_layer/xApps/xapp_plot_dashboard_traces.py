"""
Standalone PRB Dashboard xApp with episodic control over eMBB/URLLC slices.

- Reuses PRBGymEnv exactly as before:
  * episode JSON loading
  * UE registration
  * trace attachment
  * KPI aggregation
  * reward computation

- NO learning
- NO torch
- NO RIC publish/subscribe
- Manual PRB control (3 actions)
- Designed for online dashboard use
"""

import json
import math
import time
import random
import os
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np

# Dash / Plotly for lightweight control/monitoring UI
import dash
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go

import settings
from utils import load_raw_packet_csv
from .xapp_base import xAppBase

# ---------------- slice names ----------------
SL_E = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
SL_U = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")


# =====================================================================
# PRBGymEnv (dashboard-friendly; shares the 3x3 action grid with PRB Gym)
# =====================================================================
class PRBGymEnv:
    """Minimal Gym-like environment that exposes reset/step for PRB allocation."""

    def __init__(self, ric):
        self.ric = ric
        self.sim_engine = getattr(ric, "simulation_engine", None)

        self.period_steps = max(1, int(getattr(settings, "DQN_PRB_DECISION_PERIOD_STEPS", 1)))
        self.move_step = max(1, int(getattr(settings, "DQN_PRB_MOVE_STEP", 1)))

        self.norm_dl_mbps = float(getattr(settings, "DQN_NORM_MAX_DL_MBPS", 10.0))
        self.norm_buf_bytes = float(getattr(settings, "DQN_NORM_MAX_BUF_BYTES", 1e5))

        self.w_e = float(getattr(settings, "DQN_WEIGHT_EMBB", 0.5))
        self.w_u = float(getattr(settings, "DQN_WEIGHT_URLLC", 0.5))
        s = self.w_e + self.w_u
        self.w_e /= s
        self.w_u /= s

        self._episodes = self._load_episode_specs()
        self._episode_idx = -1
        self._current_episode = None
        self._steps_in_episode = 0
        self._episode_step_limit = 0

        self._managed_ues = set()
        self._trace_cache: Dict[Tuple[str, str, float], list] = {}

        # 🔑 ACTION SPACE: mirror the Gym allocator (nine combos over a shared PRB pool)
        self._action_combos = list(product([-1, 0, 1], repeat=2))

        self._state_dim = 6

    # ---------------- Episode handling ----------------
    def _load_episode_specs(self) -> List[Dict]:
        path = getattr(settings, "PRB_GYM_CONFIG_PATH", "").strip()
        if not path:
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        entries = raw.get("episodes", raw) if isinstance(raw, dict) else raw
        specs = []
        for i, e in enumerate(entries):
            spec = self._normalize_episode(e, i)
            if spec:
                specs.append(spec)
        return specs

    def _normalize_episode(self, entry, idx):
        duration = entry.get("duration_steps", 0)
        slices = entry["slices"]
        return {
            "id": entry.get("id", f"ep_{idx}"),
            # duration_steps <= 0 means "auto" (derived from trace length / sim step)
            "duration_steps": int(duration) if duration is not None else 0,
            "freeze_mobility": entry.get("freeze_mobility", True),
            "initial_prb": entry.get("initial_prb", {}),
            "slices": slices,
        }

    def reset(self):
        self._episode_idx += 1
        if self._episode_idx >= len(self._episodes):
            return None

        self._current_episode = self._episodes[self._episode_idx]
        self._episode_step_limit = self._calc_episode_step_limit(self._current_episode)
        self._steps_in_episode = 0

        self._deploy_episode(self._current_episode)

        print(f"[PRBGymEnv] Episode {self._episode_idx+1}: {self._current_episode['id']}")
        return self._get_state()

    def _deploy_episode(self, spec):
        self._clear_managed_ues()
        self._apply_initial_prb(spec.get("initial_prb", {}))
        for sl in (SL_E, SL_U):
            cfg = spec["slices"][sl]
            for i in range(cfg["ue_count"]):
                imsi = f"GIMSI_{spec['id']}_{sl}_{i}"
                self._register_episode_ue(imsi, sl, cfg)

    def _clear_managed_ues(self):
        # Remove previously managed UEs from the sim and purge their history from plots
        old_managed = set(self._managed_ues)
        if self.sim_engine:
            for imsi in list(self.sim_engine.ue_list.keys()):
                self.sim_engine.deregister_ue(imsi)
        self._managed_ues.clear()

    def _apply_initial_prb(self, prb_map):
        for cell in self.sim_engine.cell_list.values():
            quotas = dict(cell.slice_dl_prb_quota)
            quotas.update(prb_map)
            cell.slice_dl_prb_quota = quotas

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

    def _register_episode_ue(self, imsi, slice_name, cfg):
        # Retry a few times in case cell discovery/attach fails on first try
        max_attempts = 3
        attached = False
        for attempt in range(1, max_attempts + 1):
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

        if self._current_episode.get("freeze_mobility", True):
            ue.speed_mps = 0

        trace = cfg.get("trace")
        if trace:
            self._attach_trace(ue, trace, cfg)
            print(f"[TRACE] Attached {trace} → {imsi}")

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
        samples = self._trace_cache[key]
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

    # ---------------- Step ----------------
    def step(self, action_idx):
        cell = self._target_cell()
        self._apply_action(cell, action_idx)

        reward, metrics = self._reward(cell)
        state = self._get_state()

        self._steps_in_episode += 1
        done = (self._steps_in_episode >= self._episode_step_limit) or self._traces_consumed()

        info = {
            "episode_id": self._current_episode["id"],
            "step": self._steps_in_episode,
        }
        return state, (reward, metrics), done, info

    def _apply_action(self, cell, action_idx):
        combo = self._action_combos[int(action_idx)]
        delta_prb = self.move_step
        quotas = dict(getattr(cell, "slice_dl_prb_quota", {}) or {})
        for sl in (SL_E, SL_U):
            quotas.setdefault(sl, 0)
        max_prb = int(getattr(cell, "max_dl_prb", 0) or 0)
        if max_prb <= 0:
            return
        # Remove PRBs for negative moves first
        for sl, sign in zip((SL_E, SL_U), combo):
            if sign < 0:
                reduce = min(delta_prb * abs(sign), quotas[sl])
                quotas[sl] -= reduce
        # Apply positive moves
        for sl, sign in zip((SL_E, SL_U), combo):
            if sign > 0:
                quotas[sl] += delta_prb * sign
        # Keep total within pool
        total = sum(quotas.values())
        if total > max_prb:
            scale = max_prb / float(max(1, total))
            for sl in quotas:
                quotas[sl] = int(quotas[sl] * scale)
        cell.slice_dl_prb_quota = {k: int(max(0, v)) for k, v in quotas.items()}

    def _target_cell(self):
        return next(iter(self.sim_engine.cell_list.values()))

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

    def ready(self) -> bool:
        """Return True when managed UEs are attached to a cell (prevents stepping on empty sims)."""
        if not self._managed_ues or self.sim_engine is None:
            return False
        for imsi in self._managed_ues:
            ue = self.sim_engine.ue_list.get(imsi)
            if ue is None or getattr(ue, "current_cell", None) is None:
                return False
        return True

    # ---------------- KPIs ----------------
    def _get_state(self):
        cell = self._target_cell()
        agg = self._aggregate_slice_metrics(cell)
        s = []
        for sl in (SL_E, SL_U):
            s += [
                agg[sl]["slice_prb"],
                agg[sl]["tx_mbps"],
                agg[sl]["buf_bytes"],
            ]
        return np.array(s, dtype=np.float32)

    def _aggregate_slice_metrics(self, cell):
        agg = {SL_E: {}, SL_U: {}}
        for sl in (SL_E, SL_U):
            agg[sl] = {
                "slice_prb": cell.slice_dl_prb_quota.get(sl, 0),
                "tx_mbps": 0.0,
                "buf_bytes": 0.0,
                "latency": 0.0,
            }
        for ue in cell.connected_ue_list.values():
            sl = ue.slice_type
            agg[sl]["tx_mbps"] += ue.served_downlink_bitrate / 1e6
            agg[sl]["buf_bytes"] += ue.dl_buffer_bytes
            agg[sl]["latency"] += ue.downlink_latency
        return agg

    def _reward(self, cell):
        agg = self._aggregate_slice_metrics(cell)
        # Reward is not used; return 0 and metrics without a score component
        return 0.0, agg


# =====================================================================
# STANDALONE DASHBOARD xAPP
# =====================================================================
class xAppPRBDashboardStandalone(xAppBase):
    """
    Standalone PRB control + KPI dashboard xApp.
    """

    def __init__(self, ric):
        super().__init__(ric=ric)
        self.env = PRBGymEnv(ric)
        self.current_action = 4  # keep (index of (0,0) in the 3x3 action grid)
        self.last_metrics = {}
        self.running = False
        self._lock = threading.Lock()
        self._global_step = 0

        # History buffers for the dashboard (bounded to keep it light)
        self._t = deque(maxlen=400)
        self._action_hist = deque(maxlen=400)
        self._slice_tx = defaultdict(lambda: deque(maxlen=400))
        self._ue_tx = defaultdict(lambda: deque(maxlen=400))
        self._ue_prb_demand = defaultdict(lambda: deque(maxlen=400))
        self._ue_prb_sat = defaultdict(lambda: deque(maxlen=400))
        self._ue_sinr = defaultdict(lambda: deque(maxlen=400))
        self._ue_cqi = defaultdict(lambda: deque(maxlen=400))
        self._slice_prb = defaultdict(lambda: deque(maxlen=400))
        self._cell_max_prb = deque(maxlen=400)
        self._cell_eff_prb = deque(maxlen=400)
        self._slice_latency = defaultdict(lambda: deque(maxlen=400))

        # Dash server
        self._dash_app = None
        self._dash_thread = None

    def start(self):
        first_state = self.env.reset()
        if first_state is None:
            print("[xApp] No episodes available; dashboard not started.")
            self.running = False
            return
        self.running = True
        print("[xApp] PRB Dashboard started")
        self._start_dashboard()

    def step(self):
        if not self.running:
            return
        if not self.env.ready():
            # Skip stepping until new episode UEs are attached; avoids gaps with zero metrics
            return
        _, (reward, metrics), done, info = self.env.step(self.current_action)
        self._record_metrics(reward, metrics, info)
        self.last_metrics = {
            "time": time.time(),
            "action": self.current_action,
            "episode": info,
            "metrics": metrics,
        }
        if done:
            # Mark boundary so plots show the drop/reset between episodes
            self._mark_episode_boundary()
            nxt = self.env.reset()
            if nxt is None:
                print("[xApp] Episode catalog exhausted; stopping xApp.")
                self.running = False
            else:
                # Prune histories for UEs that are no longer managed in the new episode
                with self._lock:
                    active = set(getattr(self.env, "_managed_ues", set()) or [])
                    for store in (self._ue_tx, self._ue_sinr, self._ue_cqi):
                        for imsi in list(store.keys()):
                            if active and imsi not in active:
                                store.pop(imsi, None)

    # dashboard hooks
    def set_action(self, a: int):
        if 0 <= a < len(self.env._action_combos):
            self.current_action = a

    def get_metrics(self):
        return self.last_metrics

    # ---------------- internal helpers ----------------
    def _record_metrics(self, reward, metrics, info):
        step_idx = info.get("step") if isinstance(info, dict) else None
        with self._lock:
            managed = set(getattr(self.env, "_managed_ues", set()) or [])
            self._global_step += 1
            self._t.append(self._global_step if step_idx is None else self._global_step)
            self._action_hist.append(int(self.current_action))
            for sl in (SL_E, SL_U):
                m = metrics.get(sl, {}) if metrics else {}
                self._slice_tx[sl].append(float(m.get("tx_mbps", 0.0) or 0.0))
                self._slice_prb[sl].append(float(m.get("slice_prb", 0.0) or 0.0))
                self._slice_latency[sl].append(float(m.get("latency", 0.0) or 0.0))
            # Per‑UE throughput lets us verify traces are flowing
            for imsi, ue in self.ric.ue_list.items():
                if managed and imsi not in managed:
                    continue  # ignore baseline/default UEs not created by this episode
                # Prefer served_downlink_bitrate (actual) else downlink_bitrate
                bps = getattr(ue, "served_downlink_bitrate", None)
                if bps is None:
                    bps = getattr(ue, "downlink_bitrate", 0.0)
                self._ue_tx[imsi].append(float(bps) / 1e6)
                # Demand/grant and satisfaction (grant/demand)
                cell = getattr(self.env, "_target_cell", lambda: None)()
                if cell:
                    demand = float(getattr(cell, "dl_total_prb_demand", {}).get(imsi, 0) or 0)
                    granted = float(getattr(cell, "prb_ue_allocation_dict", {}).get(imsi, {}).get("downlink", 0) or 0)
                    self._ue_prb_demand[imsi].append(demand)
                    sat = 1.0 if demand <= 0 else min(1.5, granted / demand)
                    self._ue_prb_sat[imsi].append(sat)
                sinr = getattr(ue, "downlink_sinr", None)
                if sinr is not None:
                    self._ue_sinr[imsi].append(float(sinr))
                cqi = getattr(ue, "downlink_cqi", None)
                if cqi is not None:
                    self._ue_cqi[imsi].append(float(cqi))
            # Track cell max PRBs for supply context
            cell = getattr(self.env, "_target_cell", lambda: None)()
            if cell:
                self._cell_max_prb.append(int(getattr(cell, "max_dl_prb", 0) or 0))
                self._cell_eff_prb.append(int(getattr(cell, "_effective_max_dl_prb", getattr(cell, "max_dl_prb", 0)) or 0))

    def _mark_episode_boundary(self):
        """Append a boundary point with zero throughput to indicate UE removal/episode reset."""
        with self._lock:
            self._global_step += 1
            self._t.append(self._global_step)
            self._action_hist.append(self.current_action)
            for sl in (SL_E, SL_U):
                self._slice_tx[sl].append(0.0)
                self._slice_prb[sl].append(0.0)
                self._slice_latency[sl].append(0.0)
            for imsi, hist in list(self._ue_tx.items()):
                if imsi in getattr(self.env, "_managed_ues", set()):
                    hist.append(0.0)
                else:
                    # Prune legends for non-active UEs
                    self._ue_tx.pop(imsi, None)
                    self._ue_prb_demand.pop(imsi, None)
                    self._ue_prb_sat.pop(imsi, None)
                    self._ue_sinr.pop(imsi, None)
                    self._ue_cqi.pop(imsi, None)
            # Keep total PRB history aligned
            if self._cell_max_prb:
                self._cell_max_prb.append(self._cell_max_prb[-1])

    def _start_dashboard(self):
        if self._dash_thread and self._dash_thread.is_alive():
            return  # already running

        app = Dash(__name__)

        def tidy(fig, title, ytitle):
            fig.update_layout(
                title=title,
                xaxis_title="Episode step",
                yaxis_title=ytitle,
                height=320,
                margin=dict(l=45, r=20, t=40, b=40),
                legend=dict(orientation="h", y=-0.25, x=0),
                template="plotly_white",
                uirevision="trace-dash",
            )
            return fig

        def _action_label(idx: int) -> str:
            de, du = self.env._action_combos[idx]
            step = self.env.move_step
            return f"{idx}: Δe={de*step}, Δu={du*step}"

        action_labels = {i: _action_label(i) for i in range(len(self.env._action_combos))}
        action_btn_ids = [f"act-{i}" for i in range(len(self.env._action_combos))]

        app.layout = html.Div(
            style={"fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif", "padding": "12px", "maxWidth": "1200px", "margin": "0 auto"},
            children=[
                html.H2("PRB Trace Dashboard xApp"),
                html.Div([
                    html.Label("Actions (shared PRB pool — matches PRB Gym allocator)"),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(0, 1fr))", "gap": "8px"},
                        children=[
                            html.Button(action_labels[i], id=action_btn_ids[i], n_clicks=0)
                            for i in range(len(action_btn_ids))
                        ],
                    ),
                ]),
                html.Div(id="action-label", style={"marginTop": "6px", "fontSize": "14px", "color": "#333"}),
                html.Div(id="episode-label", style={"marginTop": "4px", "fontSize": "12px", "color": "#555"}),

                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px", "marginTop": "12px"}, children=[
                    dcc.Graph(id="fig-actions"),
                    html.Div(),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-prb"),
                    dcc.Graph(id="fig-throughput"),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-cell-prb"),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-latency"),
                    html.Div(id="latest-metrics", style={"fontSize": "13px", "lineHeight": "1.4"}),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-ue-throughput"),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-ue-demand"),
                    dcc.Graph(id="fig-ue-sat"),
                ]),
                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0, 1fr))", "gap": "12px"}, children=[
                    dcc.Graph(id="fig-ue-sinr"),
                    dcc.Graph(id="fig-ue-cqi"),
                ]),

                dcc.Interval(id="tick", interval=750, n_intervals=0),
            ],
        )

        @app.callback(
            [Output("action-label", "children"), Output("episode-label", "children")],
            [Input(btn_id, "n_clicks") for btn_id in action_btn_ids],
        )
        def _handle_action(*clicks):
            ctx = dash.callback_context
            if ctx.triggered:
                tid = ctx.triggered[0]["prop_id"].split(".")[0]
                if tid in action_btn_ids:
                    idx = action_btn_ids.index(tid)
                    self.set_action(idx)
            lm = self.get_metrics()
            episode = lm.get("episode", {}) if isinstance(lm, dict) else {}
            ep_text = f"Episode: {episode.get('episode_id', 'N/A')} | Step: {episode.get('step', '-')}"
            return f"Current action: {action_labels.get(self.current_action, '?')}", ep_text

        @app.callback(
            [
                Output("fig-actions", "figure"),
                Output("fig-prb", "figure"),
                Output("fig-throughput", "figure"),
                Output("fig-cell-prb", "figure"),
                Output("fig-latency", "figure"),
                Output("fig-ue-throughput", "figure"),
                Output("fig-ue-demand", "figure"),
                Output("fig-ue-sat", "figure"),
                Output("fig-ue-sinr", "figure"),
                Output("fig-ue-cqi", "figure"),
                Output("latest-metrics", "children"),
            ],
            [Input("tick", "n_intervals")],
        )
        def _update(_n):
            with self._lock:
                tx = list(self._t)
                act = list(self._action_hist)
                tx_mbps_e = list(self._slice_tx.get(SL_E, []))
                tx_mbps_u = list(self._slice_tx.get(SL_U, []))
                prb_e = list(self._slice_prb.get(SL_E, []))
                prb_u = list(self._slice_prb.get(SL_U, []))
                cell_max = list(self._cell_max_prb)
                cell_eff = list(self._cell_eff_prb)
                lat_e = list(self._slice_latency.get(SL_E, []))
                lat_u = list(self._slice_latency.get(SL_U, []))
                ue_keys = list(self._ue_tx.keys())
                ue_tx_map = {k: list(v) for k, v in self._ue_tx.items()}
                ue_demand_map = {k: list(v) for k, v in self._ue_prb_demand.items()}
                ue_sat_map = {k: list(v) for k, v in self._ue_prb_sat.items()}
                ue_sinr_map = {k: list(v) for k, v in self._ue_sinr.items()}
                ue_cqi_map = {k: list(v) for k, v in self._ue_cqi.items()}
                latest = self.last_metrics.copy()

            if not tx:
                empty = go.Figure()
                return empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, "Waiting for metrics..."

            fig_actions = tidy(go.Figure(data=[
                go.Scatter(x=tx, y=act, mode="lines+markers", name="Action", line=dict(color="#ff7f0e"), marker=dict(size=6)),
            ]), "Actions (idx: Δe, Δu)", "action idx")

            fig_prb = tidy(go.Figure(data=[
                go.Scatter(x=tx, y=prb_e, mode="lines", name="eMBB PRBs", line=dict(color="#1f77b4")),
                go.Scatter(x=tx, y=prb_u, mode="lines", name="URLLC PRBs", line=dict(color="#d62728")),
            ]), "Slice PRB Quotas", "PRBs")

            fig_tx = tidy(go.Figure(data=[
                go.Scatter(x=tx, y=tx_mbps_e, mode="lines", name="eMBB Mbps", line=dict(color="#1f77b4")),
                go.Scatter(x=tx, y=tx_mbps_u, mode="lines", name="URLLC Mbps", line=dict(color="#d62728")),
            ]), "Slice Throughput", "Mbps")

            fig_cell_prb = tidy(go.Figure(data=[
                go.Scatter(x=tx[-len(cell_max):], y=cell_max, mode="lines", name="Cell max DL PRBs", line=dict(color="#7f7f7f")),
                go.Scatter(x=tx[-len(cell_eff):], y=cell_eff, mode="lines", name="Effective DL PRBs", line=dict(color="#ffbf00")),
            ]), "Cell PRB Capacity", "PRBs")

            fig_lat = tidy(go.Figure(data=[
                go.Scatter(x=tx, y=lat_e, mode="lines", name="eMBB", line=dict(color="#1f77b4")),
                go.Scatter(x=tx, y=lat_u, mode="lines", name="URLLC", line=dict(color="#d62728")),
            ]), "Latency (s)", "seconds")

            # Per‑UE throughput plot to verify trace playback
            tr_ue = []
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for idx, imsi in enumerate(sorted(ue_keys)):
                ys = ue_tx_map.get(imsi, [])
                if not ys:
                    continue
                tr_ue.append(
                    go.Scatter(
                        x=tx[-len(ys):],
                        y=ys,
                        mode="lines",
                        name=imsi,
                        line=dict(color=palette[idx % len(palette)]),
                    )
                )
            fig_ue = tidy(go.Figure(data=tr_ue), "Per‑UE Throughput", "Mbps")

            # Demand/grant satisfaction plots
            tr_dem = []
            tr_sat = []
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for idx, imsi in enumerate(sorted(set(list(ue_demand_map.keys()) + list(ue_sat_map.keys())))):
                color = palette[idx % len(palette)]
                ys_d = ue_demand_map.get(imsi, [])
                if ys_d:
                    tr_dem.append(
                        go.Scatter(
                            x=tx[-len(ys_d):],
                            y=ys_d,
                            mode="lines",
                            name=imsi,
                            line=dict(color=color),
                        )
                    )
                ys_s = ue_sat_map.get(imsi, [])
                if ys_s:
                    tr_sat.append(
                        go.Scatter(
                            x=tx[-len(ys_s):],
                            y=ys_s,
                            mode="lines",
                            name=imsi,
                            line=dict(color=color),
                        )
                    )
            fig_dem = tidy(go.Figure(data=tr_dem), "Per‑UE PRB Demand", "PRBs")
            fig_sat = tidy(go.Figure(data=tr_sat), "Per‑UE Satisfaction (grant/demand)", "ratio")

            # Per‑UE SINR/CQI
            tr_sinr = []
            tr_cqi = []
            palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            for idx, imsi in enumerate(sorted(set(list(ue_sinr_map.keys()) + list(ue_cqi_map.keys())))):
                color = palette[idx % len(palette)]
                ys_s = ue_sinr_map.get(imsi, [])
                if ys_s:
                    tr_sinr.append(
                        go.Scatter(
                            x=tx[-len(ys_s):],
                            y=ys_s,
                            mode="lines",
                            name=imsi,
                            line=dict(color=color),
                        )
                    )
                ys_c = ue_cqi_map.get(imsi, [])
                if ys_c:
                    tr_cqi.append(
                        go.Scatter(
                            x=tx[-len(ys_c):],
                            y=ys_c,
                            mode="lines",
                            name=imsi,
                            line=dict(color=color),
                        )
                    )
            fig_sinr = tidy(go.Figure(data=tr_sinr), "Per‑UE SINR", "dB")
            fig_cqi = tidy(go.Figure(data=tr_cqi), "Per‑UE CQI", "CQI")

            def _fmt(v, digits=3):
                try:
                    return f"{float(v):.{digits}f}"
                except Exception:
                    return v if v not in (None, "") else "-"

            metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
            action_txt = action_labels.get(latest.get("action", None), "-")
            lines = [f"Action: {action_txt}"]
            for sl in (SL_E, SL_U):
                m = metrics.get(sl, {}) if metrics else {}
                lines.append(
                    f"{sl}: PRB={_fmt(m.get('slice_prb', '-'), 0)}, "
                    f"Tx={_fmt(m.get('tx_mbps', '-'))} Mbps, "
                    f"Latency={_fmt(m.get('latency', '-'))}"
                )
            latest_text = html.Ul([html.Li(t) for t in lines])

            return fig_actions, fig_prb, fig_tx, fig_cell_prb, fig_lat, fig_ue, fig_dem, fig_sat, fig_sinr, fig_cqi, latest_text

        host = os.getenv("PRB_TRACE_DASH_HOST", "0.0.0.0")
        port = int(os.getenv("PRB_TRACE_DASH_PORT", "8060"))

        def _run():
            app.run_server(host=host, port=port, debug=False, use_reloader=False)

        self._dash_app = app
        self._dash_thread = threading.Thread(target=_run, daemon=True)
        self._dash_thread.start()
        print(f"[xApp] PRB Trace dashboard running on http://{host}:{port}")
