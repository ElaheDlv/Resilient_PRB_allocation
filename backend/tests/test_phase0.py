"""Phase 0: actual simulator methods, plus small causal-boundary fixtures."""

from collections import deque
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

import settings
from network_layer.base_station import BaseStation
from network_layer.cell import Cell
from network_layer.prb_accounting import validate_quotas
from network_layer.proportional_scheduler import proportional_allocation
from network_layer.queue_measurements import QueueMeasurements
from network_layer.resilient_interval import CausalInterval
from network_layer.simulation_engine import SimulationEngine
from network_layer.xApps.xapp_resilient_dqn_allocator import (
    ACTION_LIST, ResilientDQNEnv, build_state, xAppResilientDQNAllocator,
)


@pytest.fixture
def engine(monkeypatch):
    topology = copy.deepcopy(settings.RAN_DEFAULT_BS_LIST[:1])
    topology[0]["cell_list"] = topology[0]["cell_list"][:1]
    monkeypatch.setattr(settings, "RAN_DEFAULT_BS_LIST", topology)
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    monkeypatch.setattr(settings, "SIM_STEP_TIME_DEFAULT", 0.001)
    monkeypatch.setattr(settings, "PHASE0_MEASUREMENTS", True)
    monkeypatch.setattr(settings, "PRB_GYM_CONFIG_PATH", "phase0-test-no-spawn")
    monkeypatch.setattr(settings, "SIM_FREEZE_MOBILITY", True)
    eng = SimulationEngine()
    eng.reset_network()
    eng.network_setup(xapp_classes=[])
    yield eng
    for bs in eng.base_station_list.values():
        bs.queue_measurements.close()


def make_buffer():
    bs = BaseStation.__new__(BaseStation)
    bs._dl_buf = {}
    bs._dl_buf_queue = {}
    bs._dl_last_delay_s = {}
    bs._time_s = 0.0
    bs.queue_measurements = QueueMeasurements()
    return bs


def test_partial_service_overflow_and_unfinished(monkeypatch):
    monkeypatch.setattr(settings, "TRACE_DL_BUFFER_LIMIT_BYTES", 150)
    bs = make_buffer()
    bs.queue_measurements.set_budget("u", 0.01)
    assert bs.enqueue_dl_bytes("u", 100) == 100
    bs._time_s = 0.005
    assert bs.enqueue_dl_bytes("u", 100) == 50
    assert bs.enqueue_dl_bytes("u", 30) == 0
    bs._time_s = 0.01
    assert bs.pull_dl_bytes("u", 40) == 40  # equality is on budget
    bs._time_s = 0.02
    assert bs.pull_dl_bytes("u", 70) == 70
    result = bs.queue_measurements.snapshot("u", bs._time_s)
    assert result["offered_bytes"] == 230
    assert result["overflow_bytes"] == 80
    assert result["served_bytes"] == 110
    assert result["unfinished_bytes"] == 40
    assert result["served_over_budget_bytes"] == 70
    assert result["completed_batches"] == 1
    assert result["completed_over_budget_batches"] == 1
    assert result["unfinished_batches"] == 1
    assert result["unresolved"][0]["over_budget"]
    assert bs.get_dl_head_wait_s("u") == pytest.approx(0.015)
    assert sum(size for _, size in bs._dl_buf_queue["u"]) == bs._dl_buf["u"] == 40


def test_measurements_do_not_change_queue(monkeypatch):
    monkeypatch.setattr(settings, "TRACE_DL_BUFFER_LIMIT_BYTES", 125)
    measured, legacy = make_buffer(), make_buffer()
    legacy.queue_measurements = None
    for now, arrival, capacity in [(0., 100, 25), (.001, 80, 60), (.003, 0, 200)]:
        for bs in (measured, legacy):
            bs._time_s = now
            bs.enqueue_dl_bytes("u", arrival)
            bs.pull_dl_bytes("u", capacity)
        assert measured._dl_buf == legacy._dl_buf
        assert measured._dl_buf_queue == legacy._dl_buf_queue
        assert measured._dl_last_delay_s == legacy._dl_last_delay_s


def test_measurement_budget_validation_and_reset():
    m = QueueMeasurements()
    for bad in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            m.set_budget("u", bad)
    m.set_budget("u", .01)
    m.arrival("u", 0, 100, 100)
    report = m.finalize("u", .02)
    assert report["unfinished_bytes"] == 100
    assert "u" not in m.pending
    m.arrival("u", .03, 20, 20)
    assert m.snapshot("u", .03)["offered_bytes"] == 20


def test_event_stream_retains_arrival_service_and_finalization(tmp_path):
    path=tmp_path/"events.jsonl"
    m=QueueMeasurements(str(path));m.set_budget("u",.001)
    m.arrival("u",0,100,50)
    m.service("u",.002,20)
    m.sample("u",.002)
    m.finalize("u",.003);m.close()
    events=[json.loads(line) for line in path.read_text().splitlines()]
    assert [e["kind"] for e in events]==["arrival","service","hol","unfinished"]
    assert events[0]["overflow_bytes"]==50 and events[0]["budget_s"]==.001
    assert events[1]["over_budget"] and events[-1]["summary"]["unfinished_bytes"]==30


def test_proportional_allocation_properties():
    for amounts in itertools.product(range(5), repeat=3):
        demands = dict(zip("abc", amounts))
        for quota in range(10):
            for cap in [None, 0, 2]:
                out = proportional_allocation(demands, quota, cap)
                limits = {u: min(d, cap) if cap is not None else d for u, d in demands.items()}
                assert sum(out.values()) == min(quota, sum(limits.values()))
                assert all(0 <= out[u] <= limits[u] for u in out)
                assert all(isinstance(v, int) for v in out.values())
    assert proportional_allocation(dict(a=10, b=10, c=10), 5) == dict(a=2, b=2, c=1)
    assert proportional_allocation(dict(a=100, b=50), 30) == dict(a=20, b=10)


@pytest.mark.parametrize("variable", [False, True])
def test_jammer_on_off_and_all_valid_actions(monkeypatch, variable):
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    monkeypatch.setattr(settings, "JAM_CONST_PERIOD_STEPS", 10)
    monkeypatch.setattr(settings, "JAM_CONST_DURATION_STEPS", 5)
    monkeypatch.setattr(settings, "JAM_CONST_PRBS", 20)
    monkeypatch.setattr(settings, "JAM_CONST_START_OFFSET", 0)
    c = NS(cell_id="c", max_dl_prb=218, _effective_max_dl_prb=200,
           base_dl_prb_quota={"eMBB": 140, "URLLC": 60},
           _jam_base_quota={"eMBB": 140, "URLLC": 60}, _jam_was_active=False,
           slice_dl_prb_quota={"eMBB": 149, "URLLC": 69})
    bs = BaseStation.__new__(BaseStation)
    bs.cell_list = {"c": c}
    bs.simulation_engine = NS(sim_step=0)
    env = ResilientDQNEnv.__new__(ResilientDQNEnv)
    env.ric = NS(cell_list={"c": c})
    env.sim_engine = NS(cell_list={"c": c})
    env._base_quota = {"c": {"eMBB": 140, "URLLC": 60}}
    env.actions = ACTION_LIST
    for tick in [0, 5, 10, 15]:
        bs.simulation_engine.sim_step = tick
        if variable:
            bs.apply_variable_jamming([20], [.5], 10, "fixed", "fixed")
        else:
            bs._apply_constant_jamming()
        assert validate_quotas(c, c.slice_dl_prb_quota, 18) == 18
        baseline = dict(c.base_dl_prb_quota)
        assert sum(baseline.values()) == (180 if tick % 10 == 0 else 200)
        for extra_e in range(19):
            for extra_u in range(19-extra_e):
                old = {"eMBB": baseline["eMBB"]+extra_e, "URLLC": baseline["URLLC"]+extra_u}
                c.slice_dl_prb_quota = old.copy()
                for action, valid in zip(ACTION_LIST, env._action_mask()):
                    c.slice_dl_prb_quota = old.copy()
                    if valid:
                        env._apply_action(action)
                        validate_quotas(c, c.slice_dl_prb_quota, 18)
                    else:
                        with pytest.raises(AssertionError):
                            env._apply_action(action)
                        assert c.slice_dl_prb_quota == old
        c.slice_dl_prb_quota = {s: baseline[s]+9 for s in baseline}


def test_zero_capacity_is_observable(monkeypatch):
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    c = NS(cell_id="c", max_dl_prb=218, _effective_max_dl_prb=0,
           base_dl_prb_quota={"eMBB": 0, "URLLC": 0},
           slice_dl_prb_quota={"eMBB": 0, "URLLC": 0}, connected_ue_list={})
    env = ResilientDQNEnv.__new__(ResilientDQNEnv)
    env.ric = NS(cell_list={"c": c})
    env._base_quota = {}
    assert env._collect_obs()["effective_prb_ratio"] == 0.0
    assert env._collect_obs()["effective_max_prb"] == 0.0


def test_causal_transition_alignment_and_next_mask():
    class Env:
        actions = [(0, 0), (0, 1)]
        prev_action = (0, 0)
        queue = 10
        quota = 0
        _steps_in_episode = 0
        _episode_step_limit = 2
        _current_episode = {"id": "alignment"}

        def _action_mask(self):
            return [True, self.quota < 1]

        def _collect_obs(self):
            return dict(queue=self.queue, quota=self.quota)

        def _apply_action(self, action):
            self.quota += action[1]

        def _compute_reward(self, obs, action, previous):
            return -obs["queue"], {"cost": -obs["queue"]}

        def _traces_consumed(self):
            return False

    env = Env()
    encode = lambda obs, prev: np.array([obs["queue"], obs["quota"], prev[1]], dtype=float)
    loop = CausalInterval(env, 2, encode)
    loop.begin(1)
    assert env.queue == 10  # no service/recursion in callback
    with pytest.raises(RuntimeError):
        loop.reset()
    env.queue -= env.quota  # actual first network interval
    assert loop.observe() is None
    env.queue -= env.quota
    t = loop.observe()
    np.testing.assert_array_equal(t.state, [10, 0, 0])
    np.testing.assert_array_equal(t.next_state, [8, 1, 1])
    assert t.reward == -17  # -9 + -8, never the pre-action -10
    assert t.action_idx == 1 and t.next_mask == [True, False] and not t.done
    loop.begin(0)
    loop.observe()
    assert loop.observe().done
    loop.reset()


def test_actual_cell_scheduler_and_impossible_quota(engine, monkeypatch):
    cell = next(iter(engine.cell_list.values()))
    bs = cell.base_station
    # Minimal actual UE objects for the real allocation method.
    for i, queued in enumerate([0, 10, 100]):
        imsi = str(i)
        cell.connected_ue_list[imsi] = NS(ue_imsi=imsi, slice_type="URLLC",
                qos_profile={"GBR_DL": 1e6}, downlink_mcs_data={"modulation_order":8,"target_code_rate":948})
        cell.prb_ue_allocation_dict[imsi] = {"downlink":0,"uplink":0}
        bs._dl_buf[imsi] = queued
    cell.slice_dl_prb_quota = {"URLLC": 5}
    cell.base_dl_prb_quota = {"URLLC": 5}
    monkeypatch.setattr(settings, "PRB_UE_SCHEDULER", "proportional_wc")
    cell.allocate_prb()
    grants = [cell.prb_ue_allocation_dict[str(i)]["downlink"] for i in range(3)]
    assert grants[0] == 0 and sum(grants) == 2  # useful backlog fits in two PRBs
    cell.slice_dl_prb_quota = {"URLLC": 999}
    with pytest.raises(AssertionError):
        cell.allocate_prb()


def test_real_episode_reset_and_budget_assignment(engine, monkeypatch, tmp_path):
    trace = Path("backend/notebooks/Resilient_data/urllc_dl.csv").resolve()
    entry = dict(duration_steps=2, initial_prb={"eMBB":140,"URLLC":60}, slices={
        "eMBB":dict(ue_count=0), "URLLC":dict(ue_count=1, trace=str(trace),
        ue_ip="10.0.2.1", trace_bin=.001, queue_budgets_s={"0":.002})})
    catalog = tmp_path / "episodes.json"
    catalog.write_text(json.dumps([dict(entry,id="one"),dict(entry,id="two")]))
    monkeypatch.setattr(settings,"PRB_GYM_CONFIG_PATH",str(catalog))
    monkeypatch.setattr(settings,"RES_DQN_TRANSITION_MODE","causal")
    env = ResilientDQNEnv(NS(cell_list=engine.cell_list,simulation_engine=engine), spawn_seed=3)
    env.reset()
    old_imsi = next(iter(env._managed_ues))
    bs = next(iter(engine.base_station_list.values()))
    bs.enqueue_dl_bytes(old_imsi, 100)
    assert engine.ue_list[old_imsi].queue_budget_s == .002
    env.reset()
    assert old_imsi not in bs._dl_buf_queue and old_imsi not in bs._dl_replay
    assert bs.queue_measurements.closed[-1]["unfinished_bytes"] == 100
    assert all(bs.get_dl_buf_bytes(i)==0 for i in env._managed_ues)
    env.reset()
    assert env.exhausted and not env.ready()


def test_trainer_uses_completed_interval_for_replay(monkeypatch):
    # Exercise the actual trainer callback, with only expensive learning stubbed.
    from network_layer.xApps.xapp_resilient_dqn_allocator import _ReplayBuffer
    app = xAppResilientDQNAllocator.__new__(xAppResilientDQNAllocator)
    obs = dict(urllc_latency=0, urllc_latency_deadline=.001, max_dl_prb=218)
    env = NS(exhausted=False, ready=lambda:True, prev_action=(0,0), actions=ACTION_LIST,
             _steps_in_episode=0,_episode_step_limit=100,_current_episode={"id":"trainer"},
             _collect_obs=lambda:obs.copy(),_action_mask=lambda:[True]*49,
             _apply_action=lambda action:None,_traces_consumed=lambda:False,
             _compute_reward=lambda o,a,p:(-o["urllc_latency"],{"cost":-o["urllc_latency"]}))
    app._env=env;app._causal_interval=CausalInterval(env,2,build_state)
    app._global_step=0;app._episode_done=False;app._decision_steps=0;app._episode_return=0
    app._episode_len=0;app._episode_reward_parts={"cost":0};app.train=True
    app.eps_decay=10;app.eps_start=1;app.eps_end=0;app._replay=_ReplayBuffer()
    app._select_action=lambda s:24;app._record_metrics=lambda *a:None
    app._learn=lambda:None;app._tb_log=lambda *a,**kw:None
    app._step_causal()
    assert len(app._replay)==0
    obs["urllc_latency"]=.002;app._step_causal()
    assert len(app._replay)==0
    obs["urllc_latency"]=.003;app._step_causal()
    assert len(app._replay)==1
    state,action,reward,next_state,done,mask=app._replay.buf[0]
    assert reward==pytest.approx(-.005) and action==24
    assert state[10]==0 and next_state[10]>0 and not done and all(mask)


def test_real_causal_training_and_terminal_reset(engine, monkeypatch, tmp_path):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(17)
    trace = tmp_path / "traffic.csv"
    trace.write_text("time,src_ip,dst_ip,length\n0,10.0.0.1,10.0.2.1,650\n"
                     "0.010,10.0.0.1,10.0.2.1,650\n")
    entry = dict(duration_steps=2, initial_prb={"eMBB":140,"URLLC":60},
                 slices={"eMBB":dict(ue_count=0), "URLLC":dict(ue_count=1,
                 trace=str(trace),ue_ip="10.0.2.1",trace_bin=0,queue_budget_s=.002)})
    catalog = tmp_path / "training.json"
    catalog.write_text(json.dumps([dict(entry,id="one"),dict(entry,id="two")]))
    for key,value in dict(PRB_GYM_CONFIG_PATH=str(catalog), RES_DQN_TRANSITION_MODE="causal",
                          RES_DQN_ENABLE=True,RES_DQN_TRAIN=True,RES_DQN_DECISION_PERIOD=2,
                          RES_DQN_BATCH=2,RES_DQN_MODEL_DIR="",RES_DQN_DASH_ENABLE=False,
                          RES_DQN_TB_ENABLE=False,RES_DQN_EVAL_INTERVAL=0,
                          PHASE0_MEASUREMENTS=True,JAM_RESERVED_PRBS=18,
                          SIM_FREEZE_MOBILITY=True,SIM_STEP_TIME_DEFAULT=.001).items():
        monkeypatch.setattr(settings,key,value)
    eng=SimulationEngine();eng.reset_network();eng.network_setup(xapp_classes=[xAppResilientDQNAllocator])
    app=next(iter(eng.ric.xapp_list.values()))
    initial=[p.detach().clone() for p in app._policy.parameters()]
    for _ in range(13):
        eng.sim_step+=1;eng.step(.001)
    assert app._episode_counter==2 and app._env.exhausted
    assert len(app._replay)==4
    assert sum(row[4] for row in app._replay.buf)==2
    assert any(not torch.equal(a,b) for a,b in zip(initial,app._policy.parameters()))
    assert app._causal_interval.pending is None
    for bs in eng.base_station_list.values():
        assert not bs._dl_replay and not bs._dl_buf_queue
        assert len(bs.queue_measurements.closed)==2
        bs.queue_measurements.close()


@pytest.mark.parametrize("transitions", ["legacy", "causal"])
def test_paced_unpaced_and_instrumented_replay(tmp_path, transitions):
    root=Path(__file__).resolve().parents[2]
    paths=[]
    for index,options in enumerate([[],["--paced"],["--measurements"]]):
        path=tmp_path/f"{index}.json"
        cmd=[sys.executable,"backend/tools/phase0_evaluate.py","--config",
             "backend/assets/episodes/test_config/gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json",
             "--policy","random","--steps","80","--transitions",transitions,
             "--output",str(path),*options]
        result=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        paths.append(json.loads(path.read_text()))
    assert len({r["trajectory_sha256"] for r in paths})==1
