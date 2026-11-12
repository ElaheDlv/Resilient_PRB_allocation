"""Deadline scheduling invariants and paired real-simulator integration tests."""

from collections import deque
import copy
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from network_layer.deadline_scheduler import QueueView, deadline_allocation
from network_layer.phase1_metrics import RecoveryTracker, batch_outcomes
from network_layer.queue_measurements import QueueMeasurements
from network_layer.ue_budgets import budget_assignment

ROOT = Path(__file__).resolve().parents[2]


def test_budget_assignment_is_balanced_private_and_reproducible():
    config = dict(
        queue_budget_classes_s={"tight": 0.002, "relaxed": 0.010},
        queue_budget_assignment_seed=91,
    )
    before = random.getstate()
    assignment = budget_assignment(config, 8, "episode")
    assert random.getstate() == before
    assert assignment == budget_assignment(config, 8, "episode")
    assert sum(label == "tight" for _, label in assignment) == 4
    assert sum(label == "relaxed" for _, label in assignment) == 4
    # Placement RNG consumption cannot affect a separate assignment RNG.
    for _ in range(100):
        random.random()
    assert assignment == budget_assignment(config, 8, "episode")
    random.setstate(before)
    assert budget_assignment(dict(config, queue_budgets_s={"0": 0.007}), 8, "episode")[
        0
    ] == (0.007, "override")
    for value in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            budget_assignment(dict(queue_budget_classes_s={"bad": value}), 1, "e")


def test_slack_priority_ties_partial_heads_and_no_fifo_mutation():
    queues = {
        "tight": QueueView(200, 100, 0.002, deque([(0.0, 100), (0.001, 100)])),
        "relaxed": QueueView(200, 100, 0.020, deque([(0.0, 200)])),
    }
    grants = deadline_allocation(queues, 2, 0.001, 0.001, {})
    assert grants == dict(tight=2, relaxed=0)
    assert list(queues["tight"].fifo) == [(0.0, 100), (0.001, 100)]
    equal = {u: QueueView(100, 100, 0.01, [(0, 100)]) for u in ["b", "a"]}
    assert deadline_allocation(equal, 1, 0.001, 0.001, {}) == dict(b=0, a=1)
    # This head is nearly completed by prior actual service; only one useful PRB.
    partial = {"u": QueueView(10, 100, 0.002, [(0, 10)])}
    assert deadline_allocation(partial, 5, 0.003, 0.001, {}) == {"u": 1}


def test_deadline_integer_conservation_caps_and_empty_queues():
    for sizes in itertools.product(range(4), repeat=3):
        queues = {
            u: QueueView(size, 1, 0.01, [(0, size)]) for u, size in zip("abc", sizes)
        }
        for quota in range(8):
            for cap in [None, 0, 2]:
                out = deadline_allocation(
                    queues, quota, 0.02, 0.001, {}, per_ue_cap=cap
                )
                limits = [min(s, cap) if cap is not None else s for s in sizes]
                assert sum(out.values()) == min(quota, sum(limits))
                assert all(
                    isinstance(v, int) and 0 <= v <= limit
                    for v, limit in zip(out.values(), limits)
                )
    views = {
        "empty": QueueView(0, 100, None, []),
        "zero": QueueView(100, 0, None, [(0, 100)]),
        "subbyte": QueueView(1, 0.5, 0.001, [(0, 1)]),
    }
    assert deadline_allocation(views, 1, 0.01, 0.001, {}) == dict(
        empty=0, zero=0, subbyte=0
    )
    assert deadline_allocation(views, 3, 0.01, 0.001, {}) == dict(
        empty=0, zero=0, subbyte=2
    )
    with pytest.raises(ValueError):
        deadline_allocation({"u": QueueView(1, 1, None, [(0, 1)])}, 1, 0, 0.001, {})


def test_starvation_guard_under_continuous_overload():
    views = {
        u: QueueView(1000, 1, 0.001 if u == "a" else 10, [(0, 1000)]) for u in "abc"
    }
    ages = dict.fromkeys(views, 0)
    services = dict.fromkeys(views, 0)
    longest = 0
    for tick in range(100):
        grants = deadline_allocation(
            views, 1, 1 + tick * 0.001, 0.001, ages, starvation_limit=2
        )
        assert sum(grants.values()) == 1
        for u in views:
            ages[u] = ages[u] + 1 if grants[u] == 0 else 0
            services[u] += grants[u]
        longest = max(longest, max(ages.values()))
    assert min(services.values()) > 0
    assert longest <= 2 + len(views) - 1


def test_budget_equality_partial_overflow_and_unresolved_partition():
    m = QueueMeasurements()
    m.set_budget("u", 0.002, "tight")
    m.arrival("u", 0, 100, 100)
    m.service("u", 0.002, 100)
    m.arrival("u", 0.003, 100, 50)
    m.service("u", 0.006, 50)
    m.arrival("u", 0.006, 100, 100)  # unresolved and overdue at .010
    m.arrival("u", 0.009, 100, 100)  # unresolved, not yet due
    m.arrival("u", 0.009, 100, 0)  # fully rejected
    r = m.snapshot("u", 0.010)
    o = batch_outcomes(r)
    assert o == dict(
        offered=5,
        on_time=1,
        known_failed=3,
        unresolved_not_due=1,
        unfinished_overdue=1,
        failure_fraction_lower=0.6,
        failure_fraction_upper=0.8,
    )
    assert r["completed_over_budget_batches"] == 1
    assert r["partially_admitted_completed_batches"] == 1
    assert r["served_over_budget_bytes"] == 50


def test_hol_runs_and_measurement_reset():
    m = QueueMeasurements()
    m.set_budget("u", 0.002, "tight")
    m.arrival("u", 0, 100, 100)
    for tick in range(1, 6):
        m.sample("u", tick * 0.001, 0.001)
    r = m.snapshot("u", 0.005)["hol_statistics"]
    assert r["violating_s"] == pytest.approx(0.003)
    assert r["longest_run_s"] == pytest.approx(0.003) and r["right_censored_run"]
    m.service("u", 0.006, 100)
    m.sample("u", 0.006, 0.001)
    r = m.finalize("u", 0.006)
    assert not r["hol_statistics"]["right_censored_run"]
    assert "u" not in m.hol_statistics and "u" not in m.budget_classes


def test_recovery_requires_hold_and_preserves_censoring():
    r = RecoveryTracker(0.002)
    r.observe(0.001, 0.001, True, False)
    r.observe(0.002, 0.001, False, False)
    r.observe(0.003, 0.001, False, True)
    r.observe(0.004, 0.001, False, True)
    assert r.windows[0]["recovery_s"] == pytest.approx(0.001)
    assert not r.windows[0]["censored"]
    r.observe(0.005, 0.001, True, False)
    r.observe(0.006, 0.001, False, False)
    r.observe(0.007, 0.001, True, False)
    assert r.windows[1]["recovery_s"] is None and r.windows[1]["reason"] == "next_jam"


@pytest.mark.parametrize("severity", [0, 10, 20])
def test_paired_real_scheduler_jammer_and_budget_scenarios(tmp_path, severity):
    source = (
        ROOT
        / "backend/assets/episodes/test_config/gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json"
    )
    catalog = json.loads(source.read_text())
    entry = copy.deepcopy(
        (catalog["episodes"] if isinstance(catalog, dict) else catalog)[0]
    )
    entry["duration_steps"] = 100
    entry["slices"]["eMBB"]["ue_count"] = 2
    entry["slices"]["URLLC"].update(
        ue_count=2,
        queue_budget_classes_s={"tight": 0.002, "relaxed": 0.010},
        queue_budget_assignment_seed=4,
    )
    entry["jam_var"].update(period_steps=10, prbs_list=[severity])
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(dict(episodes=[entry])))
    results = []
    for scheduler in ["proportional_wc", "deadline_slack"]:
        report = tmp_path / f"{scheduler}.json"
        cmd = [
            sys.executable,
            "backend/tools/phase0_evaluate.py",
            "--config",
            str(config),
            "--policy",
            "fixed",
            "--fixed-u",
            "3",
            "--scheduler",
            scheduler,
            "--transitions",
            "causal",
            "--phase1-metrics",
            "--steps",
            "80",
            "--output",
            str(report),
        ]
        run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert run.returncode == 0, run.stdout + run.stderr
        results.append(json.loads(report.read_text()))
    a, b = [r["phase1"] for r in results]
    assert a["exogenous_sha256"] == b["exogenous_sha256"]
    assert (
        a["quota_sha256"] == b["quota_sha256"] and a["assignments"] == b["assignments"]
    )
    for r in [a, b]:
        assert r["mean_reserve_prbs"] == pytest.approx(3)
        assert len(r["per_ue"]) == 2 and set(r["per_class"]) == {"tight", "relaxed"}
        assert len(r["recovery"]) == (0 if severity == 0 else 8)
        for stats in r["resources"].values():
            assert stats["allocated_prb_s"] + stats["unused_prb_s"] == pytest.approx(
                stats["quota_prb_s"]
            )
            assert stats["mean_allocated_prbs"] * 0.080 == pytest.approx(
                stats["allocated_prb_s"]
            )


def test_cell_starvation_state_cleared_on_deregistration():
    from network_layer.cell import Cell
    from types import SimpleNamespace as NS

    cell = Cell.__new__(Cell)
    cell.cell_id = "c"
    cell._deadline_starvation_ticks = {"u": 50}
    cell.prb_ue_allocation_dict = {"u": {}}
    cell.connected_ue_list = {"u": None}
    cell.deregister_ue(NS(ue_imsi="u"))
    assert not cell._deadline_starvation_ticks


def test_real_episode_budget_classes_and_scheduler_reset(monkeypatch, tmp_path):
    import settings
    from types import SimpleNamespace as NS
    from network_layer.simulation_engine import SimulationEngine
    from network_layer.xApps.xapp_resilient_dqn_allocator import ResilientDQNEnv

    topology = copy.deepcopy(settings.RAN_DEFAULT_BS_LIST[:1])
    topology[0]["cell_list"] = topology[0]["cell_list"][:1]
    monkeypatch.setattr(settings, "RAN_DEFAULT_BS_LIST", topology)
    monkeypatch.setattr(settings, "PHASE0_MEASUREMENTS", True)
    monkeypatch.setattr(settings, "RES_DQN_TRANSITION_MODE", "causal")
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    entry = dict(
        duration_steps=10,
        initial_prb={"eMBB": 140, "URLLC": 60},
        slices={
            "eMBB": dict(ue_count=0),
            "URLLC": dict(
                ue_count=4,
                queue_budget_classes_s={"tight": 0.002, "relaxed": 0.010},
                queue_budget_assignment_seed=8,
            ),
        },
    )
    path = tmp_path / "classes.json"
    path.write_text(
        json.dumps(dict(episodes=[dict(entry, id="one"), dict(entry, id="two")]))
    )
    monkeypatch.setattr(settings, "PRB_GYM_CONFIG_PATH", str(path))
    eng = SimulationEngine()
    eng.reset_network()
    eng.network_setup(xapp_classes=[])
    env = ResilientDQNEnv(
        NS(cell_list=eng.cell_list, simulation_engine=eng), spawn_seed=7
    )
    env.reset()
    cell = next(iter(eng.cell_list.values()))
    bs = cell.base_station
    assert sorted(u.queue_budget_class for u in eng.ue_list.values()) == [
        "relaxed",
        "relaxed",
        "tight",
        "tight",
    ]
    old = set(eng.ue_list)
    for uid in old:
        cell._deadline_starvation_ticks[uid] = 10
        bs.enqueue_dl_bytes(uid, 100)
    env.reset()
    assert not old.intersection(eng.ue_list) and not cell._deadline_starvation_ticks
    assert len(bs.queue_measurements.closed) == 4
    for uid, u in eng.ue_list.items():
        assert bs.queue_measurements.budget_classes[uid] == u.queue_budget_class
        assert bs.queue_measurements.snapshot(uid, bs._time_s)["offered_bytes"] == 0
    bs.queue_measurements.close()
