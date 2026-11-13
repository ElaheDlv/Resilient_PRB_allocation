"""Joint-control causality, telemetry validity, atomic commands and replay."""

from dataclasses import FrozenInstanceError, replace
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import settings

from network_layer.joint_control import (
    JointRICAction,
    JointRICBridge,
    URLLCSchedulerMode as Mode,
)
from network_layer.queue_measurements import QueueMeasurements
from network_layer.ue_telemetry import (
    UETelemetryCollector,
    BatchWindow,
    UETelemetry,
    TelemetryReport,
)
from network_layer.deadline_risk import encode_deadline_risk

ROOT = Path(__file__).resolve().parents[2]


def telemetry_fixture(measurements=True):
    qm = QueueMeasurements() if measurements else None
    ue = NS(
        slice_type="URLLC",
        queue_budget_s=0.002,
        queue_budget_class="tight",
        downlink_mcs_data=None,
        downlink_cqi=0,
    )
    buffers = {}
    heads = {"u": 0.0}
    bs = NS(
        _time_s=0.0,
        queue_measurements=qm,
        get_dl_buf_bytes=lambda uid: buffers.get(
            uid,
            (
                sum(b.remaining_bytes for b in qm.pending.get(uid, ()))
                if qm is not None
                else 0
            ),
        ),
        get_dl_head_wait_s=lambda uid: heads.get(uid, 0.0),
    )
    cell = NS(
        base_station=bs,
        cell_id="c",
        connected_ue_list={"u": ue},
        slice_dl_prb_quota={"URLLC": 63, "eMBB": 140},
        base_dl_prb_quota={"URLLC": 60, "eMBB": 140},
        urllc_scheduler_mode=Mode.PROPORTIONAL,
    )
    if qm:
        qm.set_budget("u", 0.002, "tight")
    return cell, qm, buffers, heads


def test_telemetry_reports_readonly_fields_units_and_missing_measurements():
    cell, qm, buf, heads = telemetry_fixture()
    c = UETelemetryCollector(cell)
    buf["u"] = 100
    heads["u"] = 0.003
    cell.base_station._time_s = 0.005
    u = c.collect(5).ues[0]
    assert (u.budget_s, u.queued_bytes, u.hol_wait_s, u.capacity_bps_per_prb) == (
        0.002,
        100,
        0.003,
        0.0,
    )
    assert (
        u.slice_quota_prbs == 63
        and u.slice_activated_reserve_prbs == 3
        and u.cell_activated_reserve_prbs == 3
    )
    assert u.window.offered_batches is None
    assert not encode_deadline_risk(c.collect(5)).valid[2]
    with pytest.raises(FrozenInstanceError):
        u.queued_bytes = 1
    assert not qm.totals and not qm.pending  # reporting did not create fake queue data
    cell.connected_ue_list["u"].downlink_cqi = None
    assert c.collect(5).ues[0].capacity_bps_per_prb is None
    cell2, _, _, _ = telemetry_fixture(False)
    assert UETelemetryCollector(cell2).collect(0).ues[0].window.known_failures is None
    c.close()
    assert not qm._listeners


def test_causal_cohorts_partial_service_overflow_equality_and_no_double_count():
    cell, qm, _, _ = telemetry_fixture()
    c = UETelemetryCollector(cell, 0.010)
    qm.arrival("u", 0.001, 10, 10)
    qm.service("u", 0.002, 5)
    cell.base_station._time_s = 0.003
    w = c.collect(3).ues[0].window
    assert (
        w.offered_batches == 1
        and w.known_failures == 0
        and w.matured_offered_batches == 0
    )
    qm.service("u", 0.003, 5)  # exact budget, on time
    qm.arrival("u", 0.004, 10, 5)
    qm.service("u", 0.008, 5)  # overflow and late count only once
    qm.arrival("u", 0.005, 10, 10)  # overdue pending
    qm.arrival("u", 0.009, 10, 0)  # fully dropped, immediately known but not mature
    cell.base_station._time_s = 0.010
    w = c.collect(10).ues[0].window
    assert (
        w.offered_batches,
        w.known_failures,
        w.matured_offered_batches,
        w.matured_failed_batches,
        w.pending_overdue_batches,
    ) == (4, 3, 3, 2, 1)
    cell.base_station._time_s = 0.020
    w = c.collect(20).ues[0].window
    assert (
        w.offered_batches == 0
        and w.known_failures == 0
        and w.pending_overdue_batches == 1
    )
    qm.close()


def test_listener_does_not_rely_on_truncated_ring_or_steal_diagnostic_sink():
    cell, qm, _, _ = telemetry_fixture()
    seen = []
    qm.event_sink = seen.append
    c = UETelemetryCollector(cell, 0.1)
    for k in range(1, 501):
        qm.arrival("u", k * 0.0001, 1, 1)
        qm.service("u", k * 0.0001, 1)
    cell.base_station._time_s = 0.1
    w = c.collect(100).ues[0].window
    assert len(qm.events) == 256 and len(seen) == 1500
    assert (
        w.offered_batches == 500
        and w.matured_offered_batches == 500
        and w.known_failures == 0
    )
    assert c.collect(100).ues[0].window == w
    c.close()
    qm.close()


def test_late_subscription_and_future_events_do_not_become_successes():
    cell, qm, _, _ = telemetry_fixture()
    qm.arrival("u", 0.010, 1, 1)
    qm.service("u", 0.011, 1)
    cell.base_station._time_s = 0.020
    c = UETelemetryCollector(cell, 0.1)
    assert not c.collect(20).ues[0].window.history_available
    qm.arrival("u", 0.030, 1, 1)
    qm.service("u", 0.031, 1)
    cell.base_station._time_s = 0.121
    w = c.collect(121).ues[0].window
    assert w.offered_batches == 1 and w.known_failures == 0 and w.history_available
    c._event(
        dict(
            kind="arrival",
            imsi="u",
            time_s=1.0,
            budget_s=0.002,
            overflow_bytes=0,
            accepted_bytes=1,
            batch_id=999,
        )
    )
    with pytest.raises(RuntimeError):
        c.collect(121)
    c.close()
    qm.close()


def test_empty_exposure_missing_budget_and_episode_counter_cleanup():
    cell, qm, _, _ = telemetry_fixture()
    c = UETelemetryCollector(cell)
    cell.base_station._time_s = 0.2
    f = encode_deadline_risk(c.collect(200))
    assert f.values[:2] == (0.0, 0.0) and f.values[2:5] == (None, None, None)
    # A positive UE budget without sidecar budget tracking is unknown, not zero.
    qm.budgets.clear()
    assert c.collect(200).ues[0].window.known_failures is None
    qm.set_budget("u", 0.002, "tight")
    qm.arrival("u", 0.2, 100, 100)
    cell.base_station._time_s = 0.21
    qm.finalize("u", 0.21)
    c.reset_episode()
    qm.set_budget("u", 0.002, "tight")
    assert c.collect(210).ues[0].window.offered_batches == 0
    assert not c.records
    c.close()
    qm.close()


def report_rows():
    def row(uid, label, budget, queued, wait, n, failed):
        return UETelemetry(
            uid,
            "URLLC",
            budget,
            label,
            queued,
            wait,
            100000.0,
            BatchWindow(0.0, 0.1, True, True, n, failed, n, failed, 0),
            63,
            3,
            3,
            Mode.FEASIBILITY_AWARE,
        )

    return TelemetryReport(
        0.1,
        100,
        "c",
        0.1,
        tuple(
            [
                row("a", "tight", 0.002, 10, 0.004, 30, 3),
                row("b", "tight", 0.002, 0, 0, 30, 0),
                row("c", "relaxed", 0.010, 10, 0.005, 40, 8),
            ]
        ),
    )


def test_separate_six_feature_definitions_and_strict_exposure():
    r = report_rows()
    f = encode_deadline_risk(r)
    assert f.values == pytest.approx((2.0, 0.5, 0.05, 0.2, 0.2, 1.0)) and all(f.valid)
    partial = replace(
        r,
        ues=(
            replace(
                r.ues[0], window=replace(r.ues[0].window, matured_offered_batches=0)
            ),
        )
        + r.ues[1:],
    )
    f = encode_deadline_risk(partial)
    assert f.values[2] is None and f.values[4] is None and f.valid[3]
    f = encode_deadline_risk(
        replace(r, ues=tuple(replace(u, urllc_scheduler_mode="legacy") for u in r.ues))
    )
    assert f.values[5] is None
    assert encode_deadline_risk(replace(r, ues=())).values == (None,) * 6


def bridge_fixture(monkeypatch):
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    monkeypatch.setattr(settings, "RES_DQN_TOTAL_RESILIENCY_PRB", 18)
    cell = NS(
        max_dl_prb=218,
        _effective_max_dl_prb=180,
        base_dl_prb_quota={"eMBB": 126, "URLLC": 54},
        slice_dl_prb_quota={"eMBB": 126, "URLLC": 54},
        _joint_urllc_mode=None,
        connected_ue_list={},
    )

    # Mimic Cell's read-only public property on a minimal real class.
    class FakeCell(NS):
        @property
        def urllc_scheduler_mode(self):
            return self._joint_urllc_mode

    cell = FakeCell(**vars(cell))
    engine = NS(cell_list={"c": cell}, sim_step=0, in_ric_callback=False)
    b = JointRICBridge(engine, cell)
    b.begin_episode(episode_token=0)
    return engine, cell, b


def test_atomic_signed_resources_policy_validation_and_boundaries(monkeypatch):
    engine, cell, b = bridge_fixture(monkeypatch)
    before = (dict(cell.slice_dl_prb_quota), cell.urllc_scheduler_mode)
    for bad in [
        JointRICAction(19, 0, Mode.FEASIBILITY_AWARE),
        JointRICAction(-1, 0, Mode.PROPORTIONAL),
    ]:
        engine.sim_step = 10
        engine.in_ric_callback = True
        with pytest.raises(AssertionError):
            b.apply(bad)
        assert (cell.slice_dl_prb_quota, cell.urllc_scheduler_mode) == before
    cell.connected_ue_list = {"u": NS(slice_type="URLLC", queue_budget_s=None)}
    with pytest.raises(ValueError):
        b.apply(JointRICAction(0, 3, Mode.FEASIBILITY_AWARE))
    assert (cell.slice_dl_prb_quota, cell.urllc_scheduler_mode) == before
    cell.connected_ue_list = {}
    engine.sim_step = 9
    with pytest.raises(RuntimeError):
        b.apply(JointRICAction(0, 3, Mode.FEASIBILITY_AWARE))
    engine.sim_step = 10
    engine.in_ric_callback = False
    with pytest.raises(RuntimeError):
        b.apply(JointRICAction(0, 3, Mode.FEASIBILITY_AWARE))
    engine.in_ric_callback = True
    receipt = b.apply(JointRICAction(0, 3, Mode.FEASIBILITY_AWARE))
    assert receipt.first_service_step == 11 and cell.slice_dl_prb_quota == {
        "eMBB": 126,
        "URLLC": 57,
    }
    assert cell.urllc_scheduler_mode is Mode.FEASIBILITY_AWARE
    with pytest.raises(RuntimeError):
        b.apply(JointRICAction(0, 0, Mode.PROPORTIONAL))
    for args in [
        (0.5, 0, "PROPORTIONAL"),
        (True, 0, "PROPORTIONAL"),
        (0, 0, "INVALID"),
    ]:
        with pytest.raises(ValueError):
            JointRICAction(*args)


def test_pool_constancy_and_midinterval_reset_cannot_bypass_validation(monkeypatch):
    engine, cell, b = bridge_fixture(monkeypatch)
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18.5)
    with pytest.raises(RuntimeError):
        b.check_physics()
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 18)
    engine.sim_step = 5
    engine.in_ric_callback = True
    with pytest.raises(RuntimeError):
        b.begin_episode(Mode.FEASIBILITY_AWARE, episode_token=1)
    assert cell.urllc_scheduler_mode is Mode.PROPORTIONAL
    engine.sim_step = 10
    b.begin_episode(Mode.FEASIBILITY_AWARE, episode_token=1)
    assert cell.urllc_scheduler_mode is Mode.FEASIBILITY_AWARE


def run_demo(
    tmp_path, steps=80, config=None, tag="demo", base_scheduler="proportional_wc"
):
    dest = tmp_path / f"{tag}.json"
    cmd = [
        sys.executable,
        str(ROOT / "backend/tools/phase2a_demo.py"),
        "--steps",
        str(steps),
        "--output",
        str(dest),
    ]
    cmd += ["--base-scheduler", base_scheduler]
    if config:
        cmd += ["--config", str(config)]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(dest.read_text())


@pytest.mark.parametrize("severity", [0, 10, 20])
def test_real_joint_switching_timing_jamming_replay_and_no_early_retroaction(
    tmp_path, severity
):
    entry = json.loads(
        (
            ROOT
            / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
        ).read_text()
    )
    entry["episodes"][0]["jam_var"]["period_steps"] = 20
    config = tmp_path / "scenario.json"
    config.write_text(json.dumps(entry))
    a = run_demo(tmp_path, 80, config, "a")
    b = run_demo(tmp_path, 80, config, "b")
    assert a["trajectory_sha256"] == b["trajectory_sha256"]
    assert a["history"] == b["history"] and a["intervals"] == b["intervals"]
    assert len(a["service_history"]) == 80 and len(a["intervals"]) == 8
    commands = [r for r in a["history"] if r["kind"] == "joint_command"]
    for record in commands:
        step = record["receipt"]["applied_after_step"]
        assert step % 10 == 0 and record["telemetry"]["service_step"] == step
        assert len(record["telemetry"]["ues"]) == 16
        current = a["service_history"][step - 1]
        previous = record["receipt"]["mode_before"]
        assert current["mode"] == (
            "deadline_feasible"
            if previous == "FEASIBILITY_AWARE"
            else "proportional_wc"
        )
        if step < 80:
            following = a["service_history"][step]
            assert following["mode"] == (
                "deadline_feasible"
                if record["action"]["urllc_scheduler_mode"] == "FEASIBILITY_AWARE"
                else "proportional_wc"
            )
        before = dict(record["receipt"]["quotas_before"])
        after = dict(record["receipt"]["quotas_after"])
        assert after["eMBB"] - before["eMBB"] == record["action"]["delta_embb_prb"]
        assert after["URLLC"] - before["URLLC"] == record["action"]["delta_urllc_prb"]
    for s in a["service_history"]:
        assert (
            s["embb_mode"] == "proportional_wc"
            and 0 <= s["activated_reserve_prbs"] <= 18
        )
        assert all(s["allocated_prbs"][sl] <= q for sl, q in s["quotas"].items())
    assert len({s["effective_prbs"] for s in a["service_history"]}) == (
        1 if severity == 0 else 2
    )


def test_real_episode_reset_clears_exposure_policy_and_action_cycle(tmp_path):
    entry = json.loads(
        (
            ROOT / "backend/evaluations/phase1/main_scenarios/scenario_j0.json"
        ).read_text()
    )["episodes"][0]
    entry["duration_steps"] = 3
    config = tmp_path / "two.json"
    config.write_text(
        json.dumps({"episodes": [dict(entry, id="one"), dict(entry, id="two")]})
    )
    r = run_demo(tmp_path, 80, config)
    assert r["complete_episodes"] == 2 and r["steps"] == 60
    starts = [v for v in r["history"] if v["kind"] == "episode_initialized"]
    assert [v["after_step"] for v in starts] == [0, 30]
    for interval in r["intervals"]:
        assert interval["ticks"] == 10
        assert interval["risk_features"]["valid"][2:5] == [False] * 3
    new = next(
        v
        for v in r["history"]
        if v["kind"] == "joint_command" and v["episode_id"] == "two"
    )
    assert new["receipt"]["applied_after_step"] == 40
    assert all(u["window"]["offered_batches"] == 10 for u in new["telemetry"]["ues"])
    assert r["service_history"][30]["mode"] == "proportional_wc"


def test_future_csv_content_does_not_change_observations_or_actions(tmp_path):
    import csv

    entry = json.loads(
        (
            ROOT / "backend/evaluations/phase1/main_scenarios/scenario_j20.json"
        ).read_text()
    )
    a = run_demo(tmp_path, 80, tag="original_prefix")
    source = ROOT / entry["episodes"][0]["slices"]["URLLC"]["trace"]
    target = tmp_path / "future.csv"
    with source.open(newline="") as f, target.open("w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if float(row["time"]) >= 1:
                row["length"] = str(int(row["length"]) * 7)
            w.writerow(row)
    entry["episodes"][0]["slices"]["URLLC"]["trace"] = str(target)
    config = tmp_path / "future.json"
    config.write_text(json.dumps(entry))
    b = run_demo(tmp_path, 80, config, "future_prefix")
    assert a["trajectory_sha256"] == b["trajectory_sha256"]
    assert a["history"] == b["history"] and a["intervals"] == b["intervals"]


def test_physical_reserve_change_is_rejected_before_real_network_service(
    monkeypatch, tmp_path
):
    from functools import partial
    from network_layer.simulation_engine import SimulationEngine
    from network_layer.xApps.xapp_scripted_joint_ric import xAppScriptedJointRIC

    topology = copy.deepcopy(settings.RAN_DEFAULT_BS_LIST[:1])
    topology[0]["cell_list"] = topology[0]["cell_list"][:1]
    monkeypatch.setattr(settings, "RAN_DEFAULT_BS_LIST", topology)
    for key, value in dict(
        PHASE0_MEASUREMENTS=True,
        RES_DQN_TRANSITION_MODE="causal",
        SIM_STEP_TIME_DEFAULT=0.001,
        JAM_RESERVED_PRBS=18,
        RES_DQN_TOTAL_RESILIENCY_PRB=18,
        PRB_GYM_CONFIG_PATH=str(
            ROOT / "backend/evaluations/phase1/main_scenarios/scenario_j20.json"
        ),
    ).items():
        monkeypatch.setattr(settings, key, value)
    engine = SimulationEngine()
    engine.reset_network()
    engine.network_setup(xapp_classes=[partial(xAppScriptedJointRIC, enabled=True)])
    app = next(iter(engine.ric.xapp_list.values()))
    bs = app.cell.base_station
    monkeypatch.setattr(settings, "JAM_RESERVED_PRBS", 17)
    engine.sim_step = 1
    with pytest.raises(RuntimeError):
        engine.step(0.001)
    assert bs._time_s == 0 and not bs.queue_measurements.totals
    app.close()
    bs.queue_measurements.close()
    engine.reset_network()


def test_unobserved_jammer_settings_do_not_enter_telemetry(monkeypatch):
    cell, qm, _, _ = telemetry_fixture()
    c = UETelemetryCollector(cell)
    before = c.collect(0)
    monkeypatch.setattr(settings, "JAM_VAR_PRBS_LIST", [199, 198, 197])
    monkeypatch.setattr(settings, "JAM_VAR_DUTY_LIST", [0.99, 0.01])
    assert c.collect(0) == before
    c.close()
    qm.close()


def test_joint_policy_keeps_preconfigured_legacy_embb_scheduler(tmp_path):
    r = run_demo(tmp_path, 40, base_scheduler="legacy")
    assert all(v["embb_mode"] == "legacy" for v in r["service_history"])
    assert {v["mode"] for v in r["service_history"]} == {
        "proportional_wc",
        "deadline_feasible",
    }
    assert all(
        all(v["allocated_prbs"][sl] <= q for sl, q in v["quotas"].items())
        for v in r["service_history"]
    )


def test_disabled_joint_xapp_is_inert_without_a_network():
    from network_layer.xApps.xapp_scripted_joint_ric import xAppScriptedJointRIC

    app = xAppScriptedJointRIC(ric=NS(), enabled=False)
    app.start()
    app.step()
    app.close()
    assert app.history == [] and not hasattr(app, "env") and not hasattr(app, "bridge")
