"""Causal service credits, resource bounds, and four-policy simulator pairing."""

from collections import deque
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from network_layer.deadline_scheduler import QueueView, deadline_allocation
from network_layer.fair_service_credit import FairServiceCredit

ROOT = Path(__file__).resolve().parents[2]


def test_credits_use_actual_service_and_admitted_queue_increments():
    c = FairServiceCredit()
    q = {"u": QueueView(13, 10, 0.002, [(0, 13)])}
    credits, ent = c.prepare(q, 3)
    assert credits["u"] == 13
    # Two PRBs can carry 20 bytes, but only 13 actual bytes are serviced.
    c.service("u", 13, 0)
    assert c.states["u"]["credit"] == 0
    credits, _ = c.prepare(q, 3)
    assert credits["u"] == 13
    c.remove("u")
    assert not c.states


def test_credits_are_bounded_and_capacity_scaled_without_future_information():
    c = FairServiceCredit(window_ticks=20)
    for tick in range(100):
        q = {
            u: QueueView((tick + 1) * 14, 10, 0.002, [(0, (tick + 1) * 14)])
            for u in "ab"
        }
        credits, ent = c.prepare(q, 2)
        assert ent == {"a": 10, "b": 10}
        assert all(0 <= v <= 200 for v in credits.values())
        for u in q:
            c.service(u, 0, q[u].queued_bytes)
    assert credits == {"a": 200, "b": 200}
    # Full drain can create negative credit, bounded on the same scale.
    c.service("a", 100000, 0)
    assert c.states["a"]["credit"] == -200
    # Zero-rate queues receive no nonfinite credit or attainable service target.
    c.prepare({"z": QueueView(10, 0, None, [(0, 10)])}, 2)
    assert c.states["z"]["credit"] == 0


def test_fraction_zero_preserves_feasibility_and_debt_can_protect_expired_head():
    views = {
        "old": QueueView(20, 10, 0.001, deque([(0, 20)])),
        "new": QueueView(20, 10, 0.010, deque([(0.003, 20)])),
    }
    baseline = deadline_allocation(views, 2, 0.004, 0.001, {}, feasibility=True)
    args = dict(
        fair_credits={"old": 20, "new": 0}, fair_entitlements={"old": 10, "new": 10}
    )
    zero = deadline_allocation(
        views, 2, 0.004, 0.001, {}, feasibility=True, fair_fraction=0, **args
    )
    fair = deadline_allocation(
        views, 2, 0.004, 0.001, {}, feasibility=True, fair_fraction=1, **args
    )
    assert zero == baseline == {"old": 0, "new": 2}
    assert fair == {"old": 2, "new": 0}
    assert list(views["old"].fifo) == [(0, 20)]
    # The independent guard still serves a protected UE with negative credit.
    out = deadline_allocation(
        views,
        1,
        0.004,
        0.001,
        {"new": 3},
        starvation_limit=3,
        feasibility=True,
        fair_fraction=1,
        **args,
    )
    assert out["new"] == 1


def test_fair_integer_useful_caps_and_work_conservation():
    for sizes in itertools.product(range(4), repeat=3):
        views = {
            u: QueueView(size, 1, 0.002, [(0, size)]) for u, size in zip("abc", sizes)
        }
        for quota in range(7):
            for cap in [None, 0, 2]:
                for fraction in [0, 0.35, 1]:
                    out = deadline_allocation(
                        views,
                        quota,
                        0.010,
                        0.001,
                        {},
                        per_ue_cap=cap,
                        feasibility=True,
                        fair_fraction=fraction,
                        fair_credits=dict(a=10, b=0, c=-5),
                        fair_entitlements=dict.fromkeys("abc", 1),
                    )
                    limits = [min(n, cap) if cap is not None else n for n in sizes]
                    assert sum(out.values()) == min(quota, sum(limits))
                    assert all(
                        isinstance(v, int) and 0 <= v <= limit
                        for v, limit in zip(out.values(), limits)
                    )
    sub = {"u": QueueView(1, 0.2, 0.010, [(0, 1)])}
    out = deadline_allocation(
        sub,
        5,
        0,
        0.001,
        {},
        feasibility=True,
        fair_fraction=0.5,
        fair_credits={"u": 1},
        fair_entitlements={"u": 1},
    )
    assert out == {
        "u": 5
    }  # quantum cannot fit fair budget, but fallback remains useful
    for fraction in [-1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            deadline_allocation(
                sub, 5, 0, 0.001, {}, feasibility=True, fair_fraction=fraction
            )


@pytest.mark.parametrize("severity", [0, 20])
def test_four_policy_pairing_zero_fraction_replay_and_jammer_transitions(
    tmp_path, severity
):
    config = json.loads(
        (
            ROOT
            / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
        ).read_text()
    )
    config["episodes"][0]["jam_var"]["period_steps"] = 10
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    rows = []
    for scheduler, fraction in [
        ("proportional_wc", 0.35),
        ("deadline_slack", 0.35),
        ("deadline_feasible", 0.35),
        ("deadline_fair", 0),
        ("deadline_fair", 0.35),
    ]:
        dest = tmp_path / f"{scheduler}_{fraction}.json"
        cmd = [
            sys.executable,
            str(ROOT / "backend/tools/phase0_evaluate.py"),
            "--config",
            str(path),
            "--scheduler",
            scheduler,
            "--policy",
            "fixed",
            "--fixed-u",
            "3",
            "--transitions",
            "causal",
            "--phase1-metrics",
            "--steps",
            "40",
            "--output",
            str(dest),
        ]
        subprocess.run(
            cmd,
            cwd=ROOT,
            env=dict(
                os.environ,
                DEADLINE_STARVATION_TICKS="3",
                DEADLINE_FAIR_FRACTION=str(fraction),
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        rows.append(json.loads(dest.read_text()))
    assert rows[2]["trajectory_sha256"] == rows[3]["trajectory_sha256"]
    for field in ["exogenous_sha256", "quota_sha256", "assignments"]:
        assert all(r["phase1"][field] == rows[0]["phase1"][field] for r in rows)
    for r in rows:
        assert len(r["phase1"]["recovery"]) == (0 if severity == 0 else 4)
        stats = r["phase1"]["resources"]["URLLC"]
        assert (
            stats["scheduler_median_us"]
            <= stats["scheduler_p95_us"]
            <= stats["scheduler_p99_us"]
        )
        assert stats["allocated_prb_s"] + stats["unused_prb_s"] == pytest.approx(
            stats["quota_prb_s"]
        )


def test_finite_deadline_bound_does_not_confuse_longrun_overload_with_finite_infeasibility():
    sys.path.insert(0, str(ROOT / "backend/tools"))
    from phase12_analyze import demand_bound

    # 1.1 PRBs/interval exceeds long-run capacity, but finite budget carry is
    # sufficient to avoid this necessary-bound violation on a short horizon.
    rates = {"u": 10.0}
    budgets = {"u": 0.010}
    assert not demand_bound(11, rates, budgets, 100, 1)["zero_miss_impossible"]
    long = demand_bound(11, rates, budgets, 200, 1)
    assert long["due_batches"] == {"u": 190}
    assert long["necessary_fluid_prbs"] == 209
    assert long["zero_miss_impossible"]
    assert not demand_bound(10, rates, budgets, 200, 1)["zero_miss_impossible"]


def test_supplementary_validation_never_uses_heldout_seeds():
    sys.path.insert(0, str(ROOT / "backend/tools"))
    from phase12_study import supplementary_protocol

    validation = supplementary_protocol("threshold_validation")
    screened = supplementary_protocol("boundary_validation")
    heldout = supplementary_protocol("threshold_heldout")
    assert validation["seeds"] == screened["seeds"] == [6001, 6002]
    assert heldout["seeds"] == list(range(9001, 9007))
    assert not set(validation["seeds"]) & set(heldout["seeds"])
    assert not set(heldout["seeds"]) & set(range(8001, 8007))
    assert validation["budget_seed"] == 2028 and heldout["budget_seed"] == 2026
    assert validation["load"] == heldout["load"] == 1.0005


def test_recovery_protocol_uses_another_untouched_heldout_cohort():
    sys.path.insert(0, str(ROOT / "backend/tools"))
    from phase12_study import supplementary_protocol

    valid = supplementary_protocol("recovery_validation")
    test = supplementary_protocol("recovery_heldout")
    assert valid["seeds"] == [6001, 6002] and valid["load"] == test["load"] == 1.05
    assert test["seeds"] == list(range(10001, 10007))
    assert set(test["seeds"]).isdisjoint(
        supplementary_protocol("threshold_heldout")["seeds"]
    )
