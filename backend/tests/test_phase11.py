"""Feasibility policy and independent event-accounting checks."""

from collections import deque
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from network_layer.deadline_scheduler import QueueView, deadline_allocation
from network_layer.scheduler_diagnostics import outcome_partition

ROOT = Path(__file__).resolve().parents[2]


def allocate(views, quota, now=0, ages=None, **kwargs):
    return deadline_allocation(
        views, quota, now, 0.001, ages or {}, feasibility=True, **kwargs
    )


def test_skip_expired_or_optimistically_infeasible_until_viable_heads_served():
    views = {
        "expired": QueueView(100, 10, 0.001, [(0, 100)]),
        "viable": QueueView(20, 10, 0.010, [(0.002, 20)]),
    }
    assert allocate(views, 2, 0.003) == {"expired": 0, "viable": 2}
    # At now=0 the first head needs ten PRBs but only 2*(1+1) attainable.
    assert allocate(views, 2, 0) == {"expired": 0, "viable": 2}
    # Fallback and guard keep expired FIFO traffic serviceable, no dropping.
    assert allocate({"expired": views["expired"]}, 2, 0.003) == {"expired": 2}
    assert allocate(views, 2, 0.003, {"expired": 5})["expired"] >= 1


def test_feasibility_equality_and_partial_fifo_and_rate_efficiency():
    views = {
        "a": QueueView(30, 10, 0.002, deque([(0, 10), (0.002, 20)])),
        "b": QueueView(100, 10, 0.010, [(0, 100)]),
    }
    snapshot = list(views["a"].fifo)
    assert allocate(views, 1, 0.002) == {"a": 1, "b": 0}  # equality is on time
    assert list(views["a"].fifo) == snapshot
    equal = {
        "slow": QueueView(20, 10, 0.01, [(0, 20)]),
        "fast": QueueView(20, 20, 0.01, [(0, 20)]),
    }
    assert allocate(equal, 1) == {"slow": 0, "fast": 1}


def test_candidate_integer_caps_subbyte_and_work_conservation():
    for sizes in itertools.product(range(4), repeat=3):
        views = {uid: QueueView(n, 1, 0.002, [(0, n)]) for uid, n in zip("abc", sizes)}
        for quota in range(7):
            for cap in [None, 0, 2]:
                out = allocate(views, quota, 0.004, per_ue_cap=cap)
                limits = [min(n, cap) if cap is not None else n for n in sizes]
                assert sum(out.values()) == min(quota, sum(limits))
                assert all(
                    isinstance(g, int) and 0 <= g <= n
                    for g, n in zip(out.values(), limits)
                )
    assert allocate({"u": QueueView(1, 0.2, 0.01, [(0, 1)])}, 5) == {"u": 5}
    assert allocate({"u": QueueView(1, 0, None, [(0, 1)])}, 5) == {"u": 0}


def test_candidate_guard_bounds_persistent_unit_quantum_starvation():
    views = {u: QueueView(100, 1, 0.001, [(0, 100)]) for u in "abc"}
    ages = dict.fromkeys(views, 0)
    for _ in range(30):
        out = allocate(views, 1, 0.010, ages, starvation_limit=3)
        ages = {u: 0 if out[u] else ages[u] + 1 for u in views}
        assert max(ages.values()) <= 3 + len(views) - 1


def test_event_partition_partial_overflow_and_terminal_censoring():
    def batch(**kwargs):
        return dict(
            time_s=0, budget_s=0.002, accepted_bytes=10, overflow_bytes=0, **kwargs
        )

    rows = [
        batch(completion_s=0.002),
        batch(completion_s=0.003),
        batch(),
        dict(time_s=0.003, budget_s=0.002, accepted_bytes=10, overflow_bytes=0),
        dict(
            time_s=0,
            budget_s=0.002,
            accepted_bytes=5,
            overflow_bytes=5,
            completion_s=0.004,
        ),
        dict(time_s=0, budget_s=0.002, accepted_bytes=0, overflow_bytes=10),
    ]
    p = outcome_partition(rows, 0.004)
    assert p["offered"] == 6 and p["completed"] == 3 and p["late_completed"] == 2
    assert p["overflow"] == 2 and p["late_only"] == 1 and p["unfinished_overdue"] == 1
    assert p["unresolved_not_due"] == 1 and p["on_time"] == 1
    assert p["known_failed"] == 4
    assert p["failure_lower"] == 4 / 6 and p["failure_upper"] == 5 / 6


@pytest.mark.parametrize("severity", [0, 10, 20])
def test_diagnostics_noninterference_and_three_policy_pairing(tmp_path, severity):
    config = (
        ROOT / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
    )
    scenario = json.loads(config.read_text())
    scenario["episodes"][0]["jam_var"]["period_steps"] = 10
    config = tmp_path / "transitions.json"
    config.write_text(json.dumps(scenario))
    rows = []
    for policy in ["proportional_wc", "deadline_slack", "deadline_feasible"]:
        for diagnostics in [False, True]:
            dest = tmp_path / f"{policy}_{diagnostics}.json"
            cmd = [
                sys.executable,
                str(ROOT / "backend/tools/phase0_evaluate.py"),
                "--config",
                str(config),
                "--policy",
                "fixed",
                "--fixed-u",
                "3",
                "--transitions",
                "causal",
                "--phase1-metrics",
                "--scheduler",
                policy,
                "--steps",
                "40",
                "--output",
                str(dest),
            ]
            if diagnostics:
                cmd += ["--diagnostics", str(tmp_path / f"{policy}.jsonl.gz")]
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            rows.append(json.loads(dest.read_text()))
        assert rows[-1]["trajectory_sha256"] == rows[-2]["trajectory_sha256"]
        assert len(rows[-1]["phase1"]["recovery"]) == (0 if severity == 0 else 4)
        for uid, ue in rows[-1]["diagnostics"]["per_ue"].items():
            o = ue["segments"]["all"]["outcomes"]
            assert (
                o["known_failed"]
                == rows[-1]["phase1"]["per_ue"][uid]["batch_outcomes"]["known_failed"]
            )
    for field in ["exogenous_sha256", "quota_sha256", "assignments"]:
        assert all(r["phase1"][field] == rows[0]["phase1"][field] for r in rows)


def test_completion_priority_avoids_spreading_every_head_at_deadline():
    # Three PRBs cannot finish all heads. Original least slack finishes one;
    # feasibility EDF + shortest remaining requirement finishes two.
    views = {
        "a": QueueView(160, 155, 0.010, [(0, 160)]),
        "b": QueueView(110, 100, 0.010, [(0, 110)]),
        "c": QueueView(10, 100, 0.010, [(0, 10)]),
    }
    old = deadline_allocation(views, 3, 0.010, 0.001, {})
    new = allocate(views, 3, 0.010)
    completed = lambda grants: sum(
        int(grants[u] * q.bytes_per_prb) >= q.queued_bytes for u, q in views.items()
    )
    assert completed(old) == 1
    assert completed(new) == 2
