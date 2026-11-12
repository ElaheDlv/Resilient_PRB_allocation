"""Paired scheduler-only pilot on unchanged Phase 0 traffic and radio models."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def summarize(rows):
    def mean(getter):
        values = [getter(row) for row in rows]
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    return dict(
        runs=len(rows),
        batch_failure_lower=mean(
            lambda r: r["phase1"]["overall"]["failure_fraction_lower"]
        ),
        batch_failure_upper=mean(
            lambda r: r["phase1"]["overall"]["failure_fraction_upper"]
        ),
        completed_batch_late=mean(
            lambda r: r["phase1"]["overall"]["completed_batch_late_fraction"]
        ),
        served_byte_late=mean(
            lambda r: r["phase1"]["overall"]["served_byte_late_fraction"]
        ),
        hol_violation=mean(lambda r: r["phase1"]["overall"]["hol_violation_fraction"]),
        longest_violation_ms=mean(
            lambda r: r["phase1"]["overall"]["longest_observed_violation_s"] * 1000
        ),
        mean_violation_ms=mean(
            lambda r: r["phase1"]["overall"]["mean_observed_violation_s"] * 1000
        ),
        embb_mbps=mean(lambda r: r["phase1"]["embb_throughput_mbps"]),
        mean_reserve_prbs=mean(lambda r: r["phase1"]["mean_reserve_prbs"]),
        allocated_urllc_prbs=mean(
            lambda r: r["phase1"]["resources"]["URLLC"]["mean_allocated_prbs"]
        ),
        unused_urllc_prbs=mean(
            lambda r: r["phase1"]["resources"]["URLLC"]["mean_unused_prbs"]
        ),
        worst_ue_failure=mean(lambda r: r["phase1"]["worst_ue_failure_fraction"]),
        scheduler_mean_us=mean(
            lambda r: r["phase1"]["resources"]["URLLC"]["scheduler_mean_us"]
        ),
        scheduler_p95_us=mean(
            lambda r: r["phase1"]["resources"]["URLLC"]["scheduler_p95_us"]
        ),
        wall_seconds=mean(lambda r: r["elapsed_seconds"]),
        classes={
            name: dict(
                batch_failure_lower=mean(
                    lambda r: r["phase1"]["per_class"][name]["failure_fraction_lower"]
                ),
                served_byte_late=mean(
                    lambda r: r["phase1"]["per_class"][name][
                        "served_byte_late_fraction"
                    ]
                ),
                hol_violation=mean(
                    lambda r: r["phase1"]["per_class"][name]["hol_violation_fraction"]
                ),
            )
            for name in rows[0]["phase1"]["per_class"]
        },
        recovery_windows=[w for r in rows for w in r["phase1"]["recovery"]],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="1234,2345,3456")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--budget-classes", default='{"tight":0.002,"relaxed":0.010}')
    p.add_argument("--budget-seed", type=int, default=2026)
    p.add_argument("--fixed-e", type=int, default=0)
    p.add_argument("--fixed-u", type=int, default=3)
    p.add_argument(
        "--checkpoint",
        default="",
        help="Add frozen causal-DQN comparisons; no retraining",
    )
    p.add_argument("--per-ue-cap", type=int)
    p.add_argument(
        "--urllc-ues",
        type=int,
        default=8,
        help="Change UE population only; per-UE trace stays unchanged",
    )
    a = p.parse_args()
    if not 0 < a.steps < 9990:
        p.error("Pilot must end before the unchanged 10-second traces are exhausted")
    if a.urllc_ues < 2:
        p.error("At least two URLLC UEs are needed for the two-class pilot")
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = (
        ROOT
        / "backend/assets/episodes/test_config/gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json"
    )
    catalog = json.loads(source.read_text())
    original = (catalog["episodes"] if isinstance(catalog, dict) else catalog)[0]
    classes = json.loads(a.budget_classes)
    if len(classes) != 2:
        p.error("This pilot requires exactly two configurable budget classes")
    sys.path.insert(0, str(ROOT / "backend"))
    from network_layer.ue_budgets import budget_assignment

    budget_assignment(dict(queue_budget_classes_s=classes), 8, "validation")
    seeds = [int(s) for s in a.seeds.split(",")]
    policies = ["fixed"] + (["dqn"] if a.checkpoint else [])
    runs = []
    pair_checks = []
    for severity in [0, 10, 20]:
        entry = copy.deepcopy(original)
        entry["id"] = "phase1_pilot"
        entry["duration_steps"] = 1000
        entry["jam_var"]["prbs_list"] = [severity]
        entry["slices"]["URLLC"].update(
            queue_budget_classes_s=classes,
            queue_budget_assignment_seed=a.budget_seed,
            ue_count=a.urllc_ues,
        )
        config = out / f"scenario_j{severity}.json"
        config.write_text(json.dumps(dict(episodes=[entry]), indent=2) + "\n")
        for seed in seeds:
            for policy in policies:
                pair = []
                for scheduler in ["proportional_wc", "deadline_slack"]:
                    label = f"j{severity}_seed{seed}_{policy}_{scheduler}"
                    report = out / f"{label}.json"
                    cmd = [
                        sys.executable,
                        str(ROOT / "backend/tools/phase0_evaluate.py"),
                        "--config",
                        str(config),
                        "--policy",
                        policy,
                        "--transitions",
                        "causal",
                        "--scheduler",
                        scheduler,
                        "--seed",
                        str(seed),
                        "--steps",
                        str(a.steps),
                        "--measurements",
                        "--phase1-metrics",
                        "--fixed-e",
                        str(a.fixed_e),
                        "--fixed-u",
                        str(a.fixed_u),
                        "--output",
                        str(report),
                    ]
                    if policy == "dqn":
                        cmd += ["--checkpoint", a.checkpoint]
                    if a.per_ue_cap is not None:
                        cmd += ["--per-ue-cap", str(a.per_ue_cap)]
                    with (out / f"{label}.log").open("w") as log:
                        subprocess.run(
                            cmd,
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    row = json.loads(report.read_text())
                    if len(row["phase1"]["per_ue"]) != a.urllc_ues:
                        raise AssertionError(
                            "Pilot cohort differs from requested URLLC population"
                        )
                    row["severity"] = severity
                    runs.append(row)
                    pair.append(row)
                    print(label, round(row["elapsed_seconds"], 3), flush=True)
                checks = dict(
                    severity=severity,
                    seed=seed,
                    policy=policy,
                    same_exogenous=pair[0]["phase1"]["exogenous_sha256"]
                    == pair[1]["phase1"]["exogenous_sha256"],
                    same_budgets=pair[0]["phase1"]["assignments"]
                    == pair[1]["phase1"]["assignments"],
                    same_quotas=pair[0]["phase1"]["quota_sha256"]
                    == pair[1]["phase1"]["quota_sha256"],
                )
                if (
                    not checks["same_exogenous"]
                    or not checks["same_budgets"]
                    or (policy == "fixed" and not checks["same_quotas"])
                ):
                    raise AssertionError(checks)
                pair_checks.append(checks)
    groups = {}
    for severity in [0, 10, 20]:
        for policy in policies:
            for scheduler in ["proportional_wc", "deadline_slack"]:
                key = f"j{severity}_{policy}_{scheduler}"
                groups[key] = summarize(
                    [
                        r
                        for r in runs
                        if r["severity"] == severity
                        and r["policy"] == policy
                        and r["scheduler"] == scheduler
                    ]
                )
    manifest = dict(
        seeds=seeds,
        steps=a.steps,
        urllc_ues=a.urllc_ues,
        budget_classes_s=classes,
        budget_seed=a.budget_seed,
        fixed_reserve=[a.fixed_e, a.fixed_u],
        per_ue_cap=a.per_ue_cap,
        input_catalog=str(source.relative_to(ROOT)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint=a.checkpoint or None,
        pair_checks=pair_checks,
        groups=groups,
    )
    (out / "summary.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
