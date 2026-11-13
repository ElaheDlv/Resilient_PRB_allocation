"""Capacity-normalized paired fairness study; validation precedes held-out tests."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REGIMES = {"nominal": 0.60, "near": 0.96, "overloaded": 1.15}
POLICIES = ["proportional_wc", "deadline_slack", "deadline_feasible", "deadline_fair"]


def run(out, label, config, seed, scheduler, fraction=0.35, steps=3000):
    report = out / f"{label}.json"
    trace = out / f"{label}.jsonl.gz"
    cmd = [
        sys.executable,
        str(ROOT / "backend/tools/phase0_evaluate.py"),
        "--config",
        str(config),
        "--policy",
        "fixed",
        "--transitions",
        "causal",
        "--scheduler",
        scheduler,
        "--seed",
        str(seed),
        "--steps",
        str(steps),
        "--phase1-metrics",
        "--fixed-e",
        "0",
        "--fixed-u",
        "3",
        "--diagnostics",
        str(trace),
        "--output",
        str(report),
    ]
    guard = 3 if scheduler in {"deadline_feasible", "deadline_fair"} else 5
    with (out / f"{label}.log").open("w") as log:
        subprocess.run(
            cmd,
            cwd=ROOT,
            env=dict(
                os.environ,
                DEADLINE_STARVATION_TICKS=str(guard),
                DEADLINE_FAIR_FRACTION=str(fraction),
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    r = json.loads(report.read_text())
    row = {
        k: r[k]
        for k in [
            "seed",
            "scheduler",
            "trajectory_sha256",
            "elapsed_seconds",
            "phase1",
            "diagnostics",
            "config_sha256",
        ]
    }
    row.update(
        label=label,
        fair_fraction=fraction if scheduler == "deadline_fair" else None,
        steps=steps,
        starvation_limit=guard,
    )
    return row


def calibration(out, seed, budget_seed):
    results = []
    for severity in [0, 20]:
        source = (
            ROOT
            / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
        )
        catalog = json.loads(source.read_text())
        catalog["episodes"][0]["slices"]["URLLC"][
            "queue_budget_assignment_seed"
        ] = budget_seed
        config = out / f"calibration_seed{seed}_j{severity}.json"
        config.write_text(json.dumps(catalog, indent=2) + "\n")
        results.append(
            run(
                out,
                f"calibration_seed{seed}_j{severity}",
                config,
                seed,
                "proportional_wc",
                steps=20,
            )
        )
    rates = {
        uid: u["segments"]["all"]["mean_bytes_per_prb"]
        for uid, u in results[0]["diagnostics"]["per_ue"].items()
    }
    for uid, rate in rates.items():
        assert (
            abs(
                rate
                - results[1]["diagnostics"]["per_ue"][uid]["segments"]["all"][
                    "mean_bytes_per_prb"
                ]
            )
            < 1e-8
        )
    qoff = results[0]["phase1"]["resources"]["URLLC"]["mean_quota_prbs"]
    qon = results[1]["phase1"]["resources"]["URLLC"]["mean_quota_prbs"]
    assert abs(qoff - 63) < 1e-9 and abs(qon - 57) < 1e-9
    return dict(
        seed=seed,
        budget_seed=budget_seed,
        rates=rates,
        quota_off=63,
        quota_on=57,
        assignments=results[0]["phase1"]["assignments"],
        method="Exogenous 20-tick rate/quota calibration; performance discarded",
    )


def scaled_trace(out, seed, regime, bytes_per_batch):
    original = ROOT / "backend/notebooks/Resilient_data/urllc_dl.csv"
    dest = out / f"urllc_seed{seed}_{regime}.csv"
    with original.open(newline="") as src, dest.open("w", newline="") as target:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(
            target, fieldnames=reader.fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for row in reader:
            row["length"] = str(
                max(1, math.floor(int(row["length"]) * bytes_per_batch / 650))
            )
            writer.writerow(row)
    return dest


def evaluate(out, stage, seeds, fractions, budget_seed):
    rows = []
    calibrations = []
    for seed in seeds:
        cal = calibration(out, seed, budget_seed)
        calibrations.append(cal)
        for regime, load in REGIMES.items():
            size = max(
                1,
                math.floor(
                    load * cal["quota_on"] / sum(1 / b for b in cal["rates"].values())
                ),
            )
            trace = scaled_trace(out, seed, regime, size)
            for severity in [0, 20]:
                source = (
                    ROOT
                    / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
                )
                catalog = copy.deepcopy(json.loads(source.read_text()))
                cfg = catalog["episodes"][0]["slices"]["URLLC"]
                cfg.update(trace=str(trace), queue_budget_assignment_seed=budget_seed)
                config = out / f"{stage}_{regime}_seed{seed}_j{severity}.json"
                config.write_text(json.dumps(catalog, indent=2) + "\n")
                policies = (
                    [("proportional_wc", None), ("deadline_feasible", None)]
                    if stage == "validation"
                    else [(p, None) for p in POLICIES[:-1]]
                )
                policies += [("deadline_fair", a) for a in fractions]
                pair = []
                for scheduler, fraction in policies:
                    label = f"{stage}_{regime}_j{severity}_seed{seed}_{scheduler}" + (
                        f"_a{fraction}" if fraction is not None else ""
                    )
                    row = run(
                        out,
                        label,
                        config,
                        seed,
                        scheduler,
                        fraction if fraction is not None else 0.35,
                    )
                    row.update(
                        stage=stage,
                        regime=regime,
                        severity=severity,
                        bytes_per_batch=size,
                        fluid_prbs=sum(size / b for b in cal["rates"].values()),
                        rounded_prbs=sum(
                            math.ceil(size / b) for b in cal["rates"].values()
                        ),
                        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    )
                    for uid, b in cal["rates"].items():
                        actual = row["diagnostics"]["per_ue"][uid]["segments"]["all"][
                            "mean_bytes_per_prb"
                        ]
                        assert abs(actual - b) < 1e-8
                    pair.append(row)
                    rows.append(row)
                    print(label, round(row["elapsed_seconds"], 3), flush=True)
                for field in ["exogenous_sha256", "quota_sha256", "assignments"]:
                    assert all(
                        r["phase1"][field] == pair[0]["phase1"][field] for r in pair
                    ), field
    return rows, calibrations


def select(rows):
    p = statistics.mean(
        r["phase1"]["overall"]["failure_fraction_lower"]
        for r in rows
        if r["scheduler"] == "proportional_wc"
    )
    f = statistics.mean(
        r["phase1"]["overall"]["failure_fraction_lower"]
        for r in rows
        if r["scheduler"] == "deadline_feasible"
    )
    bound = f + 0.5 * max(0, p - f)
    scores = {
        a: dict(
            failure=statistics.mean(
                r["phase1"]["overall"]["failure_fraction_lower"]
                for r in rows
                if r["fair_fraction"] == a
            ),
            worst=statistics.mean(
                r["phase1"]["worst_ue_failure_fraction"]
                for r in rows
                if r["fair_fraction"] == a
            ),
        )
        for a in [0.15, 0.35, 0.65]
    }
    eligible = [a for a, s in scores.items() if s["failure"] <= bound + 1e-12]
    selected = (
        min(eligible, key=lambda a: (scores[a]["worst"], scores[a]["failure"], a))
        if eligible
        else min(scores, key=lambda a: (scores[a]["failure"], scores[a]["worst"], a))
    )
    return dict(
        selected_fraction=selected,
        scores=scores,
        baseline_failure=p,
        feasible_failure=f,
        retention_threshold=bound,
        retention_constraint_satisfied=bool(eligible),
        validation_seeds=[6001, 6002],
        validation_budget_seed=2028,
        heldout_seeds=list(range(7001, 7007)),
        heldout_budget_seed=2026,
        rule="Minimize mean worst-UE failure subject to retaining >=50% of the mean aggregate feasible-vs-proportional gain; ties by aggregate then smaller fraction; if none eligible minimize aggregate",
    )


def supplementary_protocol(stage):
    if stage not in {
        "boundary_validation",
        "threshold_validation",
        "threshold_heldout",
        "recovery_validation",
        "recovery_heldout",
    }:
        raise ValueError("Unknown supplementary study stage")
    prefix = stage.split("_")[0]
    validation = stage.endswith("validation")
    heldout_seeds = (
        list(range(10001, 10007)) if prefix == "recovery" else list(range(9001, 9007))
    )
    return dict(
        regime=prefix,
        load={"boundary": 0.995, "threshold": 1.0005, "recovery": 1.05}[prefix],
        seeds=[6001, 6002] if validation else heldout_seeds,
        budget_seed=2028 if validation else 2026,
        heldout_seeds=heldout_seeds,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage",
        choices=[
            "validation",
            "heldout",
            "boundary_validation",
            "threshold_validation",
            "threshold_heldout",
            "recovery_validation",
            "recovery_heldout",
        ],
        required=True,
    )
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    global REGIMES
    if a.stage.startswith(("boundary_", "threshold_", "recovery_")):
        protocol = supplementary_protocol(a.stage)
        prefix = protocol["regime"]
        load = protocol["load"]
        REGIMES = {prefix: load}
        chosen = json.loads((out / "selection.json").read_text())
        seeds = protocol["seeds"]
        budget_seed = protocol["budget_seed"]
        (out / f"{prefix}_plan.json").write_text(
            json.dumps(
                dict(
                    load=load,
                    heldout_seeds=protocol["heldout_seeds"],
                    fair_fraction=chosen["selected_fraction"],
                    reason=(
                        "Recoverable ON-overload/OFF-feasible control, added because rho1.15 cannot recover even OFF; alpha stays frozen"
                        if prefix == "recovery"
                        else "Validation-only boundary refinement; alpha stays frozen"
                    ),
                ),
                indent=2,
            )
            + "\n"
        )
        rows, cals = evaluate(
            out, a.stage, seeds, [chosen["selected_fraction"]], budget_seed
        )
    elif a.stage == "validation":
        rows, cals = evaluate(out, a.stage, [6001, 6002], [0.15, 0.35, 0.65], 2028)
        (out / "selection.json").write_text(json.dumps(select(rows), indent=2) + "\n")
    else:
        chosen = json.loads((out / "selection.json").read_text())
        rows, cals = evaluate(
            out,
            a.stage,
            chosen["heldout_seeds"],
            [chosen["selected_fraction"]],
            chosen["heldout_budget_seed"],
        )
    (out / f"{a.stage}_summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    (out / f"{a.stage}_calibration.json").write_text(
        json.dumps(cals, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
