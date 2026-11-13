"""Predeclared validation selection, held-out paired test, and Phase 1 reproduction.

No DQN is trained or used. Full event/allocation traces are gzip JSONL files.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCHEDULERS = ["proportional_wc", "deadline_slack", "deadline_feasible"]


def evaluate(out, stage, seeds, schedulers, guard=5, budget_seed=2026):
    rows = []
    for severity in [0, 10, 20]:
        source = (
            ROOT
            / f"backend/evaluations/phase1/main_scenarios/scenario_j{severity}.json"
        )
        scenario = copy.deepcopy(json.loads(source.read_text()))
        scenario["episodes"][0]["slices"]["URLLC"][
            "queue_budget_assignment_seed"
        ] = budget_seed
        config = out / f"{stage}_j{severity}.json"
        config.write_text(json.dumps(scenario, indent=2) + "\n")
        for seed in seeds:
            pair = []
            for scheduler in schedulers:
                label = f"{stage}_j{severity}_seed{seed}_{scheduler}_s{guard}"
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
                    "5000",
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
                with (out / f"{label}.log").open("w") as log:
                    subprocess.run(
                        cmd,
                        cwd=ROOT,
                        env=dict(os.environ, DEADLINE_STARVATION_TICKS=str(guard)),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                raw = json.loads(report.read_text())
                row = {
                    k: raw[k]
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
                    stage=stage, severity=severity, starvation_limit=guard, label=label
                )
                rows.append(row)
                pair.append(row)
                print(label, round(raw["elapsed_seconds"], 3), flush=True)
            for field in ["exogenous_sha256", "quota_sha256", "assignments"]:
                assert all(
                    r["phase1"][field] == pair[0]["phase1"][field] for r in pair
                ), field
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--stage", choices=["reproduction", "validation", "heldout"], required=True
    )
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "reproduction":
        rows = evaluate(out, "reproduction", [1234, 2345, 3456], SCHEDULERS[:2])
        old = json.loads(
            (ROOT / "backend/evaluations/phase1/pilot_runs.json").read_text()
        )
        # Accommodate the stored artifact's explicit main/stress sections.
        if isinstance(old, dict):
            old = old.get("main", old.get("runs", []))
        for row in rows:
            matches = [
                r
                for r in old
                if r.get("study") == "main"
                and r["seed"] == row["seed"]
                and r["severity"] == row["severity"]
                and r["scheduler"] == row["scheduler"]
                and r["policy"] == "fixed"
            ]
            assert len(matches) == 1
            assert matches[0]["trajectory_sha256"] == row["trajectory_sha256"]
    elif a.stage == "validation":
        rows = []
        for guard in [3, 5, 10]:
            rows.extend(
                evaluate(
                    out,
                    f"validation_s{guard}",
                    [3001, 3002],
                    ["deadline_feasible"],
                    guard,
                    2027,
                )
            )
        scores = {
            guard: dict(
                mean_failure=statistics.mean(
                    r["phase1"]["overall"]["failure_fraction_lower"]
                    for r in rows
                    if r["starvation_limit"] == guard
                ),
                worst_ue=statistics.mean(
                    r["phase1"]["worst_ue_failure_fraction"]
                    for r in rows
                    if r["starvation_limit"] == guard
                ),
            )
            for guard in [3, 5, 10]
        }
        selected = min(
            scores, key=lambda g: (scores[g]["mean_failure"], scores[g]["worst_ue"], g)
        )
        (out / "selection.json").write_text(
            json.dumps(
                dict(
                    scores=scores,
                    selected_guard=selected,
                    rule="Minimum mean failed-offered lower bound; then mean worst UE; then smaller guard",
                    validation_seeds=[3001, 3002],
                    budget_seed=2027,
                    heldout_seeds=list(range(4001, 4009)),
                ),
                indent=2,
            )
            + "\n"
        )
    else:
        selection = json.loads((out / "selection.json").read_text())
        # Guard is a policy parameter: retain original slack guard=5, and tune
        # only candidate; run baseline/slack separately and assert all three pairs.
        rows = evaluate(out, "heldout", list(range(4001, 4009)), SCHEDULERS[:2], 5)
        rows.extend(
            evaluate(
                out,
                "heldout_candidate",
                list(range(4001, 4009)),
                [SCHEDULERS[2]],
                selection["selected_guard"],
            )
        )
        for severity in [0, 10, 20]:
            for seed in range(4001, 4009):
                pair = [
                    r for r in rows if r["severity"] == severity and r["seed"] == seed
                ]
                for field in ["exogenous_sha256", "quota_sha256", "assignments"]:
                    assert all(
                        r["phase1"][field] == pair[0]["phase1"][field] for r in pair
                    )
    (out / f"{a.stage}_summary.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
