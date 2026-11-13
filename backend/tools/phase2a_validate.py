"""Bounded Phase 2A demo, saved-trajectory regressions and policy equivalence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def execute(out, label, args, env=None):
    with (out / f"{label}.log").open("w") as log:
        subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    execute(
        out,
        "phase0",
        [str(ROOT / "backend/tools/phase0_smoke.py"), "--output", str(out / "phase0")],
    )
    smoke = json.loads((out / "phase0/summary.json").read_text())
    saved = json.loads(
        (ROOT / "backend/evaluations/phase0/smoke_summary.json").read_text()
    )
    phase0 = {
        label: r["trajectory_sha256"] == saved["runs"][label]["trajectory_sha256"]
        for label, r in smoke["runs"].items()
    }
    assert all(phase0.values())
    execute(
        out,
        "phase1",
        [str(ROOT / "backend/tools/phase1_pilot.py"), "--output", str(out / "phase1")],
    )
    old = json.loads((ROOT / "backend/evaluations/phase1/pilot_runs.json").read_text())
    phase1 = {}
    for row in old:
        if row["study"] != "main" or row["policy"] != "fixed":
            continue
        name = f"j{row['severity']}_seed{row['seed']}_fixed_{row['scheduler']}"
        current = json.loads((out / "phase1" / f"{name}.json").read_text())
        phase1[name] = current["trajectory_sha256"] == row["trajectory_sha256"]
    assert len(phase1) == 18 and all(phase1.values())
    config = str(ROOT / "backend/evaluations/phase1/main_scenarios/scenario_j20.json")
    equivalence = {}
    for mode, scheduler in [
        ("PROPORTIONAL", "proportional_wc"),
        ("FEASIBILITY_AWARE", "deadline_feasible"),
    ]:
        env = dict(os.environ, DEADLINE_STARVATION_TICKS="3")
        baseline = out / f"constant_{scheduler}_baseline.json"
        joint = out / f"constant_{scheduler}_joint.json"
        execute(
            out,
            f"constant_{scheduler}_baseline",
            [
                str(ROOT / "backend/tools/phase0_evaluate.py"),
                "--config",
                config,
                "--policy",
                "fixed",
                "--fixed-e",
                "0",
                "--fixed-u",
                "0",
                "--scheduler",
                scheduler,
                "--transitions",
                "causal",
                "--phase1-metrics",
                "--seed",
                "3456",
                "--steps",
                "3000",
                "--output",
                str(baseline),
            ],
            env,
        )
        execute(
            out,
            f"constant_{scheduler}_joint",
            [
                str(ROOT / "backend/tools/phase2a_demo.py"),
                "--config",
                config,
                "--fixed-mode",
                mode,
                "--seed",
                "3456",
                "--steps",
                "3000",
                "--output",
                str(joint),
            ],
            env,
        )
        b = json.loads(baseline.read_text())
        j = json.loads(joint.read_text())
        equivalence[mode] = dict(
            matches=b["trajectory_sha256"] == j["physical_trajectory_sha256"],
            baseline=b["trajectory_sha256"],
            joint=j["physical_trajectory_sha256"],
        )
    assert all(v["matches"] for v in equivalence.values())
    demos = []
    for label in ["demo", "replay"]:
        path = out / f"{label}.json"
        execute(
            out,
            label,
            [
                str(ROOT / "backend/tools/phase2a_demo.py"),
                "--steps",
                "3000",
                "--output",
                str(path),
            ],
        )
        demos.append(json.loads(path.read_text()))
    assert demos[0]["trajectory_sha256"] == demos[1]["trajectory_sha256"]
    assert (
        demos[0]["history"] == demos[1]["history"]
        and demos[0]["intervals"] == demos[1]["intervals"]
    )
    d = demos[0]
    for record in d["history"]:
        if record["kind"] != "joint_command":
            continue
        step = record["receipt"]["applied_after_step"]
        assert step % 10 == 0 and record["telemetry"]["service_step"] == step
        assert record["receipt"]["first_service_step"] == step + 1
        service = d["service_history"][step - 1]
        old = record["receipt"]["mode_before"]
        assert service["mode"] == (
            "proportional_wc" if old == "PROPORTIONAL" else "deadline_feasible"
        )
        if step < len(d["service_history"]):
            new = record["receipt"]["mode_after"]
            assert d["service_history"][step]["mode"] == (
                "proportional_wc" if new == "PROPORTIONAL" else "deadline_feasible"
            )
    assert all(s["embb_mode"] == "proportional_wc" for s in d["service_history"])
    result = dict(
        base_commit="1a29280",
        phase0_saved_hashes=phase0,
        phase1_saved_hashes=phase1,
        fixed_mode_equivalence=equivalence,
        deterministic_joint_replay=True,
        causal_boundaries_verified=True,
        demonstration={
            k: v
            for k, v in d.items()
            if k not in {"history", "intervals", "service_history"}
        },
        scope="Six saved Phase0 scenarios; eighteen fixed-quota Phase1 scenarios; two fixed-mode physical equivalence pairs; two 3-second scripted demos. No full historical publication reproduction claim.",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
