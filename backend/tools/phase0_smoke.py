"""Run a bounded source-regression and paced/unpaced smoke comparison.

Original core methods are restored from the audited commit in a temporary tree.
Only offline bootstrap hooks and lazy optional SDK imports remain from Phase 0.
No original checkpoints or traces are rewritten.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "eef3ad6"
CORE = ["network_layer/base_station.py", "network_layer/cell.py", "network_layer/ue.py",
        "network_layer/xApps/xapp_resilient_dqn_allocator.py",
        "network_layer/xApps/xapp_resilient_policy_validator.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="backend/checkpoints/res_dqn/xAppResilientDQNAllocator_20260123_215931_final.pt")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase0-reference-") as temp:
        reference = Path(temp) / "backend"
        shutil.copytree(ROOT / "backend", reference, ignore=shutil.ignore_patterns(
            "checkpoints", "tb_logs", "runs", "Results", "notebooks", "assets", "__pycache__"))
        (reference / "assets").symlink_to(ROOT / "backend/assets", target_is_directory=True)
        for name in CORE:
            data = subprocess.check_output(["git", "show", f"{BASELINE}:backend/{name}"], cwd=ROOT)
            (reference / name).write_bytes(data)
        cases = [
            ("reference_legacy", reference, []),
            ("legacy", ROOT/"backend", []),
            ("legacy_measured", ROOT/"backend", ["--measurements"]),
            ("causal", ROOT/"backend", ["--transitions", "causal", "--measurements"]),
            ("causal_paced", ROOT/"backend", ["--transitions", "causal", "--measurements", "--paced"]),
            ("causal_wc", ROOT/"backend", ["--transitions", "causal", "--scheduler", "proportional_wc", "--measurements"]),
        ]
        summaries = {}
        for label, backend, options in cases:
            report = output / f"{label}.json"
            cmd = [sys.executable, str(backend/"tools/phase0_evaluate.py"), "--config",
                "backend/assets/episodes/test_config/gym_from_testing_config_1ms_10s_8UE_jam_fixed_10.json",
                "--policy", "dqn", "--checkpoint", args.checkpoint, "--steps", "10020",
                "--output", str(report), *options]
            with (output/f"{label}.log").open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(report.read_text())
            summaries[label] = {k: v for k,v in result.items() if k not in {"samples", "batch_measurements"}}
            print(label, summaries[label]["elapsed_seconds"], flush=True)
        checks = {
            "legacy_core_trajectory_matches_original": summaries["reference_legacy"]["trajectory_sha256"] == summaries["legacy"]["trajectory_sha256"],
            "instrumentation_preserves_legacy_trajectory": summaries["legacy_measured"]["trajectory_sha256"] == summaries["legacy"]["trajectory_sha256"],
            "paced_unpaced_causal_match": summaries["causal"]["trajectory_sha256"] == summaries["causal_paced"]["trajectory_sha256"],
            "all_runs_complete_one_episode": all(s["completed_episodes"] == 1 for s in summaries.values()),
        }
        (output/"summary.json").write_text(json.dumps(dict(baseline_commit=BASELINE,
            reference_scope=CORE, checks=checks, runs=summaries), indent=2)+"\n")
        if not all(checks.values()):
            raise AssertionError(checks)


if __name__ == "__main__":
    main()
