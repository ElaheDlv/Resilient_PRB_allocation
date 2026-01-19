#!/usr/bin/env python3
"""
Batch runner for res validation across many PRB_GYM_CONFIG_PATH JSON episode configs
and multiple policies (aggressive/random/idle/dqn/qrdqn).

It runs sequentially and writes separate TensorBoard logs per (policy, config_file).
"""

import argparse
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime


def find_json_files(episodes_dir: Path, pattern: str | None = None) -> list[Path]:
    files = sorted(episodes_dir.glob("*.json"))
    if pattern:
        rx = re.compile(pattern)
        files = [p for p in files if rx.search(p.name)]
    return files


def run_one(
    *,
    config_path: Path,
    policy: str,
    tb_base_dir: Path,
    dqn_model_path: str | None,
    qrdqn_model_path: str | None,
    seed: int,
    decision_period: int,
    jam_var_start_offset: int,
    jam_reserved_prbs: int,
    dash_enable: int,
    dash_port: int,
    steps: int,
    sim_step: float,
    trace_bin: float,
    ue_spawn_radius: int,
    ue_max: int,
    preset: str,
    mode: str,
    python_exe: str,
    dry_run: bool,
) -> int:
    config_stem = config_path.stem  # filename without .json
    policy_norm = "qrdqn" if policy.lower() == "qr_dqn" else policy.lower()

    # TB dir: separate by policy + config so nothing overwrites
    tb_dir = tb_base_dir / policy_norm / config_stem
    tb_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PRB_GYM_CONFIG_PATH": str(config_path),
            "RES_VALIDATE_ENABLE": "1",
            "RES_VALIDATE_POLICY": policy_norm,
            "RES_VALIDATE_DECISION_PERIOD": str(decision_period),
            "RES_VALIDATE_DASH_ENABLE": str(dash_enable),
            "RES_VALIDATE_DASH_PORT": str(dash_port),
            "RES_VALIDATE_TB_ENABLE": "1",
            "RES_VALIDATE_TB_DIR": str(tb_dir),
            "JAM_VAR_START_OFFSET": str(jam_var_start_offset),
            "JAM_RESERVED_PRBS": str(jam_reserved_prbs),
            "RES_VALIDATE_SEED": str(seed),
        }
    )

    if policy_norm == "dqn":
        if not dqn_model_path:
            raise ValueError("Policy is 'dqn' but no --dqn-model was provided.")
        env["RES_VALIDATE_DQN_MODEL"] = dqn_model_path
    if policy_norm == "qrdqn":
        if not qrdqn_model_path:
            raise ValueError("Policy is 'qrdqn' but no --qrdqn-model was provided.")
        env["RES_VALIDATE_QRDQN_MODEL"] = qrdqn_model_path

    cmd = [
        python_exe,
        "backend/main.py",
        "--preset",
        preset,
        "--mode",
        mode,
        "--steps",
        str(steps),
        "--sim-step",
        str(sim_step),
        "--trace-bin",
        str(trace_bin),
        "--ue-spawn-radius",
        str(ue_spawn_radius),
        "--ue-max",
        str(ue_max),
    ]

    print("\n" + "=" * 100)
    print(f"RUN policy={policy_norm} config={config_path.name}")
    print(f"TB_DIR: {tb_dir}")
    if policy_norm == "dqn":
        print(f"DQN_MODEL: {dqn_model_path}")
    if policy_norm == "qrdqn":
        print(f"QRDQN_MODEL: {qrdqn_model_path}")
    print(f"DASH: enable={dash_enable} port={dash_port}")
    print("CMD:", " ".join(cmd))
    print("=" * 100)

    if dry_run:
        return 0

    # Run and stream stdout/stderr to your terminal
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--episodes-dir",
        default="backend/assets/episodes",
        help="Directory containing episode JSON files",
    )
    ap.add_argument(
        "--pattern",
        default=None,
        help="Regex filter applied to filenames (e.g. 'gym_from_testing_config_.*_jam_.*\\.json')",
    )
    ap.add_argument(
        "--policies",
        default="aggressive,random,idle",
        help="Comma-separated policies to run (e.g. aggressive,random,idle,dqn,qrdqn).",
    )
    ap.add_argument(
        "--dqn-model",
        default=None,
        help="Path to pretrained DQN model (required if 'dqn' is in --policies).",
    )
    ap.add_argument(
        "--qrdqn-model",
        default=None,
        help="Path to pretrained QR-DQN model (required if 'qrdqn' is in --policies).",
    )
    ap.add_argument(
        "--tb-base-dir",
        default="backend/tb_logs/res_validator",
        help="Base directory for TensorBoard logs. Logs go to tb_base/policy/config_stem/",
    )

    # Common env params
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decision-period", type=int, default=10)
    ap.add_argument("--jam-var-start-offset", type=int, default=0)
    ap.add_argument("--jam-reserved-prbs", type=int, default=18)

    # Dash settings
    ap.add_argument("--dash-enable", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--dash-port",
        type=int,
        default=8070,
        help="Base dash port. The script will increment per run to avoid port collisions.",
    )

    # Main command params
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--sim-step", type=float, default=0.001)
    ap.add_argument("--trace-bin", type=float, default=0.001)
    ap.add_argument("--ue-spawn-radius", type=int, default=400)
    ap.add_argument("--ue-max", type=int, default=100)
    ap.add_argument("--preset", default="simple")
    ap.add_argument("--mode", default="headless")
    ap.add_argument("--python-exe", default="python", help="Python executable to use")
    ap.add_argument("--dry-run", action="store_true", help="Print runs without executing")

    args = ap.parse_args()

    episodes_dir = Path(args.episodes_dir)
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes dir not found: {episodes_dir}")

    tb_base_dir = Path(args.tb_base_dir)
    tb_base_dir.mkdir(parents=True, exist_ok=True)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    json_files = find_json_files(episodes_dir, args.pattern)

    if not json_files:
        raise RuntimeError(f"No JSON files found in {episodes_dir} (pattern={args.pattern!r})")

    if any(p.lower() == "dqn" for p in policies) and not args.dqn_model:
        raise ValueError("You included 'dqn' in --policies but did not pass --dqn-model")
    if any(p.lower() in ("qrdqn", "qr_dqn") for p in policies) and not args.qrdqn_model:
        raise ValueError("You included 'qrdqn' in --policies but did not pass --qrdqn-model")

    print(f"Found {len(json_files)} JSON configs in {episodes_dir}")
    print("Policies:", policies)
    print("TB base:", tb_base_dir)
    print("Started at:", datetime.now().isoformat(timespec="seconds"))

    run_idx = 0
    failures: list[tuple[str, str, int]] = []

    for config_path in json_files:
        for policy in policies:
            dash_port = args.dash_port + run_idx  # avoids collisions across sequential runs
            rc = run_one(
                config_path=config_path,
                policy=policy,
                tb_base_dir=tb_base_dir,
                dqn_model_path=args.dqn_model,
                qrdqn_model_path=args.qrdqn_model,
                seed=args.seed,
                decision_period=args.decision_period,
                jam_var_start_offset=args.jam_var_start_offset,
                jam_reserved_prbs=args.jam_reserved_prbs,
                dash_enable=args.dash_enable,
                dash_port=dash_port,
                steps=args.steps,
                sim_step=args.sim_step,
                trace_bin=args.trace_bin,
                ue_spawn_radius=args.ue_spawn_radius,
                ue_max=args.ue_max,
                preset=args.preset,
                mode=args.mode,
                python_exe=args.python_exe,
                dry_run=args.dry_run,
            )
            if rc != 0:
                failures.append((policy, config_path.name, rc))
            run_idx += 1

    print("\nDone.")
    if failures:
        print("Some runs failed:")
        for pol, fname, rc in failures:
            print(f"  - policy={pol} file={fname} rc={rc}")
        raise SystemExit(1)
    else:
        print("All runs completed successfully.")


if __name__ == "__main__":
    main()
