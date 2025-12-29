#!/usr/bin/env python3
"""
Convert xApp training configs into the episodic format required by
the Gym-style PRB allocator.

Updated for:
- Independent UE counts per slice
- One traffic template CSV per slice
- DL-only deterministic traffic
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert training configs into Gym episode catalog.")
    parser.add_argument("--input", required=True, help="Path to xapp_training_configs.json")
    parser.add_argument("--output", required=True, help="Destination JSON for the Gym catalog")
    parser.add_argument(
        "--trace-root",
        required=True,
        help="Directory where trace CSVs live (filenames from the config are resolved relative to this path)",
    )
    parser.add_argument(
        "--duration-fallback",
        type=int,
        default=600,
        help="Fallback episode length (steps) if trace duration cannot be determined",
    )
    parser.add_argument(
        "--trace-speedup",
        type=float,
        default=1.0,
        help="Optional speedup factor applied to all traces",
    )
    parser.add_argument(
        "--sim-step",
        type=float,
        default=None,
        help="Override SIM_STEP_TIME_DEFAULT (seconds)",
    )
    parser.add_argument(
        "--decision-period",
        type=int,
        default=None,
        help="Override DQN_PRB_DECISION_PERIOD_STEPS",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="convert_log.txt",
        help="Optional log file for progress messages",
    )
    parser.add_argument(
        "--embb-ue-ip",
        default=None,
        help="Optional UE IP for eMBB slice (legacy support)",
    )
    parser.add_argument(
        "--urllc-ue-ip",
        default=None,
        help="Optional UE IP for URLLC slice (legacy support)",
    )
    parser.add_argument(
        "--trace-bin",
        type=float,
        default=None,
        help="Optional trace bin width per slice (0 or negative disables binning)",
    )
    return parser.parse_args()


def trace_duration_seconds(path: Path) -> float:
    """Return the final timestamp in seconds from a CSV trace."""
    if not path.exists():
        return 0.0

    try:
        last_time = 0.0
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            next(reader, None)  # skip header
            for row in reader:
                if not row:
                    continue
                try:
                    last_time = float(row[0])
                except ValueError:
                    continue
        return last_time if last_time > 0 else 0.0
    except Exception:
        return 0.0


def seconds_per_decision(args) -> float:
    sim_step = args.sim_step if args.sim_step is not None else float(
        os.getenv("SIM_STEP_TIME_DEFAULT", "1.0")
    )
    decision_period = args.decision_period if args.decision_period is not None else int(
        os.getenv("DQN_PRB_DECISION_PERIOD_STEPS", "1")
    )
    decision_period = max(1, decision_period)
    return max(1e-9, sim_step * decision_period)


def main():
    args = parse_args()

    input_path = Path(args.input)
    trace_root = Path(args.trace_root)
    output_path = Path(args.output)

    spd = seconds_per_decision(args)

    print("\n=== Converter Configuration ===")
    print(f"Input configs:   {input_path}")
    print(f"Trace root:      {trace_root}")
    print(f"Output catalog:  {output_path}")
    print(f"Sim-step:        {args.sim_step or os.getenv('SIM_STEP_TIME_DEFAULT', 'default=1.0')}")
    print(f"Decision period: {args.decision_period or os.getenv('DQN_PRB_DECISION_PERIOD_STEPS', 'default=1')}")
    print(f"Trace speedup:   {args.trace_speedup}")
    print(f"Trace bin:       {args.trace_bin}")
    print("===============================\n")

    with input_path.open("r", encoding="utf-8") as fp:
        configs = json.load(fp)

    episodes = []
    total = len(configs)
    print(f"Processing {total} configurations...\n")

    with open(args.log_file, "w", encoding="utf-8") as logf:
        for idx, cfg in enumerate(tqdm(configs, desc="Converting configs", ncols=100)):

            # ✅ NEW: independent UE counts per slice
            ue_counts = cfg.get("ue_count", {})
            embb_ues = int(ue_counts.get("embb", 1))
            urllc_ues = int(ue_counts.get("urllc", 1))

            prb_embb = int(cfg.get("embb_default_prb", 0))
            prb_urllc = int(cfg.get("urllc_default_prb", 0))

            trace_embb = trace_root / cfg.get("trace_files", {}).get("embb", "")
            trace_urllc = trace_root / cfg.get("trace_files", {}).get("urllc", "")

            duration_e = trace_duration_seconds(trace_embb)
            duration_u = trace_duration_seconds(trace_urllc)
            duration_s = max(duration_e, duration_u)

            if duration_s <= 0.0:
                steps = args.duration_fallback
                msg = f"[WARN] Fallback duration used for config {idx}"
                tqdm.write(msg)
                logf.write(msg + "\n")
            else:
                effective = duration_s / max(1e-9, args.trace_speedup)
                steps = max(1, int(effective / spd))

            embb_slice = {
                "ue_count": embb_ues,
                "trace": str(trace_embb),
                "trace_speedup": args.trace_speedup,
                **({"ue_ip": args.embb_ue_ip} if args.embb_ue_ip else {}),
            }

            urllc_slice = {
                "ue_count": urllc_ues,
                "trace": str(trace_urllc),
                "trace_speedup": args.trace_speedup,
                **({"ue_ip": args.urllc_ue_ip} if args.urllc_ue_ip else {}),
            }

            if args.trace_bin is not None:
                embb_slice["trace_bin"] = args.trace_bin
                urllc_slice["trace_bin"] = args.trace_bin

            episodes.append(
                {
                    "id": f"cfg_{idx:05d}",
                    "duration_steps": int(steps),
                    "freeze_mobility": True,
                    "initial_prb": {
                        "eMBB": prb_embb,
                        "URLLC": prb_urllc,
                    },
                    "slices": {
                        "eMBB": embb_slice,
                        "URLLC": urllc_slice,
                    },
                }
            )

            if idx % 20 == 0:
                logf.write(f"Progress: {idx}/{total} configs processed\n")
                logf.flush()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump({"episodes": episodes}, fp, indent=2)

    print(f"\n✅ Done! Wrote {len(episodes)} episodes to {output_path}")
    print(f"📝 Progress log saved to {args.log_file}")
    print("----------------------------------------------------------\n")


if __name__ == "__main__":
    main()
