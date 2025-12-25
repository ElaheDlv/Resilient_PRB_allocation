"""Generate a template episode catalog for xapp_prb_resilient_gym_allocator.

This is a convenience script to quickly build a JSON config with multiple
episodes varying UE counts, initial PRBs, and optional traces/jamming.

Usage:
    python backend/tools/generate_res_gym_config.py --out episodes.json \
        --episodes 3 --duration 400 \
        --embb-ue 6 8 10 --urllc-ue 3 4 5 \
        --embb-prb 60 --urllc-prb 20 \
        --embb-trace backend/assets/traces/embb.csv \
        --urllc-trace backend/assets/traces/urllc.csv \
        --jam-window 200 260 0.5
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import sys
from pathlib import Path

# Ensure backend root is on sys.path so settings imports cleanly when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import settings


def parse_args():
    p = argparse.ArgumentParser(description="Generate resilient gym episode catalog.")
    p.add_argument("--out", required=True, help="Path to write the episode JSON.")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to generate.")
    p.add_argument("--duration", type=int, default=300, help="duration_steps per episode.")
    p.add_argument("--freeze-mobility", action="store_true", help="Freeze mobility in episodes.")
    p.add_argument("--embb-ue", nargs="+", type=int, default=[5], help="eMBB UE counts (cycled per episode).")
    p.add_argument("--urllc-ue", nargs="+", type=int, default=[3], help="URLLC UE counts (cycled per episode).")
    p.add_argument("--embb-prb", type=int, default=50, help="Initial PRB quota for eMBB.")
    p.add_argument("--urllc-prb", type=int, default=20, help="Initial PRB quota for URLLC.")
    p.add_argument("--embb-trace", type=str, default="", help="Path to eMBB trace (optional).")
    p.add_argument("--urllc-trace", type=str, default="", help="Path to URLLC trace (optional).")
    p.add_argument("--embb-ue-ip", type=str, default="", help="UE IP for eMBB trace (optional).")
    p.add_argument("--urllc-ue-ip", type=str, default="", help="UE IP for URLLC trace (optional).")
    p.add_argument("--trace-speedup", type=float, default=1.0, help="Trace speedup applied to both slices.")
    p.add_argument("--trace-bin", type=float, default=1.0, help="Trace bin size applied to both slices.")
    p.add_argument(
        "--jam-window",
        nargs=3,
        metavar=("START", "END", "FACTOR"),
        help="Optional jamming window (start_step end_step factor).",
    )
    return p.parse_args()


def make_slice_cfg(ue_count: int, trace: str, ue_ip: str, speedup: float, bin_s: float) -> Dict:
    cfg = {"ue_count": ue_count}
    if trace:
        cfg["trace"] = trace
    if ue_ip:
        cfg["ue_ip"] = ue_ip
    cfg["trace_speedup"] = speedup
    cfg["trace_bin"] = bin_s
    return cfg


def main():
    args = parse_args()
    sl_e = getattr(settings, "NETWORK_SLICE_EMBB_NAME", "eMBB")
    sl_u = getattr(settings, "NETWORK_SLICE_URLLC_NAME", "URLLC")

    episodes: List[Dict] = []
    for i in range(max(1, args.episodes)):
        ep = {
            "id": f"ep{i+1}",
            "duration_steps": int(args.duration),
            "freeze_mobility": bool(args.freeze_mobility),
            "initial_prb": {sl_e: int(args.embb_prb), sl_u: int(args.urllc_prb)},
            "slices": {
                sl_e: make_slice_cfg(
                    args.embb_ue[i % len(args.embb_ue)],
                    args.embb_trace,
                    args.embb_ue_ip,
                    args.trace_speedup,
                    args.trace_bin,
                ),
                sl_u: make_slice_cfg(
                    args.urllc_ue[i % len(args.urllc_ue)],
                    args.urllc_trace,
                    args.urllc_ue_ip,
                    args.trace_speedup,
                    args.trace_bin,
                ),
            },
        }
        if args.jam_window:
            try:
                start, end, factor = args.jam_window
                ep["jamming"] = [{"start_step": int(start), "end_step": int(end), "factor": float(factor)}]
            except Exception:
                pass
        episodes.append(ep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump({"episodes": episodes}, fp, indent=2)
    print(f"Wrote {len(episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()
