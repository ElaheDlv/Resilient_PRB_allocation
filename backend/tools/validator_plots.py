#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.lib.io import tf_record
from tensorboard.compat.proto.event_pb2 import Event


# ----------------------------
# Event file parsing
# ----------------------------
def read_scalars(event_file: Path, wanted_tags=None):
    """
    Read all scalar tags from a TensorBoard event file into dict[tag] -> DataFrame(step, value).
    """
    scalars = defaultdict(list)
    for raw in tf_record.tf_record_iterator(str(event_file)):
        ev = Event.FromString(raw)
        if not ev.summary:
            continue
        step = int(ev.step)
        for v in ev.summary.value:
            if v.HasField("simple_value"):
                tag = v.tag
                if (wanted_tags is None) or (tag in wanted_tags):
                    scalars[tag].append({"step": step, "value": float(v.simple_value)})

    out = {}
    for tag, rows in scalars.items():
        if rows:
            out[tag] = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    return out


def merge_by_step(a: pd.DataFrame, b: pd.DataFrame, tol=5, suffixes=("_a", "_b")):
    """
    Nearest merge on 'step' within tolerance, like your earlier code.
    """
    return (
        pd.merge_asof(
            a.sort_values("step"),
            b.sort_values("step"),
            on="step",
            direction="nearest",
            tolerance=tol,
            suffixes=suffixes,
        )
        .dropna()
        .reset_index(drop=True)
    )


# ----------------------------
# Jam detection + alignment
# ----------------------------
def detect_jam_starts(df, jam_threshold, cooldown_steps):
    """
    df must have column 'effective_prb_ratio'.
    Returns list of integer row indices corresponding to jam start events, with cooldown.
    """
    jammed = df["effective_prb_ratio"] < jam_threshold
    starts = np.where(jammed & (~jammed.shift(1).fillna(False)))[0]

    # cooldown: keep first start, then ignore any start within cooldown_steps
    kept = []
    last = -10**12
    for s in starts:
        if s - last >= cooldown_steps:
            kept.append(s)
            last = s
    return kept


def event_aligned_mean_std(df, value_col, start_rows, pre_steps, post_steps):
    """
    Align windows around start_rows (row indices), return DataFrame with t_rel, mean, std.
    t_rel is in steps (1 step = 1 ms).
    """
    aligned = []
    for r in start_rows:
        s = r - pre_steps
        e = r + post_steps
        if s < 0 or e > len(df):
            continue
        w = df.iloc[s:e].copy()
        w["t_rel"] = np.arange(-pre_steps, post_steps)
        aligned.append(w[["t_rel", value_col]])

    if not aligned:
        return pd.DataFrame(columns=["t_rel", "mean", "std"])

    A = pd.concat(aligned, ignore_index=True)
    g = A.groupby("t_rel")[value_col]
    out = pd.DataFrame({"t_rel": g.mean().index, "mean": g.mean().values, "std": g.std().values})
    return out


def compute_window_metrics(df, latency_ms_col, thr_col, reserve_frac_col,
                           pre_steps, jam_steps, rec_steps,
                           deadline_ms=1.0):
    """
    Compute baseline / jam / recovery metrics from already-aligned series (t_rel exists).
    Assumes t_rel in steps (ms).
    """
    # windows in t_rel
    pre = df[(df.t_rel >= -pre_steps) & (df.t_rel < 0)]
    jam = df[(df.t_rel >= 0) & (df.t_rel < jam_steps)]
    rec = df[(df.t_rel >= jam_steps) & (df.t_rel < jam_steps + rec_steps)]

    def safe_mean(x):
        return float(np.nanmean(x)) if len(x) else np.nan

    # URLLC
    pre_lat = safe_mean(pre[latency_ms_col])
    jam_lat = safe_mean(jam[latency_ms_col])
    rec_lat = safe_mean(rec[latency_ms_col])

    jam_violation = safe_mean((jam[latency_ms_col] > deadline_ms).astype(float)) * 100.0

    # eMBB throughput
    pre_thr = safe_mean(pre[thr_col])
    jam_thr = safe_mean(jam[thr_col])
    thr_drop = (pre_thr - jam_thr) if np.isfinite(pre_thr) and np.isfinite(jam_thr) else np.nan

    # reserve usage
    jam_res = safe_mean(jam[reserve_frac_col])

    return {
        "pre_lat_ms": pre_lat,
        "jam_lat_ms": jam_lat,
        "rec_lat_ms": rec_lat,
        "jam_violation_pct": jam_violation,
        "pre_thr": pre_thr,
        "jam_thr": jam_thr,
        "thr_drop": thr_drop,
        "jam_reserve_frac": jam_res,
    }


def estimate_recovery_time(aligned_latency_mean, jam_ms=1250, baseline_ms=None,
                           band=0.10, hold_ms=50):
    """
    Recovery time after jam ends:
    first time t >= jam_ms such that latency <= baseline*(1+band) for hold_ms consecutive samples.
    """
    if aligned_latency_mean.empty:
        return np.nan

    df = aligned_latency_mean.copy()
    df["t_ms"] = df["t_rel"]  # 1 step = 1 ms
    df["lat_ms"] = df["mean"]

    if baseline_ms is None or not np.isfinite(baseline_ms):
        # estimate baseline from pre-jam region
        pre = df[df.t_ms < 0]
        baseline_ms = float(pre["lat_ms"].mean()) if len(pre) else np.nan

    if not np.isfinite(baseline_ms) or baseline_ms <= 0:
        return np.nan

    thresh = baseline_ms * (1.0 + band)
    post = df[df.t_ms >= jam_ms].reset_index(drop=True)

    if post.empty:
        return np.nan

    ok = (post["lat_ms"] <= thresh).to_numpy()
    # find first index where we have hold_ms consecutive ok's
    H = int(hold_ms)  # 1 ms resolution
    if H <= 1:
        idx = np.where(ok)[0]
        return float(post.loc[idx[0], "t_ms"] - jam_ms) if len(idx) else np.nan

    # convolution-style consecutive check
    run = np.convolve(ok.astype(int), np.ones(H, dtype=int), mode="valid")
    idx = np.where(run == H)[0]
    if len(idx) == 0:
        return np.nan
    t_recover = float(post.loc[idx[0], "t_ms"] - jam_ms)
    return t_recover


# ----------------------------
# Folder parsing
# ----------------------------
def parse_condition(test_dir_name: str):
    """
    Examples:
      gym_from_testing_config_1ms_10s_8UE_jam_fixed_5
      gym_from_testing_config_1ms_10s_8UE_jam_pattern_random
    Returns dict with:
      category in {"power","pattern"}
      power int or None
      pattern str or None
    """
    m = re.search(r"jam_fixed_(\d+)", test_dir_name)
    if m:
        return {"category": "power", "power": int(m.group(1)), "pattern": "fixed"}

    m = re.search(r"jam_pattern_([a-zA-Z]+)", test_dir_name)
    if m:
        return {"category": "pattern", "power": None, "pattern": m.group(1).lower()}

    return {"category": "unknown", "power": None, "pattern": None}


def find_event_files(root: Path):
    """
    root/
      policy/
        test_case/
          run_dir/
            events.out.tfevents...
    Returns list of dicts.
    """
    records = []
    for policy_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        policy = policy_dir.name

        for test_dir in sorted([p for p in policy_dir.iterdir() if p.is_dir()]):
            cond = parse_condition(test_dir.name)

            # inside test_dir there may be 1 run folder
            run_dirs = [p for p in test_dir.iterdir() if p.is_dir()]
            if not run_dirs:
                continue

            for run_dir in run_dirs:
                ev_files = sorted(run_dir.glob("events.out.tfevents.*"))
                if not ev_files:
                    continue
                # take the largest/latest event file
                ev_files = sorted(ev_files, key=lambda x: x.stat().st_size, reverse=True)
                records.append({
                    "policy": policy,
                    "test_dir": test_dir.name,
                    "run_dir": run_dir.name,
                    "category": cond["category"],
                    "power": cond["power"],
                    "pattern": cond["pattern"],
                    "event_file": ev_files[0],
                })
    return records


# ----------------------------
# Plot helpers
# ----------------------------
def plot_aligned_grid(aligned_dict, title, ylabel, outpath, jam_ms=1250):
    """
    aligned_dict: keyed by (facet_value, policy) -> DataFrame(t_rel, mean, std)
    Facet values are 4 (powers or patterns).
    Produces 2x2 grid with overlayed policies.
    """
    facet_values = sorted(set([k[0] for k in aligned_dict.keys()]))
    facet_values = facet_values[:4]  # expect 4

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, fv in enumerate(facet_values):
        ax = axes[i]
        ax.axvspan(0/1000.0, jam_ms/1000.0, alpha=0.12)  # jam on
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{fv}")

        policies = sorted(set([k[1] for k in aligned_dict.keys() if k[0] == fv]))
        for pol in policies:
            df = aligned_dict.get((fv, pol))
            if df is None or df.empty:
                continue
            t = df["t_rel"].to_numpy() / 1000.0  # seconds
            m = df["mean"].to_numpy()
            s = df["std"].to_numpy()
            ax.plot(t, m, label=pol)
            # optional band (comment out if you prefer cleaner)
            if np.isfinite(s).any():
                ax.fill_between(t, m - s, m + s, alpha=0.08)

        ax.grid(True, alpha=0.25)

    # labels
    for ax in axes[2:]:
        ax.set_xlabel("Time since jamming start (s)")
    for ax in axes[0::2]:
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=True)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def grouped_bar(df, facet_col, x_col, y_col, title, ylabel, outpath):
    """
    facet_col: "power" or "pattern"
    x_col: "policy"
    """
    # Make a wide table: rows = policy, cols = facet
    pivot = df.pivot_table(index=x_col, columns=facet_col, values=y_col, aggfunc="mean")
    pivot = pivot.sort_index()
    ax = pivot.plot(kind="bar", figsize=(12, 4))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def tradeoff_scatter(df, facet_col, title, outpath,
                     x="jam_thr", y="jam_violation_pct"):
    plt.figure(figsize=(8, 5))
    for fv, g in df.groupby(facet_col):
        plt.scatter(g[x], g[y], label=str(fv))
        for _, r in g.iterrows():
            plt.annotate(r["policy"], (r[x], r[y]), fontsize=8, alpha=0.8)
    plt.xlabel("Mean eMBB throughput during jam (Mbps)")
    plt.ylabel("URLLC violation rate during jam (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(title=facet_col)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to tb_logs/res_validator")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    ap.add_argument("--jam-threshold", type=float, default=0.999)
    ap.add_argument("--jam-ms", type=int, default=1250)
    ap.add_argument("--pre-ms", type=int, default=200)
    ap.add_argument("--post-ms", type=int, default=2000)
    ap.add_argument("--recovery-ms", type=int, default=750)
    ap.add_argument("--deadline-ms", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    records = find_event_files(root)
    if not records:
        raise SystemExit(f"No event files found under {root}")

    # tags we need
    tags = {
        "latency/urllc",
        "throughput/embb_mbps",
        "prb/effective_ratio",
        "prb/reserve_fraction",
    }

    per_run_metrics = []
    aligned_latency = {}   # (facet_value, policy) -> mean/std df
    aligned_thr = {}

    for rec in records:
        pol = rec["policy"]
        category = rec["category"]
        power = rec["power"]
        pattern = rec["pattern"]
        facet_value = power if category == "power" else pattern

        scal = read_scalars(rec["event_file"], wanted_tags=tags)
        if ("latency/urllc" not in scal) or ("prb/effective_ratio" not in scal) or ("throughput/embb_mbps" not in scal):
            continue

        lat = scal["latency/urllc"].rename(columns={"value": "latency_sec"})
        prb = scal["prb/effective_ratio"].rename(columns={"value": "effective_prb_ratio"})
        thr = scal["throughput/embb_mbps"].rename(columns={"value": "embb_mbps"})
        rsv = scal.get("prb/reserve_fraction", None)
        if rsv is None:
            # if missing, create dummy column to avoid crashing
            rsv = prb[["step"]].copy()
            rsv["reserve_fraction"] = np.nan
        else:
            rsv = rsv.rename(columns={"value": "reserve_fraction"})

        # merge to common step base
        df = merge_by_step(lat, prb, tol=5, suffixes=("_lat", "_prb"))
        df = df.rename(columns={"latency_sec": "latency_sec", "effective_prb_ratio": "effective_prb_ratio"})
        # add throughput + reserve
        df = merge_by_step(df, thr, tol=5, suffixes=("", "_thr"))
        df = merge_by_step(df, rsv, tol=5, suffixes=("", "_rsv"))

        # rename after merges
        if "value" in df.columns:
            df = df.rename(columns={"value": "embb_mbps"})
        if "value_rsv" in df.columns:
            df = df.rename(columns={"value_rsv": "reserve_fraction"})
        if "reserve_fraction" not in df.columns:
            # some merge combinations may rename it
            for c in df.columns:
                if "reserve" in c:
                    df = df.rename(columns={c: "reserve_fraction"})

        # add ms latency
        df["latency_ms"] = df["latency_sec"] * 1000.0

        # detect jam starts (cooldown = jam duration in ms)
        starts = detect_jam_starts(df, args.jam_threshold, cooldown_steps=args.jam_ms)

        # align windows
        lat_al = event_aligned_mean_std(df, "latency_ms", starts, args.pre_ms, args.post_ms)
        thr_al = event_aligned_mean_std(df, "embb_mbps", starts, args.pre_ms, args.post_ms)

        if not lat_al.empty:
            aligned_latency[(facet_value, pol)] = lat_al
        if not thr_al.empty:
            aligned_thr[(facet_value, pol)] = thr_al

        # Build a “combined aligned df” for metrics: use mean series (good for summary)
        if not lat_al.empty and not thr_al.empty:
            comb = pd.merge(lat_al[["t_rel", "mean"]].rename(columns={"mean": "latency_ms_mean"}),
                            thr_al[["t_rel", "mean"]].rename(columns={"mean": "embb_mbps_mean"}),
                            on="t_rel", how="inner")
            # reserve fraction: approximate using df directly by aligning raw windows then mean
            rsv_al = event_aligned_mean_std(df, "reserve_fraction", starts, args.pre_ms, args.post_ms)
            if not rsv_al.empty:
                comb = pd.merge(comb, rsv_al[["t_rel", "mean"]].rename(columns={"mean": "reserve_fraction_mean"}),
                                on="t_rel", how="left")
            else:
                comb["reserve_fraction_mean"] = np.nan

            # Convert comb into format expected by compute_window_metrics
            comb2 = comb.rename(columns={
                "latency_ms_mean": "latency_ms",
                "embb_mbps_mean": "embb_mbps",
                "reserve_fraction_mean": "reserve_fraction"
            })

            met = compute_window_metrics(
                comb2,
                latency_ms_col="latency_ms",
                thr_col="embb_mbps",
                reserve_frac_col="reserve_fraction",
                pre_steps=args.pre_ms,
                jam_steps=args.jam_ms,
                rec_steps=args.recovery_ms,
                deadline_ms=args.deadline_ms
            )

            rec_time = estimate_recovery_time(
                aligned_latency[(facet_value, pol)],
                jam_ms=args.jam_ms,
                baseline_ms=met["pre_lat_ms"],
                band=0.10,
                hold_ms=50
            )

            per_run_metrics.append({
                "policy": pol,
                "category": category,
                "power": power,
                "pattern": pattern,
                "facet": facet_value,
                "event_file": str(rec["event_file"]),
                "n_jam_events": len(starts),
                **met,
                "recovery_time_ms": rec_time,
            })

    metrics_df = pd.DataFrame(per_run_metrics)
    out.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out / "summary_metrics.csv", index=False)

    # ----------------------------
    # Plot event-aligned grids
    # ----------------------------
    # Power sweep grids
    power_aligned_latency = {k: v for k, v in aligned_latency.items() if isinstance(k[0], int)}
    power_aligned_thr     = {k: v for k, v in aligned_thr.items() if isinstance(k[0], int)}

    if power_aligned_latency:
        plot_aligned_grid(
            power_aligned_latency,
            title="Event-aligned URLLC latency (power sweep)",
            ylabel="URLLC latency (ms)",
            outpath=out / "urlcc_latency_aligned_power.png",
            jam_ms=args.jam_ms
        )

    if power_aligned_thr:
        plot_aligned_grid(
            power_aligned_thr,
            title="Event-aligned eMBB throughput (power sweep)",
            ylabel="eMBB throughput (Mbps)",
            outpath=out / "embb_throughput_aligned_power.png",
            jam_ms=args.jam_ms
        )

    # Pattern sweep grids
    pattern_aligned_latency = {k: v for k, v in aligned_latency.items() if isinstance(k[0], str)}
    pattern_aligned_thr     = {k: v for k, v in aligned_thr.items() if isinstance(k[0], str)}

    if pattern_aligned_latency:
        plot_aligned_grid(
            pattern_aligned_latency,
            title="Event-aligned URLLC latency (pattern sweep)",
            ylabel="URLLC latency (ms)",
            outpath=out / "urlcc_latency_aligned_pattern.png",
            jam_ms=args.jam_ms
        )

    if pattern_aligned_thr:
        plot_aligned_grid(
            pattern_aligned_thr,
            title="Event-aligned eMBB throughput (pattern sweep)",
            ylabel="eMBB throughput (Mbps)",
            outpath=out / "embb_throughput_aligned_pattern.png",
            jam_ms=args.jam_ms
        )

    # ----------------------------
    # Summary bar plots + tradeoff
    # ----------------------------
    if not metrics_df.empty:
        # Power bars
        power_df = metrics_df[metrics_df["category"] == "power"].copy()
        if not power_df.empty:
            grouped_bar(power_df, "power", "policy", "jam_violation_pct",
                        "URLLC deadline violation rate during jam (power sweep)",
                        "Violation rate (%)", out / "bar_urllc_violation_rate_power.png")

            grouped_bar(power_df, "power", "policy", "thr_drop",
                        "eMBB throughput drop during jam vs pre-jam (power sweep)",
                        "Throughput drop (Mbps)", out / "bar_embb_throughput_drop_power.png")

            grouped_bar(power_df, "power", "policy", "jam_reserve_frac",
                        "Reserve fraction during jam (power sweep)",
                        "Reserve fraction", out / "bar_reserve_fraction_power.png")

            grouped_bar(power_df, "power", "policy", "recovery_time_ms",
                        "Recovery time after jam end (power sweep)",
                        "Recovery time (ms)", out / "bar_recovery_time_power.png")

            tradeoff_scatter(power_df, "power",
                             "Tradeoff (power sweep): eMBB throughput vs URLLC violation",
                             out / "tradeoff_power.png")

        # Pattern bars
        pattern_df = metrics_df[metrics_df["category"] == "pattern"].copy()
        if not pattern_df.empty:
            grouped_bar(pattern_df, "pattern", "policy", "jam_violation_pct",
                        "URLLC deadline violation rate during jam (pattern sweep)",
                        "Violation rate (%)", out / "bar_urllc_violation_rate_pattern.png")

            grouped_bar(pattern_df, "pattern", "policy", "thr_drop",
                        "eMBB throughput drop during jam vs pre-jam (pattern sweep)",
                        "Throughput drop (Mbps)", out / "bar_embb_throughput_drop_pattern.png")

            grouped_bar(pattern_df, "pattern", "policy", "jam_reserve_frac",
                        "Reserve fraction during jam (pattern sweep)",
                        "Reserve fraction", out / "bar_reserve_fraction_pattern.png")

            grouped_bar(pattern_df, "pattern", "policy", "recovery_time_ms",
                        "Recovery time after jam end (pattern sweep)",
                        "Recovery time (ms)", out / "bar_recovery_time_pattern.png")

            tradeoff_scatter(pattern_df, "pattern",
                             "Tradeoff (pattern sweep): eMBB throughput vs URLLC violation",
                             out / "tradeoff_pattern.png")

    print(f"Done. Plots + CSV saved to: {out}")


if __name__ == "__main__":
    main()
