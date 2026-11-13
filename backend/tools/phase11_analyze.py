"""Aggregate independent paired seeds, and export auditable per-UE/class attribution."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import math


def interval(values):
    """Two-sided Student t interval over eight independent seed values, df=7."""
    if len(values) != 8:
        return dict(mean=statistics.mean(values), ci95=None, n=len(values))
    mean = statistics.mean(values)
    radius = 2.364624251 * statistics.stdev(values) / math.sqrt(8)
    return dict(mean=mean, ci95=[mean - radius, mean + radius], n=8)


def analyze(source, destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    rows = json.loads((source / "heldout_summary.json").read_text())
    groups = {}
    differences = {}
    for severity in [0, 10, 20]:
        for scheduler in ["proportional_wc", "deadline_slack", "deadline_feasible"]:
            selected = [
                r
                for r in rows
                if r["severity"] == severity and r["scheduler"] == scheduler
            ]
            group = {}
            for label in ["overall", "tight", "relaxed"]:
                values = [
                    (
                        r["phase1"]["overall"]
                        if label == "overall"
                        else r["phase1"]["per_class"][label]
                    )
                    for r in selected
                ]
                group[label] = {
                    key: interval([v[key] for v in values])
                    for key in [
                        "failure_fraction_lower",
                        "failure_fraction_upper",
                        "hol_violation_fraction",
                        "served_byte_late_fraction",
                    ]
                }
                group[label]["counts"] = {
                    key: sum(v[key] for v in values)
                    for key in [
                        "offered_batches",
                        "known_failed_batches",
                        "completed_batches",
                        "completed_over_budget_batches",
                        "unresolved_not_due_batches",
                    ]
                }
            group["worst_ue_failure"] = interval(
                [r["phase1"]["worst_ue_failure_fraction"] for r in selected]
            )
            group["scheduler_us"] = interval(
                [
                    r["phase1"]["resources"]["URLLC"]["scheduler_mean_us"]
                    for r in selected
                ]
            )
            group["wall_s"] = interval([r["elapsed_seconds"] for r in selected])
            group["embb_mbps"] = interval(
                [r["phase1"]["embb_throughput_mbps"] for r in selected]
            )
            group["reserve_prbs"] = interval(
                [r["phase1"]["mean_reserve_prbs"] for r in selected]
            )
            group["recovery"] = [
                dict(seed=r["seed"], **w)
                for r in selected
                for w in r["phase1"]["recovery"]
            ]
            group["starvation_max_s"] = max(
                v["segments"]["all"]["starvation_max_s"]
                for r in selected
                for v in r["diagnostics"]["per_ue"].values()
            )
            group["expired_head_grant_prbs"] = sum(
                v["segments"]["all"]["expired_grant_prbs"]
                for r in selected
                for v in r["diagnostics"]["per_ue"].values()
            )
            group["over_demand_prbs"] = sum(
                v["segments"]["all"]["over_demand_prbs"]
                for r in selected
                for v in r["diagnostics"]["per_ue"].values()
            )
            group["eligible_unused_ue_ticks"] = sum(
                v["segments"]["all"]["eligible_unused_ticks"]
                for r in selected
                for v in r["diagnostics"]["per_ue"].values()
            )
            groups[f"j{severity}_{scheduler}"] = group
        for candidate in ["deadline_slack", "deadline_feasible"]:
            diff = {}
            for label in ["overall", "tight", "relaxed"]:
                vals = []
                for seed in range(4001, 4009):
                    pair = [
                        next(
                            r
                            for r in rows
                            if r["severity"] == severity
                            and r["seed"] == seed
                            and r["scheduler"] == s
                        )
                        for s in ["proportional_wc", candidate]
                    ]
                    get = lambda r: (
                        r["phase1"]["overall"]
                        if label == "overall"
                        else r["phase1"]["per_class"][label]
                    )
                    vals.append(
                        get(pair[1])["failure_fraction_lower"]
                        - get(pair[0])["failure_fraction_lower"]
                    )
                diff[label] = interval(vals)
            harms = []
            for seed in range(4001, 4009):
                pair = [
                    next(
                        r
                        for r in rows
                        if r["severity"] == severity
                        and r["seed"] == seed
                        and r["scheduler"] == s
                    )
                    for s in ["proportional_wc", candidate]
                ]
                for uid in pair[0]["phase1"]["per_ue"]:
                    get = lambda r: r["phase1"]["per_ue"][uid]["batch_outcomes"][
                        "failure_fraction_lower"
                    ]
                    d = get(pair[1]) - get(pair[0])
                    if d > 0:
                        harms.append(dict(seed=seed, uid=uid, increase=d))
            diff["harmed_ues"] = harms
            differences[f"j{severity}_{candidate}_minus_proportional"] = diff
    (out / "heldout_analysis.json").write_text(
        json.dumps(dict(groups=groups, paired_differences=differences), indent=2) + "\n"
    )
    # Concrete matched-batch evidence; these timestamps were diagnosed on the
    # reproduction cohort, never used to select held-out scheduler parameters.
    import gzip

    evidence = {}
    for severity, times in [(10, [0.962, 0.963]), (20, [0.039, 0.072])]:
        for scheduler in ["proportional_wc", "deadline_slack"]:
            name = f"reproduction_j{severity}_seed3456_{scheduler}_s5"
            ticks = []
            completion = None
            with gzip.open(source / f"{name}.jsonl.gz", "rt") as stream:
                for line in stream:
                    tick = json.loads(line)
                    if any(abs(tick["time_s"] - t) < 1e-9 for t in times):
                        ticks.append(tick)
                    for event in tick["events"]:
                        if (
                            severity == 10
                            and event["kind"] == "batch_completion"
                            and event["batch_id"] == 15228
                        ):
                            completion = event
            evidence[name] = dict(
                ticks=ticks, matched_batch_15228_completion=completion
            )
    for stage, scheduler, guard in [
        ("heldout", "proportional_wc", 5),
        ("heldout_candidate", "deadline_feasible", 3),
    ]:
        name = f"{stage}_j20_seed4008_{scheduler}_s{guard}"
        ticks = []
        with gzip.open(source / f"{name}.jsonl.gz", "rt") as stream:
            for line in stream:
                tick = json.loads(line)
                if any(
                    abs(tick["time_s"] - t) < 1e-9 for t in [0.5, 1.3, 2.0, 3.8, 5.0]
                ):
                    ticks.append(tick)
        evidence[name] = dict(ticks=ticks)
    (out / "trace_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    allrows = []
    for stage in ["reproduction", "validation", "heldout"]:
        allrows.extend(json.loads((source / f"{stage}_summary.json").read_text()))
    for row in allrows:
        for uid, ue in row["diagnostics"]["per_ue"].items():
            segments = ue["segments"]
            outcomes = segments["all"]["outcomes"]
            sidecar = row["phase1"]["per_ue"][uid]["batch_outcomes"]
            assert outcomes["offered"] == sidecar["offered"]
            assert outcomes["known_failed"] == sidecar["known_failed"]
            assert outcomes["unresolved_not_due"] == sidecar["unresolved_not_due"]
            for key in [
                "offered",
                "overflow",
                "late_only",
                "unfinished_overdue",
                "unresolved_not_due",
                "on_time",
            ]:
                assert outcomes[key] == sum(
                    segments[p]["outcomes"][key] for p in ["before", "during", "after"]
                )
            for key in ["prbs", "served_bytes"]:
                assert segments["all"][key] == sum(
                    segments[p][key] for p in ["before", "during", "after"]
                )
            assert segments["all"]["over_demand_prbs"] == 0
            assert segments["all"]["eligible_unused_ticks"] == 0
    (out / "audit_results.json").write_text(
        json.dumps(
            dict(
                episodes=len(allrows),
                event_vs_sidecar_outcomes=True,
                phase_cohort_partitions=True,
                service_resource_segment_sums=True,
                no_over_useful_demand_grants=True,
                no_eligible_unused_capacity=True,
            ),
            indent=2,
        )
        + "\n"
    )
    records = []
    classes = []
    for r in allrows:
        meta = {
            k: r[k]
            for k in ["stage", "severity", "seed", "scheduler", "starvation_limit"]
        }
        for uid, ue in r["diagnostics"]["per_ue"].items():
            for phase, stats in ue["segments"].items():
                flat = {
                    **meta,
                    "uid": uid,
                    "deadline_class": ue["budget_class"],
                    "phase": phase,
                }
                flat.update({k: v for k, v in stats.items() if k != "outcomes"})
                flat.update({f"batch_{k}": v for k, v in stats["outcomes"].items()})
                flat["actual_served_bytes_per_allocated_prb"] = (
                    stats["served_bytes"] / stats["prbs"] if stats["prbs"] else None
                )
                records.append(flat)
        for label, segments in r["diagnostics"]["per_class"].items():
            for phase, counts in segments.items():
                members = [
                    v["segments"][phase]
                    for v in r["diagnostics"]["per_ue"].values()
                    if v["budget_class"] == label
                ]
                n = sum(v["ticks"] for v in members)
                runs = sum(v["violation_runs"] for v in members)
                classes.append(
                    {
                        **meta,
                        "deadline_class": label,
                        "phase": phase,
                        **counts,
                        "prbs": sum(v["prbs"] for v in members),
                        "served_bytes": sum(v["served_bytes"] for v in members),
                        "hol_frequency": (
                            sum(v["violating_ticks"] for v in members) / n
                            if n
                            else None
                        ),
                        "mean_violation_s": (
                            sum(v["violating_ticks"] for v in members) * 0.001 / runs
                            if runs
                            else 0
                        ),
                        "longest_violation_s": max(
                            v["longest_violation_s"] for v in members
                        ),
                        "starvation_max_s": max(v["starvation_max_s"] for v in members),
                        "mean_bytes_per_prb": (
                            sum(v["rate_sum"] for v in members) / n if n else None
                        ),
                        "mean_cqi": (
                            sum(v["cqi_sum"] for v in members) / n if n else None
                        ),
                        "mean_mcs": (
                            sum(v["mcs_sum"] for v in members) / n if n else None
                        ),
                    }
                )
    for filename, data in [
        ("per_ue_attribution.csv", records),
        ("per_class_attribution.csv", classes),
    ]:
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    analyze(args.source, args.output)


if __name__ == "__main__":
    main()
