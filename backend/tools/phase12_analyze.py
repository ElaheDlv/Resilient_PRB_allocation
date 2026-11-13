"""Operating-region certificates, paired seed intervals, and fairness attribution."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ["proportional_wc", "deadline_slack", "deadline_feasible", "deadline_fair"]


def ci(values):
    """Descriptive t interval over six independent paired held-out seeds, df=5."""
    if len(values) != 6:
        raise ValueError("This protocol prespecifies six seeds")
    mean = statistics.mean(values)
    radius = 2.570581836 * statistics.stdev(values) / math.sqrt(6)
    return dict(mean=mean, ci95=[mean - radius, mean + radius], n=6)


def demand_bound(size, rates, budgets, ticks, quota, dt=0.001):
    due = {u: max(0, math.floor(ticks - budgets[u] / dt + 1e-9)) for u in rates}
    work = sum(due[u] * size / rates[u] for u in rates)
    capacity = ticks * quota
    return dict(
        interval_ticks=ticks,
        due_batches=due,
        necessary_fluid_prbs=work,
        capacity_prbs=capacity,
        excess_prbs=work - capacity,
        zero_miss_impossible=work > capacity + 1e-8,
    )


def existing_candidate():
    rows = json.loads(
        (ROOT / "backend/evaluations/phase11/heldout_summary.json").read_text()
    )
    analysis = json.loads(
        (ROOT / "backend/evaluations/phase11/heldout_analysis.json").read_text()
    )
    result = []
    for severity in [0, 10, 20]:
        for harmed in analysis["paired_differences"][
            f"j{severity}_deadline_feasible_minus_proportional"
        ]["harmed_ues"]:
            pair = [
                next(
                    r
                    for r in rows
                    if r["severity"] == severity
                    and r["seed"] == harmed["seed"]
                    and r["scheduler"] == s
                )
                for s in ["proportional_wc", "deadline_feasible"]
            ]
            rates = {
                u: v["segments"]["all"]["mean_bytes_per_prb"]
                for u, v in pair[0]["diagnostics"]["per_ue"].items()
            }
            budgets = {u: v["budget_s"] for u, v in pair[0]["phase1"]["per_ue"].items()}
            qon = {0: 63, 10: 60, 20: 57}[severity]
            u = harmed["uid"]
            rate = rates[u]
            case = dict(
                **harmed,
                severity=severity,
                fluid_prbs=sum(650 / b for b in rates.values()),
                rounded_prbs=sum(math.ceil(650 / b) for b in rates.values()),
                quota_on=qon,
                quota_off=63,
                individual_min_quota_bytes_per_tick=math.floor(qon * rate),
                individual_arrival_bytes_per_tick=650,
                individual_service_feasible=math.floor(qon * rate) >= 650,
                bound=demand_bound(
                    650,
                    rates,
                    budgets,
                    (
                        5000
                        if severity == 0
                        else round(pair[0]["phase1"]["recovery"][0]["off_s"] / 0.001)
                        - 1
                    ),
                    qon,
                ),
                policies={},
            )
            for row in pair:
                v = row["diagnostics"]["per_ue"][u]
                s = v["segments"]["all"]
                p = row["phase1"]["per_ue"][u]
                case["policies"][row["scheduler"]] = dict(
                    budget_s=p["budget_s"],
                    budget_class=p["budget_class"],
                    mean_cqi=s["mean_cqi"],
                    mean_mcs=s["mean_mcs"],
                    bytes_per_prb=rate,
                    prbs=s["prbs"],
                    offered_bytes=5000 * 650,
                    served_bytes=s["served_bytes"],
                    service_offered_ratio=s["served_bytes"] / (5000 * 650),
                    outcomes=s["outcomes"],
                    unfinished_bytes=p["unfinished_bytes"],
                    hol=p["hol"],
                    starvation_max_s=s["starvation_max_s"],
                    segments=v["segments"],
                    all_ue_recovery=row["phase1"]["recovery"],
                )
            result.append(case)
    return result


def individual_recovery(source, rows):
    """First per-UE compliant hold within each OFF window, not joint recovery."""
    import gzip
    import sys

    sys.path.insert(0, str(ROOT / "backend"))
    from network_layer.phase1_metrics import RecoveryTracker

    results = []
    for row in rows:
        trackers = {uid: RecoveryTracker(0.010) for uid in row["phase1"]["per_ue"]}
        path = source / f"{row['label']}.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            for line in stream:
                tick = json.loads(line)
                for ue in tick["ues"]:
                    compliant = (
                        ue["queue_after_bytes"] == 0
                        or ue["hol_after_s"] <= ue["budget_s"] + 1e-12
                    )
                    trackers[ue["uid"]].observe(
                        tick["time_s"], 0.001, tick["jammed"], compliant
                    )
        results.append(
            dict(
                regime=row["regime"],
                severity=row["severity"],
                seed=row["seed"],
                scheduler=row["scheduler"],
                per_ue={
                    uid: tracker.finish(row["steps"] * 0.001)
                    for uid, tracker in trackers.items()
                },
            )
        )
    return results


def recovery_trace_evidence(source, rows):
    import gzip
    import sys

    sys.path.insert(0, str(ROOT / "backend"))
    from network_layer.fair_service_credit import FairServiceCredit
    from network_layer.deadline_scheduler import QueueView

    evidence = {}
    for policy in ["proportional_wc", "deadline_feasible", "deadline_fair"]:
        row = next(
            r
            for r in rows
            if r["regime"] == "recovery"
            and r["severity"] == 20
            and r["seed"] == 10001
            and r["scheduler"] == policy
        )
        credit = FairServiceCredit()
        samples = []
        with gzip.open(source / f"{row['label']}.jsonl.gz", "rt") as stream:
            for line in stream:
                tick = json.loads(line)
                credits = targets = None
                if policy == "deadline_fair":
                    views = {
                        u["uid"]: QueueView(
                            u["queued_bytes"], u["bytes_per_prb"], u["budget_s"], []
                        )
                        for u in tick["ues"]
                    }
                    credits, targets = credit.prepare(views, tick["ues"][0]["quota"])
                if any(
                    abs(tick["time_s"] - t) < 1e-9 for t in [0.5, 1.3, 2.0, 2.49, 3.0]
                ):
                    record = dict(
                        time_s=tick["time_s"], jammed=tick["jammed"], ues=tick["ues"]
                    )
                    if credits is not None:
                        record["reconstructed_credit_before_service"] = {
                            u: dict(
                                credit=credits[u],
                                entitlement=targets[u],
                                bound=credit.states[u]["bound"],
                            )
                            for u in credits
                        }
                    samples.append(record)
                if policy == "deadline_fair":
                    for ue in tick["ues"]:
                        credit.service(
                            ue["uid"], ue["served_bytes"], ue["queue_after_bytes"]
                        )
        evidence[policy] = dict(
            seed=10001,
            samples=samples,
            note="Queue/grant/service samples measured; optional credit state reconstructed causally from the complete trace using the documented equations",
        )
    return evidence


def analyze(source, out):
    source = Path(source)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "heldout_summary.json").read_text())
    for filename in ["threshold_heldout_summary.json", "recovery_heldout_summary.json"]:
        rows.extend(json.loads((source / filename).read_text()))
    assert len(rows) == 240, "The complete final protocol has 240 held-out episodes"
    expected = {
        "nominal": set(range(7001, 7007)),
        "near": set(range(7001, 7007)),
        "overloaded": set(range(7001, 7007)),
        "threshold": set(range(9001, 9007)),
        "recovery": set(range(10001, 10007)),
    }
    for regime, seeds in expected.items():
        assert {r["seed"] for r in rows if r["regime"] == regime} == seeds
    groups = {}
    comparisons = {}
    regions = []
    records = []
    class_records = []
    for regime in sorted({r["regime"] for r in rows}):
        for severity in [0, 20]:
            for policy in POLICIES:
                rs = sorted(
                    [
                        r
                        for r in rows
                        if r["regime"] == regime
                        and r["severity"] == severity
                        and r["scheduler"] == policy
                    ],
                    key=lambda r: r["seed"],
                )
                g = {}
                for label in ["overall", "tight", "relaxed"]:
                    vals = [
                        (
                            r["phase1"][label]
                            if label == "overall"
                            else r["phase1"]["per_class"][label]
                        )
                        for r in rs
                    ]
                    g[label] = {
                        key: ci([v[key] for v in vals])
                        for key in [
                            "failure_fraction_lower",
                            "failure_fraction_upper",
                            "served_byte_late_fraction",
                            "hol_violation_fraction",
                        ]
                    }
                    g[label]["offered"] = sum(v["offered_batches"] for v in vals)
                g["worst_ue_failure"] = ci(
                    [r["phase1"]["worst_ue_failure_fraction"] for r in rs]
                )
                service_fractions = []
                for r in rs:
                    ratios = [
                        v["served_bytes"]
                        / (v["batch_outcomes"]["offered"] * r["bytes_per_batch"])
                        for v in r["phase1"]["per_ue"].values()
                    ]
                    service_fractions.append(
                        (
                            min(ratios),
                            sum(ratios) ** 2
                            / (len(ratios) * sum(x * x for x in ratios)),
                        )
                    )
                g["worst_service_offered_fraction"] = ci(
                    [v[0] for v in service_fractions]
                )
                g["jain_service_offered_fraction"] = ci(
                    [v[1] for v in service_fractions]
                )
                g["embb_mbps"] = ci([r["phase1"]["embb_throughput_mbps"] for r in rs])
                for key in [
                    "scheduler_median_us",
                    "scheduler_p95_us",
                    "scheduler_p99_us",
                    "mean_allocated_prbs",
                    "mean_unused_prbs",
                ]:
                    g[key] = ci([r["phase1"]["resources"]["URLLC"][key] for r in rs])
                g["wall_s"] = ci([r["elapsed_seconds"] for r in rs])
                g["utilization"] = ci(
                    [
                        r["phase1"]["resources"]["URLLC"]["allocated_prb_s"]
                        / r["phase1"]["resources"]["URLLC"]["quota_prb_s"]
                        for r in rs
                    ]
                )
                g["reserve_prbs"] = ci([r["phase1"]["mean_reserve_prbs"] for r in rs])
                g["recovery"] = [
                    dict(seed=r["seed"], **w)
                    for r in rs
                    for w in r["phase1"]["recovery"]
                ]
                g["recovery_success_fraction"] = (
                    ci(
                        [
                            sum(
                                w["recovery_s"] is not None
                                for w in r["phase1"]["recovery"]
                            )
                            / len(r["phase1"]["recovery"])
                            for r in rs
                        ]
                    )
                    if severity
                    else None
                )
                groups[f"{regime}_j{severity}_{policy}"] = g
            for reference in ["proportional_wc", "deadline_feasible"]:
                result = {}
                harms = []
                for metric in ["aggregate", "worst", "tight", "relaxed", "recovery"]:
                    vals = []
                    for seed in sorted(
                        {r["seed"] for r in rows if r["regime"] == regime}
                    ):
                        pair = [
                            next(
                                r
                                for r in rows
                                if r["regime"] == regime
                                and r["severity"] == severity
                                and r["seed"] == seed
                                and r["scheduler"] == s
                            )
                            for s in [reference, "deadline_fair"]
                        ]

                        def get(r):
                            if metric == "aggregate":
                                return r["phase1"]["overall"]["failure_fraction_lower"]
                            if metric == "worst":
                                return r["phase1"]["worst_ue_failure_fraction"]
                            if metric in {"tight", "relaxed"}:
                                return r["phase1"]["per_class"][metric][
                                    "failure_fraction_lower"
                                ]
                            ws = r["phase1"]["recovery"]
                            return (
                                sum(w["recovery_s"] is not None for w in ws) / len(ws)
                                if ws
                                else None
                            )

                        values = [get(r) for r in pair]
                        if values[0] is not None:
                            vals.append(values[1] - values[0])
                        if metric == "aggregate":
                            for uid in pair[0]["phase1"]["per_ue"]:
                                ues = [r["phase1"]["per_ue"][uid] for r in pair]
                                delta = (
                                    ues[1]["batch_outcomes"]["failure_fraction_lower"]
                                    - ues[0]["batch_outcomes"]["failure_fraction_lower"]
                                )
                                if delta > 0:
                                    harms.append(
                                        dict(seed=seed, uid=uid, increase=delta)
                                    )
                    result[metric] = ci(vals) if vals else None
                result["harmed_ue_cases"] = harms
                comparisons[f"{regime}_j{severity}_fair_minus_{reference}"] = result
    for r in rows:
        budgets = {uid: v["budget_s"] for uid, v in r["phase1"]["per_ue"].items()}
        rates = {
            uid: v["segments"]["all"]["mean_bytes_per_prb"]
            for uid, v in r["diagnostics"]["per_ue"].items()
        }
        if r["scheduler"] == "proportional_wc":
            q = 57 if r["severity"] else 63
            regions.append(
                dict(
                    seed=r["seed"],
                    regime=r["regime"],
                    severity=r["severity"],
                    bytes_per_batch=r["bytes_per_batch"],
                    fluid_prbs=r["fluid_prbs"],
                    rounded_prbs=r["rounded_prbs"],
                    quota_on=57,
                    quota_off=63,
                    simultaneous_one_tick_certificate=r["rounded_prbs"] <= q,
                    finite_interval_bound=demand_bound(
                        r["bytes_per_batch"],
                        rates,
                        budgets,
                        (
                            round(r["phase1"]["recovery"][0]["off_s"] / 0.001) - 1
                            if r["severity"]
                            else 3000
                        ),
                        q,
                    ),
                )
            )
        for uid, v in r["diagnostics"]["per_ue"].items():
            s = v["segments"]["all"]
            o = s["outcomes"]
            p = r["phase1"]["per_ue"][uid]
            assert o["known_failed"] == p["batch_outcomes"]["known_failed"]
            assert o["offered"] == p["batch_outcomes"]["offered"]
            assert o["unresolved_not_due"] == p["batch_outcomes"]["unresolved_not_due"]
            for key in [
                "offered",
                "on_time",
                "late_only",
                "overflow",
                "unfinished_overdue",
                "unresolved_not_due",
            ]:
                assert o[key] == sum(
                    v["segments"][phase]["outcomes"][key]
                    for phase in ["before", "during", "after"]
                )
            for key in ["prbs", "served_bytes"]:
                assert s[key] == sum(
                    v["segments"][phase][key] for phase in ["before", "during", "after"]
                )
            assert (
                o["offered"]
                == o["on_time"] + o["known_failed"] + o["unresolved_not_due"]
            )
            assert s["over_demand_prbs"] == 0 and s["eligible_unused_ticks"] == 0
            for phase, s in v["segments"].items():
                records.append(
                    dict(
                        regime=r["regime"],
                        severity=r["severity"],
                        seed=r["seed"],
                        scheduler=r["scheduler"],
                        uid=uid,
                        deadline_class=v["budget_class"],
                        budget_s=p["budget_s"],
                        phase=phase,
                        bytes_per_batch=r["bytes_per_batch"],
                        **s["outcomes"],
                        prbs=s["prbs"],
                        served_bytes=s["served_bytes"],
                        mean_cqi=s["mean_cqi"],
                        mean_mcs=s["mean_mcs"],
                        bytes_per_prb=s["mean_bytes_per_prb"],
                        starvation_max_s=s["starvation_max_s"],
                        hol_frequency=s["hol_frequency"],
                        longest_violation_s=s["longest_violation_s"],
                        mean_violation_s=s["mean_violation_s"],
                    )
                )
        for label, v in r["phase1"]["per_class"].items():
            class_records.append(
                dict(
                    regime=r["regime"],
                    severity=r["severity"],
                    seed=r["seed"],
                    scheduler=r["scheduler"],
                    deadline_class=label,
                    **v,
                )
            )
    per_ue_recovery = individual_recovery(source, rows)
    (out / "per_ue_recovery.json").write_text(
        json.dumps(per_ue_recovery, indent=2) + "\n"
    )
    result = dict(groups=groups, comparisons=comparisons, operating_regions=regions)
    (out / "recovery_trace_evidence.json").write_text(
        json.dumps(recovery_trace_evidence(source, rows), indent=2) + "\n"
    )
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "existing_candidate_harms.json").write_text(
        json.dumps(existing_candidate(), indent=2) + "\n"
    )
    for filename, rs in [("per_ue.csv", records), ("per_class.csv", class_records)]:
        with (out / filename).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rs)
    (out / "audit.json").write_text(
        json.dumps(
            dict(
                heldout_episodes=len(rows),
                partition_agrees=True,
                no_over_useful_demand=True,
                no_eligible_unused_capacity=True,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.source, a.output)


if __name__ == "__main__":
    main()
