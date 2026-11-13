"""Separate six-feature encoder; does not alter the original 15-feature DQN."""

from dataclasses import dataclass
import math
from .joint_control import URLLCSchedulerMode
from .ue_telemetry import EPS


FEATURE_NAMES = (
    "max_hol_deadline_pressure",
    "active_overdue_fraction",
    "tight_matured_failure_fraction",
    "relaxed_matured_failure_fraction",
    "worst_ue_matured_failure_fraction",
    "feasibility_mode",
)


@dataclass(frozen=True)
class DeadlineRiskFeatures:
    values: tuple[float | None, ...]
    valid: tuple[bool, ...]
    reasons: tuple[str, ...]
    matured_exposures: tuple[int | None, ...]


def encode_deadline_risk(
    report, minimum_matured=20, tight_class="tight", relaxed_class="relaxed"
):
    if (
        not isinstance(minimum_matured, int)
        or isinstance(minimum_matured, bool)
        or minimum_matured < 1
    ):
        raise ValueError("Exposure threshold must be a positive batch count")
    rows = [u for u in report.ues if u.slice_id == "URLLC"]
    values = []
    reasons = []
    budgets_valid = bool(rows) and all(
        u.budget_s is not None and math.isfinite(u.budget_s) and u.budget_s > 0
        for u in rows
    )
    active = [u for u in rows if u.queued_bytes > 0]
    values.append(
        max((u.hol_wait_s / u.budget_s for u in active), default=0.0)
        if budgets_valid
        else None
    )
    values.append(
        sum(u.hol_wait_s > u.budget_s + EPS for u in active) / len(active)
        if budgets_valid and active
        else (0.0 if budgets_valid else None)
    )
    reasons.extend(["valid" if budgets_valid else "no_urllc_or_missing_budget"] * 2)

    def rate(group):
        if not group:
            return None, None, "class_absent"
        if any(
            u.budget_s is None or not math.isfinite(u.budget_s) or u.budget_s <= 0
            for u in group
        ):
            return None, None, "missing_budget"
        if not all(
            u.window.history_available
            and u.window.full_window
            and u.window.matured_offered_batches is not None
            for u in group
        ):
            return None, None, "missing_history_or_window_warmup"
        if any(u.window.matured_offered_batches < minimum_matured for u in group):
            return (
                None,
                sum(u.window.matured_offered_batches for u in group),
                "insufficient_per_ue_exposure",
            )
        n = sum(u.window.matured_offered_batches for u in group)
        return sum(u.window.matured_failed_batches for u in group) / n, n, "valid"

    exposures = []
    for name in [tight_class, relaxed_class]:
        v, n, reason = rate([u for u in rows if u.budget_class == name])
        values.append(v)
        exposures.append(n)
        reasons.append(reason)
    per_ue = [rate([u]) for u in rows]
    if rows and all(v is not None for v, _, _ in per_ue):
        values.append(max(v for v, _, _ in per_ue))
        reasons.append("valid")
    else:
        values.append(None)
        reasons.append("incomplete_worst_ue_exposure")
    modes = {u.urllc_scheduler_mode for u in rows}
    if len(modes) == 1 and isinstance(next(iter(modes)), URLLCSchedulerMode):
        values.append(float(next(iter(modes)) is URLLCSchedulerMode.FEASIBILITY_AWARE))
        reasons.append("valid")
    else:
        values.append(None)
        reasons.append("missing_or_inconsistent_mode")
    return DeadlineRiskFeatures(
        tuple(values),
        tuple(v is not None for v in values),
        tuple(reasons),
        tuple(exposures),
    )
