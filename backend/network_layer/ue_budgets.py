"""Fixed budget assignment using a private RNG, independent of radio/placement."""

import hashlib
import math
import random


def budget_assignment(config, ue_count, episode_id):
    classes = config.get("queue_budget_classes_s", {})
    if not isinstance(classes, dict):
        raise ValueError("queue_budget_classes_s must map class names to seconds")
    for name, value in classes.items():
        if (
            not isinstance(name, str)
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("Budget classes require names and positive finite seconds")
    assigned = [None] * ue_count
    if classes:
        names = sorted(classes)
        assigned = [names[i % len(names)] for i in range(ue_count)]
        material = (
            f"{config.get('queue_budget_assignment_seed', 0)}:{episode_id}".encode()
        )
        private_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        random.Random(private_seed).shuffle(assigned)
    overrides = config.get("queue_budgets_s", {})
    result = []
    for index, name in enumerate(assigned):
        value = classes[name] if name is not None else config.get("queue_budget_s")
        if str(index) in overrides:
            value, name = overrides[str(index)], "override"
        value = float(value) if value is not None else None
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError("queue_budget_s must be positive finite seconds")
        result.append((value, name))
    return result
