"""Canonical multi-algorithm selections shared by GUI, CLI and preflight."""

TRACKING = ("csd_act", "ukf", "dti_det", "saf")
BUNDLES = ("wmql", "wma", "tractcloud")
LABELS = {"csd_act": "CSD / ACT", "ukf": "UKF", "dti_det": "DTI deterministic", "saf": "CSD + SAF",
          "wmql": "WMQL", "wma": "WMA", "tractcloud": "TractCloud",
          "dti": "DTI", "freewater": "FW", "noddi": "NODDI", "dki": "DKI", "ifw": "iFW"}


def selection_choices(choices):
    from itertools import combinations
    return [",".join(group) for size in range(1, len(choices) + 1) for group in combinations(choices, size)]


def algorithms(value, choices=TRACKING, default="csd_act"):
    text = (value or default).strip().lower().replace("+", ",").replace(";", ",")
    selected = [part.strip() for part in text.split(",") if part.strip()]
    if "all" in selected:
        selected = [*choices, *(part for part in selected if part != "all")]
    if not selected or set(selected) - set(choices):
        raise ValueError("Select one or more of: " + ", ".join(choices))
    return tuple(dict.fromkeys(selected))


def contains_tracking(value, backend):
    return backend in algorithms(value)


def choice_value(value, choices=TRACKING, default="csd_act"):
    """Normalize shared CLI settings for order-independent MRML Choice values."""
    selected = algorithms(value, choices, default)
    return ",".join(item for item in choices if item in selected)
