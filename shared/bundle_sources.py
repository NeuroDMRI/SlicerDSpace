"""Tractography source of each bundle algorithm (batch 2, item 5).

Pure Python (no Slicer, Qt or heavy imports): the CLI, the closure, the tracking variants and the GUI share it.

Stored value ``auto`` is the default and keeps the routing that existed before this setting:

* WMQL and TractCloud run on every ticked tractography.
* WMA runs on UKF (the closure adds UKF when it is not ticked).

An explicit value names one tracking backend (``csd_act``, ``ukf``, ``dti_det``). For WMA, ``all`` runs WMA once per
ticked tracking backend. ``saf`` is never a source: SAF is a short-fibre product, not a whole-brain tractogram.
"""

ALGORITHMS = ("wmql", "wma", "tractcloud")
SOURCES = ("csd_act", "ukf", "dti_det")
# Values each algorithm accepts. "auto" is today's rule. WMA "ukf" is the same as "auto" (UKF is the WMA default).
CHOICES = {
    "wmql": ("auto", *SOURCES),
    "tractcloud": ("auto", *SOURCES),
    "wma": ("auto", "csd_act", "dti_det", "all"),
}
# Bundle algorithm -> concrete orchestrator target.
TARGETS = {"wmql": "bundles", "wma": "wma", "tractcloud": "tractcloud"}
ALGORITHM_OF_TARGET = {target: algorithm for algorithm, target in TARGETS.items()}
# GUI parameter-node attributes (SlicerSpaceParameterNode), one per bundle algorithm.
NODE_ATTRIBUTES = {
    "wmql": "orchestrationWmqlTractographySource",
    "wma": "orchestrationWmaTractographySource",
    "tractcloud": "orchestrationTractcloudTractographySource",
}


def normalize_bundle_sources(value, *, strict=True) -> dict:
    """``{algorithm: source}`` with only explicit (non-``auto``) values.

    ``value`` is a mapping, ``None``, or an iterable of ``ALG=SOURCE`` tokens. ``strict=False`` drops an unknown
    algorithm or source (a saved user config); ``strict=True`` raises ``ValueError`` (the CLI).
    """

    if not value:
        return {}
    if isinstance(value, dict):
        pairs = list(value.items())
    else:
        pairs = []
        for token in value:
            text = str(token).strip()
            if "=" not in text:
                if strict:
                    raise ValueError(f"--bundle-source expects ALG=SOURCE (for example wma=csd_act), got {text!r}")
                continue
            algorithm, source = text.split("=", 1)
            pairs.append((algorithm, source))
    result: dict = {}
    for algorithm, source in pairs:
        algorithm = str(algorithm).strip().lower()
        source = str(source or "auto").strip().lower() or "auto"
        if algorithm not in CHOICES:
            if strict:
                raise ValueError(f"Unknown bundle algorithm {algorithm!r} for --bundle-source; use one of: "
                                 + ", ".join(ALGORITHMS))
            continue
        if algorithm == "wma" and source == "ukf":
            source = "auto"  # UKF is the WMA default
        if source == "saf":
            if strict:
                raise ValueError("SAF is a short-fibre product, not a whole-brain tractogram; "
                                 "it cannot be a bundle source.")
            continue
        if source not in CHOICES[algorithm]:
            if strict:
                raise ValueError(f"Unknown tractography source {source!r} for {algorithm}; use one of: "
                                 + ", ".join(CHOICES[algorithm]))
            continue
        if algorithm in result and result[algorithm] != source and strict:
            raise ValueError(f"--bundle-source names {algorithm} twice with different sources")
        if source == "auto":
            result.pop(algorithm, None)
        else:
            result[algorithm] = source
    return {algorithm: result[algorithm] for algorithm in ALGORITHMS if algorithm in result}


def cli_tokens(sources) -> list:
    """``--bundle-source ALG=SOURCE`` tokens for the explicit values only (nothing for an all-``auto`` map)."""

    tokens: list = []
    for algorithm, source in normalize_bundle_sources(sources, strict=False).items():
        tokens.extend(["--bundle-source", f"{algorithm}={source}"])
    return tokens


def explicit_sources(sources) -> dict:
    """The explicit values of a stored map (``auto`` and unknown values dropped); never raises."""

    return normalize_bundle_sources(sources, strict=False)


def target_source(sources, target):
    """The explicit source of a concrete bundle target (``bundles``, ``wma``, ``tractcloud``), or ``None``."""

    algorithm = ALGORITHM_OF_TARGET.get(target)
    return explicit_sources(sources).get(algorithm) if algorithm else None


def target_runs_on(sources, target, backend) -> bool:
    """False when an explicit source keeps ``target`` out of the tracking variant ``backend``."""

    source = target_source(sources, target)
    if source is None:
        return True
    if source == "all":
        return backend != "saf"  # SAF is not a whole-brain source
    return backend == source


def wma_explicit(sources) -> bool:
    """True when the user chose a WMA source other than the UKF default."""

    return bool(explicit_sources(sources).get("wma"))
