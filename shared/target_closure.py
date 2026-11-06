"""Automatic prerequisite closure for product targets.

Clinicians tick one target. The plugin must run every prerequisite stage that
target needs, and prepare only the dependencies of that closure. This module is
pure Python (no Slicer, Qt or heavy imports) so the orchestrator, the headless
CLI, the cohort planner and the GUI setup plan can share one rule set.

Rules (applied until nothing changes):

* microstructure_freewater / _noddi / _dki  -> microstructure
* microstructure_ifw                        -> microstructure, microstructure_freewater, segmentation
* surface_postprocess                       -> surface
* saf                                       -> surface_postprocess, tractography
* bundles (WMQL), connectome                -> parcellation, tractography
* tractcloud                                -> tractography
* wma                                       -> tractography, with ``ukf`` added to
                                               the tractography algorithms
* a supplied tractogram file removes the ``tractography`` prerequisite of wma and
  tractcloud only. SAF keeps it: the tractography stage makes the CSD FOD that SAF
  needs, and the SAF stage still uses the supplied tractogram (K3).
* ``excluded`` targets (``--skip-targets``) are never added and never expanded (K5).
* tractography with csd_act (or saf)        -> segmentation, parcellation
* tractography with saf                     -> surface_postprocess, saf
* preprocessing is never added.

Brain mask, native tensor fit and registration are implicit stages and are not
targets, so they do not appear here.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

TARGET_ORDER = (
    "validation",
    "preprocessing",
    "segmentation",
    "parcellation",
    "surface",
    "surface_postprocess",
    "microstructure",
    "microstructure_freewater",
    "microstructure_noddi",
    "microstructure_dki",
    "microstructure_ifw",
    "tractography",
    "bundles",
    "saf",
    "wma",
    "tractcloud",
    "connectome",
)

# Targets that need the brain mask (shared by the GUI setup plan and docs).
MASK_TARGETS = (
    "preprocessing", "segmentation", "parcellation", "surface", "microstructure",
    "microstructure_freewater", "microstructure_noddi", "microstructure_dki", "microstructure_ifw",
    "tractography", "saf",
)

# Static prerequisites: target -> targets it always needs.
STATIC_REQUIREMENTS = {
    "microstructure_freewater": ("microstructure",),
    "microstructure_noddi": ("microstructure",),
    "microstructure_dki": ("microstructure",),
    "microstructure_ifw": ("microstructure", "microstructure_freewater", "segmentation"),
    "surface_postprocess": ("surface",),
    "saf": ("surface_postprocess", "tractography"),
    "bundles": ("parcellation", "tractography"),
    "connectome": ("parcellation", "tractography"),
    "tractcloud": ("tractography",),
    "wma": ("tractography",),
}

# Targets whose tractography prerequisite a supplied tractogram file replaces.
PROVIDED_TRACTOGRAM_TARGETS = ("wma", "tractcloud")

# Algorithms of the tractography target that pull in the ACT anatomy.
ACT_ALGORITHMS = ("csd_act", "saf")

# Product names as shown in the GUI.
DISPLAY_NAMES = {
    "preprocessing": "DWI preprocessing",
    "segmentation": "DDSeg",
    "parcellation": "DDParcel",
    "surface": "DDSurfer",
    "surface_postprocess": "Surface postprocess",
    "microstructure": "Microstructure",
    "microstructure_freewater": "Free Water",
    "microstructure_noddi": "NODDI",
    "microstructure_dki": "DKI",
    "microstructure_ifw": "iFW",
    "tractography": "Tractography",
    "bundles": "WMQL bundles",
    "saf": "SAF",
    "wma": "WMA",
    "tractcloud": "TractCloud",
    "connectome": "Connectome",
}


@dataclass(frozen=True)
class Closure:
    """Result of :func:`expand_closure`."""

    targets: tuple[str, ...]
    tractography_algorithms: tuple[str, ...]
    requested: tuple[str, ...]
    # Added target -> targets that required it (TARGET_ORDER order).
    required_by: dict = field(default_factory=dict)
    # Added algorithm -> targets that required it.
    added_algorithms: dict = field(default_factory=dict)

    @property
    def added(self) -> tuple[str, ...]:
        return tuple(target for target in self.targets if target in self.required_by)

    def provenance(self, target: str) -> dict:
        """``{"requested": bool, "required_by": [...]}`` for one target."""
        required = tuple(self.required_by.get(target, ()))
        return {"requested": not required, "required_by": list(required)}

    @property
    def algorithms_text(self) -> str:
        return ",".join(self.tractography_algorithms)


def _expand_selection(targets, bundle_backends, free_water, noddi, dki, ifw=False) -> list[str]:
    """Product-level targets -> concrete targets (existing orchestrator semantics)."""
    expanded: list[str] = []
    for target in targets:
        if target == "microstructure":
            expanded.append("microstructure")
            if free_water:
                expanded.append("microstructure_freewater")
            if noddi:
                expanded.append("microstructure_noddi")
            if dki:
                expanded.append("microstructure_dki")
            if ifw:
                expanded.append("microstructure_ifw")
        elif target == "bundles":
            for backend in bundle_backends:
                expanded.append("bundles" if backend == "wmql" else backend)
        else:
            expanded.append(target)
    return expanded


def requirements(target: str, algorithms: Iterable[str], provided_tractography: Iterable[str] = ()) -> tuple[str, ...]:
    """Targets that ``target`` needs, given the tractography algorithm set.

    ``provided_tractography`` names targets (wma, tractcloud) whose tractogram the user
    supplies as a file; they do not need the tractography target. ``saf`` is not among
    them: SAF needs the CSD FOD that only the tractography stage makes (K3).
    """
    needed = list(STATIC_REQUIREMENTS.get(target, ()))
    if target in set(provided_tractography) and target in PROVIDED_TRACTOGRAM_TARGETS:
        needed = [name for name in needed if name != "tractography"]
    if target == "tractography":
        selected = set(algorithms)
        if selected & set(ACT_ALGORITHMS):
            needed += ["segmentation", "parcellation"]
        if "saf" in selected:
            needed += ["surface_postprocess", "saf"]
    return tuple(needed)


def expand_closure(
    targets: Iterable[str],
    *,
    tractography_algorithms: Iterable[str] = ("csd_act",),
    bundle_backends: Iterable[str] = ("wmql",),
    microstructure_free_water: bool = False,
    microstructure_noddi: bool = False,
    microstructure_dki: bool = False,
    microstructure_ifw: bool = False,
    wma_source_explicit: bool = False,
    provided_tractography: Iterable[str] = (),
    excluded: Iterable[str] = (),
) -> Closure:
    """Return the prerequisite closure of ``targets``.

    ``excluded`` is the skip set: an excluded target is never requested or added, and its
    own requirements are never expanded (K5).

    ``wma_source_explicit`` is True when the user chose the tractography source
    of the bundle targets by hand; the closure then does not add ``ukf``.
    """

    algorithms = list(dict.fromkeys(tractography_algorithms)) or ["csd_act"]
    requested_raw = tuple(dict.fromkeys(str(t).strip().lower() for t in targets if str(t).strip()))
    unknown = sorted(set(requested_raw) - set(TARGET_ORDER))
    if unknown:
        raise ValueError(f"Unsupported orchestrator target(s): {', '.join(unknown)}")
    skip = frozenset(str(t).strip().lower() for t in excluded if str(t).strip())
    requested = {target for target in _expand_selection(
        tuple(t for t in requested_raw if t not in skip), tuple(bundle_backends),
        microstructure_free_water, microstructure_noddi, microstructure_dki, microstructure_ifw) if target not in skip}
    selected = set(requested)
    added_algorithms: dict = {}
    provided = frozenset(provided_tractography)
    if ("wma" in requested and "ukf" not in algorithms and not wma_source_explicit and "wma" not in provided
            and "tractography" not in skip):
        # WMA needs a UKF tractogram. When nothing else asked for tracking, run UKF alone
        # instead of the default CSD/ACT as well. (No rule adds wma, so `requested` is final here.)
        others = {"tractography", "bundles", "connectome", "tractcloud", "saf"}
        algorithms = ["ukf"] if not (requested & others) else [*algorithms, "ukf"]
        added_algorithms["ukf"] = ("wma",)
    changed = True
    while changed:
        changed = False
        for target in list(selected):
            for needed in requirements(target, algorithms, provided):
                if needed not in selected and needed not in skip:
                    selected.add(needed)
                    changed = True

    if "saf" in algorithms and "tractography" in selected and "saf" not in skip:
        # Q4: picking the SAF tracking algorithm requests the saf target, whether tractography
        # was ticked or only added by the closure (the user chose the SAF backend by hand).
        requested.add("saf")
        selected.add("saf")
    final = tuple(target for target in TARGET_ORDER if target in selected)
    required_by: dict = {}
    for target in final:
        for needed in requirements(target, algorithms, provided):
            if needed not in requested and needed in selected:
                required_by.setdefault(needed, []).append(target)
    ordered = {target: tuple(required_by[target]) for target in final if target in required_by}
    return Closure(targets=final, tractography_algorithms=tuple(algorithms),
                   requested=tuple(target for target in TARGET_ORDER if target in requested),
                   required_by=ordered, added_algorithms=added_algorithms)


def _requester_label(requester: str, closure: Closure) -> str:
    if requester == "tractography":
        if set(closure.tractography_algorithms) & set(ACT_ALGORITHMS):
            return "CSD/ACT tractography"
    return DISPLAY_NAMES.get(requester, requester)


ALGORITHM_NAMES = {"csd_act": "CSD/ACT", "dti_det": "DTI deterministic", "ukf": "UKF", "saf": "SAF"}


def closure_notice(closure: Closure) -> str:
    """One plain line naming the targets and tracking algorithms that run automatically, or ``""``."""
    groups: dict = {}
    for target in closure.added:
        groups.setdefault(closure.required_by[target], []).append(target)
    added_algorithms = {name: tuple(who) for name, who in closure.added_algorithms.items()
                        if "tractography" in closure.targets}  # Q3
    parts = []
    for requesters, names in groups.items():
        label = " and ".join(_requester_label(item, closure) for item in requesters)
        shown = []
        for name in names:
            if (name == "tractography" and added_algorithms
                    and set(closure.tractography_algorithms) == set(added_algorithms)
                    and {who for who in added_algorithms.values()} == {requesters}):
                shown.append("UKF tractography")  # tractography runs only because of the added algorithm
                added_algorithms = {}
            else:
                shown.append(DISPLAY_NAMES.get(name, name))
        parts.append(", ".join(shown) + f" (needed by {label})")
    for algorithm, requesters in added_algorithms.items():
        label = " and ".join(_requester_label(item, closure) for item in requesters)
        parts.append(f"{ALGORITHM_NAMES.get(algorithm, algorithm)} tractography (needed by {label})")
    if not parts:
        return ""
    return "Also runs automatically: " + "; ".join(parts)


def provenance_map(closure: Optional[Closure], extra_required_by: Optional[dict] = None) -> dict:
    """``target -> {"requested", "required_by"}`` merged with caller-supplied provenance."""
    merged: dict = {}
    if closure is not None:
        merged.update({target: list(value) for target, value in closure.required_by.items()})
    for target, value in (extra_required_by or {}).items():
        merged.setdefault(target, [])
        merged[target] = list(dict.fromkeys([*merged[target], *value]))
    return merged
