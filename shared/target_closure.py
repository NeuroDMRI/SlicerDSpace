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
* an explicit bundle source (``bundle_sources``, batch 2 item 5) adds its tracking
  algorithm when it is not ticked and records the bundle targets that asked for it in
  ``added_algorithms``; with every source at ``auto`` the rules above are unchanged
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
    "microstructure",
    "segmentation",
    "parcellation",
    "surface",
    "surface_postprocess",
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
    "segmentation": ("microstructure",),
    "parcellation": ("microstructure",),
    "surface": ("microstructure",),
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
    "microstructure": "DTI",
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


def _expand_selection(targets, bundle_backends, free_water, noddi, dki, ifw=False, dti=True) -> list[str]:
    """Product-level targets -> concrete targets (existing orchestrator semantics)."""
    expanded: list[str] = []
    for target in targets:
        if target == "microstructure":
            if dti or not (free_water or noddi or dki or ifw):
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
    microstructure_dti: bool = True,
    wma_source_explicit: bool = False,
    provided_tractography: Iterable[str] = (),
    excluded: Iterable[str] = (),
    bundle_sources: Optional[dict] = None,
) -> Closure:
    """Return the prerequisite closure of ``targets``.

    ``excluded`` is the skip set: an excluded target is never requested or added, and its
    own requirements are never expanded (K5).

    ``wma_source_explicit`` is True when the user chose the tractography source
    of the bundle targets by hand; the closure then does not add ``ukf``.

    ``bundle_sources`` maps a bundle algorithm (wmql, wma, tractcloud) to its tractography
    source (batch 2 item 5). A missing key or ``auto`` keeps today's rule. A source is
    ignored for an algorithm that is not requested or whose tractogram file is supplied.
    """

    algorithms = list(dict.fromkeys(tractography_algorithms)) or ["csd_act"]
    requested_raw = tuple(dict.fromkeys(str(t).strip().lower() for t in targets if str(t).strip()))
    unknown = sorted(set(requested_raw) - set(TARGET_ORDER))
    if unknown:
        raise ValueError(f"Unsupported orchestrator target(s): {', '.join(unknown)}")
    skip = frozenset(str(t).strip().lower() for t in excluded if str(t).strip())
    requested = {target for target in _expand_selection(
        tuple(t for t in requested_raw if t not in skip), tuple(bundle_backends),
        microstructure_free_water, microstructure_noddi, microstructure_dki, microstructure_ifw,
        dti=microstructure_dti) if target not in skip}
    selected = set(requested)
    added_algorithms: dict = {}
    provided = frozenset(provided_tractography)
    explicit = _explicit_sources(bundle_sources, requested, provided) if "tractography" not in skip else {}
    if explicit:
        algorithms = _algorithms_for_sources(algorithms, explicit, requested, provided, wma_source_explicit,
                                             added_algorithms)
    elif ("wma" in requested and "ukf" not in algorithms and not wma_source_explicit and "wma" not in provided
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


def _explicit_sources(bundle_sources, requested, provided) -> dict:
    """Explicit (non-``auto``) bundle sources of requested bundle targets that read a tracking result."""

    if not bundle_sources:
        return {}
    from shared.bundle_sources import TARGETS, normalize_bundle_sources
    return {algorithm: source for algorithm, source in normalize_bundle_sources(bundle_sources, strict=False).items()
            if TARGETS[algorithm] in requested and TARGETS[algorithm] not in provided}


def _algorithms_for_sources(ticked, explicit, requested, provided, wma_source_explicit, added_algorithms) -> list:
    """Tracking algorithms when at least one bundle algorithm has an explicit source (batch 2 item 5).

    A source that is not ticked is appended and recorded in ``added_algorithms`` with the bundle targets that asked
    for it, like WMA adds UKF today. When no selected product reads every ticked tracking result, only the tracking
    algorithms that a bundle source needs run (the same rule as WMA alone, which runs UKF alone); a ticked SAF still
    runs. WMA ``all`` with no ticked whole-brain tracking (only SAF) behaves as ``auto``.
    """

    from shared.bundle_sources import SOURCES, TARGETS

    ticked = list(ticked)
    needs: dict = {}
    for algorithm, source in explicit.items():
        if source != "all":
            needs.setdefault(source, []).append(TARGETS[algorithm])
    wma = explicit.get("wma")
    if wma == "all" and not any(name != "saf" for name in ticked):
        wma = None  # R4: SAF is never a whole-brain source; `all` with no other ticked tracking behaves as auto
    if "wma" in requested and wma is None and not wma_source_explicit and "wma" not in provided:
        # WMA at auto keeps its UKF rule. Recorded also when UKF is ticked (fix S2): only `missing` decides what is
        # added, and the record keeps a ticked UKF from being dropped below.
        needs.setdefault("ukf", []).append("wma")
    uses_all_ticked = bool(requested & {"tractography", "connectome", "saf"}) or wma == "all" or any(
        TARGETS[algorithm] in requested and TARGETS[algorithm] not in provided and algorithm not in explicit
        for algorithm in ("wmql", "tractcloud"))
    missing = [source for source in SOURCES if source in needs and source not in ticked]
    if uses_all_ticked:
        algorithms = [*ticked, *missing]
    else:
        # R1 (fix S3): a ticked algorithm that no selected product reads is dropped, but a ticked SAF always runs.
        algorithms = [*(name for name in ticked if name in needs or name == "saf"), *missing]
    for source in missing:
        added_algorithms[source] = tuple(target for target in TARGET_ORDER if target in needs[source])
    return algorithms


def _requester_label(requester: str, closure: Closure, *, zh: bool = False) -> str:
    if requester == "tractography":
        if set(closure.tractography_algorithms) & set(ACT_ALGORITHMS):
            return "CSD/ACT 纤维追踪" if zh else "CSD/ACT tractography"
    if zh:
        return DISPLAY_NAMES_ZH.get(requester, DISPLAY_NAMES.get(requester, requester))
    return DISPLAY_NAMES.get(requester, requester)


DISPLAY_NAMES_ZH = {
    "preprocessing": "DWI 预处理",
    "segmentation": "DDSeg",
    "parcellation": "DDParcel",
    "surface": "DDSurfer",
    "surface_postprocess": "表面形态学后处理",
    "microstructure": "DTI",
    "microstructure_freewater": "Free Water",
    "microstructure_noddi": "NODDI",
    "microstructure_dki": "DKI",
    "microstructure_ifw": "iFW",
    "tractography": "纤维追踪",
    "bundles": "WMQL 纤维束",
    "saf": "SAF",
    "wma": "WMA",
    "tractcloud": "TractCloud",
    "connectome": "连接组",
}


ALGORITHM_NAMES = {"csd_act": "CSD/ACT", "dti_det": "DTI", "ukf": "UKF", "saf": "SAF"}


def closure_notice(closure: Closure, *, language: str = "en") -> str:
    """One plain line naming the targets and tracking algorithms that run automatically, or ``""``."""
    zh = str(language or "en").strip().lower().startswith("zh")
    groups: dict = {}
    for target in closure.added:
        groups.setdefault(closure.required_by[target], []).append(target)
    added_algorithms = {name: tuple(who) for name, who in closure.added_algorithms.items()
                        if "tractography" in closure.targets}  # Q3
    parts = []
    join_req = "、" if zh else " and "
    join_shown = "、" if zh else ", "
    for requesters, names in groups.items():
        label = join_req.join(_requester_label(item, closure, zh=zh) for item in requesters)
        shown = []
        for name in names:
            if (name == "tractography" and added_algorithms
                    and set(closure.tractography_algorithms) == set(added_algorithms)
                    and {who for who in added_algorithms.values()} == {requesters}):
                # tractography runs only because of the added algorithm(s): UKF for WMA, or a bundle source
                alg_text = join_req.join(ALGORITHM_NAMES.get(item, item) for item in added_algorithms)
                shown.append((alg_text + " 纤维追踪") if zh else (alg_text + " tractography"))
                added_algorithms = {}
            else:
                shown.append(DISPLAY_NAMES_ZH.get(name, DISPLAY_NAMES.get(name, name)) if zh else DISPLAY_NAMES.get(name, name))
        if zh:
            parts.append(join_shown.join(shown) + f"（被 {label} 需要）")
        else:
            parts.append(join_shown.join(shown) + f" (needed by {label})")
    for algorithm, requesters in added_algorithms.items():
        label = join_req.join(_requester_label(item, closure, zh=zh) for item in requesters)
        alg = ALGORITHM_NAMES.get(algorithm, algorithm)
        if zh:
            parts.append(f"{alg} 纤维追踪（被 {label} 需要）")
        else:
            parts.append(f"{alg} tractography (needed by {label})")
    if not parts:
        return ""
    if zh:
        return "同时自动运行：" + "；".join(parts)
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
