"""Canonical and legacy-compatible output directory layout for SlicerDSpace."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Optional
from uuid import uuid4


ANATOMY_OUTPUT_NAME = "01_anatomical_priors"
LEGACY_ANATOMY_OUTPUT_NAMES = ("03_anatomy", "03_structural")
LEGACY_ANATOMY_OUTPUT_NAME = "03_anatomy"

CANONICAL_OUTPUT_TOP_DIRS = (
    "00_run_metadata",
    "01_anatomical_priors",
    "02_microstructure",
    "03_tractography",
    "04_bundles",
    "05_connectome",
    "06_qc",
)

ANATOMY_PRODUCT_SUBDIRS = {
    "ddseg": "tissue_segmentation",
    "tissue_segmentation": "tissue_segmentation",
    "ddparcel": "brain_parcellation",
    "brain_parcellation": "brain_parcellation",
    "ddsurfer": "cortical_surface",
    "cortical_surface": "cortical_surface",
    "ddsurfer/postprocess": "surface_morphometry",
    "ddsurfer_postprocess": "surface_morphometry",
    "surface_morphometry": "surface_morphometry",
}


def anatomy_input_dir(output_root: Path, product: str = "") -> Path:
    """Return canonical ``01_anatomical_priors`` first, or legacy ``03_anatomy``/``03_structural`` read-only fallback."""

    canonical_root = output_root / ANATOMY_OUTPUT_NAME
    if not product:
        if canonical_root.exists() or canonical_root.is_symlink():
            return canonical_root
        for legacy_name in LEGACY_ANATOMY_OUTPUT_NAMES:
            legacy = output_root / legacy_name
            if legacy.exists() or legacy.is_symlink():
                return legacy
        return canonical_root

    mapped = ANATOMY_PRODUCT_SUBDIRS.get(product, product)
    product_path = Path(product)
    mapped_path = Path(mapped)
    if (
        product_path.is_absolute()
        or mapped_path.is_absolute()
        or ".." in product_path.parts
        or ".." in mapped_path.parts
    ):
        raise ValueError(f"Anatomy product path escapes output root: {product!r}")
    canonical = canonical_root / mapped
    try:
        canonical_root_resolved = canonical_root.resolve()
        if not canonical.is_symlink() and not canonical.resolve().is_relative_to(canonical_root_resolved):
            raise ValueError(f"Anatomy product path escapes output root: {product!r}")
    except OSError as exc:
        raise ValueError(f"Invalid anatomy product path: {product!r}") from exc
    if canonical.exists() or canonical.is_symlink():
        return canonical
    if mapped != product:
        direct_canonical = canonical_root / product
        if direct_canonical.exists() or direct_canonical.is_symlink():
            return direct_canonical
    for legacy_name in LEGACY_ANATOMY_OUTPUT_NAMES:
        for rel in (product, mapped) if mapped != product else (product,):
            legacy = output_root / legacy_name / rel
            if legacy.exists() or legacy.is_symlink():
                return legacy
    return canonical


@dataclass(frozen=True)
class SubjectIOLayout:
    """Canonical output, preprocessed, and cache directories for one subject root."""

    subject_root: Path
    output_root: Optional[Path] = None

    @property
    def workspace_root(self) -> Path:
        if self.output_root is None:
            return self.subject_root
        if self.output_root.name.lower() in {"output", "outputs"}:
            return self.output_root.parent
        return self.output_root

    @property
    def outputs(self) -> Path:
        return self.output_root if self.output_root is not None else self.subject_root / "outputs"

    @property
    def cache(self) -> Path:
        return self.workspace_root / "cache"

    @property
    def cache_stages(self) -> Path:
        return self.cache / "stages"

    @property
    def cache_reports(self) -> Path:
        return self.cache / "reports"

    @property
    def display_previews(self) -> Path:
        return self.cache / "display_previews"

    def stage_cache(self, subpath: str) -> Path:
        return self.cache_stages / subpath

    @property
    def run_metadata(self) -> Path:
        return self.outputs / "00_run_metadata"

    @property
    def preproc(self) -> Path:
        return self.workspace_root / "preprocessed"

    @property
    def preproc_legacy(self) -> Path:
        return self.outputs / "01_preproc"

    # 01_anatomical_priors and task deliverables
    @property
    def anatomical_priors(self) -> Path:
        return self.outputs / ANATOMY_OUTPUT_NAME

    @property
    def anatomy(self) -> Path:
        return self.anatomical_priors

    @property
    def structural(self) -> Path:
        return self.anatomical_priors

    def anatomy_input_dir(self, product: str = "") -> Path:
        return anatomy_input_dir(self.outputs, product)

    @property
    def tissue_segmentation(self) -> Path:
        return self.anatomical_priors / "tissue_segmentation"

    @property
    def brain_parcellation(self) -> Path:
        return self.anatomical_priors / "brain_parcellation"

    @property
    def cortical_surface(self) -> Path:
        return self.anatomical_priors / "cortical_surface"

    @property
    def surface_morphometry(self) -> Path:
        return self.anatomical_priors / "surface_morphometry"

    # Legacy anatomy property aliases
    @property
    def ddseg(self) -> Path:
        return self.tissue_segmentation

    @property
    def ddsurfer(self) -> Path:
        return self.cortical_surface

    @property
    def ddsurfer_postprocess(self) -> Path:
        return self.surface_morphometry

    @property
    def ddparcel(self) -> Path:
        return self.brain_parcellation

    @property
    def registered_to_dwi(self) -> Path:
        return self.stage_cache("01_anatomical_priors/registered_to_dwi")

    # 02_microstructure and task deliverables
    @property
    def microstructure(self) -> Path:
        return self.outputs / "02_microstructure"

    @property
    def diffusion_tensor(self) -> Path:
        return self.microstructure / "diffusion_tensor"

    @property
    def free_water(self) -> Path:
        return self.microstructure / "free_water"

    @property
    def interstitial_free_water(self) -> Path:
        return self.microstructure / "interstitial_free_water"

    @property
    def neurite_microstructure(self) -> Path:
        return self.microstructure / "neurite_microstructure"

    @property
    def diffusion_kurtosis(self) -> Path:
        return self.microstructure / "diffusion_kurtosis"

    # Legacy microstructure property aliases
    @property
    def microstructure_dti(self) -> Path:
        return self.diffusion_tensor

    @property
    def microstructure_freewater(self) -> Path:
        return self.free_water

    @property
    def microstructure_noddi(self) -> Path:
        return self.neurite_microstructure

    @property
    def microstructure_dki(self) -> Path:
        return self.diffusion_kurtosis

    @property
    def microstructure_ifw(self) -> Path:
        return self.interstitial_free_water

    @property
    def regional_microstructure(self) -> Path:
        return self.microstructure

    # 03_tractography and task deliverables
    @property
    def tractography(self) -> Path:
        return self.outputs / "03_tractography"

    @property
    def single_tensor_tractography(self) -> Path:
        return self.tractography / "single_tensor_tractography"

    @property
    def multi_tensor_tractography(self) -> Path:
        return self.tractography / "multi_tensor_tractography"

    @property
    def anatomically_constrained_tractography(self) -> Path:
        return self.tractography / "anatomically_constrained_tractography"

    @property
    def short_association_fibers(self) -> Path:
        return self.tractography / "short_association_fibers"

    # Tractography caches & legacy aliases
    @property
    def csd(self) -> Path:
        return self.stage_cache("03_tractography/csd")

    @property
    def act_prep(self) -> Path:
        return self.stage_cache("03_tractography/act_prep")

    @property
    def tract_cloud(self) -> Path:
        return self.stage_cache("03_tractography/tract_cloud")

    @property
    def tractograms(self) -> Path:
        return self.stage_cache("03_tractography/tractograms")

    @property
    def tractograms_dti_det(self) -> Path:
        return self.single_tensor_tractography

    @property
    def tractograms_csd_prob(self) -> Path:
        return self.stage_cache("03_tractography/csd_prob")

    @property
    def tractograms_csd_act(self) -> Path:
        return self.anatomically_constrained_tractography

    @property
    def saf(self) -> Path:
        return self.short_association_fibers

    # 04_bundles and task deliverables
    @property
    def bundles(self) -> Path:
        return self.outputs / "04_bundles"

    @property
    def query_bundle_extraction(self) -> Path:
        return self.bundles / "query_bundle_extraction"

    @property
    def fiber_clustering(self) -> Path:
        return self.bundles / "fiber_clustering"

    @property
    def bundle_identification(self) -> Path:
        return self.bundles / "bundle_identification"

    @property
    def bundles_wma(self) -> Path:
        return self.fiber_clustering

    @property
    def bundles_tractcloud(self) -> Path:
        return self.bundle_identification

    @property
    def bundles_wmql(self) -> Path:
        return self.query_bundle_extraction

    @property
    def bundle_profiles(self) -> Path:
        return self.bundles

    # 05_connectome and 06_qc
    @property
    def connectome(self) -> Path:
        return self.outputs / "05_connectome"

    @property
    def qc(self) -> Path:
        return self.outputs / "06_qc"

    # Reports and legacy statistics containers
    @property
    def slicer_scene_report(self) -> Path:
        return self.cache_reports

    @property
    def statistics(self) -> Path:
        return self.stage_cache("statistics")

    @property
    def ddsurfer_postprocess_stats(self) -> Path:
        return self.surface_morphometry


def subject_layout(subject_root: Path, output_root: Optional[Path] = None) -> SubjectIOLayout:
    return SubjectIOLayout(
        subject_root=Path(subject_root).expanduser().resolve(),
        output_root=Path(output_root).expanduser().resolve() if output_root is not None else None,
    )


def normalize_run_name(run_name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in (run_name or "").strip()).strip("_")
    return cleaned or "Default"


def connectome_stage_cache_dir(
    subject_root: Path,
    run_name: str,
    metric: str,
    *,
    output_root: Optional[Path] = None,
) -> Path:
    """Return the internal stage-cache directory for connectome matrix intermediates."""

    layout = subject_layout(subject_root, output_root=output_root)
    run_dir = layout.stage_cache(f"05_connectome/{normalize_run_name(run_name)}")
    return run_dir if metric == "streamline_count" else (run_dir / metric)


def canonical_connectome_output_dir(
    subject_root: Path,
    run_name: str = "Default",
    metric: str = "streamline_count",
    *,
    output_root: Optional[Path] = None,
) -> Path:
    """Return the canonical published connectome output directory."""

    _ = (run_name, metric)
    return subject_layout(subject_root, output_root=output_root).connectome


def _parse_legacy_connectome_dir_name(name: str) -> tuple[str, str] | None:
    if name == "connectome":
        return ("Default", "streamline_count")
    if name.startswith("connectome_") and len(name) > len("connectome_"):
        metric = name[len("connectome_"):]
        return ("Default", metric)
    for metric in (
        "sift2_weighted_streamline_count",
        "streamline_count",
        "mean_fa",
        "mean_md",
        "mean_ad",
        "mean_rd",
        "mean_length",
        "radial_diffusivity",
        "axial_diffusivity",
    ):
        suffix = f"_{metric}"
        if name.endswith(suffix) and len(name) > len(suffix):
            return (name[: -len(suffix)], metric)
    return None


def default_task_output_dir(
    subject_root: Path,
    task_name: str,
    *,
    output_root: Optional[Path] = None,
) -> Path:
    """Return the canonical output folder for a named SlicerDSpace task."""

    layout = subject_layout(subject_root, output_root=output_root)
    mapping = {
        "validation": layout.run_metadata / "validation",
        "preproc": layout.preproc,
        "preprocessing": layout.preproc,
        "dwi_preprocessing": layout.preproc,
        "microstructure": layout.microstructure,
        "microstructure_dti": layout.diffusion_tensor,
        "diffusion_tensor": layout.diffusion_tensor,
        "dti": layout.diffusion_tensor,
        "microstructure_freewater": layout.free_water,
        "free_water": layout.free_water,
        "freewater": layout.free_water,
        "microstructure_ifw": layout.interstitial_free_water,
        "interstitial_free_water": layout.interstitial_free_water,
        "ifw": layout.interstitial_free_water,
        "microstructure_noddi": layout.neurite_microstructure,
        "neurite_microstructure": layout.neurite_microstructure,
        "noddi": layout.neurite_microstructure,
        "microstructure_dki": layout.diffusion_kurtosis,
        "diffusion_kurtosis": layout.diffusion_kurtosis,
        "dki": layout.diffusion_kurtosis,
        "ddseg": layout.tissue_segmentation,
        "segmentation": layout.tissue_segmentation,
        "tissue_segmentation": layout.tissue_segmentation,
        "ddsurfer": layout.cortical_surface,
        "surface": layout.cortical_surface,
        "cortical_surface": layout.cortical_surface,
        "ddsurfer_postprocess": layout.surface_morphometry,
        "surface_morphometry": layout.surface_morphometry,
        "ddparcel": layout.brain_parcellation,
        "parcellation": layout.brain_parcellation,
        "brain_parcellation": layout.brain_parcellation,
        "registered_to_dwi": layout.registered_to_dwi,
        "tractography": layout.tractography,
        "csd": layout.csd,
        "act_prep": layout.act_prep,
        "tract_cloud": layout.tract_cloud,
        "tractograms": layout.tractography,
        "tractography_dti": layout.single_tensor_tractography,
        "dti_det": layout.single_tensor_tractography,
        "single_tensor_tractography": layout.single_tensor_tractography,
        "tractography_ukf": layout.multi_tensor_tractography,
        "ukf": layout.multi_tensor_tractography,
        "multi_tensor_tractography": layout.multi_tensor_tractography,
        "tractograms_csd_prob": layout.tractograms_csd_prob,
        "tractography_act": layout.anatomically_constrained_tractography,
        "tractograms_csd_act": layout.anatomically_constrained_tractography,
        "csd_act": layout.anatomically_constrained_tractography,
        "anatomically_constrained_tractography": layout.anatomically_constrained_tractography,
        "saf": layout.short_association_fibers,
        "short_association_fibers": layout.short_association_fibers,
        "bundles": layout.bundles,
        "wmql": layout.query_bundle_extraction,
        "query_bundle_extraction": layout.query_bundle_extraction,
        "wma_org_atlas": layout.fiber_clustering,
        "wma": layout.fiber_clustering,
        "fiber_clustering": layout.fiber_clustering,
        "tractcloud": layout.bundle_identification,
        "bundle_identification": layout.bundle_identification,
        "connectome": layout.connectome,
        "qc": layout.qc,
        "regional_microstructure": layout.microstructure,
        "bundle_profiles": layout.bundles,
        "slicer_scene_report": layout.slicer_scene_report,
        "display_previews": layout.display_previews,
    }
    if task_name not in mapping:
        raise ValueError(f"Unsupported task output directory lookup: {task_name}")
    return mapping[task_name]


def default_task_output_dir_for_dependency(dependency_path: Path, task_name: str) -> Path:
    """Infer the subject root from a dependency path and return the task's canonical directory."""

    resolved = Path(dependency_path).expanduser().resolve()
    parts = resolved.parts
    for marker in ("outputs", "cache", "preprocessed", "input"):
        if marker in parts:
            idx = len(parts) - 1 - parts[::-1].index(marker)
            if idx > 0:
                return default_task_output_dir(Path(*parts[:idx]), task_name)
    return resolved.parent / task_name


def publish_output_file(src: Path, dest: Path, *, replace: bool = True) -> Path:
    """Publish a real non-symlink file at ``dest`` using an atomic hardlink with copy fallback."""

    source = Path(src).expanduser()
    resolved_src = source.resolve()
    if not resolved_src.is_file():
        raise FileNotFoundError(f"Cannot publish missing file: {src}")
    target = Path(dest).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not target.is_symlink():
        try:
            if os.path.samefile(resolved_src, target):
                return target
        except OSError:
            pass
        if not replace:
            return target

    tmp_path = target.parent / f".{target.name}.{uuid4().hex[:8]}.tmp"
    try:
        try:
            os.link(resolved_src, tmp_path)
        except OSError:
            shutil.copy2(resolved_src, tmp_path)
        os.replace(tmp_path, target)
    finally:
        if tmp_path.exists() or tmp_path.is_symlink():
            tmp_path.unlink(missing_ok=True)
    return target


def publish_output_directory_flat(
    src_dir: Path,
    dest_dir: Path,
    *,
    suffixes: tuple[str, ...] = (),
    exclude_names: Iterable[str] = (),
    recursive: bool = False,
    replace: bool = True,
) -> list[Path]:
    """Publish matching regular files from ``src_dir`` flat into ``dest_dir`` without symlinks."""

    source_root = Path(src_dir).expanduser()
    if not source_root.is_dir():
        return []
    excluded = set(exclude_names)
    suffix_tuple = tuple(s.lower() for s in suffixes)
    candidates = sorted(source_root.rglob("*") if recursive else source_root.iterdir())
    published: list[Path] = []
    seen_names: set[str] = set()
    for item in candidates:
        if item.name.startswith("._") or item.name in excluded:
            continue
        if not item.is_file() or (item.is_symlink() and not item.resolve().is_file()):
            continue
        lower_name = item.name.lower()
        if suffix_tuple and not any(lower_name.endswith(sfx) for sfx in suffix_tuple):
            continue
        if item.name in seen_names:
            continue
        seen_names.add(item.name)
        published.append(publish_output_file(item, dest_dir / item.name, replace=replace))
    return published


def _safe_link_or_copy(src: Path, dest: Path, *, mode: str, replace: bool = False) -> None:
    """Legacy wrapper retained for internal helpers; publishes non-symlink hardlinks or copies."""

    _ = mode
    if not (src.exists() or src.is_symlink()):
        return
    if src.is_dir() and not src.is_symlink():
        dest.mkdir(parents=True, exist_ok=True)
        for child in sorted(src.iterdir()):
            _safe_link_or_copy(child, dest / child.name, mode="copy", replace=replace)
        return
    resolved = src.resolve()
    if not resolved.is_file():
        return
    publish_output_file(resolved, dest, replace=replace)


def _archive_legacy_path(src: Path, archive_root: Path) -> None:
    if not (src.exists() or src.is_symlink()):
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / src.name
    if target.exists() or target.is_symlink():
        if src.is_dir() and not src.is_symlink() and target.is_dir():
            for child in list(src.iterdir()):
                _archive_legacy_path(child, target)
            try:
                src.rmdir()
            except OSError:
                pass
            return
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
    shutil.move(str(src), str(target))


def _archive_debug_path(src: Path, archive_root: Path) -> Optional[Path]:
    if not (src.exists() or src.is_symlink()):
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / src.name
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.move(str(src), str(target))
    return target


def _merge_directory_contents(src: Path, dest: Path, *, mode: str, files_only: bool = False) -> None:
    if not src.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        if files_only and item.is_dir() and not item.is_symlink():
            continue
        _safe_link_or_copy(item, dest / item.name, mode=mode)


def _remove_self_referential_connectome_links(layout: SubjectIOLayout) -> None:
    if not layout.connectome.is_dir():
        return
    for entry in sorted(layout.connectome.iterdir()):
        if entry.is_symlink():
            try:
                if entry.resolve() == entry.parent.resolve():
                    entry.unlink()
            except OSError:
                pass


def _refresh_microstructure_manifest_flags(layout: SubjectIOLayout, *, roi_csv: Optional[Path] = None) -> None:
    candidate_paths = [
        layout.stage_cache("02_microstructure/microstructure_manifest.json"),
        layout.microstructure / "microstructure_manifest.json",
    ]
    for manifest_path in candidate_paths:
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        outputs = payload.get("outputs") if isinstance(payload, dict) else None
        if not isinstance(outputs, dict):
            continue
        updated = False
        if outputs.get("fw_fraction") is not None and not payload.get("run_free_water"):
            payload["run_free_water"] = True
            updated = True
        if outputs.get("noddi_ndi") is not None and not payload.get("run_noddi"):
            payload["run_noddi"] = True
            updated = True
        if roi_csv is not None and roi_csv.is_file() and not outputs.get("roi_stats_csv"):
            outputs["roi_stats_csv"] = roi_csv.as_posix()
            updated = True
        if updated:
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sync_tree_into_cache(src: Path, dest: Path, *, move: bool) -> None:
    """Safely move or hardlink-copy ``src`` into ``dest`` without propagating broken symlinks."""

    if not (src.exists() or src.is_symlink()):
        return
    if src.is_symlink():
        if move:
            src.unlink(missing_ok=True)
        return
    if src.is_file():
        publish_output_file(src, dest, replace=True)
        if move:
            src.unlink(missing_ok=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.iterdir()):
        _sync_tree_into_cache(item, dest / item.name, move=move)
    if move:
        try:
            src.rmdir()
        except OSError:
            shutil.rmtree(src, ignore_errors=True)


def _infer_subject_id_for_migration(layout: SubjectIOLayout) -> str:
    if layout.run_metadata.is_dir():
        for manifest_path in sorted(layout.run_metadata.glob("*_orchestration_manifest.json"), reverse=True):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                sid = str(data.get("subject_id") or "").strip()
                if sid:
                    return sid
            except (OSError, ValueError):
                pass
    for search_root in (
        layout.outputs / "02_microstructure" / "dti",
        layout.stage_cache("02_microstructure/dti"),
        layout.diffusion_tensor,
        layout.preproc_legacy,
        layout.preproc,
    ):
        if search_root.is_dir():
            for f in sorted(search_root.iterdir()):
                for suffix in ("_dti_FA.nii.gz", "_FA.nii.gz", "_dwi_final.nii.gz"):
                    if f.name.endswith(suffix):
                        return f.name[: -len(suffix)]
    return layout.subject_root.name


def _find_first_file(roots: Iterable[Path], patterns: Iterable[str]) -> Optional[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            matches = [
                p for p in sorted(root.rglob(pattern))
                if not p.name.startswith("._") and p.resolve().is_file()
            ]
            if matches:
                # Prefer non-symlink regular files when both exist
                matches.sort(key=lambda p: (p.is_symlink(), -p.stat().st_mtime))
                return matches[0]
    return None


def _remove_empty_dirs(root: Path, *, stop_at: Path) -> None:
    if not root.is_dir():
        return
    for item in sorted(root.iterdir()):
        if item.is_dir() and not item.is_symlink():
            _remove_empty_dirs(item, stop_at=stop_at)
    if root != stop_at:
        try:
            next(root.iterdir())
        except StopIteration:
            root.rmdir()


def _bundle_files(folder: Path, pattern: str) -> list[Path]:
    """Published bundle files: the old flat layout and the batch 2 per-tractography subfolders (``<backend>/``)."""

    if not folder.is_dir():
        return []
    return sorted(folder.glob(pattern)) + sorted(folder.glob("*/" + pattern))


def _migrate_qc_from_stage_caches(layout: SubjectIOLayout, subject_id: str) -> dict[str, Path]:
    from shared.qc_text_summary import write_qc_text_summaries

    task_records: list[dict[str, Any]] = []
    # Check latest orchestration manifest first if present
    manifest_tasks: dict[str, dict[str, Any]] = {}
    run_name: Optional[str] = None
    if layout.run_metadata.is_dir():
        manifests = sorted(
            layout.run_metadata.glob("*_orchestration_manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for mpath in manifests:
            try:
                data = json.loads(mpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            run_name = run_name or data.get("run_name")
            for t in data.get("tasks") or []:
                if isinstance(t, dict) and t.get("name"):
                    manifest_tasks.setdefault(str(t["name"]), t)

    def maybe_add(task_name: str, check_files: list[Path], qc_patterns: tuple[Path, ...] = ()) -> None:
        existing_outputs = [p for p in check_files if p.is_file()]
        if not existing_outputs:
            return
        m_task = manifest_tasks.get(task_name, {})
        qc_path = None
        for qp in qc_patterns:
            if qp.is_file():
                qc_path = qp
                break
        if qc_path is None and m_task.get("qc_path") and Path(str(m_task["qc_path"])).is_file():
            qc_path = Path(str(m_task["qc_path"]))
        status = str(m_task.get("status") or "completed")
        if status in {"planned", "will_run", "blocked"}:
            status = "completed"
        task_records.append(
            {
                "name": task_name,
                "status": status,
                "reason": m_task.get("reason") or "",
                "runtime_seconds": m_task.get("runtime_seconds"),
                "regional_statistics": m_task.get("regional_statistics"),
                "qc_path": qc_path,
                "outputs": {str(i): p.as_posix() for i, p in enumerate(existing_outputs)},
            }
        )

    sc = layout.cache_stages
    maybe_add(
        "preprocessing.dwi",
        [layout.preproc / f"{subject_id}_dwi_final.nii.gz"],
        tuple(sorted((sc / "00_preprocessing").rglob("*qc.json"))) if (sc / "00_preprocessing").is_dir() else (),
    )
    maybe_add(
        "segmentation.ddseg",
        [layout.tissue_segmentation / "SegmentationMap_GMWMCSF.nii.gz"],
        tuple(sorted((sc / "01_anatomical_priors/tissue_segmentation").rglob("*qc.json")))
        if (sc / "01_anatomical_priors/tissue_segmentation").is_dir()
        else (),
    )
    maybe_add(
        "parcellation.ddparcel",
        [layout.brain_parcellation / f"{subject_id}_parcellation.nii.gz"],
        tuple(sorted((sc / "01_anatomical_priors/brain_parcellation").rglob("*qc.json")))
        if (sc / "01_anatomical_priors/brain_parcellation").is_dir()
        else (),
    )
    maybe_add(
        "surface.ddsurfer",
        [layout.cortical_surface / "lh.white.obj", layout.cortical_surface / "rh.white.obj"],
        tuple(sorted((sc / "01_anatomical_priors/cortical_surface").rglob("*qc.json")))
        if (sc / "01_anatomical_priors/cortical_surface").is_dir()
        else (),
    )
    maybe_add(
        "surface.ddsurfer_postprocess",
        [layout.surface_morphometry / "lh.aparc.stats"],
        tuple(sorted((sc / "01_anatomical_priors/surface_morphometry").rglob("*qc.json")))
        if (sc / "01_anatomical_priors/surface_morphometry").is_dir()
        else (),
    )
    maybe_add(
        "microstructure.dti",
        [layout.diffusion_tensor / f"{subject_id}_FA.nii.gz"],
        (sc / "02_microstructure/dti/dti_qc.json",),
    )
    maybe_add(
        "microstructure.freewater",
        [layout.free_water / f"{subject_id}_FW.nii.gz"],
        (sc / "02_microstructure/freewater/fw_qc.json",),
    )
    maybe_add(
        "microstructure.ifw",
        [layout.interstitial_free_water / f"{subject_id}_ifw.nii.gz"],
        (sc / "02_microstructure/ifw/ifw_manifest.json", sc / "02_microstructure/ifw/ifw_qc.json"),
    )
    maybe_add(
        "microstructure.noddi",
        [layout.neurite_microstructure / f"{subject_id}_NDI.nii.gz"],
        (sc / "02_microstructure/noddi/noddi_qc.json",),
    )
    maybe_add(
        "microstructure.dki",
        [layout.diffusion_kurtosis / f"{subject_id}_MK.nii.gz"],
        (sc / "02_microstructure/dki/dki_qc.json",),
    )
    maybe_add(
        "tractography.dti_det",
        [layout.single_tensor_tractography / f"{subject_id}_dti.trk"],
        tuple(sorted((sc / "03_tractography").rglob("*dti*qc.json"))) if (sc / "03_tractography").is_dir() else (),
    )
    maybe_add(
        "tractography.ukf",
        [layout.multi_tensor_tractography / f"{subject_id}_ukf.trk", layout.multi_tensor_tractography / f"{subject_id}_ukf.vtk"],
        tuple(sorted((sc / "03_tractography").rglob("*ukf*qc.json"))) if (sc / "03_tractography").is_dir() else (),
    )
    maybe_add(
        "tractography.csd_act",
        [layout.anatomically_constrained_tractography / f"{subject_id}_act.tck"],
        tuple(sorted((sc / "03_tractography").rglob("*csd_act_qc.json"))) if (sc / "03_tractography").is_dir() else (),
    )
    maybe_add(
        "tractography.saf",
        [layout.short_association_fibers / f"{subject_id}_saf.tck", layout.short_association_fibers / f"{subject_id}_saf.vtk"],
        tuple(sorted((sc / "03_tractography").rglob("*saf_qc.json"))) if (sc / "03_tractography").is_dir() else (),
    )
    maybe_add(
        "bundles.wmql",
        _bundle_files(layout.query_bundle_extraction, "*.trk"),
        tuple(sorted((sc / "04_bundles").rglob("*wmql_qc.json"))) if (sc / "04_bundles").is_dir() else (),
    )
    maybe_add(
        "bundles.wma",
        _bundle_files(layout.fiber_clustering, "T_*.vtp"),
        tuple(sorted((sc / "04_bundles").rglob("*wma_org_atlas_qc.json"))) if (sc / "04_bundles").is_dir() else (),
    )
    maybe_add(
        "bundles.tractcloud",
        _bundle_files(layout.bundle_identification, "*.vtp"),
        tuple(sorted((sc / "04_bundles").rglob("*tractcloud_qc.json"))) if (sc / "04_bundles").is_dir() else (),
    )
    maybe_add(
        "connectome.streamline_count",
        [layout.connectome / f"{subject_id}_streamline_count.csv"],
        tuple(sorted((sc / "05_connectome").rglob("*streamline_count_qc.json"))) if (sc / "05_connectome").is_dir() else (),
    )
    maybe_add(
        "connectome.sift2_weighted",
        [layout.connectome / f"{subject_id}_sift2_weighted.csv"],
        tuple(sorted((sc / "05_connectome").rglob("*sift2*qc.json"))) if (sc / "05_connectome").is_dir() else (),
    )

    return write_qc_text_summaries(
        layout.qc,
        task_records,
        subject_id=subject_id,
        run_name=run_name,
        subject_root=layout.subject_root,
    )


def migrate_subject_output_tree(
    subject_root: Path,
    *,
    output_root: Optional[Path] = None,
    clean_legacy: bool = True,
) -> dict[str, Any]:
    """Migrate a legacy or partial SlicerDSpace subject tree into the canonical layout."""

    layout = subject_layout(subject_root, output_root=output_root)
    outputs = layout.outputs
    stage_cache = layout.cache_stages
    if not outputs.exists() and not stage_cache.exists() and not layout.preproc.exists():
        return {"subject_id": layout.subject_root.name, "published": {}, "qc": {}}
    subject_id = _infer_subject_id_for_migration(layout)
    published: dict[str, Path] = {}

    # 1. Stage-cache preservation of legacy folders before cleanup
    if (outputs / "01_preproc").exists():
        _sync_tree_into_cache(outputs / "01_preproc", stage_cache / "00_preprocessing", move=clean_legacy)

    for legacy_anat in LEGACY_ANATOMY_OUTPUT_NAMES:
        anat_dir = outputs / legacy_anat
        if not anat_dir.exists():
            continue
        for sub, target_rel in (
            ("ddseg", "01_anatomical_priors/tissue_segmentation"),
            ("tissue_segmentation", "01_anatomical_priors/tissue_segmentation"),
            ("ddparcel", "01_anatomical_priors/brain_parcellation"),
            ("brain_parcellation", "01_anatomical_priors/brain_parcellation"),
            ("ddsurfer/postprocess", "01_anatomical_priors/surface_morphometry"),
            ("surface_morphometry", "01_anatomical_priors/surface_morphometry"),
            ("ddsurfer", "01_anatomical_priors/cortical_surface"),
            ("cortical_surface", "01_anatomical_priors/cortical_surface"),
            ("registered_to_dwi", "01_anatomical_priors/registered_to_dwi"),
        ):
            cand = anat_dir / sub
            if cand.exists() or cand.is_symlink():
                _sync_tree_into_cache(cand, stage_cache / target_rel, move=clean_legacy)
        if clean_legacy:
            _sync_tree_into_cache(anat_dir, layout.cache / "legacy_outputs" / legacy_anat, move=True)

    micro_dir = outputs / "02_microstructure"
    if micro_dir.is_dir():
        for legacy_sub in ("dti", "freewater", "noddi", "dki", "ifw", "regional_microstructure"):
            cand = micro_dir / legacy_sub
            if cand.exists() or cand.is_symlink():
                _sync_tree_into_cache(cand, stage_cache / "02_microstructure" / legacy_sub, move=clean_legacy)
        for jfile in sorted(micro_dir.glob("*.json")):
            _sync_tree_into_cache(jfile, stage_cache / "02_microstructure" / jfile.name, move=clean_legacy)

    tract_dir = outputs / "04_tractography"
    # Batch 2 item 5: only bundle folders that come from a legacy ``04_tractography/bundles`` tree are published flat
    # below (old results). Current runs publish their own ``04_bundles/<product>/<backend>/`` folders.
    legacy_bundle_roots: dict[str, list[Path]] = {}
    for bundle_sub in ("wmql", "wma_org_atlas", "tract_cloud_dl"):
        legacy = tract_dir / "bundles" / bundle_sub
        if legacy.is_dir():
            children = sorted(legacy.iterdir())
            roots = [stage_cache / "04_bundles" / bundle_sub / child.name for child in children if child.is_dir()]
            if any(child.is_file() for child in children):
                roots.append(stage_cache / "04_bundles" / bundle_sub)
            legacy_bundle_roots[bundle_sub] = roots
    if tract_dir.is_dir():
        for sub in ("act_prep", "csd", "saf", "tract_cloud", "tractograms"):
            cand = tract_dir / sub
            if cand.exists() or cand.is_symlink():
                _sync_tree_into_cache(cand, stage_cache / "03_tractography" / sub, move=clean_legacy)
        for bundle_sub in ("wmql", "wma_org_atlas", "tract_cloud_dl"):
            cand = tract_dir / "bundles" / bundle_sub
            if cand.exists() or cand.is_symlink():
                _sync_tree_into_cache(cand, stage_cache / "04_bundles" / bundle_sub, move=clean_legacy)
        if (tract_dir / "bundles").exists():
            _sync_tree_into_cache(tract_dir / "bundles", stage_cache / "04_bundles", move=clean_legacy)
        conn_legacy = tract_dir / "connectome"
        if conn_legacy.exists():
            for citem in sorted(conn_legacy.iterdir()):
                cparsed = _parse_legacy_connectome_dir_name(citem.name)
                if cparsed is not None:
                    c_run, c_metric = cparsed
                    c_target = connectome_stage_cache_dir(layout.subject_root, c_run, c_metric, output_root=layout.output_root)
                    _sync_tree_into_cache(citem, c_target, move=clean_legacy)
            _sync_tree_into_cache(conn_legacy, stage_cache / "05_connectome", move=clean_legacy)
        for item in sorted(tract_dir.iterdir()):
            parsed = _parse_legacy_connectome_dir_name(item.name)
            if parsed is not None:
                run_n, metric_n = parsed
                target_c = connectome_stage_cache_dir(layout.subject_root, run_n, metric_n, output_root=layout.output_root)
                _sync_tree_into_cache(item, target_c, move=clean_legacy)
        if clean_legacy:
            _sync_tree_into_cache(tract_dir, layout.cache / "legacy_outputs" / "04_tractography", move=True)

    stats_dir = outputs / "05_statistics"
    if stats_dir.exists() or stats_dir.is_symlink():
        if (stats_dir / "ddsurfer" / "postprocess").exists():
            _sync_tree_into_cache(
                stats_dir / "ddsurfer" / "postprocess",
                stage_cache / "01_anatomical_priors/surface_morphometry/stats",
                move=clean_legacy,
            )
        if (stats_dir / "regional_microstructure").exists():
            _sync_tree_into_cache(
                stats_dir / "regional_microstructure",
                stage_cache / "02_microstructure/regional_microstructure",
                move=clean_legacy,
            )
        if clean_legacy:
            _sync_tree_into_cache(stats_dir, layout.cache / "legacy_outputs" / "05_statistics", move=True)

    report_dir = outputs / "06_slicer_scene_report"
    if report_dir.exists() or report_dir.is_symlink():
        if (report_dir / "display_previews").exists():
            _sync_tree_into_cache(report_dir / "display_previews", layout.display_previews, move=clean_legacy)
        _sync_tree_into_cache(report_dir, layout.cache_reports, move=clean_legacy)

    for dev_name in (".debug_outputs", ".legacy_flat_outputs", "_archive_dev", "display_previews"):
        dev_path = outputs / dev_name
        if dev_path.exists() or dev_path.is_symlink():
            dest_dev = layout.display_previews if dev_name == "display_previews" else (layout.cache / "legacy_outputs" / dev_name)
            _sync_tree_into_cache(dev_path, dest_dev, move=clean_legacy)

    # Move non-canonical subdirs inside 01_anatomical_priors if any
    if layout.anatomical_priors.is_dir():
        for leg_sub, cache_rel in (
            ("ddseg", "01_anatomical_priors/tissue_segmentation"),
            ("ddparcel", "01_anatomical_priors/brain_parcellation"),
            ("ddsurfer", "01_anatomical_priors/cortical_surface"),
            ("registered_to_dwi", "01_anatomical_priors/registered_to_dwi"),
        ):
            cand = layout.anatomical_priors / leg_sub
            if cand.exists() or cand.is_symlink():
                _sync_tree_into_cache(cand, stage_cache / cache_rel, move=clean_legacy)

    # 2. Publish canonical deliverables into preprocessed/ and outputs/01..05
    # Preprocessing -> <subject_root>/preprocessed/
    preproc_sources = [stage_cache / "00_preprocessing", layout.preproc]
    for pattern, target_name in (
        ("*_dwi_final.nii.gz", f"{subject_id}_dwi_final.nii.gz"),
        ("*_dwi_final.bval", f"{subject_id}_dwi_final.bval"),
        ("*_dwi_final.bvec", f"{subject_id}_dwi_final.bvec"),
        ("*_brain_mask_final.nii.gz", f"{subject_id}_brain_mask_final.nii.gz"),
    ):
        src = _find_first_file(preproc_sources, (pattern,))
        if src is not None:
            dest = layout.preproc / target_name
            publish_output_file(src, dest, replace=True)
            published[f"preproc:{target_name}"] = dest

    # 01_anatomical_priors
    seg_sources = [stage_cache / "01_anatomical_priors/tissue_segmentation", layout.tissue_segmentation]
    for fname in ("SegmentationMap_GMWMCSF.nii.gz", "ProbabilisticMap_WM.nii.gz", "ProbabilisticMap_GM.nii.gz", "ProbabilisticMap_CSF.nii.gz"):
        src = _find_first_file(seg_sources, (fname,))
        if src is not None:
            dest = layout.tissue_segmentation / fname
            publish_output_file(src, dest, replace=True)
            published[f"tissue_segmentation:{fname}"] = dest

    parcel_sources = [stage_cache / "01_anatomical_priors/brain_parcellation", layout.brain_parcellation]
    parcel_src = _find_first_file(parcel_sources, (f"{subject_id}_parcellation.nii.gz", "*_parcellation.nii.gz", "*-DDSurfer-wmparc.nii.gz", "*wmparc_labels.nii.gz"))
    if parcel_src is not None:
        dest = layout.brain_parcellation / f"{subject_id}_parcellation.nii.gz"
        publish_output_file(parcel_src, dest, replace=True)
        published["brain_parcellation"] = dest

    surf_sources = [stage_cache / "01_anatomical_priors/cortical_surface", layout.cortical_surface]
    for obj_name in ("lh.white.obj", "lh.pial.obj", "rh.white.obj", "rh.pial.obj"):
        src = _find_first_file(surf_sources, (obj_name,))
        if src is not None:
            dest = layout.cortical_surface / obj_name
            publish_output_file(src, dest, replace=True)
            published[f"cortical_surface:{obj_name}"] = dest

    morph_sources = [stage_cache / "01_anatomical_priors/surface_morphometry", layout.surface_morphometry]
    morph_patterns = (
        "lh.aparc.stats", "rh.aparc.stats", "lh.aparc.a2009s.stats", "rh.aparc.a2009s.stats",
        "lh.thickness", "rh.thickness", "lh.curv", "rh.curv", "lh.sulc", "rh.sulc",
        "lh.area", "rh.area", "lh.inflated", "rh.inflated",
        "lh.aparc.annot", "rh.aparc.annot", "lh.aparc.a2009s.annot", "rh.aparc.a2009s.annot",
    )
    for mname in morph_patterns:
        src = _find_first_file(morph_sources, (mname,))
        if src is not None:
            dest = layout.surface_morphometry / mname
            publish_output_file(src, dest, replace=True)
            published[f"surface_morphometry:{mname}"] = dest

    # 02_microstructure
    roi_src = _find_first_file(
        [stage_cache / "02_microstructure/regional_microstructure", stage_cache / "02_microstructure", layout.microstructure],
        (f"{subject_id}_ROIWise-Microstructure.csv", "*_ROIWise-Microstructure.csv"),
    )
    if roi_src is not None:
        dest = layout.microstructure / f"{subject_id}_ROIWise-Microstructure.csv"
        publish_output_file(src=roi_src, dest=dest, replace=True)
        published["microstructure_roi_csv"] = dest

    dti_sources = [stage_cache / "02_microstructure/dti", stage_cache / "02_microstructure/diffusion_tensor", layout.diffusion_tensor]
    for patterns, dest_name in (
        ((f"{subject_id}_FA.nii.gz", f"{subject_id}_dti_FA.nii.gz", "*_dti_FA.nii.gz"), f"{subject_id}_FA.nii.gz"),
        ((f"{subject_id}_MD.nii.gz", f"{subject_id}_dti_MD.nii.gz", "*_dti_MD.nii.gz"), f"{subject_id}_MD.nii.gz"),
        ((f"{subject_id}_AD.nii.gz", f"{subject_id}_dti_AD.nii.gz", "*_dti_AD.nii.gz"), f"{subject_id}_AD.nii.gz"),
        ((f"{subject_id}_RD.nii.gz", f"{subject_id}_dti_RD.nii.gz", "*_dti_RD.nii.gz"), f"{subject_id}_RD.nii.gz"),
        ((f"{subject_id}_DEC.nrrd", f"{subject_id}-dti-DEC.nrrd", "*DEC.nrrd"), f"{subject_id}_DEC.nrrd"),
    ):
        src = _find_first_file(dti_sources, patterns)
        if src is not None:
            dest = layout.diffusion_tensor / dest_name
            publish_output_file(src, dest, replace=True)
            published[f"dti:{dest_name}"] = dest

    fw_sources = [stage_cache / "02_microstructure/freewater", stage_cache / "02_microstructure/free_water", layout.free_water]
    for patterns, dest_name in (
        ((f"{subject_id}_FW.nii.gz", f"{subject_id}_fw_fraction.nii.gz", "*_fw_fraction.nii.gz"), f"{subject_id}_FW.nii.gz"),
        ((f"{subject_id}_fw_FA.nii.gz", "*_fw_FA.nii.gz"), f"{subject_id}_fw_FA.nii.gz"),
        ((f"{subject_id}_fw_MD.nii.gz", "*_fw_MD.nii.gz"), f"{subject_id}_fw_MD.nii.gz"),
    ):
        src = _find_first_file(fw_sources, patterns)
        if src is not None:
            dest = layout.free_water / dest_name
            publish_output_file(src, dest, replace=True)
            published[f"fw:{dest_name}"] = dest

    ifw_sources = [stage_cache / "02_microstructure/ifw", stage_cache / "02_microstructure/interstitial_free_water", layout.interstitial_free_water]
    for patterns, dest_name in (
        ((f"{subject_id}_ifw.nii.gz", "*_ifw.nii.gz"), f"{subject_id}_ifw.nii.gz"),
        ((f"{subject_id}_ROIWise-iFW.csv", "*_ROIWise-iFW.csv"), f"{subject_id}_ROIWise-iFW.csv"),
    ):
        src = _find_first_file(ifw_sources, patterns)
        if src is not None:
            dest = layout.interstitial_free_water / dest_name
            publish_output_file(src, dest, replace=True)
            published[f"ifw:{dest_name}"] = dest

    noddi_sources = [stage_cache / "02_microstructure/noddi", stage_cache / "02_microstructure/neurite_microstructure", layout.neurite_microstructure]
    for metric in ("NDI", "ODI", "IsoVF"):
        src = _find_first_file(noddi_sources, (f"{subject_id}_{metric}.nii.gz", f"{subject_id}_noddi_{metric}.nii.gz", f"*_noddi_{metric}.nii.gz"))
        if src is not None:
            dest = layout.neurite_microstructure / f"{subject_id}_{metric}.nii.gz"
            publish_output_file(src, dest, replace=True)
            published[f"noddi:{metric}"] = dest

    dki_sources = [stage_cache / "02_microstructure/dki", stage_cache / "02_microstructure/diffusion_kurtosis", layout.diffusion_kurtosis]
    for metric in ("MK", "AK", "RK", "KFA", "MKT", "RTK"):
        src = _find_first_file(dki_sources, (f"{subject_id}_{metric}.nii.gz", f"{subject_id}_dki_{metric}.nii.gz", f"*_dki_{metric}.nii.gz"))
        if src is not None:
            dest = layout.diffusion_kurtosis / f"{subject_id}_{metric}.nii.gz"
            publish_output_file(src, dest, replace=True)
            published[f"dki:{metric}"] = dest

    # 03_tractography
    tract_sources = [stage_cache / "03_tractography", layout.tractography]
    for target_dir, specs in (
        (layout.single_tensor_tractography, ((("*_dti_det.trk", "*_dti.trk", "*-dti-deterministic.trk", "*dti_det*.trk"), f"{subject_id}_dti.trk"), (("*_dti_det.vtk", "*_dti.vtk", "*-dti-deterministic.vtk", "*dti_det*.vtk"), f"{subject_id}_dti.vtk"))),
        (layout.multi_tensor_tractography, ((("*_ukf.trk",), f"{subject_id}_ukf.trk"), (("*_ukf.vtk",), f"{subject_id}_ukf.vtk"))),
        (layout.anatomically_constrained_tractography, ((("*_csd_act*.tck", "*_act.tck"), f"{subject_id}_act.tck"), (("*_csd_act*.trk", "*_act.trk"), f"{subject_id}_act.trk"), (("*_csd_act*.vtk", "*_act.vtk"), f"{subject_id}_act.vtk"))),
        (layout.short_association_fibers, ((("SAF.tck", "*_saf.tck"), f"{subject_id}_saf.tck"), (("SAF.vtk", "*_saf.vtk"), f"{subject_id}_saf.vtk"))),
    ):
        for patterns, dest_name in specs:
            src = _find_first_file(tract_sources, patterns)
            if src is not None:
                dest = target_dir / dest_name
                publish_output_file(src, dest, replace=True)
                published[f"tract:{dest_name}"] = dest

    # 04_bundles
    wmql_src = _find_first_file(legacy_bundle_roots.get("wmql", ()), ("*.trk",))
    if wmql_src is not None:
        for p in publish_output_directory_flat(wmql_src.parent, layout.query_bundle_extraction, suffixes=(".trk",)):
            published[f"wmql:{p.name}"] = p

    wma_root = _find_first_file(legacy_bundle_roots.get("wma_org_atlas", ()), ("T_*.vtp", "*.vtp"))
    if wma_root is not None:
        for p in publish_output_directory_flat(wma_root.parent, layout.fiber_clustering, suffixes=(".vtp",)):
            published[f"wma:{p.name}"] = p
        meas = _find_first_file([wma_root.parent.parent], ("bundle_measurements.csv", "*measurements.csv"))
        if meas is not None:
            dest = layout.fiber_clustering / "bundle_measurements.csv"
            publish_output_file(meas, dest, replace=True)
            published["wma:bundle_measurements.csv"] = dest

    tc_src = _find_first_file(legacy_bundle_roots.get("tract_cloud_dl", ()), ("*.vtp",))
    if tc_src is not None:
        for p in publish_output_directory_flat(tc_src.parent, layout.bundle_identification, suffixes=(".vtp",)):
            published[f"tractcloud:{p.name}"] = p

    # 05_connectome
    conn_sources = [stage_cache / "05_connectome", layout.connectome]
    for patterns, dest_name in (
        ((f"{subject_id}_streamline_count.csv", "*-connectome-streamline_count.csv", "*streamline_count.csv"), f"{subject_id}_streamline_count.csv"),
        ((f"{subject_id}_sift2_weighted.csv", "*-connectome-sift2_weighted*.csv", "*sift2_weighted*.csv"), f"{subject_id}_sift2_weighted.csv"),
        ((f"{subject_id}_nodes.tsv", "*-connectome-nodes.tsv", "*_nodes.tsv"), f"{subject_id}_nodes.tsv"),
        ((f"{subject_id}_node_parcellation.nii.gz", "*-connectome-node-parcellation.nii.gz", "*_node_parcellation.nii.gz"), f"{subject_id}_node_parcellation.nii.gz"),
    ):
        src = _find_first_file(conn_sources, patterns)
        if src is not None and "_dense" not in src.name.lower():
            dest = layout.connectome / dest_name
            publish_output_file(src, dest, replace=True)
            published[f"connectome:{dest_name}"] = dest
    # Also publish any scalar metric connectomes
    for croot in conn_sources:
        if not croot.is_dir():
            continue
        for cfile in sorted(croot.rglob("*-connectome-*.csv")):
            lower = cfile.name.lower()
            if "dense" in lower or "streamline_count" in lower or "sift2_weighted" in lower:
                continue
            m = re.search(r"-connectome-([a-z0-9_]+)\.csv$", lower)
            if m:
                dest = layout.connectome / f"{subject_id}_{m.group(1)}.csv"
                publish_output_file(cfile, dest, replace=True)
                published[f"connectome:{dest.name}"] = dest

    _refresh_microstructure_manifest_flags(layout, roi_csv=published.get("microstructure_roi_csv") if isinstance(published.get("microstructure_roi_csv"), Path) else None)
    qc_written = _migrate_qc_from_stage_caches(layout, subject_id)
    for k, qpath in qc_written.items():
        published[f"qc:{k}"] = qpath

    for top in CANONICAL_OUTPUT_TOP_DIRS:
        if (outputs / top).is_dir():
            _remove_empty_dirs(outputs / top, stop_at=outputs)
    return {"subject_id": subject_id, "published": published, "qc": qc_written}


def materialize_canonical_output_tree(
    subject_root: Path,
    *,
    output_root: Optional[Path] = None,
    mode: str = "copy",
    archive_legacy: bool = False,
) -> dict[str, Path]:
    """Ensure existing legacy deliverables are exposed in the canonical layout (symlink-free)."""

    _ = mode
    result = migrate_subject_output_tree(
        subject_root,
        output_root=output_root,
        clean_legacy=archive_legacy,
    )
    return {k: v for k, v in (result.get("published") or {}).items() if isinstance(v, Path)}


def archive_development_outputs(
    subject_root: Path,
    *,
    output_root: Optional[Path] = None,
    current_run_name: str = "csd_act",
    current_tractography_backend: str = "csd_act",
    archive_alternate_tractograms: bool = True,
) -> dict[str, Path]:
    """Move non-canonical development/scratch folders into ``cache/`` while preserving legacy numbered trees."""

    layout = subject_layout(subject_root, output_root=output_root)
    archived: dict[str, Path] = {}
    outputs = layout.outputs
    archive_root = layout.cache / "legacy_outputs"
    preserved_top_dirs = set(CANONICAL_OUTPUT_TOP_DIRS) | {
        "input",
        "cache",
        "preprocessed",
        "01_preproc",
        "03_anatomy",
        "03_structural",
        "04_tractography",
        "05_statistics",
        "06_slicer_scene_report",
    }
    if outputs.is_dir():
        for entry in sorted(outputs.iterdir()):
            if entry.name in preserved_top_dirs:
                continue
            moved = _archive_debug_path(entry, archive_root)
            if moved is not None:
                archived[entry.as_posix()] = moved
        if layout.tractography.is_dir():
            allowed_tract_dirs = {
                "single_tensor_tractography",
                "multi_tensor_tractography",
                "anatomically_constrained_tractography",
                "short_association_fibers",
            }
            for entry in sorted(layout.tractography.iterdir()):
                if (entry.is_dir() or entry.is_symlink()) and entry.name not in allowed_tract_dirs:
                    moved = _archive_debug_path(entry, archive_root / "03_tractography")
                    if moved is not None:
                        archived[entry.as_posix()] = moved
        if layout.bundles.is_dir():
            allowed_bundle_dirs = {
                "query_bundle_extraction",
                "fiber_clustering",
                "bundle_identification",
            }
            for entry in sorted(layout.bundles.iterdir()):
                if (entry.is_dir() or entry.is_symlink()) and entry.name not in allowed_bundle_dirs:
                    moved = _archive_debug_path(entry, archive_root / "04_bundles")
                    if moved is not None:
                        archived[entry.as_posix()] = moved
        if layout.connectome.is_dir():
            for entry in sorted(layout.connectome.iterdir()):
                is_metadata_json = entry.is_file() and (
                    entry.name.endswith("_qc.json") or entry.name.endswith("_manifest.json")
                )
                if entry.is_dir() or entry.is_symlink() or is_metadata_json:
                    moved = _archive_debug_path(entry, archive_root / "05_connectome")
                    if moved is not None:
                        archived[entry.as_posix()] = moved
    if archive_alternate_tractograms and layout.tractograms.is_dir():
        alt_root = layout.cache / "alternate_runs" / "tractograms"
        run_name = str(current_run_name or "").strip()
        backend_name = str(current_tractography_backend or "").strip()
        known_backends = {"dti_det", "ukf", "csd_act", "csd_prob", "saf"}
        for entry in sorted(layout.tractograms.iterdir()):
            if not entry.is_dir() and not entry.is_symlink():
                continue
            if entry.name == run_name:
                continue
            if entry.is_dir() and entry.name in known_backends and entry.name == backend_name:
                for subentry in sorted(entry.iterdir()):
                    if (subentry.is_dir() or subentry.is_symlink()) and subentry.name != run_name:
                        moved = _archive_debug_path(subentry, alt_root / entry.name)
                        if moved is not None:
                            archived[subentry.as_posix()] = moved
            else:
                moved = _archive_debug_path(entry, alt_root)
                if moved is not None:
                    archived[entry.as_posix()] = moved
    return archived
