"""Canonical subject I/O layout for SlicerDSpace outputs.

Pipeline code should use this module for default locations while still honoring
explicit user-provided output paths for backward compatibility. Generated
output-tree HTML reports are local artifacts and should be written under an
ignored report directory unless a caller explicitly requests another path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json
import shutil

ANATOMY_OUTPUT_NAME = "03_anatomy"
LEGACY_ANATOMY_OUTPUT_NAME = "03_structural"


def anatomy_input_dir(output_root: Path, product: str = "") -> Path:
    """Read one intact product directory, preferring the new layout, without mutation.

    An existing new directory takes precedence even if incomplete or a broken
    symlink; never fill its missing files from a different historical product.
    Writers must use SubjectIOLayout.anatomy and its canonical child properties.
    """
    relative = Path(product)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("anatomy product must be a relative path without parent traversal")
    current = Path(output_root) / ANATOMY_OUTPUT_NAME / relative
    legacy = Path(output_root) / LEGACY_ANATOMY_OUTPUT_NAME / relative
    if current.exists() or current.is_symlink():
        return current
    return legacy if legacy.exists() else current


@dataclass(frozen=True)
class SubjectIOLayout:
    """Canonical output and cache directories for one subject root."""

    subject_root: Path
    output_root: Optional[Path] = None

    @property
    def outputs(self) -> Path:
        return self.output_root if self.output_root is not None else self.subject_root / "outputs"

    @property
    def run_metadata(self) -> Path:
        return self.outputs / "00_run_metadata"

    @property
    def preproc(self) -> Path:
        return self.outputs / "01_preproc"

    @property
    def microstructure(self) -> Path:
        return self.outputs / "02_microstructure"

    @property
    def microstructure_dti(self) -> Path:
        return self.microstructure / "dti"

    @property
    def microstructure_freewater(self) -> Path:
        return self.microstructure / "freewater"

    @property
    def microstructure_noddi(self) -> Path:
        return self.microstructure / "noddi"

    @property
    def microstructure_dki(self) -> Path:
        return self.microstructure / "dki"

    @property
    def microstructure_ifw(self) -> Path:
        return self.microstructure / "ifw"

    @property
    def anatomy(self) -> Path:
        return self.outputs / ANATOMY_OUTPUT_NAME

    @property
    def structural(self) -> Path:
        """Backward-compatible Python API alias for the canonical anatomy output."""
        return self.anatomy

    def anatomy_input_dir(self, product: str = "") -> Path:
        return anatomy_input_dir(self.outputs, product)

    @property
    def ddseg(self) -> Path:
        return self.anatomy / "ddseg"

    @property
    def ddsurfer(self) -> Path:
        return self.anatomy / "ddsurfer"

    @property
    def ddsurfer_postprocess(self) -> Path:
        return self.ddsurfer / "postprocess"

    @property
    def ddparcel(self) -> Path:
        return self.anatomy / "ddparcel"

    @property
    def registered_to_dwi(self) -> Path:
        return self.anatomy / "registered_to_dwi"

    @property
    def tractography(self) -> Path:
        return self.outputs / "04_tractography"

    @property
    def csd(self) -> Path:
        return self.tractography / "csd"

    @property
    def act_prep(self) -> Path:
        return self.tractography / "act_prep"

    @property
    def tract_cloud(self) -> Path:
        return self.tractography / "tract_cloud"

    @property
    def tractograms(self) -> Path:
        return self.tractography / "tractograms"

    @property
    def tractograms_dti_det(self) -> Path:
        return self.tractograms / "dti_det"

    @property
    def tractograms_csd_prob(self) -> Path:
        return self.tractograms / "csd_prob"

    @property
    def tractograms_csd_act(self) -> Path:
        return self.tractograms / "csd_act"

    @property
    def saf(self) -> Path:
        return self.tractography / "saf"

    @property
    def bundles(self) -> Path:
        return self.tractography / "bundles"

    @property
    def bundles_wma(self) -> Path:
        return self.bundles / "wma_org_atlas"

    @property
    def bundles_tractcloud(self) -> Path:
        return self.bundles / "tract_cloud_dl"

    @property
    def bundles_wmql(self) -> Path:
        return self.bundles / "wmql"

    @property
    def connectome(self) -> Path:
        return self.tractography / "connectome"

    @property
    def statistics(self) -> Path:
        return self.outputs / "05_statistics"

    @property
    def ddsurfer_postprocess_stats(self) -> Path:
        return self.statistics / "ddsurfer" / "postprocess"

    @property
    def regional_microstructure(self) -> Path:
        return self.statistics / "regional_microstructure"

    @property
    def bundle_profiles(self) -> Path:
        return self.statistics / "bundle_profiles"

    @property
    def slicer_scene_report(self) -> Path:
        return self.outputs / "06_slicer_scene_report"

    @property
    def cache(self) -> Path:
        return self.subject_root / "cache"


def subject_layout(subject_root: Path, output_root: Optional[Path] = None) -> SubjectIOLayout:
    return SubjectIOLayout(
        subject_root=subject_root.expanduser().resolve(),
        output_root=output_root.expanduser().resolve() if output_root is not None else None,
    )


def normalize_run_name(run_name: str) -> str:
    normalized = run_name.strip().replace("/", "_")
    return normalized or "csd_act"


def canonical_connectome_output_dir(
    subject_root: Path,
    run_name: str,
    metric: str,
    *,
    output_root: Optional[Path] = None,
) -> Path:
    layout = subject_layout(subject_root, output_root=output_root)
    run_dir = layout.connectome / normalize_run_name(run_name)
    return run_dir if metric == "streamline_count" else (run_dir / metric)


def _parse_legacy_connectome_dir_name(name: str) -> tuple[str, str] | None:
    if name.endswith("_sift2"):
        return name[: -len("_sift2")], "sift2_weighted_streamline_count"
    for metric in ("mean_fa", "mean_md", "mean_length_mm", "total_length_mm", "mrtrix_dense_compressed"):
        suffix = f"_{metric}"
        if name.endswith(suffix):
            return name[: -len(suffix)], metric
    if name.startswith("csd_act"):
        return name, "streamline_count"
    return None


def default_task_output_dir(subject_root: Path, task_name: str) -> Path:
    """Return the canonical default output directory for a workflow task."""

    layout = subject_layout(subject_root)
    normalized = task_name.lower().replace("-", "_")
    mapping = {
        "validation": layout.run_metadata / "validation",
        "input_validation": layout.run_metadata / "validation",
        "preproc": layout.preproc,
        "microstructure": layout.microstructure,
        "dti": layout.microstructure_dti,
        "freewater": layout.microstructure_freewater,
        "noddi": layout.microstructure_noddi,
        "segmentation": layout.ddseg,
        "ddseg": layout.ddseg,
        "surface": layout.ddsurfer,
        "ddsurfer": layout.ddsurfer,
        "surface_postprocess": layout.ddsurfer_postprocess,
        "ddsurfer_postprocess": layout.ddsurfer_postprocess,
        "parcellation": layout.ddparcel,
        "ddparcel": layout.ddparcel,
        "tractography": layout.tractography,
        "tractography_dti": layout.tractograms_dti_det,
        "dti_tractography": layout.tractograms_dti_det,
        "tractography_act": layout.tractograms_csd_act,
        "act": layout.tractograms_csd_act,
        "act_prep": layout.act_prep,
        "saf": layout.saf,
        "short_association_fibres": layout.saf,
        "short_association_fibers": layout.saf,
        "wma": layout.bundles_wma,
        "wma_org_atlas": layout.bundles_wma,
        "wmql": layout.bundles_wmql,
        "wmql_bundle_segmentation": layout.bundles_wmql,
        "tractcloud": layout.bundles_tractcloud,
        "tract_cloud": layout.tract_cloud,
        "connectome": layout.connectome,
        "statistics": layout.statistics,
        "regional_microstructure": layout.regional_microstructure,
        "slicer_scene_report": layout.slicer_scene_report,
    }
    return mapping.get(normalized, layout.outputs / normalized)


def default_task_output_dir_for_dependency(dependency_path: Path, task_name: str) -> Path:
    """Infer a canonical output directory from a dependency already under a subject tree."""

    resolved = dependency_path.expanduser().resolve()
    parts = resolved.parts
    if "outputs" in parts:
        outputs_index = parts.index("outputs")
        if outputs_index > 0:
            return default_task_output_dir(Path(*parts[:outputs_index]), task_name)
    if "input" in parts:
        input_index = parts.index("input")
        if input_index > 0:
            return default_task_output_dir(Path(*parts[:input_index]), task_name)
    return resolved.parent / task_name


def _safe_link_or_copy(src: Path, dest: Path, *, mode: str, replace: bool = False) -> None:
    if dest.is_symlink():
        try:
            if dest.resolve(strict=True) == src.resolve():
                return
        except (FileNotFoundError, OSError):
            dest.unlink()
        else:
            if replace:
                dest.unlink()
            else:
                return
    elif dest.exists():
        if replace:
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        else:
            return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            dest.symlink_to(src.resolve(), target_is_directory=src.is_dir())
            return
        except OSError:
            pass
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
    else:
        shutil.copy2(src, dest)


def _archive_legacy_path(src: Path, archive_root: Path) -> None:
    if not src.exists() or src.is_symlink():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / src.name
    if dest.exists():
        suffix = 1
        while (archive_root / f"{src.name}.{suffix}").exists():
            suffix += 1
        dest = archive_root / f"{src.name}.{suffix}"
    shutil.move(src.as_posix(), dest.as_posix())


def _archive_debug_path(src: Path, archive_root: Path) -> Optional[Path]:
    if not src.exists() or src.is_symlink():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    dest = archive_root / src.name
    if dest.exists():
        suffix = 1
        while (archive_root / f"{src.name}.{suffix}").exists():
            suffix += 1
        dest = archive_root / f"{src.name}.{suffix}"
    shutil.move(src.as_posix(), dest.as_posix())
    return dest


def _merge_directory_contents(src: Path, dest: Path, *, mode: str, files_only: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        if files_only and child.is_dir():
            continue
        _safe_link_or_copy(child, dest / child.name, mode=mode)


def _remove_self_referential_connectome_links(layout: SubjectIOLayout) -> None:
    if not layout.connectome.is_dir():
        return
    for run_dir in sorted(path for path in layout.connectome.iterdir() if path.is_dir()):
        candidate = run_dir / "streamline_count"
        if not candidate.is_symlink():
            continue
        try:
            if candidate.resolve(strict=True) == run_dir.resolve(strict=True):
                candidate.unlink()
        except (FileNotFoundError, OSError):
            candidate.unlink(missing_ok=True)


def _refresh_microstructure_manifest_flags(
    manifest_path: Path,
    *,
    freewater_enabled: bool,
    noddi_enabled: bool,
) -> None:
    if not manifest_path.is_file():
        return
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    changed = False
    freewater_model = payload.get("free_water_model")
    if isinstance(freewater_model, dict) and freewater_enabled and not freewater_model.get("run_enabled"):
        freewater_model["run_enabled"] = True
        changed = True

    noddi_model = payload.get("noddi_model")
    if isinstance(noddi_model, dict) and noddi_enabled and not noddi_model.get("run_enabled"):
        noddi_model["run_enabled"] = True
        changed = True

    if changed:
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _legacy_csd_run_sources(layout: SubjectIOLayout) -> list[Path]:
    sources: list[Path] = []
    if layout.tractography.is_dir():
        sources.extend(
            sorted(
                path
                for path in layout.tractography.iterdir()
                if path.is_dir() and path.name == "csd_act"
            )
        )
    debug_legacy_runs = layout.subject_root / ".debug_outputs" / "04_tractography" / "legacy_csd_runs"
    if debug_legacy_runs.is_dir():
        sources.extend(sorted(path for path in debug_legacy_runs.iterdir() if path.is_dir() and path.name == "csd_act"))
    return sources


def _legacy_connectome_dir_sources(layout: SubjectIOLayout) -> list[Path]:
    sources: list[Path] = []
    if layout.connectome.is_dir():
        sources.extend(sorted(path for path in layout.connectome.iterdir() if path.is_dir()))
    debug_legacy_layout = layout.subject_root / ".debug_outputs" / "04_tractography" / "connectome_legacy_layout"
    if debug_legacy_layout.is_dir():
        sources.extend(sorted(path for path in debug_legacy_layout.iterdir() if path.is_dir()))
    return sources


def materialize_canonical_output_tree(
    subject_root: Path,
    *,
    output_root: Optional[Path] = None,
    mode: str = "symlink",
    archive_legacy: bool = False,
) -> dict[str, Path]:
    """Create canonical output-tree views for known legacy SlicerDSpace outputs.

    ``mode`` controls how existing legacy files are exposed in the canonical tree:
    ``symlink`` avoids duplicating large neuroimaging files, while ``copy`` creates
    standalone files. If ``archive_legacy`` is true, legacy top-level directories
    are moved under ``.legacy_flat_outputs`` after canonical links/copies are
    created so ``outputs`` remains a clean product tree.
    """

    if mode not in {"symlink", "copy"}:
        raise ValueError("mode must be 'symlink' or 'copy'")
    if archive_legacy and mode == "symlink":
        raise ValueError("archive_legacy=True requires mode='copy' so canonical links do not break")

    layout = subject_layout(subject_root, output_root=output_root)
    outputs = layout.outputs
    archive_root = layout.subject_root / ".legacy_flat_outputs"
    created: dict[str, Path] = {}

    directory_map = {
        "validation": layout.run_metadata / "validation",
        "ddseg": layout.ddseg,
        "ddseg_cpu": layout.anatomy / "ddseg_cpu",
        "ddseg_cuda": layout.anatomy / "ddseg_cuda",
        "ddsurfer": layout.ddsurfer,
        "ddparcel": layout.ddparcel,
    }
    for legacy_name, canonical_dir in directory_map.items():
        if canonical_dir.is_relative_to(layout.anatomy):
            previous = outputs / LEGACY_ANATOMY_OUTPUT_NAME / canonical_dir.relative_to(layout.anatomy)
            if previous.exists() or previous.is_symlink():
                continue  # A numbered historical product outranks an older flat view.
        legacy_dir = outputs / legacy_name
        if legacy_dir.exists():
            _safe_link_or_copy(legacy_dir, canonical_dir, mode=mode)
            created[legacy_name] = canonical_dir

    legacy_microstructure_roots = [
        outputs / "microstructure",
        archive_root / "microstructure",
    ]
    for legacy_microstructure in legacy_microstructure_roots:
        if not legacy_microstructure.exists():
            continue
        subject_id = layout.subject_root.name
        freewater_enabled = False
        noddi_enabled = False
        file_map = {
            f"{subject_id}-dti-FractionalAnisotropy.nii.gz": layout.microstructure_dti / f"{subject_id}_dti_FA.nii.gz",
            f"{subject_id}-dti-MeanDiffusivity.nii.gz": layout.microstructure_dti / f"{subject_id}_dti_MD.nii.gz",
            f"{subject_id}-freewater-Fraction.nii.gz": layout.microstructure_freewater / f"{subject_id}_fw_fraction.nii.gz",
            f"{subject_id}-freewater-TissueFA.nii.gz": layout.microstructure_freewater / f"{subject_id}_fw_FA.nii.gz",
            f"{subject_id}-freewater-TissueMD.nii.gz": layout.microstructure_freewater / f"{subject_id}_fw_MD.nii.gz",
            f"{subject_id}-noddi-NDI.nii.gz": layout.microstructure_noddi / f"{subject_id}_noddi_NDI.nii.gz",
            f"{subject_id}-noddi-ODI.nii.gz": layout.microstructure_noddi / f"{subject_id}_noddi_ODI.nii.gz",
            f"{subject_id}-noddi-FWF.nii.gz": layout.microstructure_noddi / f"{subject_id}_noddi_IsoVF.nii.gz",
            "freewater_manifest.json": layout.microstructure_freewater / "freewater_manifest.json",
            "noddi_manifest.json": layout.microstructure_noddi / "noddi_manifest.json",
            f"{subject_id}-ROIWise-FA.csv": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.csv",
            f"{subject_id}-ROIWise-FA.json": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.json",
            f"{subject_id}-ROIWise-Microstructure.csv": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.csv",
            f"{subject_id}-ROIWise-Microstructure.json": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.json",
        }
        for legacy_name, canonical_path in file_map.items():
            legacy_path = legacy_microstructure / legacy_name
            if legacy_path.exists():
                _safe_link_or_copy(legacy_path, canonical_path, mode=mode)
                created[legacy_name] = canonical_path
                if "freewater-" in legacy_name or legacy_name == "freewater_manifest.json":
                    freewater_enabled = True
                if "noddi-" in legacy_name or legacy_name == "noddi_manifest.json":
                    noddi_enabled = True
        legacy_manifest = legacy_microstructure / "manifest.json"
        if legacy_manifest.exists():
            _safe_link_or_copy(legacy_manifest, layout.microstructure / "manifest.json", mode=mode, replace=True)
            _refresh_microstructure_manifest_flags(
                layout.microstructure / "manifest.json",
                freewater_enabled=freewater_enabled,
                noddi_enabled=noddi_enabled,
            )
            created["microstructure_manifest"] = layout.microstructure / "manifest.json"
        amico_dir = legacy_microstructure / "_amico_noddi"
        if amico_dir.exists():
            amico_cache_dir = layout.microstructure / "cache" / "noddi" / "_amico_noddi"
            _safe_link_or_copy(amico_dir, amico_cache_dir, mode=mode)
            created["_amico_noddi"] = amico_cache_dir

    legacy_tractography = outputs / "tractography"
    if legacy_tractography.exists():
        for legacy_path in legacy_tractography.iterdir():
            if legacy_path.is_file():
                _safe_link_or_copy(legacy_path, layout.tractograms_dti_det / legacy_path.name, mode=mode)
                created[f"tractography/{legacy_path.name}"] = layout.tractograms_dti_det / legacy_path.name

    for legacy_csd_run in _legacy_csd_run_sources(layout):
        if not legacy_csd_run.is_dir():
            continue
        if legacy_csd_run.parent == layout.csd:
            continue
        if not legacy_csd_run.name.startswith("csd_act"):
            continue
        target = layout.csd / legacy_csd_run.name
        _safe_link_or_copy(legacy_csd_run, target, mode=mode)
        created[legacy_csd_run.name] = target

    legacy_roots = [outputs]
    debug_legacy_root = layout.subject_root / ".debug_outputs" / "legacy_outputs_root"
    if debug_legacy_root.is_dir():
        legacy_roots.append(debug_legacy_root)
    for legacy_root in legacy_roots:
        for legacy_wma_run in sorted(path for path in legacy_root.glob("wma_org_execute*") if path.is_dir()):
            target = layout.bundles_wma / legacy_wma_run.name
            _safe_link_or_copy(legacy_wma_run, target, mode=mode)
            created[legacy_wma_run.name] = target

    for legacy_connectome_dir in _legacy_connectome_dir_sources(layout):
        if not legacy_connectome_dir.is_dir():
            continue
        parsed = _parse_legacy_connectome_dir_name(legacy_connectome_dir.name)
        if parsed is None:
            continue
        legacy_run_name, metric = parsed
        if legacy_run_name != "csd_act":
            continue
        target = canonical_connectome_output_dir(layout.subject_root, legacy_run_name, metric, output_root=output_root)
        if legacy_connectome_dir == target:
            continue
        if metric == "streamline_count":
            _merge_directory_contents(legacy_connectome_dir, target, mode=mode, files_only=True)
        else:
            _safe_link_or_copy(legacy_connectome_dir, target, mode=mode)
        created[legacy_connectome_dir.name] = target

    debug_legacy_root_files = layout.subject_root / ".debug_outputs" / "04_tractography" / "connectome_legacy_root_files"
    if debug_legacy_root_files.is_dir():
        for root_file in sorted(debug_legacy_root_files.iterdir()):
            if not root_file.is_file():
                continue
            if not root_file.name.startswith(f"{layout.subject_root.name}-connectome-"):
                continue
            target = layout.connectome / "csd_act" / root_file.name
            _safe_link_or_copy(root_file, target, mode=mode)
            created[f"csd_act/{root_file.name}"] = target

    _remove_self_referential_connectome_links(layout)

    tractcloud_root_map = {
        "AnatomicalTracts": layout.bundles_tractcloud / "AnatomicalTracts",
        "AnatomicalTracts_separated": layout.bundles_tractcloud / "AnatomicalTracts_separated",
    }
    for legacy_name, canonical_dir in tractcloud_root_map.items():
        legacy_dir = layout.subject_root / legacy_name
        if legacy_dir.exists():
            _safe_link_or_copy(legacy_dir, canonical_dir, mode=mode)
            created[legacy_name] = canonical_dir

    if archive_legacy:
        for legacy_name in [*directory_map.keys(), "microstructure", "microstructure_direct", "microstructure_regression", "tractography"]:
            _archive_legacy_path(outputs / legacy_name, archive_root)
        for legacy_name in tractcloud_root_map:
            _archive_legacy_path(layout.subject_root / legacy_name, archive_root)

    return created


def archive_development_outputs(
    subject_root: Path,
    *,
    output_root: Optional[Path] = None,
    current_run_name: str = "csd_act",
    current_tractography_backend: str = "csd_act",
    archive_alternate_tractograms: bool = True,
) -> dict[str, Path]:
    """Move development-only output folders out of the canonical product tree.

    This keeps ``outputs/`` aligned with the documented user-facing layout while
    preserving exploratory artifacts under ``.debug_outputs`` for later audits.
    Set ``archive_alternate_tractograms`` false for partial or plan-only runs so
    they cannot redefine the canonical tractography run.
    """

    layout = subject_layout(subject_root, output_root=output_root)
    outputs = layout.outputs
    debug_root = layout.subject_root / ".debug_outputs"
    moved: dict[str, Path] = {}

    def _move_if_exists(path: Path, dest_root: Path) -> None:
        dest = _archive_debug_path(path, dest_root)
        if dest is not None:
            moved[path.as_posix()] = dest

    allowed_top_level = {
        "00_run_metadata",
        "01_preproc",
        "02_microstructure",
        ANATOMY_OUTPUT_NAME,
        LEGACY_ANATOMY_OUTPUT_NAME,
        "04_tractography",
        "05_statistics",
        "06_slicer_scene_report",
    }
    if outputs.is_dir():
        for child in outputs.iterdir():
            if child.name not in allowed_top_level:
                _move_if_exists(child, debug_root / "legacy_outputs_root")

    structural = layout.anatomy
    if structural.is_dir():
        for child in structural.iterdir():
            if child.name.endswith("_dryrun") or child.name.endswith("_smoke"):
                _move_if_exists(child, debug_root / ANATOMY_OUTPUT_NAME)

    tractography = layout.tractography
    if tractography.is_dir():
        for child in tractography.iterdir():
            if child.name.startswith("saf_") and child.name != "saf":
                _move_if_exists(child, debug_root / "04_tractography")
            elif child.name.endswith("_smoke") or child.name.endswith("_dryrun"):
                _move_if_exists(child, debug_root / "04_tractography")
            elif child.is_dir() and child.name.startswith("csd_act") and child.name != "csd":
                canonical_target = layout.csd / child.name
                if canonical_target.exists():
                    _move_if_exists(child, debug_root / "04_tractography" / "legacy_csd_runs")

    tractograms = layout.tractograms
    if archive_alternate_tractograms and tractograms.is_dir():
        backend = (current_tractography_backend or "csd_act").strip().lower()
        run_name = (current_run_name or "csd_act").strip() or "csd_act"
        allowed_top_dirs = {backend} if backend in {"dti_det", "ukf"} else {run_name}
        for child in tractograms.iterdir():
            if not child.is_dir() or child.name in allowed_top_dirs:
                continue
            _move_if_exists(child, layout.tractography / "cache" / "alternate_runs" / "tractograms")
        if backend in {"dti_det", "ukf"}:
            dti_root = layout.tractograms / backend
            if dti_root.is_dir():
                for child in dti_root.iterdir():
                    if child.is_dir() and child.name != run_name:
                        _move_if_exists(child, layout.tractography / "cache" / "alternate_runs" / "tractograms" / backend)

    bundles = layout.bundles
    if bundles.is_dir():
        for child in bundles.iterdir():
            if child.name.startswith("wmql_probe_"):
                _move_if_exists(child, debug_root / "04_tractography" / "bundles")

    connectome = layout.connectome
    if connectome.is_dir():
        for child in connectome.iterdir():
            if child.name.endswith("_check") or child.name.endswith("_smoke") or child.name.endswith("_dryrun"):
                _move_if_exists(child, debug_root / "04_tractography" / "connectome")
            else:
                parsed = _parse_legacy_connectome_dir_name(child.name) if child.is_dir() else None
                if parsed is not None:
                    legacy_run_name, metric = parsed
                    canonical_target = canonical_connectome_output_dir(
                        layout.subject_root,
                        legacy_run_name,
                        metric,
                        output_root=output_root,
                    )
                    if canonical_target != child and canonical_target.exists():
                        _move_if_exists(child, debug_root / "04_tractography" / "connectome_legacy_layout")
        for child in list(connectome.iterdir()):
            if child.is_file() and child.name.startswith(f"{layout.subject_root.name}-connectome-"):
                _move_if_exists(child, debug_root / "04_tractography" / "connectome_legacy_root_files")

    return moved
