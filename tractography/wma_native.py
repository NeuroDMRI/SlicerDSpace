"""Run the pinned WMA ORG stages in Python on a fresh staging directory."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import shutil

from tractography.wma_atlases import atlas_definition


_HEMISPHERE_GROUPS = ("tracts_commissural", "tracts_left_hemisphere", "tracts_right_hemisphere")
_CLUSTER_NAMES = tuple(f"cluster_{index:05d}.vtp" for index in range(1, 801))
_HEMISPHERE_LOG = "cluster_location_by_hemisphere.log"
_HEMISPHERE_DONE = "<wm_assess_cluster_location_by_hemisphere.py> Done!!!"


def _has_product(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _require_product(stage: str, path: Path) -> None:
    if not _has_product(path):
        raise RuntimeError(f"WMA {stage} did not produce {path}")


def _require_clusters(stage: str, directory: Path) -> None:
    for name in _CLUSTER_NAMES:
        _require_product(stage, directory / name)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def run_org_atlas_native(
    *,
    input_tractography: Path,
    output_dir: Path,
    atlas_dir: Path,
    registration_mode: str,
    num_threads: int,
    clean_files: int,
    transform_path: Path | None,
    run_stage: Callable[[str, list[str]], None],
) -> None:
    """Apply ORG using author Python commands and the Slicer core hardener.

    The callback launches the named script in the proper runtime and raises on
    subprocess failure. Each stage also checks its output because some author
    scripts exit with code zero when a required operation fails.
    """
    if registration_mode not in {"rig", "nonrig"}:
        raise ValueError("registration_mode must be 'rig' or 'nonrig'.")
    if type(num_threads) is not int or num_threads < 1:
        raise ValueError("num_threads must be >= 1.")
    if clean_files not in {0, 1, 2}:
        raise ValueError("clean_files must be 0, 1, or 2.")

    source = input_tractography
    case_dir = output_dir / source.stem
    case_dir.mkdir(parents=True)
    registration_name, clustering_name = atlas_definition("org")["directories"]
    registration_atlas = atlas_dir / registration_name / "registration_atlas.vtk"
    clustering_atlas = atlas_dir / clustering_name
    hemisphere_reference = clustering_atlas / "cluster_hemisphere_location.txt"
    registration_dir = case_dir / "TractRegistration"
    clustering_dir = case_dir / "FiberClustering"

    def harden(stage: str, tractography: Path, transform: Path, inverse: str, destination: Path) -> None:
        run_stage(stage, [
            "run_wma_harden_transform.py", str(tractography), str(transform), inverse, str(destination),
        ])

    if transform_path is not None:
        pretransformed_dir = case_dir / "TransformedTracts"
        harden("pretransform", source, transform_path, "0", pretransformed_dir)
        source = pretransformed_dir / source.name
        _require_product("pretransform", source)

    registered = source
    transforms = []
    modes = ("rigid_affine_fast",) if registration_mode == "rig" else ("affine", "nonrigid")
    for mode in modes:
        stage = f"registration-{mode}"
        run_stage(stage, [
            "wm_register_to_atlas_new.py", "-mode", mode,
            str(registered), str(registration_atlas), str(registration_dir),
        ])
        registered_id = registered.stem
        registration_output = registration_dir / registered_id / "output_tractography"
        registered = registration_output / f"{registered_id}_reg.vtk"
        transform = registration_output / f"itk_txform_{registered_id}.tfm"
        _require_product(stage, registered)
        _require_product(stage, transform)
        transforms.append(transform)

    initial_root = clustering_dir / "InitialClusters"
    initial = initial_root / registered.stem
    run_stage("clustering", [
        "wm_cluster_from_atlas.py", "-j", str(num_threads),
        str(registered), str(clustering_atlas), str(initial_root), "-norender",
    ])
    _require_clusters("clustering", initial)

    outlier_root = clustering_dir / "OutlierRemovedClusters"
    outliers = outlier_root / f"{registered.stem}_outlier_removed"
    run_stage("outlier-removal", [
        "wm_cluster_remove_outliers.py", "-j", str(num_threads),
        str(initial), str(clustering_atlas), str(outlier_root),
    ])
    _require_clusters("outlier-removal", outliers)

    run_stage("hemisphere-assessment", [
        "wm_assess_cluster_location_by_hemisphere.py", "-clusterLocationFile",
        str(hemisphere_reference), str(outliers),
    ])
    hemisphere_log = outliers / _HEMISPHERE_LOG
    _require_product("hemisphere-assessment", hemisphere_log)
    if _HEMISPHERE_DONE not in hemisphere_log.read_text(encoding="utf-8"):
        raise RuntimeError(f"WMA hemisphere assessment did not finish: {hemisphere_log}")

    transformed_root = clustering_dir / "TransformedClusters"
    transformed = transformed_root / input_tractography.stem
    to_transform = outliers
    if registration_mode == "nonrig":
        to_transform = transformed / "tmp"
        harden("inverse-nonrigid", outliers, transforms[1], "1", to_transform)
        _require_clusters("inverse-nonrigid", to_transform)
    harden("inverse-registration", to_transform, transforms[0], "1", transformed)
    _require_clusters("inverse-registration", transformed)

    separated = clustering_dir / "SeparatedClusters"
    run_stage("hemisphere-split", [
        "wm_separate_clusters_by_hemisphere.py", str(transformed), str(separated),
    ])
    for group in _HEMISPHERE_GROUPS:
        _require_clusters("hemisphere-split", separated / group)

    append_source = separated
    if transform_path is not None:
        append_source = case_dir / "InvTransformedTracts"
        for group in _HEMISPHERE_GROUPS:
            stage = f"inverse-pretransform-{group}"
            harden(stage, separated / group, transform_path, "1", append_source / group)
            _require_clusters(stage, append_source / group)

    anatomical = case_dir / "AnatomicalTracts"
    run_stage("anatomical-tracts", [
        "wm_append_clusters_to_anatomical_tracts.py",
        str(append_source), str(clustering_atlas), str(anatomical),
    ])
    _require_product("anatomical-tracts", anatomical / "T_UF_right.vtp")
    _require_product("anatomical-tracts", anatomical / "T_MCP.vtp")
    # The author shell checks a minimum of 73 files. The append script
    # determines the exact bundle inventory from its ORG definitions.
    if sum(_has_product(path) for path in anatomical.glob("*.vtp")) < 73:
        raise RuntimeError(f"WMA anatomical-tracts produced fewer than 73 tract files in {anatomical}")

    if clean_files == 1:
        shutil.rmtree(initial_root)
        shutil.rmtree(transformed_root)
    elif clean_files == 2:
        for transform in transforms:
            for registered_path in transform.parent.glob("*.vtk"):
                registered_path.unlink()
            for iteration in transform.parent.parent.glob("iteration*"):
                _remove(iteration)
        for directory in (initial_root, outlier_root, transformed_root):
            for product in directory.iterdir():
                _remove(product)
