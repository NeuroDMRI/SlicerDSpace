"""Harden WMA transforms with Slicer core's Model/Transform nodes.

No SlicerDMRI FiberBundle reader, writer, or TractographyDisplay module is
needed. The pinned WMA author command calls ``Slicer --python-script`` with
four positional arguments. The caller provides the same four arguments here.
This adapter deliberately retains WMA's transform inversion and empty-cluster
behavior; the registration/clustering algorithms are unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import runpy
import shutil
import sys
import time


def _ensure_numpy_core_multiarray() -> None:
    """Attach ``numpy.core.multiarray`` on NumPy 1.26.x ``numpy._core`` stubs."""
    try:
        import numpy as _np

        core_mod = importlib.import_module("numpy._core")
        if not hasattr(core_mod, "multiarray"):
            legacy = getattr(_np, "core", None)
            if legacy is not None and hasattr(legacy, "multiarray"):
                core_mod.multiarray = legacy.multiarray
    except Exception:
        pass


def _add_slicer_module_search_paths(slicer_module) -> None:
    try:
        for path in slicer_module.app.moduleManager().factoryManager().searchPaths:
            if path not in sys.path:
                sys.path.append(path)
    except Exception:
        pass


def _has_slicerdmri_fiber_bundle_logic(slicer_module) -> bool:
    _add_slicer_module_search_paths(slicer_module)
    try:
        import vtkSlicerTractographyDisplayModuleLogicPython  # noqa: F401
        return True
    except Exception:
        return False


def _wait_for_input_status(status_path: str) -> None:
    while not os.path.isfile(status_path):
        time.sleep(0.05)
    if Path(status_path).read_text(encoding="utf-8").strip() != "ready":
        raise RuntimeError("Input preparation failed; no files were processed.")


def _prepare_wma_split_modules(wma_root: Path):
    _ensure_numpy_core_multiarray()
    for path in (str(wma_root), str(wma_root / "bin")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import whitematteranalysis as wma
    import wm_separate_clusters_by_hemisphere as separation

    return wma, separation


def _run_fast_core_harden(args, *, wma_root: Path | None = None) -> int:
    """Fast Slicer-core ModelNode batch hardening with optional inline hemisphere split.

    Works under ``Slicer --no-main-window --disable-modules`` without requiring the
    optional SlicerDMRI extension, reusing a single MRML Model/Storage node inside
    ``BatchProcessState`` and supporting ``--input-status`` and ``--split-output-directory``.
    """
    import slicer
    import vtk

    vtk.vtkMultiThreader.SetGlobalMaximumNumberOfThreads(1)
    vtk.vtkMultiThreader.SetGlobalDefaultNumberOfThreads(1)
    _add_slicer_module_search_paths(slicer)
    from vtkSlicerTransformsModuleLogicPython import vtkSlicerTransformLogic

    logic = vtkSlicerTransformLogic()
    wma_mod = separation_mod = None
    if args.split_output_directory is not None:
        if wma_root is None or not wma_root.is_dir():
            raise RuntimeError("WMA root is required for --split-output-directory")
        wma_mod, separation_mod = _prepare_wma_split_modules(wma_root)
    if args.input_status is not None:
        _wait_for_input_status(args.input_status)

    source = Path(args.polydata)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted((*source.glob("*.vtk"), *source.glob("*.vtp")))
    else:
        raise FileNotFoundError(f"WMA polydata not found: {source}")
    paths = paths[args.partition_index::args.partition_count]
    if not paths:
        return 0

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_output_directory) if args.split_output_directory else None
    if split_dir is not None:
        for folder in ("tracts_left_hemisphere", "tracts_right_hemisphere", "tracts_commissural"):
            (split_dir / folder).mkdir(parents=True, exist_ok=True)

    transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
    transform_storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformStorageNode")
    transform_node.SetAndObserveStorageNodeID(transform_storage.GetID())
    transform_storage.SetFileName(str(args.transform))
    if not transform_storage.ReadData(transform_node):
        slicer.mrmlScene.RemoveNode(transform_node)
        slicer.mrmlScene.RemoveNode(transform_storage)
        raise RuntimeError(f"Could not load transform file: {args.transform}")

    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    storage_node = model_node.CreateDefaultStorageNode()
    slicer.mrmlScene.AddNode(storage_node)
    model_node.SetAndObserveStorageNodeID(storage_node.GetID())

    def _split_poly(polydata, poly_path: Path) -> None:
        found, labels = separation_mod.read_mask_location_from_vtk(polydata)
        if not found:
            raise RuntimeError(f"HemisphereLocation is missing from transformed file: {poly_path}")
        if (labels == 0).sum() > 1:
            raise RuntimeError(f"Hemisphere labels are incomplete: {poly_path}")
        separation_mod.split_polydata(polydata, labels, poly_path.name, str(split_dir))

    try:
        if str(args.inverse) == "1":
            transform_node.Inverse()
        for poly_path in paths:
            out_path = outdir / poly_path.name
            if out_path.exists():
                if split_dir is None:
                    continue
                split_files = [
                    split_dir / folder / poly_path.name
                    for folder in ("tracts_left_hemisphere", "tracts_right_hemisphere", "tracts_commissural")
                ]
                if all(p.is_file() for p in split_files):
                    continue
                _split_poly(wma_mod.io.read_polydata(str(out_path)), out_path)
                continue

            model_node.SetAndObservePolyData(None)
            storage_node.SetCoordinateSystem(storage_node.CoordinateSystemRAS)
            storage_node.SetFileName(str(poly_path))
            if not storage_node.ReadData(model_node):
                raise RuntimeError(f"Could not load polydata file: {poly_path}")
            data = model_node.GetPolyData()
            if data is None:
                raise RuntimeError(f"Could not load polydata geometry: {poly_path}")
            if data.GetNumberOfCells() == 0:
                shutil.copyfile(poly_path, out_path)
            else:
                model_node.SetAndObserveTransformNodeID(transform_node.GetID())
                if not logic.hardenTransform(model_node):
                    raise RuntimeError(f"Could not harden transform: {poly_path}")
                data = model_node.GetPolyData()
                if split_dir is None or out_path.suffix.lower() != ".vtp":
                    storage_node.SetFileName(str(out_path))
                    storage_node.SetCoordinateSystem(storage_node.CoordinateSystemRAS)
                    if not storage_node.WriteData(model_node):
                        raise RuntimeError(f"Could not save transformed polydata: {out_path}")
                else:
                    space = vtk.vtkStringArray()
                    space.SetName("SPACE")
                    space.InsertNextValue("RAS")
                    data.GetFieldData().AddArray(space)
                    wma_mod.io.write_polydata(data, str(out_path))
            if split_dir is not None:
                _split_poly(data, out_path)
    finally:
        owned = [model_node.GetNthDisplayNode(i) for i in range(model_node.GetNumberOfDisplayNodes())]
        owned.append(model_node.GetStorageNode())
        slicer.mrmlScene.RemoveNode(model_node)
        for node in owned:
            if node is not None and node.GetScene() == slicer.mrmlScene:
                slicer.mrmlScene.RemoveNode(node)
        t_storage = transform_node.GetStorageNode()
        slicer.mrmlScene.RemoveNode(transform_node)
        if t_storage is not None and t_storage.GetScene() == slicer.mrmlScene:
            slicer.mrmlScene.RemoveNode(t_storage)
        slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)
    return 0


def harden_transform(polydata, transform, inverse, outdir, *, slicer_module=None, transformation=None):
    """Transform one .vtk/.vtp cluster using only Slicer-core MRML classes."""
    if slicer_module is None:
        import slicer as slicer_module
    slicer = slicer_module
    source = Path(polydata)
    destination = Path(outdir) / source.name
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    model = owned_transformation = storage = None
    try:
        # loadModel uses Slicer's core ModelFile reader, unlike loadFiberBundle
        # whose FiberBundleFile reader is registered by SlicerDMRI.
        model = slicer.util.loadModel(str(source))
        if model is None or model.GetPolyData() is None:
            raise RuntimeError(f"Could not load model polydata: {source}")
        if model.GetPolyData().GetNumberOfCells() == 0:
            shutil.copyfile(source, destination)
            return destination

        if transformation is None:
            owned_transformation = slicer.util.loadTransform(str(transform))
            if owned_transformation is None:
                raise RuntimeError(f"Could not load transform: {transform}")
            if str(inverse) == "1":
                owned_transformation.Inverse()
            active_transform = owned_transformation
        else:
            active_transform = transformation
        model.SetAndObserveTransformNodeID(active_transform.GetID())
        slicer.vtkSlicerTransformLogic().hardenTransform(model)

        # Explicit Model storage prevents SlicerDMRI's optional TRX/FiberBundle
        # writers from changing how .vtk/.vtp data are serialized.
        storage = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelStorageNode")
        if storage is None:
            raise RuntimeError("Slicer core vtkMRMLModelStorageNode is unavailable")
        storage.SetFileName(str(destination))
        if not storage.WriteData(model):
            raise RuntimeError(f"Failed to save transformed model: {destination}")
        return destination
    finally:
        for node in (storage, model, owned_transformation):
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)


def main(argv=None, *, slicer_module=None):
    parser = argparse.ArgumentParser(description="WMA transform using Slicer core; SlicerDMRI is not required")
    parser.add_argument("polydata")
    parser.add_argument("transform")
    parser.add_argument("inverse", choices=("0", "1"))
    parser.add_argument("outdir")
    parser.add_argument("partition_index", type=int, nargs="?", default=0)
    parser.add_argument("partition_count", type=int, nargs="?", default=1)
    parser.add_argument("--split-output-directory", default=None)
    parser.add_argument("--profile-output", default=None)
    parser.add_argument("--input-status", default=None)
    args = parser.parse_args(argv)
    source = Path(args.polydata)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted((*source.glob("*.vtk"), *source.glob("*.vtp")))
    else:
        raise FileNotFoundError(f"WMA polydata not found: {source}")
    paths = paths[args.partition_index::args.partition_count]
    if not paths:
        return 0
    if slicer_module is None:
        import slicer as slicer_module
    shared_transform = None
    try:
        if len(paths) > 1:
            shared_transform = slicer_module.util.loadTransform(str(args.transform))
            if shared_transform is not None and str(args.inverse) == "1":
                shared_transform.Inverse()
        for path in paths:
            harden_transform(
                path, args.transform, args.inverse, args.outdir,
                slicer_module=slicer_module, transformation=shared_transform,
            )
    finally:
        if shared_transform is not None:
            slicer_module.mrmlScene.RemoveNode(shared_transform)
    return 0


def run(author=None):
    """Compatibility entrypoint used by tractography/wma.py's existing shim.

    Delegates to ``whitematteranalysis_gpu``'s partitioned/split hardener when
    present AND SlicerDMRI's FiberBundle logic is installed, and otherwise uses
    Slicer-core Model/Transform batch hardening (supporting ``--input-status``
    and ``--split-output-directory`` without requiring SlicerDMRI).
    """
    author_str = author or os.environ.get("WMA_AUTHOR_HARDEN_HELPER")
    wma_root: Path | None = None
    if os.environ.get("WMA_ROOT"):
        wma_root = Path(os.environ["WMA_ROOT"]).resolve()
    if author_str:
        author_path = Path(author_str)
        if author_path.is_file():
            candidate_root = author_path.resolve().parents[1]
            if (candidate_root / "whitematteranalysis" / "gpu" / "__init__.py").is_file():
                wma_root = wma_root or candidate_root
                import slicer as _slicer

                if _has_slicerdmri_fiber_bundle_logic(_slicer):
                    _ensure_numpy_core_multiarray()
                    runpy.run_path(str(author_path), run_name="__main__")
                    return 0
                parser = argparse.ArgumentParser(description="Fast Slicer-core WMA hardening")
                parser.add_argument("polydata")
                parser.add_argument("transform")
                parser.add_argument("inverse", choices=("0", "1"))
                parser.add_argument("outdir")
                parser.add_argument("partition_index", type=int, nargs="?", default=0)
                parser.add_argument("partition_count", type=int, nargs="?", default=1)
                parser.add_argument("--split-output-directory", default=None)
                parser.add_argument("--profile-output", default=None)
                parser.add_argument("--input-status", default=None)
                args = parser.parse_args(sys.argv[1:])
                return _run_fast_core_harden(args, wma_root=wma_root)
    return main(sys.argv[1:])


if __name__ == "__main__":
    if os.environ.get("WMA_AUTHOR_HARDEN_HELPER") or "--split-output-directory" in sys.argv or "--input-status" in sys.argv:
        try:
            rc = run()
        except Exception:
            import traceback
            traceback.print_exc()
            sys.exit(1)
        if rc:
            sys.exit(rc)
    else:
        sys.exit(main())
