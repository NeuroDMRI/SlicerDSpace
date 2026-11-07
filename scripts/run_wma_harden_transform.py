"""Harden WMA transforms with Slicer core's Model/Transform nodes.

No SlicerDMRI FiberBundle reader, writer, or TractographyDisplay module is
needed. The pinned WMA author command calls ``Slicer --python-script`` with
four positional arguments. The caller provides the same four arguments here.
This adapter deliberately retains WMA's transform inversion and empty-cluster
behavior; the registration/clustering algorithms are unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


def harden_transform(polydata, transform, inverse, outdir, *, slicer_module=None):
    """Transform one .vtk/.vtp cluster using only Slicer-core MRML classes."""
    if slicer_module is None:
        import slicer as slicer_module
    slicer = slicer_module
    source = Path(polydata)
    destination = Path(outdir) / source.name
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    model = transformation = storage = None
    try:
        # loadModel uses Slicer's core ModelFile reader, unlike loadFiberBundle
        # whose FiberBundleFile reader is registered by SlicerDMRI.
        model = slicer.util.loadModel(str(source))
        if model is None or model.GetPolyData() is None:
            raise RuntimeError(f"Could not load model polydata: {source}")
        if model.GetPolyData().GetNumberOfCells() == 0:
            shutil.copyfile(source, destination)
            return destination

        transformation = slicer.util.loadTransform(str(transform))
        if transformation is None:
            raise RuntimeError(f"Could not load transform: {transform}")
        if str(inverse) == "1":
            transformation.Inverse()
        model.SetAndObserveTransformNodeID(transformation.GetID())
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
        for node in (storage, model, transformation):
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)


def main(argv=None, *, slicer_module=None):
    parser = argparse.ArgumentParser(description="WMA transform using Slicer core; SlicerDMRI is not required")
    parser.add_argument("polydata")
    parser.add_argument("transform")
    parser.add_argument("inverse", choices=("0", "1"))
    parser.add_argument("outdir")
    args = parser.parse_args(argv)
    source = Path(args.polydata)
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted((*source.glob("*.vtk"), *source.glob("*.vtp")))
    else:
        raise FileNotFoundError(f"WMA polydata not found: {source}")
    for path in paths:
        harden_transform(path, args.transform, args.inverse, args.outdir, slicer_module=slicer_module)
    return 0


def run(author=None):
    """Compatibility entrypoint used by tractography/wma.py's existing shim.

    ``author`` is kept only for historical call-site compatibility. No author
    script is executed, because its loadFiberBundle() requires SlicerDMRI.
    """
    del author
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
