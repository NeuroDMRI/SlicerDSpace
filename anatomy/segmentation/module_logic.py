"""Segmentation workflow logic glue for SlicerDSpace."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from shared import ensure_native_dti_bundle, find_subject_root, resolve_pythonslicer_executable
from shared.bundled_python import inject_bundled_python_env
from shared.environment import resolve_auto_torch_device
from shared.external_process import is_slicer_python, text_subprocess_kwargs

DDSEG_PYTHON_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("nibabel>=5.0", "nibabel"),
    ("pynrrd", "nrrd"),
    ("torchvision", "torchvision"),
)


class SegmentationLogicMixin:
    """Logic entrypoints for the DDSeg workflow."""

    def setupDDSegRequirements(self) -> None:
        self._ensure_torch_requirements(DDSEG_PYTHON_REQUIREMENTS)

    def runDDSegPipeline(
        self,
        *,
        mask_path: Optional[Path] = None,
        dwi_path: Optional[Path] = None,
        bval_path: Optional[Path] = None,
        bvec_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        weights_dir: Path,
        slicer_base: Optional[Path] = None,
        slicer_ext: Optional[Path] = None,
        device: str = "auto",
        apply_softmax: bool = False,
        subject_root: Optional[Path] = None,
        subject_id: Optional[str] = None,
    ) -> None:
        self._ensure_ddseg_on_path()
        python_slicer = resolve_pythonslicer_executable()
        resolved_device = resolve_auto_torch_device(device or "auto", Path(python_slicer))
        self.log(f"[DDSeg] Using torch device: {resolved_device} (requested: {device or 'auto'})")

        if not self.segmentation_dir.exists():
            raise FileNotFoundError(f"DDSeg package not found at {self.segmentation_dir}")

        resolved_output_dir: Path
        dti_feature_dir: Optional[Path] = None
        resolved_mask_path: Path
        resolved_dwi_path: Optional[Path]
        resolved_bval_path: Optional[Path]
        resolved_bvec_path: Optional[Path]

        if subject_root is not None:
            if not subject_id:
                raise ValueError("subject_id is required when running DDSeg from a subject root.")
            raw_bundle = find_subject_root(subject_root, subject_id)
            resolved_output_dir = self._resolve_optional_output_dir(
                output_dir,
                subject_root=raw_bundle.subject_root,
                task_name="ddseg",
            )
            native_bundle = ensure_native_dti_bundle(raw_bundle, fit_method="ls", reuse=True, log=self.log)
            dti_feature_dir = native_bundle.bundle_dir
            resolved_mask_path = raw_bundle.mask_path
            resolved_dwi_path = None
            resolved_bval_path = None
            resolved_bvec_path = None
        else:
            if mask_path is None or dwi_path is None or bval_path is None or bvec_path is None:
                raise ValueError(
                    "mask_path, dwi_path, bval_path, and bvec_path are required when no subject root is provided."
                )
            if output_dir is None:
                raise ValueError("output_dir is required when no subject root is provided.")
            resolved_output_dir = self._resolve_optional_output_dir(
                output_dir,
                subject_root=None,
                task_name="ddseg",
            )
            resolved_mask_path = mask_path.expanduser().resolve()
            resolved_dwi_path = dwi_path.expanduser().resolve()
            resolved_bval_path = bval_path.expanduser().resolve()
            resolved_bvec_path = bvec_path.expanduser().resolve()

        runner_script = (self.segmentation_dir / "run_ddseg.py").resolve()
        if not runner_script.is_file():
            raise FileNotFoundError(f"DDSeg runner script not found: {runner_script}")

        weights_dir = weights_dir.expanduser().resolve()
        cmd = [
            python_slicer,
            runner_script.as_posix(),
            "--mask",
            resolved_mask_path.as_posix(),
            "--output",
            resolved_output_dir.as_posix(),
            "--weights-dir",
            weights_dir.as_posix(),
            "--parameter-type",
            "DTI",
            "--device",
            resolved_device,
        ]
        if apply_softmax:
            cmd.append("--apply-softmax")
        if dti_feature_dir is not None:
            cmd.extend(["--dti-feature-dir", dti_feature_dir.as_posix()])
        if resolved_dwi_path is not None:
            cmd.extend(["--dwi-nii", resolved_dwi_path.as_posix()])
        if resolved_bval_path is not None:
            cmd.extend(["--bval", resolved_bval_path.as_posix()])
        if resolved_bvec_path is not None:
            cmd.extend(["--bvec", resolved_bvec_path.as_posix()])
        if slicer_base is not None:
            cmd.extend(["--slicer-base", slicer_base.expanduser().resolve().as_posix()])
        if slicer_ext is not None:
            cmd.extend(["--slicer-ext", slicer_ext.expanduser().resolve().as_posix()])

        self.log(f"[DDSeg] Running in PythonSlicer subprocess: {' '.join(cmd)}")
        child_env = os.environ.copy()
        if is_slicer_python(python_slicer):
            inject_bundled_python_env(child_env)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_env,
            **text_subprocess_kwargs(bufsize=1),
        )
        # H8: a surviving grandchild must not keep the stage open after the direct child has exited.
        from shared.line_stream import stream_process_lines
        return_code = stream_process_lines(
            process,
            lambda line: self.log(f"[DDSeg] {line.rstrip()}"),
            notice=lambda message: self.log(f"[DDSeg] {message}"),
        )
        if return_code != 0:
            raise RuntimeError(f"DDSeg subprocess failed with exit code {return_code}")
