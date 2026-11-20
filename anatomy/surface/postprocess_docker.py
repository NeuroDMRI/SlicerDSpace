#!/usr/bin/env python3
"""
Docker postprocessing implementation for SlicerDSpace surfaces.

This script stages the required inputs (MRI + predicted STL surfaces) into the
postprocess-release repository and calls its docker/run_container.sh helper to
run FreeSurfer-based refinement without requiring a local FreeSurfer install.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from shared.external_process import text_subprocess_kwargs
from shared.staging_paths import resolve_staging_subject_dir

REQUIRED_MRI_FILES = ("brain.mgz", "DDSurfer-wmparc.nii.gz")


def _emit(msg: str, callback: Optional[Callable[[str], None]] = None) -> None:
    if callback:
        callback(msg)
    else:
        print(msg, flush=True)


def copy_file(src: Path, dest: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Required file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def stage_subject_inputs(
    subject_id: str,
    native_dir: Path,
    mri_dir: Path,
    inputs_dir: Path,
    clobber: bool,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    subject_input_dir = resolve_staging_subject_dir(inputs_dir, subject_id, for_delete=clobber)
    surf_dir = subject_input_dir / "surf"
    mri_out_dir = subject_input_dir / "mri"

    if clobber and subject_input_dir.exists():
        shutil.rmtree(subject_input_dir)
    surf_dir.mkdir(parents=True, exist_ok=True)
    mri_out_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_MRI_FILES:
        copy_file(mri_dir / filename, mri_out_dir / filename)
        _emit(f"Copied {filename} to {mri_out_dir}", log_callback)

    pattern = f"{subject_id}_predicted_*_orig.stl"
    stl_paths = sorted(native_dir.glob(pattern))
    if not stl_paths:
        raise FileNotFoundError(
            f"No STL files matching '{pattern}' under {native_dir}. "
            "Ensure SlicerDSpace prediction completed successfully."
        )
    for stl in stl_paths:
        copy_file(stl, surf_dir / stl.name)
        _emit(f"Copied {stl.name} to {surf_dir}", log_callback)

    return subject_input_dir


def run_docker(
    subject_id: str,
    mode: str,
    postprocess_repo: Path,
    license_path: Path,
    inputs_dir: Path,
    outputs_dir: Path,
    image_tag: str | None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    docker_script = postprocess_repo / "docker" / "run_container.sh"
    if not docker_script.is_file():
        raise FileNotFoundError(f"Docker helper not found: {docker_script}")
    if not os.access(docker_script, os.X_OK):
        docker_script.chmod(0o755)
    env = os.environ.copy()
    env["FREESURFER_LICENSE"] = str(license_path)
    env["INPUTS_DIR"] = str(inputs_dir)
    env["OUTPUTS_DIR"] = str(outputs_dir)
    if image_tag:
        env["DDSURFER_POST_IMAGE"] = image_tag

    cmd = ["bash", str(docker_script), subject_id, mode]
    _emit(f"Running docker command: {' '.join(cmd)}", log_callback)
    process = subprocess.Popen(
        cmd,
        env=env,
        cwd=str(postprocess_repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **text_subprocess_kwargs(bufsize=1),
    )
    # H8: a surviving grandchild must not keep the stage open after the direct child has exited.
    from shared.line_stream import stream_process_lines
    return_code = stream_process_lines(
        process,
        lambda line: _emit(line.rstrip(), log_callback),
        notice=lambda message: _emit(message, log_callback),
    )
    if return_code != 0:
        raise RuntimeError(f"Docker postprocess failed with exit code {return_code}")


def run_postprocess(
    subject_id: str,
    native_dir: Path,
    mri_dir: Path,
    postprocess_repo: Path,
    mode: str = "whole",
    license_path: Path | None = None,
    image_tag: str | None = None,
    inputs_dir: Path | None = None,
    outputs_dir: Path | None = None,
    force: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
) -> None:
    if license_path is None:
        license_path = Path.home() / ".freesurfer-license"
    if inputs_dir is None:
        inputs_dir = postprocess_repo / "inputs"
    if outputs_dir is None:
        outputs_dir = postprocess_repo / "outputs"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    staged_dir = stage_subject_inputs(
        subject_id=subject_id,
        native_dir=native_dir,
        mri_dir=mri_dir,
        inputs_dir=inputs_dir,
        clobber=force,
        log_callback=log_callback,
    )
    _emit(f"Inputs staged under {staged_dir}", log_callback)

    run_docker(
        subject_id=subject_id,
        mode=mode,
        postprocess_repo=postprocess_repo,
        license_path=license_path,
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        image_tag=image_tag,
        log_callback=log_callback,
    )
    _emit("Docker postprocess completed.", log_callback)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage SlicerDSpace outputs and run postprocess-release inside Docker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("subject_id", help="Subject identifier (e.g., 877168)")
    parser.add_argument("native_dir", help="Path to SlicerDSpace native output directory (contains *_orig.stl)")
    parser.add_argument("mri_dir", help="Path to MRI directory holding brain.mgz and DDSurfer-wmparc.nii.gz")
    parser.add_argument("--postprocess-repo", default="/data/chengjin/postprocess-release", help="postprocess-release repository root")
    parser.add_argument("--mode", choices=["left", "right", "whole"], default="whole", help="Hemisphere mode for postprocess run.sh")
    parser.add_argument("--license", default=os.environ.get("FREESURFER_LICENSE", str(Path.home() / ".freesurfer-license")), help="Path to FreeSurfer license file")
    parser.add_argument("--image-tag", default=None, help="Custom Docker image tag (defaults to docker/run_container.sh default)")
    parser.add_argument("--inputs-dir", default=None, help="Override inputs directory passed to docker (defaults to <repo>/inputs)")
    parser.add_argument("--outputs-dir", default=None, help="Override outputs directory passed to docker (defaults to <repo>/outputs)")
    parser.add_argument("--force", action="store_true", help="Remove any existing staged subject folder before copying inputs")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    subject_id = args.subject_id
    native_dir = Path(args.native_dir).expanduser().resolve()
    mri_dir = Path(args.mri_dir).expanduser().resolve()
    postprocess_repo = Path(args.postprocess_repo).expanduser().resolve()
    inputs_dir = Path(args.inputs_dir).expanduser().resolve() if args.inputs_dir else postprocess_repo / "inputs"
    outputs_dir = Path(args.outputs_dir).expanduser().resolve() if args.outputs_dir else postprocess_repo / "outputs"
    license_path = Path(args.license).expanduser().resolve()

    for path, desc in [(native_dir, "native output directory"), (mri_dir, "MRI directory"), (postprocess_repo, "postprocess repo")]:
        if not path.exists():
            raise FileNotFoundError(f"{desc} not found: {path}")
    if not license_path.is_file():
        raise FileNotFoundError(f"FreeSurfer license not found: {license_path}")

    _emit(f"Launching Docker postprocess for {subject_id}...", None)
    run_postprocess(
        subject_id=subject_id,
        native_dir=native_dir,
        mri_dir=mri_dir,
        postprocess_repo=postprocess_repo,
        mode=args.mode,
        license_path=license_path,
        image_tag=args.image_tag,
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
