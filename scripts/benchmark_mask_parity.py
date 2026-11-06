#!/usr/bin/env python3
"""M-A mask parity harness: managed ANTs PNL route (reference) vs the native SimpleITK route.

T2 design v4, section 3 M-A (items 1-6) and work item W-A4. Ops tool for fan93; it runs the existing
routes and computes the M-A quantities, and it adds no rule beyond M-A.

Driver (Slicer's Python, which has SimpleITK, nibabel and torch)::

    PythonSlicer scripts/benchmark_mask_parity.py --cases cases.json --out /data04/chengjin/r5_mask \\
        --model-dir <assets>/brain_mask/pnl_cnn_torch --pnl-python <pnl-cnn-torch prefix>/bin/python \\
        [--ants-bin <prefix>/bin] [--torch-python <torch 2.2.2 venv>/bin/python] [--gpu] \\
        [--steps reference,native,t1,t2,t4,t6] [--variant default|sampling100|estimate_once]

``cases.json`` is a list of ``{"id", "dwi", "bval", "bvec"}``. Inputs are read only, through symlinks
under ``<out>/cases/<id>/input``. Results: ``<out>/cases/<id>/result.json`` per case, and
``<out>/summary.json`` with ``<out>/summary.tsv``. A run with a completed record is reused; an
incomplete run folder is renamed (``.stale-<time>``), never deleted.

Workers (started by the driver):

* ``--reference-worker`` runs in the managed PNL Python (pnl_brain_mask package, ANTs 2.3.0): a copy of
  the package's ``run`` that reads the adapter baseline file (A5), keeps its work folder, passes
  ``-n <threads>`` to ``antsRegistrationSyNQuick.sh`` and sets ``ANTS_RANDOM_SEED`` (fixes C6, O8).
* ``--native-worker`` runs ``shared.pnl_mask_native.run`` (A1-A6) with the selected A3 variant.
* ``--inference-worker`` runs the vendored three-view inference on a normalised template-grid array (T4).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_NAME = "IITmean_b0_256.nii.gz"
B0_THRESHOLD = 50.0
PERCENTILE = 99
ROTATION_DEGREES = 2.0  # M-A 1 run B2; a setup choice, reported with the result
QFORM_SFORM_TOLERANCE_MM = 1e-4
THREADS_LINE = re.compile(r"^\s*Number of threads:\s*(\d+)\s*$", re.MULTILINE)
# M-A 1: run -> (seed, -n threads, baseline file). A_repeat is A' (exact repeat of A).
REFERENCE_RUNS = {
    "A": (1, 1, "mean_b0.nii.gz"),
    "A_repeat": (1, 1, "mean_b0.nii.gz"),
    # B1: as in the package's run, -n also sets torch.set_num_threads (production-faithful coupling).
    "B1": (1, 8, "mean_b0.nii.gz"),
    "B2": (1, 1, "mean_b0_rot2.nii.gz"),
    "S2": (2, 1, "mean_b0.nii.gz"),
}
PERTURBATIONS = ("A_repeat", "B1", "B2", "S2")
# M-A 4: the native route at 1 thread (T3) and at 8 threads (diagnostic only).
NATIVE_RUNS = {"N1": 1, "N8": 8}
# M-A failure clause: one A3 change at a time.
VARIANTS = {"default": {}, "sampling100": {"sampling_percentage": 1.0},
            "estimate_once": {"learning_rate_estimation": "Once"}}
STEPS = ("reference", "native", "t1", "t2", "t4", "t6")


# --------------------------------------------------------------------------- shared helpers
def repository_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".part")
    staged.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    staged.replace(path)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def printed_threads(text: str):
    """The last "Number of threads:" value that antsRegistrationSyNQuick.sh printed (synquick.sh:228)."""
    found = THREADS_LINE.findall(text)
    return int(found[-1]) if found else None


def run_child(command, env, log_path: Path) -> dict:
    """Run one child process; its output goes to ``log_path``. Wall time and the child's peak RSS."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([str(part) for part in command], env=env, stdout=stream, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    # ru_maxrss: bytes on macOS, KiB on Linux.
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    return {"returncode": process.returncode, "seconds": time.monotonic() - started,
            "max_rss_mb": usage.ru_maxrss / scale, "log": str(log_path)}


def set_aside(folder: Path) -> None:
    """Rename an incomplete run folder; the harness never deletes."""
    if folder.exists() or folder.is_symlink():
        folder.rename(folder.with_name(f"{folder.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"))


# --------------------------------------------------------------------------- reference worker (managed PNL Python)
def reference_worker(args) -> int:
    """Copy of ``pnl_brain_mask.pipeline.run`` @ 8d201a5 with the M-A changes (W-A4).

    Changes: the input is the adapter's float32 baseline file (A5, the package's 3-D branch);
    the work folder is kept; ``-n <threads>`` reaches antsRegistrationSyNQuick.sh (C6);
    ``ANTS_RANDOM_SEED`` is set (O8); the synquick output is kept for its "Number of threads" line.
    ANTs commands, normalisation, inference and ``finish_native`` are the package's own.
    """
    import nibabel as nib
    import torch
    from pnl_brain_mask import __version__
    from pnl_brain_mask.inference import normalize, predict, select_device
    from pnl_brain_mask.pipeline import ants_workspace, finish_native, mean_b0
    from pnl_brain_mask.weights import SOURCE_COMMIT, TEMPLATE_SHA256, check

    work = Path(args.work).absolute()
    work.mkdir(parents=True)  # a new folder for each run
    check(args.template, TEMPLATE_SHA256)
    selection = select_device(args.device)
    torch.set_num_threads(args.threads)
    phases = {}
    started = time.monotonic()
    image = nib.load(str(args.baseline), keep_file_open=True)
    baseline_data = mean_b0(image, None, B0_THRESHOLD)
    private_ants, private_template = ants_workspace(work, args.template, args.ants_bin)
    baseline = work / "baseline.nii.gz"
    nib.save(nib.Nifti1Image(baseline_data, image.affine, image.header), str(baseline))
    prefix = work / "rigid-"
    # As the package's run_ants, plus the seed; synquick sets the ITK thread count from -n (synquick.sh:394-396).
    env = os.environ.copy()
    env["PATH"] = str(private_ants) + os.pathsep + env.get("PATH", "")
    env["ANTSPATH"] = str(private_ants)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    env["ANTS_RANDOM_SEED"] = str(args.seed)
    phases["prepare"] = time.monotonic() - started
    started = time.monotonic()
    command = [private_ants / "antsRegistrationSyNQuick.sh", "-d", "3", "-f", private_template, "-m", baseline,
               "-t", "r", "-o", prefix, "-n", str(args.threads)]
    log_path = work / "synquick.log"
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(part) for part in command], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        stream.write(result.stdout)
    if result.returncode:
        raise RuntimeError(f"antsRegistrationSyNQuick.sh failed ({result.returncode}); see {log_path}")
    phases["registration"] = time.monotonic() - started
    started = time.monotonic()
    registered = nib.load(str(prefix) + "Warped.nii.gz")
    data = normalize(registered.get_fdata(), PERCENTILE)
    _, _, mask = predict(data, args.model_dir, device=selection["resolved"], batch_size=1)
    phases["inference"] = time.monotonic() - started
    started = time.monotonic()
    candidate = work / "native_mask.nii.gz"
    finish_native(mask, registered, baseline, str(prefix) + "0GenericAffine.mat", candidate,
                  ants_bin=private_ants, threads=args.threads)
    output = nib.load(str(candidate))
    if output.shape != image.shape[:3] or not np.allclose(output.affine, image.affine, rtol=0, atol=1e-4):
        raise RuntimeError("Native geometry changed")
    phases["inverse_and_fill"] = time.monotonic() - started
    printed = printed_threads(result.stdout)
    write_json(Path(args.record), {
        **selection, "status": "completed", "route": "managed-ants", "seed": args.seed, "threads": args.threads,
        "printed_threads": printed, "valid": printed == args.threads, "package_version": __version__,
        "source_commit": SOURCE_COMMIT, "torch_version": torch.__version__, "numpy_version": np.__version__,
        "python": sys.executable, "baseline": str(args.baseline), "baseline_sha256": sha256(args.baseline),
        "mask": str(candidate), "output_sha256": sha256(candidate), "work": str(work),
        "transform": str(prefix) + "0GenericAffine.mat", "warped": str(prefix) + "Warped.nii.gz",
        "phase_seconds": phases})
    return 0


# --------------------------------------------------------------------------- native and inference workers
def native_worker(args) -> int:
    repository_on_path()
    from shared import pnl_mask_native as native

    parameters = replace(native.PARAMETERS, **VARIANTS[args.variant])
    record = native.run(args.baseline, args.output, args.model_dir, args.template, device=args.device,
                        threads=args.threads, keep_dir=args.keep_dir, parameters=parameters)
    write_json(Path(args.record), dict(record, variant=args.variant, baseline=str(args.baseline),
                                       mask=str(args.output), keep_dir=str(args.keep_dir)))
    return 0


def inference_worker(args) -> int:
    """T4: vendored three-view inference of one normalised template-grid array, CPU.

    Imports ``pnl_cnn`` without the ``shared`` package, whose imports a plain torch venv lacks.
    """
    sys.path.insert(0, str(ROOT / "shared"))
    import torch
    from pnl_cnn.inference import predict, select_device

    torch.set_num_threads(args.threads)
    selection = select_device("cpu")
    data = np.load(args.input)
    _, _, mask = predict(data, args.model_dir, device="cpu", batch_size=1)
    np.save(args.output, np.asarray(mask))
    write_json(Path(args.record), dict(selection, status="completed", threads=args.threads, python=sys.executable,
                                       torch_version=torch.__version__, numpy_version=np.__version__,
                                       input_sha256=sha256(args.input)))
    return 0


# --------------------------------------------------------------------------- inputs and pre-checks
def load_cases(path: Path) -> list[dict]:
    # SPEC-GAP: M-A names the cases, not a file format; a JSON list of {id, dwi, bval, bvec} is used.
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    seen = set()
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(case.get("id", ""))) or case["id"] in seen:
            raise ValueError(f"Case id must be unique and use only letters, digits, '.', '_', '-': {case.get('id')!r}")
        seen.add(case["id"])
        for key in ("dwi", "bval", "bvec"):
            case[key] = Path(case[key]).expanduser().absolute()
    return cases


def check_case(case: dict) -> dict:
    """Volume, b-value and b-vector counts; qform/sform codes and the affine difference (M-A pre-check)."""
    import nibabel as nib

    image = nib.load(str(case["dwi"]))
    bvals = np.loadtxt(str(case["bval"])).reshape(-1)
    bvecs = np.atleast_2d(np.loadtxt(str(case["bvec"])))
    volumes = int(image.shape[3]) if image.ndim == 4 else 0
    vectors = bvecs.shape[1] if bvecs.shape[0] == 3 else bvecs.shape[0]
    qform, qcode = image.header.get_qform(coded=True)
    sform, scode = image.header.get_sform(coded=True)
    difference = (float(np.max(np.abs(image.header.get_qform() - image.header.get_sform())))
                  if qform is not None and sform is not None else None)
    # M-A stop rule: volume and b-value counts. The b-vector count is recorded; _mean_b0 refuses a mismatch.
    return {"shape": list(image.shape), "volumes": volumes, "bvals": int(bvals.size), "bvecs": int(vectors),
            "counts_match": volumes == bvals.size and volumes > 0, "bvecs_match": vectors == volumes,
            "orientation": "".join(nib.aff2axcodes(image.affine)),  # M-A limitation: all inventory cases LAS
            "qform_code": int(qcode), "sform_code": int(scode), "qform_sform_max_difference": difference,
            "qform_sform_flag": difference is not None and difference > QFORM_SFORM_TOLERANCE_MM}


def rotated_baseline(source: Path, output: Path, degrees: float = ROTATION_DEGREES) -> np.ndarray:
    """M-A 1 B2: A' = T(c) R_z(degrees) T(-c) A, c = world point of the voxel-grid centre; qform and sform
    both A' with the input's codes; voxel data unchanged."""
    import nibabel as nib

    image = nib.load(str(source))
    affine = image.affine
    centre = affine @ np.r_[(np.asarray(image.shape[:3], dtype=float) - 1) / 2, 1.0]
    angle = math.radians(degrees)
    rotation = np.eye(4)
    rotation[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    shift, back = np.eye(4), np.eye(4)
    shift[:3, 3], back[:3, 3] = centre[:3], -centre[:3]
    rotated = shift @ rotation @ back @ affine
    header = image.header.copy()
    _, qcode = image.header.get_qform(coded=True)
    _, scode = image.header.get_sform(coded=True)
    header.set_qform(rotated, code=int(qcode))
    header.set_sform(rotated, code=int(scode))
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), rotated, header), str(output))
    return rotated


def prepare_case(case: dict, folder: Path) -> dict:
    """Read-only input symlinks, the adapter baseline (A5) and the B2 baseline; their SHA-256."""
    repository_on_path()
    from shared.brain_mask import _mean_b0

    inputs, baselines = folder / "input", folder / "baseline"
    inputs.mkdir(parents=True, exist_ok=True)
    baselines.mkdir(parents=True, exist_ok=True)
    links = {}
    for key, name in (("dwi", "dwi" + "".join(case["dwi"].suffixes[-2:])), ("bval", "bvals"), ("bvec", "bvecs")):
        link = inputs / name
        if not link.is_symlink():
            link.symlink_to(case[key])
        elif Path(os.readlink(link)) != case[key]:
            raise ValueError(f"{link} points to {os.readlink(link)}, not {case[key]}")
        links[key] = link
    baseline = baselines / "mean_b0.nii.gz"
    rotated = baselines / "mean_b0_rot2.nii.gz"
    record_path = baselines / "baseline.json"
    record = read_json(record_path)
    if not (record and baseline.is_file() and rotated.is_file() and record.get("sha256") == sha256(baseline)):
        for path in (baseline, rotated):
            set_aside(path)
        count = _mean_b0(links["dwi"], links["bval"], links["bvec"], baseline, B0_THRESHOLD)
        affine = rotated_baseline(baseline, rotated)
        record = {"b0_volumes": count, "threshold": B0_THRESHOLD, "sha256": sha256(baseline),
                  "rotated_sha256": sha256(rotated), "rotation_degrees": ROTATION_DEGREES,
                  "rotated_affine": affine.tolist()}
        write_json(record_path, record)
    return record


# --------------------------------------------------------------------------- arms
def reference_run(folder: Path, name: str, args, baseline: dict) -> dict:
    """One reference run; a completed run on the same baseline file (SHA-256) is reused."""
    seed, threads, baseline_name = REFERENCE_RUNS[name]
    expected = baseline["rotated_sha256" if baseline_name == "mean_b0_rot2.nii.gz" else "sha256"]
    run_dir = folder / "reference" / name
    record_path = run_dir / "reference.json"
    record = read_json(record_path)
    if record and record.get("status") == "completed" and record.get("baseline_sha256") == expected:
        return record
    set_aside(run_dir)
    run_dir.mkdir(parents=True)
    repository_on_path()
    from shared.managed_pnl import pnl_python_environment

    env = pnl_python_environment(args.pnl_python, {"PYTHONUNBUFFERED": "1"},
                                 path_prepend=[args.ants_bin] if args.ants_bin else [])
    # SPEC-GAP: M-A 1 does not name the reference arm's inference device; CPU is used (T3 is a CPU comparison).
    command = [args.pnl_python, Path(__file__).resolve(), "--reference-worker",
               "--baseline", folder / "baseline" / baseline_name, "--template", args.template,
               "--model-dir", args.model_dir, "--work", run_dir / "work", "--threads", threads,
               "--seed", seed, "--device", "cpu", "--record", record_path]
    if args.ants_bin:
        command += ["--ants-bin", args.ants_bin]
    process = run_child(command, env, run_dir / "worker.log")
    record = read_json(record_path) or {"status": "failed", "seed": seed, "threads": threads}
    record["process"] = process
    if process["returncode"]:
        record["status"] = "failed"
    write_json(record_path, record)
    return record


def native_run(folder: Path, name: str, threads: int, args, baseline: dict, *, device: str = "cpu",
               group: str = "native") -> dict:
    """One native run on the reference arm's baseline file; reused on the same baseline SHA-256."""
    run_dir = folder / group / variant_name(name, args.variant, "")
    record_path = run_dir / "native.json"
    record = read_json(record_path)
    if record and record.get("status") == "completed" and record.get("input_sha256") == baseline["sha256"]:
        return record
    set_aside(run_dir)
    run_dir.mkdir(parents=True)
    command = [sys.executable, Path(__file__).resolve(), "--native-worker",
               "--baseline", folder / "baseline" / "mean_b0.nii.gz", "--template", args.template,
               "--model-dir", args.model_dir, "--output", run_dir / "mask.nii.gz", "--keep-dir", run_dir / "keep",
               "--threads", threads, "--device", device, "--variant", args.variant, "--record", record_path]
    process = run_child(command, dict(os.environ, PYTHONUNBUFFERED="1"), run_dir / "worker.log")
    record = read_json(record_path) or {"status": "failed", "threads": threads, "device": device}
    record["process"] = process
    if process["returncode"]:
        record["status"] = "failed"
    write_json(record_path, record)
    return record


# --------------------------------------------------------------------------- comparisons
def load_mask(path: Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).dataobj)


def compare(mask: np.ndarray, reference: np.ndarray) -> dict:
    """Dice and relative voxel-count difference against the reference, in voxel space."""
    if mask.shape != reference.shape:
        raise ValueError(f"Mask shapes differ: {mask.shape} vs {reference.shape}")
    a, b = np.asarray(mask) != 0, np.asarray(reference) != 0
    count, reference_count, overlap = int(a.sum()), int(b.sum()), int((a & b).sum())
    dice = 2.0 * overlap / (count + reference_count) if count + reference_count else 1.0
    # SPEC-GAP: M-A names the "relative voxel-count difference" without a formula; |n - n_A| / n_A is used.
    relative = abs(count - reference_count) / reference_count if reference_count else (0.0 if not count else math.inf)
    differing = int((a ^ b).sum())
    return {"dice": dice, "one_minus_dice": 1.0 - dice, "relative_voxel_difference": relative,
            "voxels": count, "reference_voxels": reference_count, "differing_voxels": differing,
            "equal": differing == 0}


def a4_from_reference(template_mask: np.ndarray, reference_record: dict, template: Path) -> dict:
    """A4 on run A's transform: inverse (float32), FillHoles rule, uint8 mask (shared.pnl_mask_native)."""
    import SimpleITK as sitk

    repository_on_path()
    from shared import pnl_mask_native as native

    transform = sitk.ReadTransform(str(reference_record["transform"]))
    fixed = sitk.ReadImage(str(template), sitk.sitkFloat64)
    moving = sitk.ReadImage(str(Path(reference_record["work"]) / "baseline.nii.gz"), sitk.sitkFloat64)
    inverse = native.inverse_mask(template_mask, fixed, moving, transform)
    filled = native.fill_holes(inverse)
    return {"inverse": inverse, "filled": filled, "mask": native.finish_mask(filled)}


def t1_case(reference: dict, template: Path) -> dict:
    """M-A 2: run A's transform and template-grid mask through A4; the result must equal run A's mask."""
    import nibabel as nib

    work = Path(reference["work"])
    template_mask = nib.load(str(work / "template_mask.nii.gz")).get_fdata()
    produced = a4_from_reference(template_mask, reference, template)
    expected = load_mask(Path(reference["mask"]))
    result = compare(produced["mask"], expected)
    # M-A 2: the differing voxels are reported ([i, j, k] on the native grid).
    result["differing_indices"] = np.argwhere((produced["mask"] != 0) ^ (expected != 0)).tolist()
    result["inverse_max_abs_difference"] = float(np.max(np.abs(
        produced["inverse"] - nib.load(str(work / "inverse.nii.gz")).get_fdata())))
    result["filled_differing_voxels"] = int(np.count_nonzero(
        produced["filled"] != nib.load(str(work / "filled.nii.gz")).get_fdata()))
    result["pass"] = result["equal"]
    return result


def affine_matrix(transform) -> np.ndarray:
    """4x4 matrix of an affine (or composite of affine) SimpleITK transform, physical LPS points."""
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    matrix = np.eye(4)
    for axis in range(3):
        unit = [0.0, 0.0, 0.0]
        unit[axis] = 1.0
        matrix[:3, axis] = np.asarray(transform.TransformPoint(tuple(unit))) - origin
    matrix[:3, 3] = origin
    return matrix


def t2_case(reference: dict, native_record: dict, template: Path) -> dict:
    """M-A 3 diagnostics (no pass rule): A3 vs run A transform and warped b0."""
    import nibabel as nib
    import SimpleITK as sitk

    native_matrix = affine_matrix(sitk.ReadTransform(str(Path(native_record["keep_dir"]) / "transform.tfm")))
    reference_matrix = affine_matrix(sitk.ReadTransform(str(reference["transform"])))
    fixed = sitk.ReadImage(str(template), sitk.sitkFloat64)
    template_mask = nib.load(str(Path(reference["work"]) / "template_mask.nii.gz")).get_fdata() > 0
    indices = np.argwhere(template_mask)
    low, high = indices.min(axis=0), indices.max(axis=0)
    corners = [np.asarray(fixed.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in corner)))
               for corner in np.array(np.meshgrid(*zip(low, high), indexing="ij")).reshape(3, -1).T]
    centre = np.asarray(fixed.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in (low + high) / 2)))

    def displacement(point):
        point = np.r_[point, 1.0]
        return float(np.linalg.norm((native_matrix @ point)[:3] - (reference_matrix @ point)[:3]))

    relative = np.linalg.inv(reference_matrix) @ native_matrix
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    warped_native = nib.load(str(Path(native_record["keep_dir"]) / "Warped.nii.gz")).get_fdata().ravel()
    warped_reference = nib.load(str(reference["warped"])).get_fdata().ravel()
    return {
        "rotation_degrees": math.degrees(math.acos(float(cosine))),
        # SPEC-GAP: M-A 3 does not name the point for "translation"; the brain bounding-box centre is used.
        "translation_mm": displacement(centre),
        # SPEC-GAP: the bounding box is run A's template-grid CNN mask (template space, voxel centres).
        "max_corner_displacement_mm": max(displacement(corner) for corner in corners),
        # SPEC-GAP: NCC over all template-grid voxels.
        "warped_ncc": float(np.corrcoef(warped_native, warped_reference)[0, 1]),
    }


def normalized_input(reference: dict) -> np.ndarray:
    """T4 input: run A's Warped image through the vendored normalisation, once for both torch arms."""
    import nibabel as nib

    repository_on_path()
    from shared.pnl_cnn.inference import normalize

    return normalize(nib.load(str(reference["warped"])).get_fdata(), PERCENTILE)


def t4_case(folder: Path, reference: dict, args) -> dict:
    """M-A 5: the same normalised input through two torch versions; both masks through A4 with run A's transform."""
    t4 = folder / "t4"
    t4.mkdir(parents=True, exist_ok=True)
    source = t4 / "input.npy"
    if not source.is_file():
        np.save(source, normalized_input(reference))
    # The driver runs in Slicer's Python (Slicer torch arm); --torch-python is the torch 2.2.2 + numpy 1.26 venv.
    arms = {"slicer": sys.executable, "venv": args.torch_python}
    outputs = {}
    for arm, python in arms.items():
        mask_path, record_path = t4 / f"{arm}_template_mask.npy", t4 / f"{arm}.json"
        record = read_json(record_path)
        if not (record and record.get("status") == "completed" and mask_path.is_file()):
            # SPEC-GAP: M-A 5 names CPU but no thread count; both arms use 1 thread (production count).
            command = [python, Path(__file__).resolve(), "--inference-worker", "--input", source,
                       "--model-dir", args.model_dir, "--output", mask_path, "--record", record_path,
                       "--threads", 1]
            if arm == "slicer":
                env = dict(os.environ, PYTHONUNBUFFERED="1")
            else:
                from shared.external_process import python_tool_environment
                env = python_tool_environment(python, {"PYTHONUNBUFFERED": "1"})
            process = run_child(command, env, t4 / f"{arm}.log")
            record = read_json(record_path) or {"status": "failed"}
            record["process"] = process
            write_json(record_path, record)
            if process["returncode"]:
                return {"status": "failed", "arm": arm, "record": record}
        outputs[arm] = {"record": record, "template_mask": np.load(mask_path)}
    template_difference = int(np.count_nonzero(outputs["slicer"]["template_mask"] != outputs["venv"]["template_mask"]))
    finals = {arm: a4_from_reference(item["template_mask"], reference, args.template)["mask"]
              for arm, item in outputs.items()}
    # SPEC-GAP: M-A 5 does not name the reference of the pair; the Slicer torch mask is the denominator.
    result = compare(finals["venv"], finals["slicer"])
    result.update(status="completed", template_grid_differing_voxels=template_difference,
                  torch_versions={arm: item["record"].get("torch_version") for arm, item in outputs.items()},
                  numpy_versions={arm: item["record"].get("numpy_version") for arm, item in outputs.items()})
    return result


# --------------------------------------------------------------------------- evaluation and summary
def evaluate(cases: dict) -> dict:
    """M-A 1 noise floor, T1, T3 (pass bounds from the perturbations) and T4 verdicts."""
    perturbation_values = {"one_minus_dice": [], "relative_voxel_difference": []}
    all_equal, missing = True, []
    for case_id, case in cases.items():
        for run in PERTURBATIONS:
            item = case.get("reference_vs_A", {}).get(run)
            if item is None:
                missing.append(f"{case_id}:{run}")
                continue
            all_equal &= item["equal"]
            # A' joins the perturbations only when it differs from A.
            if run == "A_repeat" and item["equal"]:
                continue
            for key in perturbation_values:
                perturbation_values[key].append(item[key])
    bounds = {key: max(values) if values else None for key, values in perturbation_values.items()}
    noise_zero = all_equal and not missing

    def verdict(name):
        values = [(case_id, case.get(name)) for case_id, case in cases.items()]
        absent = [case_id for case_id, value in values if not value or value.get("status") == "failed"]
        if noise_zero:
            return {"status": "stop", "reason": "every perturbation equals run A on every case; no bound (M-A 1)"}
        if absent or missing or bounds["one_minus_dice"] is None:
            return {"status": "incomplete", "missing": absent + missing}
        failing = [case_id for case_id, value in values
                   if value["one_minus_dice"] > bounds["one_minus_dice"]
                   or value["relative_voxel_difference"] > bounds["relative_voxel_difference"]]
        return {"status": "fail" if failing else "pass", "failing_cases": failing}

    t1 = [case.get("t1") for case in cases.values()]
    repeat_differs = [case_id for case_id, case in cases.items()
                      if case.get("reference_vs_A", {}).get("A_repeat", {}).get("equal") is False]
    invalid = [f"{case_id}:{run}" for case_id, case in cases.items()
               for run, record in case.get("reference", {}).items()
               if record.get("status") != "completed" or not record.get("valid")]
    return {"perturbation_bounds": bounds, "noise_floor_zero": noise_zero, "missing_reference_runs": missing,
            "a_repeat_differs": repeat_differs, "invalid_reference_runs": invalid,
            "t1": {"status": "incomplete" if any(item is None for item in t1) else
                   ("pass" if all(item["pass"] for item in t1) else "fail"),
                   "failing_cases": [case_id for case_id, case in cases.items()
                                     if case.get("t1") and not case["t1"]["pass"]]},
            "t3": verdict("t3"), "t4": verdict("t4")}


TABLE_COLUMNS = ("case", "shape", "baseline_sha256", "qform_sform_flag", "A_valid",
                 "A_repeat_dice", "B1_dice", "B2_dice", "S2_dice", "t1_equal", "t2_rotation_deg",
                 "t2_translation_mm", "t2_corner_mm", "t2_ncc", "t3_dice", "t3_rel_diff", "N8_dice",
                 "t4_dice", "t4_rel_diff", "t4_template_diff", "N1_seconds", "N1_max_rss_mb")


def table_row(case_id: str, case: dict) -> list:
    def get(*keys):
        value = case
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, float):
            return f"{value:.6g}"
        return "" if value is None else str(value)

    return [case_id, "x".join(str(v) for v in case.get("check", {}).get("shape", [])[:3]),
            get("baseline", "sha256")[:12], get("check", "qform_sform_flag"), get("reference", "A", "valid"),
            *(get("reference_vs_A", run, "dice") for run in PERTURBATIONS), get("t1", "equal"),
            get("t2", "rotation_degrees"), get("t2", "translation_mm"), get("t2", "max_corner_displacement_mm"),
            get("t2", "warped_ncc"), get("t3", "dice"), get("t3", "relative_voxel_difference"), get("n8", "dice"),
            get("t4", "dice"), get("t4", "relative_voxel_difference"), get("t4", "template_grid_differing_voxels"),
            get("native", "N1", "process", "seconds"), get("native", "N1", "process", "max_rss_mb")]


def variant_name(stem: str, variant: str, suffix: str) -> str:
    return f"{stem}{suffix}" if variant == "default" else f"{stem}-{variant}{suffix}"


def summarise(out: Path, results: dict, args) -> dict:
    summary = {"design": "T2 portable routes v4, section 3 M-A", "variant": args.variant,
               "rotation_degrees": ROTATION_DEGREES, "cases": sorted(results),
               "evaluation": evaluate(results)}
    write_json(out / variant_name("summary", args.variant, ".json"), summary)
    lines = ["\t".join(TABLE_COLUMNS)] + ["\t".join(table_row(case_id, results[case_id])) for case_id in sorted(results)]
    (out / variant_name("summary", args.variant, ".tsv")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def t6_cases(results: dict) -> list[str]:
    """M-A 6: smallest, median and largest baseline grid."""
    ordered = sorted(results, key=lambda case_id: (int(np.prod(results[case_id]["check"]["shape"][:3])), case_id))
    return list(dict.fromkeys([ordered[0], ordered[len(ordered) // 2], ordered[-1]])) if ordered else []


# --------------------------------------------------------------------------- driver
def drive(args) -> int:
    repository_on_path()
    out = Path(args.out).absolute()
    if any(char.isspace() for char in str(out)):
        raise SystemExit("--out must not contain whitespace (the reference ANTs work folders live under it)")
    args.template = Path(args.template or Path(args.model_dir) / TEMPLATE_NAME).absolute()
    steps = set(args.steps.split(",")) if args.steps else set(STEPS)
    unknown = steps - set(STEPS)
    if unknown:
        raise SystemExit("Unknown steps: " + ", ".join(sorted(unknown)))
    if "reference" in steps and not args.pnl_python:
        raise SystemExit("--pnl-python is required for the reference step")
    if "t4" in steps and not args.torch_python:
        raise SystemExit("--torch-python is required for the t4 step")
    cases = load_cases(args.cases)
    checks = {case["id"]: check_case(case) for case in cases}
    mismatched = {case_id: check for case_id, check in checks.items() if not check["counts_match"]}
    write_json(out / "precheck.json", checks)
    if mismatched:
        for case_id, check in mismatched.items():
            print(f"[mask parity] {case_id}: {check['volumes']} volumes, {check['bvals']} b-values, "
                  f"{check['bvecs']} b-vectors", file=sys.stderr)
        raise SystemExit("Pre-check failed: volume and b-value counts differ; no run started")
    results = {}
    for case in cases:
        folder = out / "cases" / case["id"]
        print(f"[mask parity] case {case['id']}", flush=True)
        # Steps not selected in this call keep their earlier results.
        earlier = read_json(folder / variant_name("result", args.variant, ".json")) or {}
        result = {key: earlier[key] for key in ("t1", "t2", "t4", "t6") if key in earlier}
        result.update(check=checks[case["id"]], baseline=prepare_case(case, folder), reference={}, native={})
        if "reference" in steps:
            for name in REFERENCE_RUNS:
                result["reference"][name] = reference_run(folder, name, args, result["baseline"])
        else:
            for name in REFERENCE_RUNS:
                record = read_json(folder / "reference" / name / "reference.json")
                if record:
                    result["reference"][name] = record
        if "native" in steps:
            for name, threads in NATIVE_RUNS.items():
                result["native"][name] = native_run(folder, name, threads, args, result["baseline"])
        else:
            for name in NATIVE_RUNS:
                record = read_json(folder / "native" / variant_name(name, args.variant, "") / "native.json")
                if record:
                    result["native"][name] = record
        reference_a = result["reference"].get("A")
        a_ok = bool(reference_a and reference_a.get("status") == "completed" and reference_a.get("valid"))
        if a_ok:
            mask_a = load_mask(Path(reference_a["mask"]))
            result["reference_vs_A"] = {
                name: dict(compare(load_mask(Path(record["mask"])), mask_a), valid=record.get("valid"))
                for name, record in result["reference"].items()
                if name != "A" and record.get("status") == "completed" and record.get("valid")}
            for name, key in (("N1", "t3"), ("N8", "n8")):
                record = result["native"].get(name)
                if record and record.get("status") == "completed":
                    result[key] = compare(load_mask(Path(record["mask"])), mask_a)
            if "t1" in steps:
                result["t1"] = t1_case(reference_a, args.template)
            native_one = result["native"].get("N1")
            if "t2" in steps and native_one and native_one.get("status") == "completed":
                result["t2"] = t2_case(reference_a, native_one, args.template)
            if "t4" in steps:
                result["t4"] = t4_case(folder, reference_a, args)
        write_json(folder / variant_name("result", args.variant, ".json"), result)
        results[case["id"]] = result
    if "t6" in steps:
        # SPEC-GAP: M-A 6 names no route or memory metric; the native route is timed, memory is the worker's
        # host peak RSS (os.wait4); CUDA device memory is not recorded.
        for case_id in t6_cases(results):
            folder = out / "cases" / case_id
            baseline = results[case_id]["baseline"]
            timing = {"cpu4": native_run(folder, "cpu4", 4, args, baseline, group="t6")}
            if args.gpu:
                timing["gpu"] = native_run(folder, "gpu", 4, args, baseline, device="cuda", group="t6")
            results[case_id]["t6"] = {name: {"status": record.get("status"), "process": record.get("process"),
                                             "phase_seconds": record.get("phase_seconds"),
                                             "device": record.get("device")} for name, record in timing.items()}
            write_json(folder / variant_name("result", args.variant, ".json"), results[case_id])
    summary = summarise(out, results, args)
    print(json.dumps(summary["evaluation"], indent=2, default=_json_default))
    return 0


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference-worker", action="store_true")
    mode.add_argument("--native-worker", action="store_true")
    mode.add_argument("--inference-worker", action="store_true")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--pnl-python", type=Path, help="managed pnl-cnn-torch Python (reference arm)")
    parser.add_argument("--ants-bin", type=Path, help="ANTs 2.3.0 bin folder for the reference arm")
    parser.add_argument("--torch-python", type=Path, help="T4 second torch Python (torch 2.2.2, numpy 1.26)")
    parser.add_argument("--steps", help="comma list of " + ",".join(STEPS) + " (default: all)")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="default",
                        help="A3 change for the M-A T3 failure clause")
    parser.add_argument("--gpu", action="store_true", help="T6: add one CUDA run per timing case")
    # Worker arguments.
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-dir", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    worker = args.reference_worker or args.native_worker or args.inference_worker
    if not worker and not (args.cases and args.out and args.model_dir):
        parser.error("the driver needs --cases, --out and --model-dir")
    return args


def main(argv=None) -> int:
    args = parse_arguments(argv)
    if args.reference_worker:
        return reference_worker(args)
    if args.native_worker:
        return native_worker(args)
    if args.inference_worker:
        return inference_worker(args)
    return drive(args)


if __name__ == "__main__":
    raise SystemExit(main())
