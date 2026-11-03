"""Invoke the installed official UKFTractography CLI, never a reimplementation."""

from dataclasses import asdict, dataclass, fields, replace
from shared.cohort_schedule import scheduled_stage
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


REFERENCE = "https://github.com/pnlbwh/ukftractography/blob/2d2b661539700aa825c1949823119601e11e1823/UKFTractography/UKFTractography.xml"


@dataclass(frozen=True)
class UKFParameters:
    # Explicit author-module defaults, not a subject-tuned ORG protocol.
    numTensor: int = 2
    seedsPerVoxel: int = 1
    seedingThreshold: float = 0.18
    stoppingFA: float = 0.15
    stoppingThreshold: float = 0.1
    stepLength: float = 0.3
    recordLength: float = 0.9
    maxHalfFiberLength: float = 250.0
    numThreads: int = -1
    freeWater: bool = False
    recordFA: bool = True
    recordTensors: bool = True
    Qm: float = 0.0
    Ql: float = 0.0
    Qw: float = 0.0
    Rs: float = 0.0


def parameters(values=None):
    if values is not None and not isinstance(values, dict):
        raise ValueError("UKF parameters must be a JSON object")
    unknown = set(values or {}) - {field.name for field in fields(UKFParameters)}
    if unknown:
        raise ValueError("Unknown UKF parameters: " + ", ".join(sorted(unknown)))
    result = UKFParameters(**(values or {}))
    for key, value in asdict(result).items():
        if key in ("freeWater", "recordFA", "recordTensors"):
            if not isinstance(value, bool):
                raise ValueError(key + " must be boolean")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be finite numeric")
    if result.numTensor not in (1, 2) or type(result.numTensor) is not int:
        raise ValueError("UKF numTensor must be integer 1 or 2")
    if type(result.seedsPerVoxel) is not int or not 1 <= result.seedsPerVoxel <= 50:
        raise ValueError("UKF seedsPerVoxel must be integer in [1,50]")
    if type(result.numThreads) is not int or result.numThreads != -1 and result.numThreads < 1:
        raise ValueError("UKF numThreads must be -1 (author automatic) or positive integer")
    if not all(0 <= getattr(result, key) <= 1 for key in ("seedingThreshold", "stoppingFA", "stoppingThreshold")):
        raise ValueError("UKF thresholds must be in [0,1]")
    if (
        not 0.1 <= result.stepLength <= 1
        or not 0.1 <= result.recordLength <= 4
        or not 1 <= result.maxHalfFiberLength <= 500
    ):
        raise ValueError("UKF step/output lengths exceed documented author limits")
    for key, maximum in (("Qm", 0.25), ("Ql", 1000), ("Qw", 0.25), ("Rs", 0.25)):
        if not 0 <= getattr(result, key) <= maximum:
            raise ValueError(f"UKF {key} must be in [0,{maximum}]; zero selects author automatic noise")
    if not result.recordFA or not result.recordTensors:
        raise ValueError("UKF output must retain the author-default FA and tensor arrays")
    if os.environ.get('SLICERDSPACE_COHORT_SCHEDULE'):
        budget = int(os.environ['SLICERDSPACE_STAGE_CPUS'])
        if result.numThreads == -1:
            result = replace(result, numThreads=budget)
        elif result.numThreads > budget:
            raise ValueError('UKF thread request exceeds the admitted cohort CPU reservation')
    return result


def resolve_executable(explicit=None, name="UKFTractography"):
    candidate = explicit
    if not candidate and name == "UKFTractography":
        candidate = os.environ.get("SLICERDSPACE_UKF_EXECUTABLE")
    if not candidate and name == "DWIConvert":
        candidate = os.environ.get("SLICERDSPACE_DWICONVERT_EXECUTABLE")
    if candidate:
        path = Path(candidate).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(str(path))
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    try:
        from shared.pipeline import resolve_cli_executable

        return Path(resolve_cli_executable(name.lower())).resolve()
    except (RuntimeError, AttributeError, ImportError, FileNotFoundError) as exc:
        # Background PythonSlicer does not instantiate the GUI module registry.
        roots = [Path(value) for key in ("SLICER_HOME", "Slicer_HOME") if (value := os.environ.get(key))]
        roots.append(Path(sys.executable).resolve().parent.parent)
        for root in roots:
            for pattern in ("lib/Slicer-*/cli-modules/*", "slicer.org/Extensions-*/*/lib/Slicer-*/cli-modules/*"):
                for path in root.glob(pattern):
                    if path.is_file() and path.name.lower() in (name.lower(), name.lower() + ".exe"):
                        return path.resolve()
        raise RuntimeError(
            "Install UKFTractography and SlicerDMRI through Slicer Extension Manager, restart Slicer, "
            "and select UKF again. Headless users may pass --ukf-executable. Missing CLI: " + name
        ) from exc


def paths(subject, directory):
    directory = Path(directory)
    return dict(
        tractogram=directory / (subject + "_ukf.vtk"),
        tractogram_trk=directory / (subject + "_ukf.trk"),
        qc=directory / "ukf_qc.json",
        manifest=directory / "ukf_manifest.json",
        schema=directory / "ukf_cli.xml",
    )


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def executable_identity(executable):
    executable = Path(executable).resolve()
    libraries = {}
    roots = {executable.parent, *list(executable.parents)[:3]}
    for root in roots:
        for pattern in ("*UKF*", "*DWIConvert*"):
            for path in root.glob(pattern):
                name = path.name.lower()
                if path.is_file() and (name.endswith((".dll", ".dylib", ".so")) or ".so." in name):
                    resolved = str(path.resolve())
                    if resolved not in libraries:
                        libraries[resolved] = digest(path)
    return dict(path=str(executable), sha256=digest(executable), companion_libraries=libraries)


def signature(raw, options, executable, converter=None):
    converter = converter or resolve_executable(name="DWIConvert")
    return dict(
        version=4,
        subject_id=raw.subject_id,
        parameters=asdict(parameters(options)),
        inputs={
            key: dict(path=str(getattr(raw, key).resolve()), sha256=digest(getattr(raw, key)))
            for key in ("dwi_path", "bval_path", "bvec_path", "mask_path")
        },
        executable=executable_identity(executable),
        converter=executable_identity(converter),
        protocol="author-module-defaults-with-explicit-overrides",
        shell_policy="all supplied volumes; no implicit shell filtering",
        conversion_policy="official-gradient-geometry-probe-with-lossless-native-payload-v1",
    )


def current(raw, directory, options=None, executable=None, converter=None):
    try:
        products = paths(raw.subject_id, directory)
        record = json.loads(products["manifest"].read_text())
        if record["status"] != "completed" or record["signature"] != signature(
            raw, options, resolve_executable(executable), resolve_executable(converter, name="DWIConvert")
        ):
            return False
        qc = json.loads(products["qc"].read_text())
        return qc.get("automatic_qc_passed") is True and all(
            path.is_file() and record["artifacts"][key] == digest(path)
            for key, path in products.items()
            if key != "manifest"
        )
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError):
        return False


def validate_inputs(raw):
    import nibabel as nib
    import numpy as np

    image = nib.load(raw.dwi_path)
    mask_image = nib.load(raw.mask_path)
    b = np.loadtxt(raw.bval_path).reshape(-1)
    g = np.loadtxt(raw.bvec_path, ndmin=2)
    if g.shape == (3, len(b)):
        g = g.T
    if len(image.shape) != 4 or image.shape[3] != len(b) or g.shape != (len(b), 3):
        raise ValueError("UKF DWI and gradient dimensions disagree")
    if not np.isfinite(b).all() or np.any(b < 0) or not np.isfinite(g).all():
        raise ValueError("UKF gradients must be finite, b-values nonnegative")
    nonzero = b > 50
    if (
        not np.any(~nonzero)
        or np.count_nonzero(nonzero) < 6
        or np.any(abs(np.linalg.norm(g[nonzero], axis=1) - 1) > 0.01)
    ):
        raise ValueError("UKF needs b0, at least six diffusion measurements and unit b-vectors")
    directions = g[nonzero]
    design = np.column_stack(
        [
            directions[:, 0] ** 2,
            directions[:, 1] ** 2,
            directions[:, 2] ** 2,
            directions[:, 0] * directions[:, 1],
            directions[:, 0] * directions[:, 2],
            directions[:, 1] * directions[:, 2],
        ]
    )
    if np.linalg.matrix_rank(design) < 6:
        raise ValueError("UKF diffusion directions do not span a tensor design (rank < 6)")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid UKF spatial affine")
    if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("UKF mask and DWI must share the native grid")
    mask = np.asanyarray(mask_image.dataobj)
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all() or not mask.any():
        raise ValueError("UKF mask must be nonempty, finite and binary")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("UKF DWI contains NaN or Inf; repair the source data before fitting")
    baseline = data[..., ~nonzero].mean(axis=3)
    if np.any(baseline[mask > 0] <= 0):
        raise ValueError("UKF brain mask includes voxels without positive b0 signal")
    return dict(
        volumes=len(b),
        b0_volumes=int(np.count_nonzero(~nonzero)),
        brain_voxels=int(mask.sum()),
        approximate_shells=np.unique(np.round(b[nonzero] / 100) * 100).tolist(),
        input_coordinate_system="NIfTI RAS+, gradients converted by official DWIConvert",
    )


def output_qc(path, raw, selected=None):
    import nibabel as nib
    import numpy as np
    from connectome.pipeline import _iter_streamlines
    import vtk  # Require the full polyline reader, never an endpoint-only fallback.
    from shared.output_contracts import polydata_streamline_count
    from vtk.util.numpy_support import vtk_to_numpy

    count = polydata_streamline_count(path)
    if count is None or count < 1:
        raise ValueError("UKF output is empty or malformed")
    image = nib.load(raw.mask_path)
    inverse = np.linalg.inv(image.affine)
    mask = np.asanyarray(image.dataobj) > 0
    lengths = []
    inside = total = 0
    for line in _iter_streamlines(path):
        points = np.asarray(line)
        if not np.isfinite(points).all():
            raise ValueError("UKF output has nonfinite coordinates")
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
        voxels = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
        valid = ((voxels >= 0) & (voxels < mask.shape)).all(axis=1)
        inside += int(mask[tuple(voxels[valid].T)].sum())
        total += len(points)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    arrays_valid = True
    for index in range(1, (selected or parameters()).numTensor + 1):
        for name, components in ((f"tensor{index}", 9), (f"FA{index}", 1)):
            array = data.GetPointData().GetArray(name)
            valid_array = (
                array is not None
                and array.GetNumberOfTuples() == data.GetNumberOfPoints()
                and array.GetNumberOfComponents() == components
            )
            if valid_array:
                values = vtk_to_numpy(array)
                valid_array = bool(np.isfinite(values).all())
                if name.startswith("FA"):
                    valid_array = valid_array and bool(((values >= -1e-6) & (values <= 1 + 1e-6)).all())
            arrays_valid = arrays_valid and valid_array
    checks = dict(
        nonempty_geometry=count == len(lengths),
        positive_lengths=bool(np.all(np.asarray(lengths) > 0)),
        native_mask_overlap=inside / max(1, total) >= 0.9,
        requested_tensor_and_fa_arrays=arrays_valid,
    )
    return dict(
        status="completed",
        automatic_qc_passed=all(checks.values()),
        checks=checks,
        streamline_count=count,
        length_mm_percentiles=np.percentile(lengths, [0, 25, 50, 75, 100]).tolist(),
        points_inside_brain_fraction=inside / max(1, total),
        human_review="pending",
        scientific_scope="Geometry/native-mask sanity only; not anatomical accuracy or reliable ORG sampling",
    )


def build_command(binary, dwi, mask, target, selected, schema):
    """Use SEM switches explicitly: XML GUI defaults do not enable CLI switches."""
    nodes = {node.findtext("name"): node for node in schema.findall("./parameters/*") if node.find("name") is not None}
    if not set(asdict(selected)).union(("dwiFile", "maskFile", "tracts")).issubset(nodes):
        raise RuntimeError("Installed UKF CLI lacks requested parameters; no fallback")
    command = [str(binary), "--dwiFile", str(dwi), "--maskFile", str(mask), "--tracts", str(target)]
    for key, value in asdict(selected).items():
        if isinstance(value, bool):
            default = nodes[key].findtext("default", "").strip().lower()
            expected = "false" if key == "freeWater" else "true"
            if default != expected:
                raise RuntimeError(f"Installed UKF boolean default differs for {key}; inspect its protocol")
            if value:
                command.append("--" + key)
        else:
            command.extend(("--" + key, str(value)))
    return command


def inspect_schema(binary, env):
    from shared.external_process import text_subprocess_kwargs

    description = subprocess.run(
        [str(binary), "--xml"],
        capture_output=True,
        timeout=30,
        env=env,
        **text_subprocess_kwargs(),
    )
    start = description.stdout.find("<?xml")
    if start < 0:
        start = description.stdout.find("<executable")
    end = description.stdout.find("</executable>", max(0, start))
    if start < 0 or end < 0:
        raise RuntimeError("UKF CLI schema inspection failed: " + description.stderr[-1000:])
    xml = description.stdout[start : end + len("</executable>")]
    # Some upstream builds return from PARSE_ARGS on --xml but then continue
    # into LoadFiles in ModuleEntryPoint. Accept only complete, parseable XML;
    # conversion and tracking themselves must still return zero.
    return ET.fromstring(xml), xml, description.returncode


def export_trk(source, reference, target):
    import nibabel as nib
    import numpy as np
    from nibabel.streamlines import Field
    from connectome.pipeline import _iter_streamlines

    image = nib.load(reference)
    header = nib.streamlines.TrkFile.create_empty_header()
    header[Field.VOXEL_TO_RASMM] = image.affine
    header[Field.VOXEL_SIZES] = image.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = image.shape[:3]
    header[Field.VOXEL_ORDER] = "RAS"
    tractogram = nib.streamlines.LazyTractogram(
        streamlines=lambda: _iter_streamlines(source), affine_to_rasmm=np.eye(4)
    )
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram, header), str(target))


def verify_trk(source, target):
    import itertools
    import nibabel as nib
    import numpy as np
    from connectome.pipeline import _iter_streamlines

    count = 0
    converted = nib.streamlines.load(target, lazy_load=True).streamlines
    for original, restored in itertools.zip_longest(_iter_streamlines(source), converted):
        if original is None or restored is None or np.shape(original) != np.shape(restored):
            raise ValueError("UKF VTK/TRK streamline count or point count changed")
        if not np.allclose(original, restored, atol=1e-4, rtol=0):
            raise ValueError("UKF VTK/TRK native RAS coordinates changed beyond 0.0001 mm")
        count += 1
    if count == 0:
        raise ValueError("Empty UKF TRK conversion")
    return count


def write_conversion_probe(raw, target):
    """Ask the official converter about geometry/gradients without quantizing DWI."""
    import nibabel as nib
    import numpy as np

    image = nib.load(raw.dwi_path)
    values = np.arange(8 * image.shape[3], dtype=np.int64)
    if values[-1] > np.iinfo(np.int16).max:
        raise ValueError("Too many volumes for the lossless UKF conversion probe")
    probe = values.astype(np.int16).reshape((2, 2, 2, image.shape[3]))
    header = image.header.copy()
    header.set_data_dtype(np.int16)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(probe, image.affine, header), target)
    return probe


def restore_lossless_payload(raw, probe, converted, dwi, mask):
    """Validate a pure axis permutation/flip, then copy native signals bit-for-bit.

    DWIConvert supplies ALL gradient metadata and coordinate conventions. Its
    short-only image IO is used only for a tiny integer probe. The actual DWI
    and mask share the same verified mapping and never pass through a short cast.
    """
    import nibabel as nib
    import nrrd
    import numpy as np

    converted_probe, header = nrrd.read(str(converted))
    directions = np.asarray(header["space directions"], dtype=float)
    spatial = np.flatnonzero(np.isfinite(directions).all(axis=1))
    channels = np.flatnonzero(~np.isfinite(directions).all(axis=1))
    if len(spatial) != 3 or len(channels) != 1:
        raise ValueError("DWIConvert did not return three spatial axes plus one volume axis")
    flips = {
        "left-posterior-superior": [-1, -1, 1],
        "right-anterior-superior": [1, 1, 1],
        "left-anterior-superior": [-1, 1, 1],
    }
    if header.get("space") not in flips:
        raise ValueError("Unsupported DWIConvert coordinate frame")
    to_ras = np.diag(flips[header["space"]])
    affine = np.eye(4)
    affine[:3, :3] = to_ras @ directions[spatial].T
    affine[:3, 3] = to_ras @ header["space origin"]
    image = nib.load(raw.dwi_path)
    mapping = np.linalg.inv(image.affine) @ affine
    signed = np.rint(mapping[:3, :3]).astype(int)
    if (
        not np.allclose(mapping[:3, :3], signed, atol=1e-5, rtol=0)
        or not np.array_equal(abs(signed).sum(axis=0), np.ones(3))
        or not np.array_equal(abs(signed).sum(axis=1), np.ones(3))
    ):
        raise ValueError("DWIConvert requires resampling, not a lossless axis transformation")
    order = tuple(np.argmax(abs(signed), axis=0).tolist())
    negative = [bool(signed[old, new] < 0) for new, old in enumerate(order)]
    probe_shift = np.zeros(3)
    for new, old in enumerate(order):
        probe_shift[old] = int(negative[new])
    if not np.allclose(mapping[:3, 3], probe_shift, atol=1e-4, rtol=0):
        raise ValueError("DWIConvert origin does not match a lossless probe transformation")

    def reorder(array):
        axes = order + tuple(range(3, array.ndim))
        transformed = np.transpose(array, axes)
        for axis, reverse in enumerate(negative):
            if reverse:
                transformed = np.flip(transformed, axis)
        return transformed

    channel = int(channels[0])
    if not np.array_equal(np.moveaxis(converted_probe, channel, -1), reorder(probe)):
        raise ValueError("DWIConvert changed probe intensities or volume order")
    gradients = [value for key, value in header.items() if key.startswith("DWMRI_gradient_")]
    if len(gradients) != image.shape[3] or not all(
        np.isfinite(np.fromstring(value, sep=" ")).all() and len(np.fromstring(value, sep=" ")) == 3
        for value in gradients
    ):
        raise ValueError("Incomplete DWIConvert gradient metadata")
    full_shift = np.zeros(3)
    for new, old in enumerate(order):
        full_shift[old] = image.shape[old] - 1 if negative[new] else 0
    origin_ras = image.affine[:3, :3] @ full_shift + image.affine[:3, 3]
    header["space origin"] = to_ras @ origin_ras
    for key in ("data file", "sizes", "type", "endian"):
        header.pop(key, None)
    header["encoding"] = "gzip"
    data = np.asanyarray(image.dataobj)
    nrrd.write(str(dwi), np.moveaxis(reorder(data), -1, channel), header)
    mask_header = {
        "space": header["space"],
        "space directions": directions[spatial],
        "space origin": header["space origin"],
        "kinds": ["domain"] * 3,
        "encoding": "gzip",
    }
    nrrd.write(str(mask), reorder(np.asanyarray(nib.load(raw.mask_path).dataobj)).astype(np.uint8), mask_header)
    return dict(
        policy="lossless-native-payload",
        probe_voxels=8,
        gradient_count=len(gradients),
        native_dtype=str(data.dtype),
        axis_order=list(order),
        reversed_axes=negative,
        output_origin_ras=origin_ras.tolist(),
        resampled=False,
        allow_lossy_conversion=False,
    )


@scheduled_stage('ukf')
def run_ukf(raw, directory, *, options=None, executable=None, converter=None, force=False, log=None):
    from shared.external_process import merged_startup_environment
    from shared.pipeline import build_cli_environment

    selected = parameters(options)
    binary = resolve_executable(executable)
    converter = resolve_executable(converter, name="DWIConvert")
    products = paths(raw.subject_id, Path(directory).expanduser().resolve())
    products["manifest"].parent.mkdir(parents=True, exist_ok=True)
    if not force and current(raw, directory, options, binary, converter):
        return dict(status="cached", outputs=products)
    started = time.monotonic()
    env = merged_startup_environment(build_cli_environment(str(binary)))
    schema, xml, probe_code = inspect_schema(binary, env)
    build_command(binary, "DWI.nrrd", "Mask.nrrd", "tracts.vtk", selected, schema)
    inputs = validate_inputs(raw)
    before = signature(raw, options, binary, converter)
    record = dict(
        status="running",
        signature=before,
        inputs=inputs,
        reference=REFERENCE,
        schema_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        schema_probe_exit_code=probe_code,
        commands=[],
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    def save():
        temporary = products["manifest"].with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(products["manifest"])

    save()
    products["schema"].write_text(xml, encoding="utf-8")
    try:
        converter_env = merged_startup_environment(build_cli_environment(str(converter)))
        with tempfile.TemporaryDirectory(prefix=".ukf-", dir=products["manifest"].parent) as temporary:
            work = Path(temporary)
            dwi, mask, tracts = work / "DWI.nrrd", work / "Mask.nrrd", work / "tracts.vtk"
            probe_path, converted_probe = work / "geometry_probe.nii.gz", work / "geometry_probe.nrrd"
            probe = write_conversion_probe(raw, probe_path)
            conversion = [
                str(converter),
                "--conversionMode",
                "FSLToNrrd",
                "--inputVolume",
                str(probe_path),
                "--inputBValues",
                str(raw.bval_path),
                "--inputBVectors",
                str(raw.bvec_path),
                "--outputVolume",
                str(converted_probe),
            ]
            command = build_command(binary, dwi, mask, tracts, selected, schema)
            record["commands"] = [conversion, command]
            save()
            for index, cmd in enumerate((conversion, command)):
                log_path = products["manifest"].parent / ("ukf_" + str(index) + ".log")
                if log:
                    log(
                        "[UKF] "
                        + ("Converting gradients with DWIConvert" if index == 0 else "Running official UKFTractography")
                    )
                with log_path.open("wb") as stream:
                    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                               env=converter_env if index == 0 else env)
                    try:
                        if process.wait():
                            raise RuntimeError("UKF stage failed; see " + str(log_path))
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                if index == 0:
                    record["conversion"] = restore_lossless_payload(raw, probe, converted_probe, dwi, mask)
                    save()
            qc = output_qc(tracts, raw, selected)
            if not qc["automatic_qc_passed"]:
                products["qc"].write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
                failed = work.with_name(work.name + "-failed.vtk")
                shutil.copyfile(tracts, failed)
                record["failed_tractogram"] = str(failed)
                raise RuntimeError("UKF geometry/native-mask QC failed; see " + str(failed))
            export_trk(tracts, raw.dwi_path, work / "tracts.trk")
            qc["vtk_trk_verified_streamlines"] = verify_trk(tracts, work / "tracts.trk")
            if signature(raw, options, binary, converter) != before:
                raise RuntimeError("UKF inputs or executable changed during execution")
            shutil.move(tracts, products["tractogram"])
            shutil.move(work / "tracts.trk", products["tractogram_trk"])
        products["qc"].write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")
        record.update(
            status="completed",
            artifacts={key: digest(path) for key, path in products.items() if key != "manifest"},
            runtime_seconds=time.monotonic() - started,
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        save()
        return dict(status="completed", outputs=products)
    except BaseException as exc:
        record.update(
            status="cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error=str(exc),
            runtime_seconds=time.monotonic() - started,
        )
        save()
        raise
