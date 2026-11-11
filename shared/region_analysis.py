"""Optional, manifest-scoped regional postprocessing. Never refits scientific models."""

import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import re
from pathlib import Path
import shutil
import tempfile
import time

import nibabel as nib
import numpy as np

from shared.result_scene import record_qc_state
from shared.region_selection import resolve_regions


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, record):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_path(value):
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError("Missing absolute input file: " + str(value))
    return path.resolve()


def signatures_current(record):
    """Honor stored input/output signatures without guessing files from directories."""
    if isinstance(record, dict):
        if isinstance(record.get("path"), str) and Path(record["path"]).is_absolute():
            path = Path(record["path"])
            if any(key in record for key in ("size", "size_bytes", "mtime_ns", "sha256")):
                if not path.is_file():
                    return False
                stat = path.stat()
                if "size" in record and record["size"] != stat.st_size:
                    return False
                if "size_bytes" in record and record["size_bytes"] != stat.st_size:
                    return False
                if "mtime_ns" in record and record["mtime_ns"] != stat.st_mtime_ns:
                    return False
                if "sha256" in record and record["sha256"] != sha256(path):
                    return False
        # Some stage manifests store an absolute-path -> digest mapping instead.
        for key in ("source_sha256", "artifact_sha256"):
            for path, digest in (record.get(key) or {}).items():
                if not Path(path).is_file() or sha256(path) != digest:
                    return False
        return all(signatures_current(value) for value in record.values())
    if isinstance(record, list):
        return all(signatures_current(value) for value in record)
    return True


def usable_task(task):
    if task.get("status") not in {"completed", "reused", "skipped"}:
        raise ValueError("Task did not complete")
    raw_qc = task.get("qc_path") or (task.get("outputs") or {}).get("qc")
    if not raw_qc and task.get("name") == "microstructure.ifw":
        raw_qc = task.get("manifest_path")
    qc_path = file_path(raw_qc)
    qc = read_json(qc_path)
    if not isinstance(qc, dict):
        raise ValueError("Malformed task QC record")
    if record_qc_state(qc) not in {"completed", "passed", "pass", "success"}:
        raise ValueError("Task QC failed or is unavailable")
    if not signatures_current(qc):
        raise ValueError("QC source/output signatures are stale")
    stage_path = task.get("manifest_path")
    stage = read_json(file_path(stage_path)) if stage_path else {}
    if not isinstance(stage, dict):
        raise ValueError("Malformed stage manifest")
    if stage.get("status") in {
        "failed",
        "partial_failed",
        "error",
        "fail",
        "running",
        "planned",
        "cancelled",
        "blocked",
    }:
        raise ValueError("Stage manifest is not completed")
    if not signatures_current(stage):
        raise ValueError("Stage source/output signatures are stale")
    for key, digest in (stage.get("artifacts") or {}).items():
        source = (task.get("outputs") or {}).get(key)
        if source and isinstance(digest, str) and len(digest) == 64:
            if sha256(file_path(source)) != digest:
                raise ValueError("Stage output digest is stale: " + key)
    for key, digest in (stage.get("output_sha256") or {}).items():
        source = (task.get("outputs") or {}).get(key) or (stage.get("outputs") or {}).get(key)
        if source and isinstance(digest, str) and len(digest) == 64:
            if sha256(file_path(source)) != digest:
                raise ValueError("Stage output digest is stale: " + key)
    return qc, stage


def atlas_data(path):
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Atlas must be finite, nonnegative and 3D")
    if np.any(abs(values - np.rint(values)) > 1e-3) or np.max(values) > np.iinfo(np.int32).max:
        raise ValueError("Atlas must contain discrete nonnegative int32 labels")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("Atlas has an invalid spatial affine")
    return image, np.rint(values).astype(np.int32)


def _metric_specs(task, outputs, warnings):
    name = task["name"]
    candidates = []
    if name == "microstructure.dti":
        candidates = [(key, key, None) for key in ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")]
    elif name == "microstructure.freewater":
        mask = outputs.get("valid_mask") or outputs.get("fw_tissue_valid_mask")
        candidates = [
            ("fwf", "fwf", None),
            ("fw_tissue_fa", "fw_tissue_fa" if outputs.get("fw_tissue_fa") else "fw_fa", mask),
            ("fw_tissue_md", "fw_tissue_md" if outputs.get("fw_tissue_md") else "fw_md", mask),
        ]
        if not mask:
            candidates = candidates[:1]
            warnings.append("FW tissue maps unavailable: explicit validity mask is required")
    elif name == "microstructure.noddi":
        mask = outputs.get("noddi_tissue_valid_mask") or outputs.get("valid_mask")
        candidates = [("noddi_fwf", "noddi_fwf" if outputs.get("noddi_fwf") else "isovf", None)]
        if mask:
            candidates += [
                (key, key if outputs.get(key) else key.removeprefix("noddi_"), mask)
                for key in ("noddi_ndi", "noddi_odi")
            ]
        else:
            warnings.append("NDI/ODI unavailable: explicit tissue validity mask is required")
    elif name in {"microstructure.dki", "microstructure.mkcurve"}:
        mask = outputs.get("valid_mask")
        if mask:
            candidates = [("dki_" + key, key + "_raw", mask) for key in ("mk", "ak", "rk", "kfa", "mkt", "rtk")]
        for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk"):
            field = "mkcurve_corrected_" + key
            validity = outputs.get(field + "_valid_mask") or outputs.get("mkcurve_corrected_valid_mask")
            if validity:
                candidates.append((field, field, validity))
    elif name == "microstructure.ifw":
        candidates = [
            ("ifw", "ifw", None),
            ("fw_volume_fraction", "fw_volume_fraction", None),
        ]
    result = {}
    for key, output_key, mask in candidates:
        if not outputs.get(output_key):
            continue
        try:
            result[key] = dict(
                path=str(file_path(outputs[output_key])),
                valid_mask=str(file_path(mask)) if mask else None,
                task=name,
                qc_path=str(file_path(task["qc_path"])),
            )
        except (OSError, ValueError) as exc:
            warnings.append(key + ": " + str(exc))
    return result


def inspect_run(manifest_path, *, atlas_override=None, lut_path=None):
    """Discover only explicitly recorded, successfully checked products of one run."""
    from microstructure.pipeline import _default_lut_path, _parse_label_names
    from shared.orchestrator import _resolve_analysis_inputs

    path = file_path(manifest_path)
    parent = read_json(path)
    if parent.get("kind") != "slicerddsurfer_product_orchestration":
        raise ValueError("Choose a single-subject orchestration manifest, not a batch or preview")
    if parent.get("status") in {"running", "planned"} or any(
        task.get("status") == "running" for task in parent.get("tasks", [])
    ):
        raise ValueError("Wait for this run to finish before regional analysis")
    warnings, metrics, tracts, connectomes, validation_files = [], {}, {}, {}, set()
    tissue, surface_task = None, None
    atlas = None
    metric_brain_masks = {}
    # An appended failed retry must supersede an earlier successful task, not leak old outputs.
    latest_tasks = {task["name"]: task for task in parent.get("tasks", [])}
    for task in latest_tasks.values():
        name = task.get("name", "")
        if not name.startswith(("parcellation.", "microstructure.", "tractography.", "connectome.", "segmentation.", "surface.")):
            continue
        try:
            qc, stage = usable_task(task)
            resolved_qc = task.get("qc_path") or (task.get("outputs") or {}).get("qc")
            if not resolved_qc and name == "microstructure.ifw":
                resolved_qc = task.get("manifest_path")
            task = dict(task, qc_path=str(file_path(resolved_qc)))
            outputs = task.get("outputs") or {}
            validation_files.add(task["qc_path"])
            if task.get("manifest_path"):
                validation_files.add(str(file_path(task["manifest_path"])))
            if name.startswith("parcellation.") and atlas is None:
                for key in ("wmparc", "wmparc_labels", "parcellation", "labels", "label_map", "aparc"):
                    if outputs.get(key):
                        atlas = file_path(outputs[key])
                        break
            if name == "segmentation.ddseg" and outputs.get("segmentation"):
                tissue = dict(path=str(file_path(outputs["segmentation"])), qc_path=task["qc_path"])
            if name == "surface.ddsurfer_postprocess":
                surface_task = task
            if name.startswith("microstructure."):
                available = _metric_specs(task, outputs, warnings)
                recorded_mask = (stage.get("raw_input_bundle") or qc.get("source_signature") or {}).get("mask")
                if isinstance(recorded_mask, dict):
                    recorded_mask = recorded_mask.get("path")
                if recorded_mask:
                    recorded_mask = str(file_path(recorded_mask))
                metric_brain_masks.update({key: recorded_mask for key in available})
                metrics.update(available)
            if name.startswith("tractography."):
                for key in ("tractogram", "tractogram_trk", "tractogram_vtk"):
                    if outputs.get(key):
                        source = file_path(outputs[key])
                        if "preview" in source.name.lower():
                            raise ValueError("Display previews cannot be quantitative inputs")
                        tracts[name] = dict(
                            path=str(source),
                            qc_path=str(file_path(task["qc_path"])),
                            weights=str(file_path(outputs["sift2_weights"])) if outputs.get("sift2_weights") else None,
                        )
                        break
            if name.startswith("connectome."):
                matrix = outputs.get("matrix_path") or outputs.get("matrix")
                nodes = outputs.get("nodes_path") or outputs.get("nodes")
                if matrix and nodes:
                    source_atlas = (stage.get("config") or {}).get("parcellation_path") or stage.get(
                        "parcellation_path"
                    )
                    # Connectome manifests also retain their explicit input mapping.
                    source_atlas = (
                        source_atlas
                        or (stage.get("inputs") or {}).get("parcellation")
                        or (stage.get("dependencies") or {}).get("ddparcel_parcellation")
                    )
                    if isinstance(source_atlas, dict):
                        source_atlas = source_atlas.get("path")
                    if not source_atlas:
                        raise ValueError("Connectome does not record its source atlas")
                    if stage.get("symmetric") is False:
                        raise ValueError("Directed connectomes are not supported by this regional viewer")
                    connectomes[name] = dict(
                        matrix=str(file_path(matrix)),
                        nodes=str(file_path(nodes)),
                        atlas=str(file_path(source_atlas)),
                        backend=qc.get("backend") or stage.get("backend") or parent.get("connectome_backend"),
                        qc_path=str(file_path(task["qc_path"])),
                    )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warnings.append(name + ": " + str(exc))
    if atlas_override:
        atlas = file_path(atlas_override)
    if atlas is None:
        raise ValueError("No QC-eligible DDParcel atlas. Run parcellation or select an atlas override")
    image, labels = atlas_data(atlas)
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    if not len(ids):
        raise ValueError("Atlas contains no non-background regions")
    lut = file_path(lut_path) if lut_path else (_default_lut_path() if not atlas_override else None)
    names = _parse_label_names([int(value) for value in ids], lut) if lut else {}
    brain_mask = None
    try:
        inputs = _resolve_analysis_inputs(
            Path(parent["subject_root"]),
            str(parent["subject_id"]),
            Path(parent["output_root"]) if parent.get("output_root") else None,
        )
        brain_mask = next((path for path in metric_brain_masks.values() if path), None) or str(
            file_path(inputs["mask"])
        )
        brain_hash = sha256(brain_mask)
        for key, mask_path in metric_brain_masks.items():
            if mask_path and sha256(mask_path) != brain_hash:
                metrics.pop(key, None)
                warnings.append(key + ": recorded fitting brain mask differs from other selected-run maps")
    except (OSError, ValueError, KeyError):
        warnings.append("Brain mask unavailable; scalar statistics cannot run")
        metrics = {}
    atlas_hash = sha256(atlas)
    for key, value in list(connectomes.items()):
        if sha256(value["atlas"]) != atlas_hash:
            del connectomes[key]
            warnings.append(key + ": source atlas differs from the chosen atlas")
    regions = [dict(id=int(key), name=names.get(int(key), "ROI-" + str(key)), voxels=int(count),
                    volume_mm3=float(count * abs(np.linalg.det(image.affine[:3, :3]))))
               for key, count in zip(ids, counts)]
    surfaces = {}
    if surface_task:
        from shared.region_reports import discover_surfaces

        try:
            surfaces = discover_surfaces(surface_task, regions)
        except (OSError, ValueError, IndexError) as exc:
            warnings.append("Cortical morphology unavailable: " + str(exc))
    return dict(
        version=1,
        source_manifest=str(path),
        manifest_sha256=sha256(path),
        subject_id=str(parent["subject_id"]),
        run_name=str(parent.get("run_name") or "run"),
        output_root=str(parent.get("output_root") or path.parent.parent),
        atlas=str(atlas),
        atlas_sha256=atlas_hash,
        atlas_override=str(atlas_override or ""),
        lut_path=str(lut) if lut else None,
        lut_sha256=sha256(lut) if lut else None,
        brain_mask=brain_mask,
        regions=regions,
        tissue=tissue,
        surfaces=surfaces,
        metrics=metrics,
        tractograms=tracts,
        connectomes=connectomes,
        warnings=warnings,
        validation_files=sorted(validation_files),
    )


class VTKSpool:
    """Bounded-memory legacy VTK writer; original RAS coordinates remain doubles."""

    def __init__(self, folder, stem):
        self.points_path, self.cells_path = Path(folder) / (stem + ".points"), Path(folder) / (stem + ".cells")
        self.points = self.points_path.open("wb")
        self.cells = self.cells_path.open("wb")
        self.point_count = self.line_count = 0

    def append(self, points):
        size = len(points)
        if self.point_count + size >= 2**31 - 1:
            raise ValueError("Selected geometry exceeds legacy VTK integer capacity")
        self.points.write(np.asarray(points, dtype=">f8").tobytes())
        ids = np.r_[size, np.arange(self.point_count, self.point_count + size)].astype(">i4")
        self.cells.write(ids.tobytes())
        self.point_count += size
        self.line_count += 1

    def finish(self, target):
        self.points.close()
        self.cells.close()
        with Path(target).open("wb") as stream:
            stream.write(
                b"# vtk DataFile Version 3.0\nSelected streamlines, geometry only. SPACE=RAS\nBINARY\nDATASET POLYDATA\n"
            )
            stream.write(f"POINTS {self.point_count} double\n".encode())
            with self.points_path.open("rb") as source:
                shutil.copyfileobj(source, stream)
            stream.write(f"\nLINES {self.line_count} {self.point_count + self.line_count}\n".encode())
            with self.cells_path.open("rb") as source:
                shutil.copyfileobj(source, stream)
            stream.write(b"\n")

    def close(self):
        self.points.close()
        self.cells.close()


def native_streamlines(path):
    from connectome.pipeline import _iter_streamlines

    path = Path(path)
    if path.suffix.lower() not in {".vtk", ".vtp"}:
        yield from _iter_streamlines(path)
        return
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader() if path.suffix.lower() == ".vtp" else vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data.GetPoints() is None or data.GetNumberOfLines() < 1:
        raise ValueError("Tractogram has no polyline geometry")
    space = "RAS"
    header_space = None
    if path.suffix.lower() == ".vtk":
        with path.open("rb") as stream:
            stream.readline(512)
            header = stream.readline(512).decode("ascii", errors="ignore")
        match = re.search(r"\bSPACE\s*=\s*([A-Za-z]+)", header)
        if match:
            header_space = match.group(1).upper()
            space = header_space
    for key in ("SPACE", "CoordinateSystem"):
        field = data.GetFieldData().GetAbstractArray(key)
        if field is not None and field.GetNumberOfValues():
            value = field.GetVariantValue(0).ToString().upper()
            if header_space and value != header_space:
                raise ValueError("Conflicting VTK coordinate-system metadata")
            space = value
    if space not in {"RAS", "LPS"}:
        raise ValueError("Unsupported tractogram coordinate system: " + space)
    points = vtk_to_numpy(data.GetPoints().GetData())
    cells = data.GetLines()
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    for start, end in zip(offsets[:-1], offsets[1:]):
        line = np.asarray(points[connectivity[start:end]], dtype=np.float64)
        if space == "LPS":
            line[:, :2] *= -1
        yield line


def _endpoint_selection(batch, image, selected_mask, both, target_mask=None):
    inverse = np.linalg.inv(image.affine)
    result = []
    for points in batch:
        voxel = np.floor(points[[0, -1]] @ inverse[:3, :3].T + inverse[:3, 3] + 0.5).astype(np.int64)
        valid = ((voxel >= 0) & (voxel < selected_mask.shape)).all(axis=1)
        hits = np.zeros(2, dtype=bool)
        hits[valid] = selected_mask[tuple(voxel[valid].T)]
        if target_mask is not None:
            target_hits = np.zeros(2, dtype=bool)
            target_hits[valid] = target_mask[tuple(voxel[valid].T)]
            accepted = (hits[0] and target_hits[1]) or (hits[1] and target_hits[0])
        else:
            accepted = hits.all() if both else hits.any()
        if accepted:
            result.append(points)
    return result


def select_streamlines(source, image, labels, selected, mode, directory, *, weights_path=None, preview_limit=30000,
                       target_labels=(), include_labels=(), exclude_labels=()):
    """Select complete lines, not clipped fragments or a new anatomical bundle definition."""
    from dipy.tracking.utils import target_line_based

    if mode not in {"through_any", "through_all", "endpoint_any", "endpoint_both", "endpoint_pair"}:
        raise ValueError("Unknown ROI streamline rule")
    if mode == "through_all" and len(selected) > 16:
        raise ValueError("Through-all supports at most 16 regions; use through-any for larger selections")
    if preview_limit < 1:
        raise ValueError("Preview line limit must be positive")
    if mode == "endpoint_pair" and (not target_labels or set(selected) & set(target_labels)):
        raise ValueError("A-B endpoint selection requires nonempty, disjoint source and target regions")
    if mode != "endpoint_pair" and target_labels:
        raise ValueError("Target regions require the A-B endpoint rule")
    if len(include_labels) > 16:
        raise ValueError("At most 16 through-all inclusion regions are supported")
    union = np.isin(labels, selected)
    target_mask = np.isin(labels, target_labels) if target_labels else None
    exclude_mask = np.isin(labels, exclude_labels) if exclude_labels else None
    inclusion_masks = [labels == label for label in include_labels]
    inputs = iter(native_streamlines(source))
    rng = np.random.default_rng(42)
    preview, total, kept, degenerate, longest_preview_line = [], 0, 0, 0, 1
    preview_written = False
    length_sum = length_square_sum = 0.0
    weights = np.loadtxt(weights_path, ndmin=1) if weights_path else None
    if weights is not None and (weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Invalid matching SIFT2 weights")
    source_ids = directory / "selected_streamline_ids.csv"
    output, preview_path = directory / "selected_streamlines.vtk", directory / "selected_streamlines_preview.vtk"
    with (
        tempfile.TemporaryDirectory(prefix=".geometry-", dir=directory) as temporary,
        source_ids.open("w", newline="") as ids,
    ):
        spool = VTKSpool(temporary, "full")
        writer = csv.writer(ids)
        writer.writerow(
            ["selected_index", "source_index", "sift2_weight"]
            if weights is not None
            else ["selected_index", "source_index"]
        )
        try:
            while True:
                raw = list(itertools.islice(inputs, 4096))
                if not raw:
                    break
                batch, indices = [], {}
                for offset, line in enumerate(raw):
                    points = np.ascontiguousarray(line, dtype=np.float64)
                    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                        raise ValueError("Malformed or nonfinite source streamline")
                    if len(points) < 2:
                        degenerate += 1
                        continue
                    batch.append(points)
                    indices[id(points)] = total + offset
                if mode.startswith("endpoint"):
                    matches = _endpoint_selection(batch, image, union, mode == "endpoint_both", target_mask)
                elif mode == "through_any":
                    matches = list(target_line_based(batch, image.affine, union)) if batch else []
                else:
                    matches = batch
                    for label in selected:
                        if not matches:
                            break
                        matches = list(target_line_based(matches, image.affine, labels == label))
                for mask in inclusion_masks:
                    matches = list(target_line_based(matches, image.affine, mask)) if matches else []
                if matches and exclude_mask is not None:
                    matches = list(target_line_based(matches, image.affine, exclude_mask, include=False))
                for points in matches:
                    source_index = indices[id(points)]
                    if weights is not None and source_index >= len(weights):
                        raise ValueError("Tractogram and SIFT2 weight counts disagree")
                    writer.writerow(
                        [kept, source_index, float(weights[source_index])]
                        if weights is not None
                        else [kept, source_index]
                    )
                    spool.append(points)
                    kept += 1
                    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                    length_sum += length
                    length_square_sum += length * length
                    # Bound reservoir memory even for exceptionally long source lines.
                    if len(points) > 1_000_000:
                        continue
                    longest_preview_line = max(len(points), longest_preview_line)
                    bounded_limit = min(preview_limit, max(1, 1_000_000 // longest_preview_line))
                    if len(preview) > bounded_limit:
                        preview = preview[:bounded_limit]
                    if len(preview) < bounded_limit:
                        preview.append(points.astype(np.float32))
                    else:
                        index = int(rng.integers(kept))
                        if index < bounded_limit:
                            preview[index] = points.astype(np.float32)
                total += len(raw)
                print(f"[Regions] Tractography {total} processed; {kept} selected", flush=True)
            if weights is not None and len(weights) != total:
                raise ValueError("Tractogram and SIFT2 weight counts disagree")
            if total == 0:
                raise ValueError("Source tractogram is empty")
            if kept:
                spool.finish(output)
                visual = VTKSpool(temporary, "preview")
                try:
                    budget = 1_000_000
                    for points in preview:
                        if len(points) <= budget:
                            visual.append(points)
                            budget -= len(points)
                    if visual.line_count:
                        visual.finish(preview_path)
                        preview_written = True
                finally:
                    visual.close()
        finally:
            spool.close()
    return dict(
        input_streamlines=total,
        selected_streamlines=kept,
        degenerate_streamlines=degenerate,
        selection_rule=mode,
        source_labels=list(selected), target_labels=list(target_labels),
        include_labels=list(include_labels), exclude_labels=list(exclude_labels),
        length_mean_mm=length_sum / kept if kept else None,
        length_std_mm=float(np.sqrt(max(0.0, length_square_sum / kept - (length_sum / kept)**2))) if kept else None,
        coordinates="RAS mm",
        algorithm="DIPY target_line_based for through rules; exact endpoint voxel lookup otherwise",
        geometry_only=True,
        tensor_arrays_preserved=False,
        full_geometry_precision="float64",
        output=str(output) if kept else None,
        preview=str(preview_path) if preview_written else None,
        source_indices=str(source_ids),
        preview_policy="deterministic seed-42 bounded reservoir, at most 30000 lines/1000000 points; not for measurement",
    )


def extract_connectome(source, atlas, selected, directory):
    """Preserve original edge values; export an induced subgraph and selected-node connections."""
    if not source.get("atlas") or sha256(file_path(source["atlas"])) != sha256(atlas):
        raise ValueError("Connectome atlas differs from the selected atlas; labels cannot be silently remapped")
    with Path(source["nodes"]).open(encoding="utf-8", newline="") as stream:
        nodes = list(csv.DictReader(stream, delimiter="\t"))
    labels = [int(row["label"]) for row in nodes]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Invalid connectome node table")
    names = {int(row["label"]): row.get("name", "") for row in nodes}
    absent = set(selected) - set(labels)
    if absent:
        raise ValueError("Selected regions are not connectome nodes: " + ", ".join(map(str, sorted(absent))))
    edges, metric = {}, "connectivity_strength"
    with Path(source["matrix"]).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        sparse = header[:2] == ["source_node", "target_node"]
        if sparse:
            if len(header) != 3:
                raise ValueError("Expected one explicitly named connectivity metric")
            metric = header[2]
            for row in reader:
                if len(row) != 3:
                    raise ValueError("Malformed connectome edge row")
                a, b, value = int(row[0]), int(row[1]), float(row[2])
                if a not in names or b not in names or not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid connectome edge")
                key = tuple(sorted((a, b)))
                if key in edges and edges[key] != value:
                    raise ValueError("Conflicting symmetric connectome edge values")
                edges[key] = value
    if not sparse:
        if not str(source.get("backend", "")).startswith("mrtrix"):
            raise ValueError("Unrecognized dense matrix convention")
        matrix = np.loadtxt(source["matrix"], delimiter=",", ndmin=2)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
        ):
            raise ValueError("Invalid dense connectome matrix")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("Asymmetric matrix: do not silently symmetrize")
        lookup = (
            {key: index for index, key in enumerate(sorted(labels))}
            if len(matrix) == len(labels)
            else {key: key - 1 for key in labels}
        )
        if any(index < 0 or index >= len(matrix) for index in lookup.values()):
            raise ValueError("Connectome labels do not match matrix dimensions")
        for a in selected:
            for b in labels:
                edges[tuple(sorted((a, b)))] = float(matrix[lookup[a], lookup[b]])
    matrix_path, edges_path = (
        directory / "selected_connectome_matrix.csv",
        directory / "selected_region_connections.csv",
    )
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label_id", *selected])
        for a in selected:
            writer.writerow([a, *(edges.get(tuple(sorted((a, b))), 0.0) for b in selected)])
    with edges_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["selected_label", "selected_name", "partner_label", "partner_name", metric])
        for a in selected:
            for b in sorted(labels):
                value = edges.get(tuple(sorted((a, b))), 0.0)
                if value != 0:
                    writer.writerow([a, names[a], b, names[b], value])
    return dict(
        matrix=str(matrix_path),
        connections=str(edges_path),
        metric=metric,
        policy="Unchanged parent weights; induced subgraph plus all nonzero connections of each selected node",
    )


def run_analysis(request):
    started = time.monotonic()
    print("[Regions] Checking source run and QC", flush=True)
    catalog = inspect_run(
        request["source_manifest"], atlas_override=request.get("atlas_override"), lut_path=request.get("lut_path")
    )
    for key in ("manifest_sha256", "atlas_sha256", "lut_sha256"):
        if request.get(key) and request[key] != catalog[key]:
            raise ValueError("Source run/atlas changed; reload the region list")
    request = dict(request)
    if "selected_regions" not in request:
        ids = request.get("selected_labels")
        if not isinstance(ids, list) or any(type(value) is not int or value <= 0 for value in ids):
            raise ValueError("selected_labels requires positive integer IDs; use selected_regions for names")
    primary = resolve_regions(catalog["regions"], request.get("selected_regions", request.get("selected_labels")))
    request["selected_labels"] = primary
    for field in ("target", "include", "exclude"):
        request[field + "_labels"] = resolve_regions(
            catalog["regions"], request.get(field + "_regions", request.get(field + "_labels")), allow_empty=True
        )
    selected = sorted(set(primary) | set(request["target_labels"]))
    metric_keys = request.get("metrics", [])
    if not isinstance(metric_keys, list) or any(key not in catalog["metrics"] for key in metric_keys):
        raise ValueError("Selected metric is unavailable or failed QC")
    tract_key, connectome_key = request.get("tractogram"), request.get("connectome")
    if (
        tract_key
        and tract_key not in catalog["tractograms"]
        or connectome_key
        and connectome_key not in catalog["connectomes"]
    ):
        raise ValueError("Selected tractography/connectome is unavailable or failed QC")
    if request.get("tissue") and not catalog.get("tissue"):
        raise ValueError("QC-eligible DDSeg segmentation is unavailable")
    if request.get("surface") and not any(str(label) in catalog["surfaces"] for label in selected):
        raise ValueError("No selected cortical region has a matching aparc surface annotation")
    if not tract_key and any(request[field + "_labels"] for field in ("target", "include", "exclude")):
        raise ValueError("Target/include/exclude regions require a tractogram")
    if not metric_keys and not tract_key and not connectome_key and not request.get("tissue") and not request.get("surface"):
        raise ValueError("Select microstructure, tissue, surface, tractography or connectome analysis")
    directory = Path(request["output_dir"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path, qc_path = directory / "region_analysis_manifest.json", directory / "region_analysis_qc.json"
    if any(path.name != "request.json" for path in directory.iterdir()):
        raise ValueError("Choose a fresh regional output directory; previous results are not overwritten")
    sources = {Path(catalog["source_manifest"]), Path(catalog["atlas"])}
    sources.update(Path(value) for value in catalog["validation_files"])
    if catalog["lut_path"]:
        sources.add(Path(catalog["lut_path"]))
    if request.get("tissue"):
        sources.update(Path(value) for value in catalog["tissue"].values())
    if request.get("surface"):
        for label in selected:
            spec = catalog["surfaces"].get(str(label), {})
            sources.update(Path(spec[key]) for key in ("annotation", "thickness", "area", "curv") if key in spec)
    for key in metric_keys:
        item = catalog["metrics"][key]
        sources.update(
            Path(value) for value in (item["path"], item["valid_mask"], item["qc_path"], catalog["brain_mask"]) if value
        )
    if tract_key:
        sources.update(Path(value) for value in catalog["tractograms"][tract_key].values() if value)
    if connectome_key:
        sources.update(
            Path(value)
            for key, value in catalog["connectomes"][connectome_key].items()
            if key in {"matrix", "nodes", "atlas", "qc_path"} and value
        )
    print("[Regions] Fingerprinting selected quantitative inputs", flush=True)
    signatures = {str(path): sha256(path) for path in sources}
    record = dict(
        kind="slicerddsurfer_regional_analysis",
        status="running",
        subject_id=catalog["subject_id"],
        run_name=catalog["run_name"] + " / regions / " + directory.name,
        output_root=str(directory),
        source_manifest=catalog["source_manifest"],
        request=request,
        selected_region_ids=selected,
        source_sha256=signatures,
        atlas_lut_path=catalog["lut_path"],
        implementation_sha256={
            name: sha256(Path(__file__).resolve().parents[1] / name)
            for name in ("shared/region_analysis.py", "shared/region_selection.py", "shared/region_reports.py", "microstructure/pipeline.py")
        },
        tasks=[],
        started_at=datetime.now(timezone.utc).isoformat(),
        human_review="pending",
    )
    write_json(manifest_path, record)
    warnings = []
    try:
        from shared.region_reports import bilateral_statistics, surface_statistics, tissue_statistics, write_report, write_table

        image, labels = atlas_data(catalog["atlas"])
        regions = [row for row in catalog["regions"] if row["id"] in selected]
        metric_rows, tissue_rows, surface_rows = [], [], []
        tract_info = graph_info = None
        region_path = directory / "selected_regions.nii.gz"
        subset = np.where(np.isin(labels, selected), labels, 0).astype(np.int32)
        nib.save(nib.Nifti1Image(subset, image.affine), region_path)
        summary_path = directory / "selected_regions.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "name", "voxels", "volume_mm3"])
            writer.writeheader()
            writer.writerows(row for row in catalog["regions"] if row["id"] in selected)
        outputs = {"selected_regions": str(region_path), "region_summary": str(summary_path)}
        if metric_keys:
            from microstructure.pipeline import _compute_roi_stats

            print("[Regions] Computing selected-region microstructure statistics", flush=True)
            stats_csv, stats_json = (
                directory / "selected_microstructure.csv",
                directory / "selected_microstructure.json",
            )
            _compute_roi_stats(
                metric_paths={key: Path(catalog["metrics"][key]["path"]) for key in metric_keys},
                roi_label_map=Path(catalog["atlas"]),
                lut_path=Path(catalog["lut_path"]) if catalog["lut_path"] else None,
                csv_path=stats_csv,
                json_path=stats_json,
                validity_mask_paths={
                    key: Path(catalog["metrics"][key]["valid_mask"])
                    for key in metric_keys
                    if catalog["metrics"][key]["valid_mask"]
                },
                brain_mask_path=Path(catalog["brain_mask"]),
                selected_labels=tuple(selected),
                include_quantiles=True,
            )
            outputs["roi_stats_csv"] = str(stats_csv)
            metric_rows = read_json(stats_json)["rows"]
            for row in metric_rows:
                for key in metric_keys:
                    if row[key + "_valid_voxel_count"] == 0:
                        warnings.append(f'Region {row["label_id"]}: no valid {key} estimates; blank, not zero')
        if request.get("tissue"):
            tissue_rows = tissue_statistics(catalog["tissue"]["path"], image, labels, regions)
            path = directory / "selected_tissue_statistics.csv"
            write_table(path, tissue_rows)
            record["tasks"].append(dict(name="statistics.regional_tissue", status="completed",
                                        outputs={"tissue_stats_csv": str(path)}, qc_path=str(qc_path)))
        if request.get("surface"):
            surface_rows = surface_statistics(catalog["surfaces"], regions)
            unmapped = [row["name"] for row in regions if str(row["id"]) not in catalog["surfaces"]]
            if unmapped:
                warnings.append("No exact cortical aparc match for: " + ", ".join(unmapped))
            path = directory / "selected_surface_statistics.csv"
            write_table(path, surface_rows)
            record["tasks"].append(dict(name="statistics.regional_surface", status="completed",
                                        outputs={"surface_stats_csv": str(path)}, qc_path=str(qc_path)))
        bilateral_rows = bilateral_statistics(regions, metric_rows, metric_keys, surface_rows)
        if bilateral_rows:
            path = directory / "bilateral_statistics.csv"
            write_table(path, bilateral_rows)
            record["tasks"].append(dict(name="statistics.regional_bilateral", status="completed",
                                        outputs={"bilateral_stats_csv": str(path)}, qc_path=str(qc_path)))
        record["tasks"].append(
            dict(name="statistics.regions", status="completed", outputs=outputs, qc_path=str(qc_path))
        )
        if tract_key:
            item = catalog["tractograms"][tract_key]
            info = select_streamlines(
                item["path"],
                image,
                labels,
                primary,
                request.get("tract_rule", "through_any"),
                directory,
                weights_path=item.get("weights"),
                target_labels=request["target_labels"], include_labels=request["include_labels"],
                exclude_labels=request["exclude_labels"],
            )
            tract_info = info
            write_json(directory / "selected_tractography.json", info)
            if not info["selected_streamlines"]:
                warnings.append("No streamlines match the ROI rule; no tract geometry invented")
            record["tasks"].append(
                dict(
                    name="tractography.regions",
                    status="completed",
                    qc_path=str(qc_path),
                    outputs={
                        key: info[field]
                        for key, field in (
                            ("tractogram", "output"),
                            ("display_preview", "preview"),
                            ("source_indices", "source_indices"),
                        )
                        if info.get(field)
                    },
                )
            )
        if connectome_key:
            info = extract_connectome(
                catalog["connectomes"][connectome_key], Path(catalog["atlas"]), selected, directory
            )
            graph_info = info
            write_json(directory / "selected_connectome.json", info)
            record["tasks"].append(
                dict(
                    name="connectome.regions",
                    status="completed",
                    qc_path=str(qc_path),
                    outputs={key: info[key] for key in ("matrix", "connections")},
                )
            )
        report = write_report(directory, catalog, regions, metric_rows, metric_keys, tissue_rows, surface_rows,
                              bilateral_rows, tract_info, graph_info, warnings)
        record["report_path"] = str(report)
        print("[Regions] Verifying inputs remained unchanged", flush=True)
        if any(sha256(path) != signature for path, signature in signatures.items()):
            raise RuntimeError("A quantitative input changed during regional analysis; results not accepted")
        qc = dict(
            status="completed",
            automatic_qc_passed=True,
            checks=dict(selected_labels_present=True, quantitative_inputs_unchanged=True, no_refitting=True),
            warnings=warnings,
            human_review="pending",
            scope="Regional postprocessing consistency only; not new upstream scientific validation",
        )
        write_json(qc_path, qc)
        record.update(
            status="completed",
            runtime_seconds=time.monotonic() - started,
            warnings=warnings,
            artifact_sha256={
                str(path): sha256(path) for path in directory.iterdir() if path.is_file() and path != manifest_path
            },
        )
        write_json(manifest_path, record)
        print("[Regions] Completed: " + str(manifest_path), flush=True)
        return manifest_path
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        record.update(status=state, error=str(exc) or state, runtime_seconds=time.monotonic() - started)
        for task in record["tasks"]:
            task["status"] = state
        write_json(qc_path, dict(status=state, automatic_qc_passed=False, error=str(exc) or state))
        write_json(manifest_path, record)
        raise
