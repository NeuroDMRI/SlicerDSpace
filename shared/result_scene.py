"""Manifest-scoped result selection and ownership-aware Slicer scene management."""

from pathlib import Path
import json
import os
from typing import Optional
from shared.bundle_qc import bundle_record_error

PREFIX = "SlicerDDSurfer."
FREESURFER_LUT = Path(__file__).resolve().parents[1] / "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt"
SUFFIXES = (".nii.gz", ".nii", ".mgz", ".csv", ".tsv", ".stl", ".obj", ".vtk", ".vtp", ".trk", ".tck")
_NATIVE_TRACT_SUFFIXES = (".vtk", ".vtp")
# D6: the DEC colour FA map is the only NRRD result; other .nrrd files in the tree stay intermediates.
DEC_SUFFIX = "-dti-dec.nrrd"
DEC_SUFFIXES = ("-dti-dec.nrrd", "_dec.nrrd")


def result_file_suffix(path):
    """Compound extensions must not absorb dots in product/subject filenames."""
    name = Path(path).name.lower()
    if name.endswith(DEC_SUFFIXES):
        return ".nrrd"
    return next((suffix for suffix in sorted(SUFFIXES, key=len, reverse=True) if name.endswith(suffix)), "")


def tract_file_coordinate_system(path):
    """Honor explicit Slicer SPACE metadata; pipeline tract points default to RAS."""
    try:
        import vtk
        reader = vtk.vtkXMLPolyDataReader() if Path(path).suffix.lower() == ".vtp" else vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        fields = reader.GetOutput().GetFieldData()
        for key in ("SPACE", "CoordinateSystem"):
            array = fields.GetAbstractArray(key)
            if array is not None and array.GetNumberOfValues():
                value = str(array.GetVariantValue(0).ToString()).upper()
                if value in {"RAS", "LPS"}:
                    return value
    except (ImportError, AttributeError, ValueError):
        pass
    return "RAS"


def bundle_outputs(task, *, allow_directory_scan=True, accept_qc_not_passed=False):
    """Resolve bundles from the successful task's own QC, not shared folders.

    ``accept_qc_not_passed`` (Results tree and one clicked row only): also resolve the files that a readable record
    names when its QC verdict is not passed; the caller marks them."""
    task = dict(task, name=str(task.get("name", "")).split("@", 1)[0])
    if task.get("name") not in {"bundles.wmql", "bundles.wma", "bundles.tractcloud"}:
        return {}
    path = task.get("qc_path") or (task.get("outputs") or {}).get("qc_path")
    if accept_qc_not_passed and _unreadable_record(path):
        # Review M1: no readable QC record. The task's own bundle manifest names the files (no directory scan).
        return _bundle_files_from_task_manifest(task)
    try:
        if not path or not Path(path).is_absolute():
            return {}
        qc = json.loads(Path(path).read_text(encoding="utf-8"))
        if not accept_qc_not_passed and (
                qc.get("status") not in {"completed", "passed"} or bundle_record_error(task['name'], qc)):
            return {}
        summary = qc.get("output_summary") or {}
        files = summary.get("bundle_files") or summary.get("anatomical_tract_files") or []
        if not files and task["name"] == "bundles.wma" and allow_directory_scan:
            # Older WMA QC records name the exact anatomical-tract directory.
            # Require the recorded count; never scan the entire output root.
            folder = Path(str(summary.get("anatomical_tracts_dir") or ""))
            if folder.is_absolute() and folder.is_dir():
                files = [str(path) for path in sorted(folder.glob("*.vtp"))]
                if len(files) != int(summary.get("anatomical_tract_count") or 0):
                    return {}
        empty = set(summary.get("empty_bundle_files") or [])
        empty_names = set(summary.get("anatomical_empty_tracts") or [])
        published_dir = (task.get("outputs") or {}).get("output_dir")
        published_dir_path = Path(str(published_dir)) if published_dir else None
        resolved_files: dict[str, str] = {}
        for value in files:
            if not isinstance(value, str) or value in empty or Path(value).name in empty_names:
                continue
            name_only = Path(value).name
            published_file = published_dir_path / name_only if published_dir_path is not None else None
            if published_file is not None and published_file.is_file() and _published_copy_of(value, published_file):
                resolved_files["bundle_" + name_only] = published_file.as_posix()
            else:
                resolved_files["bundle_" + name_only] = value
        return resolved_files
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _unreadable_record(path) -> bool:
    """True when ``path`` names no absolute, readable JSON object (missing, invalid JSON or not an object)."""

    try:
        if not path or not Path(str(path)).is_absolute():
            return True
        return not isinstance(json.loads(Path(str(path)).read_text(encoding="utf-8")), dict)
    except (OSError, ValueError, TypeError):
        return True


def _bundle_files_from_task_manifest(task) -> dict:
    """Review M1: the files of a bundle task whose QC record cannot be read, from the task's own bundle manifest.

    The backends (WMQL, TractCloud: ``outputs.summary.bundle_files``; WMA: ``outputs.summary.anatomical_tract_files``)
    name their stage files; each one is looked up by name in the task's recorded ``output_dir``, where the orchestrator
    published it. A file is accepted only when it exists and, resolved, lies inside that folder. Empty bundles are left
    out as on the QC route. No directory scan; a missing or unreadable manifest gives nothing.
    """

    outputs = task.get("outputs") or {}
    manifest_path, output_dir = task.get("manifest_path") or outputs.get("manifest_path"), outputs.get("output_dir")
    try:
        if not manifest_path or not output_dir or not Path(str(manifest_path)).is_absolute():
            return {}
        folder = Path(str(output_dir))
        if not folder.is_absolute() or not folder.is_dir():
            return {}
        record = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
        summary = (record.get("outputs") or {}).get("summary") or {}
        files = summary.get("bundle_files") or summary.get("anatomical_tract_files") or []
        empty = set(summary.get("empty_bundle_files") or [])
        empty_names = set(summary.get("anatomical_empty_tracts") or [])
        root = folder.resolve()
        resolved_files: dict[str, str] = {}
        for value in files:
            if not isinstance(value, str) or not value or value in empty or Path(value).name in empty_names:
                continue
            name_only = Path(value).name
            published_file = folder / name_only
            if not name_only or not published_file.is_file():
                continue
            if not published_file.resolve().is_relative_to(root):
                continue
            if not _published_copy_of(value, published_file):
                continue  # a same-name file of another tracking variant or run (as on the QC route)
            resolved_files["bundle_" + name_only] = str(published_file)
        return resolved_files
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _published_copy_of(stage_file, published_file):
    """Batch 2 item 5: True unless the published file is another tracking variant's file of the same name.

    Before the per-tractography subfolders, every variant published into one flat folder, so an old flat file can
    belong to a later variant. When the task's own stage file still exists and differs from the published file, the
    stage file is the right one; when the stage file is gone, the published file is all there is.
    """
    stage = Path(stage_file)
    try:
        if not stage.is_file():
            return True
        if stage.samefile(published_file):
            return True
        first, second = stage.stat(), Path(published_file).stat()
        return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns
    except OSError:
        return True


def source_stamp(path):
    stat = Path(path).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def record_qc_state(record):
    """Explicit failure wins over a stale or contradictory pass flag."""
    status = str(record.get('status') or '').strip().lower()
    checks = record.get('checks')
    failed_check = (any(value is False for value in checks.values()) if isinstance(checks, dict) else
                    any(isinstance(value, dict) and value.get('passed') is False for value in checks)
                    if isinstance(checks, list) else False)
    if (record.get('kind') == 'wmql_bundle_segmentation_qc' and record.get('version') == 6
            and (record.get('official_multibundle') is True or record.get('coverage_report_mode') is True)
            and status == 'completed'
            and record.get('execution_passed') is True and record.get('output_integrity_passed') is True
            and not failed_check and not record.get('error')
            and record.get('bundle_assessment', {}).get('bundles')
            and record.get('automatic_qc_passed') is False):
        return 'warning'  # Usable candidates, not all-bundle scientific acceptance.
    if (status in {'fail', 'failed', 'partial_failed', 'error'}
            or record.get('automatic_qc_passed') is False or failed_check or record.get('error')):
        return 'failed'
    if status == 'warning' or record.get('sampling_warnings'):
        return 'warning'
    if record.get('automatic_qc_passed') is True:
        return 'passed'
    return status or 'unavailable'


def recorded_qc_summary(manifest):
    """Report recorded QC states without treating execution status as QC."""
    counts = {"passed": 0, "issues": 0, "unavailable": 0}
    for task in manifest.get("tasks", []) or []:
        if not isinstance(task, dict) or task.get("status") in {"planned", "blocked", "cancelled"}:
            continue
        if task.get('status') in {'failed', 'partial_failed', 'error'}:
            counts['issues'] += 1
            continue
        value = task.get("qc_path") or (task.get("outputs") or {}).get("qc")
        try:
            if not value or not Path(value).is_absolute():
                raise ValueError("Missing absolute QC path")
            record = json.loads(Path(value).read_text(encoding='utf-8'))
            status = record_qc_state(record)
            issue = status in {'failed', 'warning'} or bundle_record_error(task.get('name'), record)
            key = "issues" if issue else "passed" if status in {"passed", "completed", "pass", "success"} else "unavailable"
            counts[key] += 1
        except (OSError, ValueError, TypeError, AttributeError):
            counts["unavailable"] += 1
    return counts


def review_state_text(qc, *, issues_found):
    """Text of the review label: recorded QC counts, then the pending human review on its own line."""
    text = (f"Recorded QC: {qc['passed']} passed · {qc['issues']} issues · {qc['unavailable']} unavailable"
            "\nVisual review: pending")
    return "Execution: issues found\n" + text if issues_found else text


def _is_internal_cache_path(path: Path, manifest: dict) -> bool:
    roots: list[Path] = []
    subject_root = str(manifest.get("subject_root") or "").strip()
    output_root_raw = str(manifest.get("output_root") or "").strip()
    if subject_root:
        roots.append(Path(subject_root) / "cache")
        if output_root_raw:
            from shared.io_layout import subject_layout

            roots.append(subject_layout(Path(subject_root), output_root=Path(output_root_raw)).cache)
    elif output_root_raw:
        out = Path(output_root_raw)
        roots.append((out.parent if out.name.lower() in {"output", "outputs"} else out) / "cache")
    for cache_root in roots:
        try:
            path.relative_to(cache_root)
            return True
        except ValueError:
            try:
                path.resolve().relative_to(cache_root.resolve())
                return True
            except (OSError, ValueError):
                pass
    return False


def _is_under_input_dir(path: Path, manifest: dict) -> bool:
    subject_root_raw = str(manifest.get("subject_root") or "").strip()
    if not subject_root_raw:
        return False
    input_dir = Path(subject_root_raw).expanduser() / "input"
    try:
        path.resolve().relative_to(input_dir.resolve())
        return True
    except (OSError, ValueError):
        return False


def _has_preprocessing_mask_candidate(candidates: list[dict]) -> bool:
    for item in candidates:
        product = str(item.get("product") or "")
        key = str(item.get("key") or "").lower().removesuffix("_path")
        if product.startswith("preprocessing.") and key in {
            "mask", "brain_mask", "brainmask", "final_mask", "brain_mask_final"
        }:
            return True
    return False


def _resolve_manifest_brain_mask_candidate(manifest: dict) -> Optional[Path]:
    raw_mask_path = manifest.get("brain_mask_path")
    if isinstance(raw_mask_path, str) and raw_mask_path.strip():
        cand = Path(raw_mask_path).expanduser()
        if (
            cand.is_absolute()
            and cand.name.lower().endswith((".nii.gz", ".nii", ".mgz"))
            and cand.is_file()
            and not _is_internal_cache_path(cand, manifest)
            and not _is_under_input_dir(cand, manifest)
        ):
            return cand
        return None
    if manifest.get("execute") is False:
        return None
    subject_root_raw = str(manifest.get("subject_root") or "").strip()
    output_root_raw = str(manifest.get("output_root") or "").strip()
    cand_preproc: Optional[Path] = None
    if subject_root_raw:
        from shared.io_layout import subject_layout

        layout = subject_layout(
            Path(subject_root_raw),
            output_root=Path(output_root_raw) if output_root_raw else None,
        )
        cand_preproc = layout.preproc / "nodif_brain_mask.nii.gz"
    elif output_root_raw:
        out = Path(output_root_raw)
        workspace = out.parent if out.name.lower() in {"output", "outputs"} else out
        cand_preproc = workspace / "preprocessed" / "nodif_brain_mask.nii.gz"
    if (
        cand_preproc is not None
        and cand_preproc.is_absolute()
        and cand_preproc.is_file()
        and not _is_internal_cache_path(cand_preproc, manifest)
    ):
        return cand_preproc
    return None


def native_tract_stem_key(path) -> tuple[str, str]:
    """Case-normalized ``(parent_dir, stem)`` key for matching a ``.trk`` file to a sibling ``.vtk``/``.vtp``."""
    resolved = Path(path)
    return (os.path.normcase(str(resolved.parent)), resolved.stem.lower())


def _filter_redundant_trk_candidates(task_candidates: list[dict]) -> list[dict]:
    """Omit ``.trk`` candidates when Slicer-native ``.vtk``/``.vtp`` is available for the same task or stem.

    ``.trk`` files remain generated on disk and recorded in manifests/QC for downstream consumers
    (e.g. WMQL, connectome), while Slicer Results visualization uses native ``.vtk``/``.vtp`` directly.
    Tasks that only emit ``.trk`` (such as WMQL bundles or standalone DTI-det) retain ``.trk`` as fallback.
    """
    from shared.fiber_display import KIND_TRACTOGRAPHY, streamline_kind

    has_native_tractography = any(
        Path(entry["path"]).suffix.lower() in _NATIVE_TRACT_SUFFIXES
        and streamline_kind(entry) == KIND_TRACTOGRAPHY
        for entry in task_candidates
    )
    native_stems = {
        native_tract_stem_key(entry["path"])
        for entry in task_candidates
        if Path(entry["path"]).suffix.lower() in _NATIVE_TRACT_SUFFIXES
    }
    if not has_native_tractography and not native_stems:
        return task_candidates
    kept: list[dict] = []
    for entry in task_candidates:
        if Path(entry["path"]).suffix.lower() == ".trk":
            if (has_native_tractography and streamline_kind(entry) == KIND_TRACTOGRAPHY) or (
                native_tract_stem_key(entry["path"]) in native_stems
            ):
                continue
        kept.append(entry)
    return kept


def result_candidates(manifest, *, allow_bundle_directory_scan=True, include_preprocessing=False, include_brain_mask=False,
                      include_surface_morphometry=False, include_qc_not_passed=False):
    """``include_surface_morphometry`` (off by default, so bulk and automatic loads are unchanged): also list the
    published files of ``surface.ddsurfer_postprocess`` (shared.surface_morphometry_display), accepted by product and
    file name, in the order lh then rh: thickness, curv, sulc, area, aparc, aparc.a2009s, inflated, stats.

    ``include_qc_not_passed`` (off by default; on only for the Results tree and the load of one clicked row): a
    completed bundle task whose recorded QC failed or is unreadable is listed, not skipped. Each of its candidates has
    ``qc_mark`` ("failed" or "unreadable") and ``qc_reason`` (the recorded reason). A task that is not completed stays
    skipped."""
    from shared.surface_morphometry_display import morphometry_row, row_order

    candidates, skipped, seen = [], [], set()
    for task in manifest.get("tasks", []) or []:
        if not isinstance(task, dict):
            continue
        name = str(task.get("name") or "task")
        reused = task.get("status") == "skipped" and task.get("reason") in {
            "completed outputs found", "completed DTI microstructure outputs found",
            "completed Free Water outputs found", "completed NODDI outputs found",
        }
        if task.get("status") not in {"completed", "reused"} and not reused:
            entry = {"label": name, "product": name, "reason": "Task is not completed or explicitly reused"}
            if task.get('status') in {'failed', 'partial_failed', 'error', 'cancelled'}:
                entry['reason_category'] = 'task_failed'
            skipped.append(entry)
            continue
        if name.startswith("preprocessing.") and not include_preprocessing:
            continue
        qc_mark = None
        if name.split('@', 1)[0] in {'bundles.wmql', 'bundles.wma', 'bundles.tractcloud'}:
            value = task.get('qc_path') or (task.get('outputs') or {}).get('qc_path')
            unreadable = not value
            try:
                record = json.loads(Path(str(value)).read_text(encoding='utf-8')) if value else {}
                error = bundle_record_error(name, record)
            except (OSError, ValueError, TypeError):
                error, unreadable = 'Unreadable bundle QC', True
            if error and not include_qc_not_passed:
                skipped.append({'label': name, 'product': name, 'reason': error, 'reason_category': 'qc_failed'})
                continue
            if error:
                qc_mark = ("unreadable" if unreadable else "failed", error)
        outputs = dict(task.get("outputs") or {})
        outputs.update(bundle_outputs(task, allow_directory_scan=allow_bundle_directory_scan,
                                      accept_qc_not_passed=qc_mark is not None))
        task_candidates: list[dict] = []
        for key, value in outputs.items():
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                skipped.append({"path": value, "reason": "Relative output path needs an explicit source root"})
                continue
            morphometry = include_surface_morphometry and morphometry_row(name, path) is not None
            if not morphometry and not path.name.lower().endswith(SUFFIXES + DEC_SUFFIXES):
                continue
            if _is_internal_cache_path(path, manifest):
                continue
            if path.name.endswith("_orig.stl") and path.with_name(path.name.replace("_orig.stl", ".stl")).is_file():
                continue
            identity = str(path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            if not path.is_file():
                skipped.append({"path": value, "reason": "Manifest output is missing"})
                continue
            # Prefer small anatomical/scalar products before expensive tractograms.
            suffix_lower = path.suffix.lower()
            priority = 2 if suffix_lower in {".tck", ".trk", ".vtk", ".vtp"} else 1 if suffix_lower in {".stl", ".obj"} else 0
            entry = {"path": identity, "label": name + "_" + str(key), "key": str(key), "product": name, "priority": priority}
            if morphometry:
                entry.update(priority=1, order=row_order(name, path))
            if qc_mark is not None:
                entry.update(qc_mark=qc_mark[0], qc_reason=qc_mark[1])
            task_candidates.append(entry)
        candidates.extend(_filter_redundant_trk_candidates(task_candidates))
    if include_brain_mask and not _has_preprocessing_mask_candidate(candidates):
        mask_path = _resolve_manifest_brain_mask_candidate(manifest)
        if mask_path is not None:
            identity = str(mask_path.resolve())
            if identity not in seen:
                seen.add(identity)
                candidates.append({
                    "path": identity,
                    "label": "preprocessing.brain_mask_mask",
                    "key": "brain_mask",
                    "product": "preprocessing.brain_mask",
                    "priority": 0,
                })
    # A post-processing row sorts by its decided order (a tuple), every other row by its label as before.
    return sorted(candidates, key=lambda item: (item["priority"], item.get("order", ()), item["label"])), skipped


def result_nodes(slicer):
    # Tables are storable MRML nodes but do not inherit DisplayableNode.
    nodes = {}
    for class_name in ("vtkMRMLDisplayableNode", "vtkMRMLTableNode"):
        for node in slicer.util.getNodesByClass(class_name):
            nodes[node.GetID()] = node
    return list(nodes.values())


def owned_nodes(slicer, run_id):
    targets = {str(run_id)}
    try:
        targets.add(str(Path(str(run_id)).expanduser().resolve()))
    except OSError:
        pass
    return [node for node in result_nodes(slicer)
            if node.GetAttribute(PREFIX + "Run") in targets]


def clear_run(slicer, run_id):
    nodes = owned_nodes(slicer, run_id)
    for node in nodes:
        slicer.mrmlScene.RemoveNode(node)
    return len(nodes)


def register_node(slicer, node, run_id, subject, run_name, product, source):
    for key, value in {"Run": run_id, "Subject": subject, "Product": product, "SourcePath": source,
                       "SourceStamp": source_stamp(source)}.items():
        node.SetAttribute(PREFIX + key, value)
    hierarchy = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)

    def folder(parent, name, key):
        import vtk
        children = vtk.vtkIdList()
        hierarchy.GetItemChildren(parent, children)
        for index in range(children.GetNumberOfIds()):
            item = children.GetId(index)
            if hierarchy.GetItemAttribute(item, PREFIX + "FolderKey") == key:
                return item
        item = hierarchy.CreateFolderItem(parent, name)
        hierarchy.SetItemAttribute(item, PREFIX + "FolderKey", key)
        return item

    parent = folder(hierarchy.GetSceneItemID(), subject, "subject:" + subject)
    parent = folder(parent, run_name, "run:" + run_id)
    parent = folder(parent, product, "product:" + product)
    hierarchy.SetItemParent(hierarchy.GetItemByDataNode(node), parent)


def scalar_preset(node, label):
    """Stable conventional display ranges; scientific voxel data are unchanged."""
    display = node.GetDisplayNode()
    if display is None:
        return
    label = label.lower()
    if "md" in label or "ad" in label or "rd" in label:
        display.SetAndObserveColorNodeID("vtkMRMLColorTableNodeGrey")
    elif any(token in label for token in ("_fa", "_ndi", "_odi", "fraction")):
        display.SetAutoWindowLevel(False)
        display.SetWindowLevel(1.0, 0.5)


def annotate_automatic_result(node, candidate):
    """Recorded QC is not a human review, nor a claim of scientific validity."""
    for key, value in {"AutomaticLoaded": "true", "DisplayRole": candidate["display_role"],
                       "QCState": candidate.get("qc_state", "unavailable")}.items():
        node.SetAttribute(PREFIX + key, str(value))
    if not node.GetAttribute(PREFIX + "HumanReviewState"):
        node.SetAttribute(PREFIX + "HumanReviewState", "pending")
    if candidate["display_role"].startswith(("tractography", "bundle:")):
        node.SetAttribute(PREFIX + "DisplayPreview", "true")
        node.SetAttribute(PREFIX + "QuantitativeSourcePath", candidate.get('quantitative_source_path', candidate["path"]))
        node.SetAttribute(PREFIX + "MeasurementWarning", "Display preview; use the original file for quantitative measurements")


def regional_color_node(slicer, candidate):
    """Interpret FreeSurfer's unused alpha column as opaque foreground for display."""
    labels = sorted(set(candidate.get('region_label_ids') or []))
    if not labels:
        return None
    if min(labels) <= 0 or max(labels) > 1_000_000:
        raise ValueError('Regional display supports positive label IDs up to 1000000; quantitative files remain unchanged')
    lut_path = candidate.get('color_lut_path')
    stamp = source_stamp(lut_path) if lut_path and Path(lut_path).is_file() else None
    identity = str(lut_path) if stamp else 'generated:' + ','.join(map(str, labels))
    color = next((item for item in slicer.util.getNodesByClass('vtkMRMLColorTableNode')
                  if item.GetAttribute(PREFIX + 'LUTSourcePath') == identity
                  and item.GetAttribute(PREFIX + 'LUTSourceStamp') == (stamp or 'generated')), None)
    if color is None:
        color = slicer.util.loadColorTable(str(lut_path)) if stamp else None
        if color is None:
            color = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode', 'Selected region colors')
            color.SetTypeToUser()
        color.SetAttribute(PREFIX + 'LUTSourcePath', identity)
        color.SetAttribute(PREFIX + 'LUTSourceStamp', stamp or 'generated')
    supplied = {label: (color.GetColorName(label), color.GetLookupTable().GetTableValue(label)[:3])
                for label in labels if label < color.GetNumberOfColors() and color.GetColorName(label) not in {None, '', '(none)', 'NoName'}}
    if color.GetNumberOfColors() <= max(labels):
        color.SetNumberOfColors(max(labels) + 1)
    color.GetLookupTable().SetTableRange(0, color.GetNumberOfColors() - 1)
    palette = ((.31, .52, .70), (.78, .47, .31), (.35, .64, .53), (.68, .54, .72), (.77, .65, .30))
    color.SetColor(0, 'Background', 0., 0., 0., 0.)
    for index, label in enumerate(labels):
        name, rgb = supplied.get(label, ('ROI-' + str(label), palette[index % len(palette)]))
        color.SetColor(label, name, *rgb, 1.)
    return color


def freesurfer_color_node(slicer):
    """Slicer's FreeSurfer colour table, else the bundled FreeSurferColorLUT.txt, loaded once.

    Slicer 5.x keeps the FreeSurfer tables in the SlicerFreeSurfer extension; without it, DDParcel labels fell back
    to the generic Labels table. FreeSurfer's unused alpha column is read as opaque, as in regional_color_node.
    """
    node = slicer.mrmlScene.GetNodeByID("vtkMRMLColorTableNodeFileFreeSurferColorLUT.txt")
    if node is not None:
        return node
    if not FREESURFER_LUT.is_file():
        return None
    identity, stamp = "parcellation:" + str(FREESURFER_LUT), source_stamp(FREESURFER_LUT)
    color = next((item for item in slicer.util.getNodesByClass('vtkMRMLColorTableNode')
                  if item.GetAttribute(PREFIX + 'LUTSourcePath') == identity
                  and item.GetAttribute(PREFIX + 'LUTSourceStamp') == stamp), None)
    if color is not None:
        return color
    color = slicer.util.loadColorTable(str(FREESURFER_LUT))
    if color is None:
        return None
    table = color.GetLookupTable()
    for label in range(1, color.GetNumberOfColors()):
        name = color.GetColorName(label)
        if name not in {None, '', '(none)', 'NoName'}:
            color.SetColor(label, name, *table.GetTableValue(label)[:3], 1.)
    color.SetAttribute(PREFIX + 'LUTSourcePath', identity)
    color.SetAttribute(PREFIX + 'LUTSourceStamp', stamp)
    return color


_MASK_KEYS = {"mask", "brain_mask", "brainmask", "final_mask", "brain_mask_final"}
SEGMENTATION_FILL_OPACITY = 0.35


def categorical_role(candidate):
    """The categorical display role of a label volume, the same for automatic and manual loads; None if not one."""
    role = str(candidate.get("display_role") or "")
    if role.startswith("categorical"):
        return role
    product = str(candidate.get("product") or "")
    key = str(candidate.get("key") or "").lower().removesuffix("_path")
    if product == "statistics.regions" and key == "selected_regions":
        return "categorical:region_selection"
    if product.startswith("connectome.") and key == "node_parcellation":
        return "categorical:node_parcellation"  # the DDParcel ids kept as nodes, not renumbered
    if product.startswith("preprocessing.") and key in _MASK_KEYS:
        return "categorical:mask"
    if product.startswith(("segmentation.", "parcellation.")) and any(
            word in key for word in ("label", "segmentation", "wmparc", "aparc")):
        return "categorical"
    return None


def categorical_context(candidate, manifest):
    """Add what the colour source needs from the manifest: the recorded .ctbl, or the region ids and atlas LUT."""
    role = categorical_role(candidate) or ""
    extra = {}
    if role == "categorical:region_selection":
        extra["color_lut_path"] = manifest.get("atlas_lut_path")
        extra["region_label_ids"] = (manifest.get("selected_region_ids")
                                     or (manifest.get("request") or {}).get("selected_labels", []))
    elif str(candidate.get("product") or "").startswith("parcellation."):
        task = next((task for task in manifest.get("tasks") or [] if isinstance(task, dict)
                     and str(task.get("name") or "") == candidate["product"]), {})
        extra["color_table_path"] = (task.get("outputs") or {}).get("color_table")
    return {**candidate, **{key: value for key, value in extra.items() if key not in candidate}}


def is_segmentation_result(candidate):
    """B2: the DDParcel parcellation loads as a named segmentation; DDSeg stays a labelmap for its edit copy."""
    return str(candidate.get("product") or "").startswith("parcellation.") and categorical_role(candidate) is not None


def _reused_color_node(slicer, identity, stamp):
    return next((item for item in slicer.util.getNodesByClass('vtkMRMLColorTableNode')
                 if item.GetAttribute(PREFIX + 'LUTSourcePath') == identity
                 and item.GetAttribute(PREFIX + 'LUTSourceStamp') == stamp), None)


def tissue_color_node(slicer):
    """DDSeg tissue names and colours (segmentation_review.TISSUES) as one shared colour table."""
    from shared.segmentation_review import TISSUES

    identity, stamp = "tissues:ddseg", ";".join(f"{value}:{name}:{rgb}" for value, name, rgb in TISSUES)
    color = _reused_color_node(slicer, identity, stamp)
    if color is not None:
        return color
    color = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLColorTableNode', 'DDSeg tissue colors')
    color.SetTypeToUser()
    color.SetNumberOfColors(max(value for value, _, _ in TISSUES) + 1)
    color.GetLookupTable().SetTableRange(0, color.GetNumberOfColors() - 1)
    color.SetColor(0, 'Background', 0., 0., 0., 0.)
    for value, name, rgb in TISSUES:
        color.SetColor(value, name, *rgb, 1.)
    color.SetAttribute(PREFIX + 'LUTSourcePath', identity)
    color.SetAttribute(PREFIX + 'LUTSourceStamp', stamp)
    return color


def parcellation_color_node(slicer, candidate):
    """The run's recorded .ctbl when it is not older than the label file, else the FreeSurfer table."""
    table, labels = candidate.get("color_table_path"), Path(candidate["path"])
    try:
        current = (bool(table) and Path(table).is_file() and labels.is_file()
                   and Path(table).stat().st_mtime_ns >= labels.stat().st_mtime_ns)
    except OSError:
        current = False
    if current:
        identity, stamp = "parcellation-table:" + str(Path(table).resolve()), source_stamp(table)
        color = _reused_color_node(slicer, identity, stamp)
        if color is None:
            try:
                color = slicer.util.loadColorTable(str(table))
            except Exception:  # b2fix D3: Slicer raises on an unreadable .ctbl; the FreeSurfer table below still colours it
                color = None
            if color is not None:
                color.SetAttribute(PREFIX + 'LUTSourcePath', identity)
                color.SetAttribute(PREFIX + 'LUTSourceStamp', stamp)
        if color is not None:
            return color
    return freesurfer_color_node(slicer)


def apply_categorical_colors(slicer, node, candidate):
    """The one colour source of a categorical label volume, for automatic and manual loads alike."""
    display = node.GetDisplayNode() if hasattr(node, "GetDisplayNode") else None
    if display is None:
        return None
    role = categorical_role(candidate) or ""
    product = str(candidate.get("product") or "")
    if role == "categorical:region_selection":
        color, fallbacks = regional_color_node(slicer, candidate), ()
    elif product.startswith("parcellation."):
        color, fallbacks = parcellation_color_node(slicer, candidate), ("vtkMRMLColorTableNodeLabels",)
    elif role == "categorical:node_parcellation":
        color, fallbacks = freesurfer_color_node(slicer), ("vtkMRMLColorTableNodeLabels",)
    elif product.startswith("segmentation."):
        color, fallbacks = tissue_color_node(slicer), ()
    else:
        color, fallbacks = None, ("vtkMRMLColorTableNodeGenericAnatomyColors", "vtkMRMLColorTableNodeLabels")
    if color is not None:
        display.SetAndObserveColorNodeID(color.GetID())
        return color
    for identity in fallbacks:
        if slicer.mrmlScene.GetNodeByID(identity) is not None:
            display.SetAndObserveColorNodeID(identity)
            return slicer.mrmlScene.GetNodeByID(identity)
    return None


def load_categorical_result(slicer, path, name, candidate):
    """Load a label volume with its colours; a parcellation becomes a named segmentation (2D fill and outline).

    The segmentation path loads a temporary labelmap, sets its colour node, imports it, and removes it again.
    No closed surface is built; the user can still choose Show 3D.
    """
    labelmap = slicer.util.loadLabelVolume(str(path), properties={"name": name, "show": False})
    if not labelmap:
        return None
    if not is_segmentation_result(candidate):
        try:
            apply_categorical_colors(slicer, labelmap, candidate)
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap)  # a manual load must not leave an unregistered node behind
            raise
        return labelmap
    segmentation = None
    try:
        apply_categorical_colors(slicer, labelmap, candidate)
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
        segmentation.CreateDefaultDisplayNodes()
        if not slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap, segmentation):
            raise RuntimeError("Slicer could not convert the parcellation labels to a segmentation")
        display = segmentation.GetDisplayNode()
        display.SetVisibility2DFill(True)
        display.SetVisibility2DOutline(True)
        display.SetOpacity2DFill(SEGMENTATION_FILL_OPACITY)
    except Exception:
        if segmentation is not None:
            slicer.mrmlScene.RemoveNode(segmentation)
        raise
    finally:
        slicer.mrmlScene.RemoveNode(labelmap)
    return segmentation


_DWI_KEYS = {"dwi", "final_dwi", "preprocessed_dwi"}


def is_dwi_result(candidate) -> bool:
    """True when ``candidate`` names a 4D diffusion-weighted NIfTI output (for example ``preprocessing.dwi``)."""
    product = str(candidate.get("product") or "")
    key = str(candidate.get("key") or "").lower().removesuffix("_path")
    suffix = result_file_suffix(candidate.get("path") or "")
    if suffix not in {".nii", ".nii.gz"}:
        return False
    return product.startswith("preprocessing.") and key in _DWI_KEYS


def dwi_gradient_paths(path, candidate, manifest) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve matching ``(bval, bvec)`` files for a DWI result from the manifest task or adjacent files."""
    dwi_path = Path(path)
    product = str(candidate.get("product") or "")
    bval_path: Optional[Path] = None
    bvec_path: Optional[Path] = None
    for task in (manifest or {}).get("tasks") or []:
        if not isinstance(task, dict) or str(task.get("name") or "") != product:
            continue
        outputs = task.get("outputs") or {}
        for key in ("bval", "final_bval", "bval_path"):
            raw = outputs.get(key)
            if isinstance(raw, str) and raw.strip():
                cand = Path(raw).expanduser()
                if cand.is_file():
                    bval_path = cand
                    break
        for key in ("bvec", "final_bvec", "bvec_path"):
            raw = outputs.get(key)
            if isinstance(raw, str) and raw.strip():
                cand = Path(raw).expanduser()
                if cand.is_file():
                    bvec_path = cand
                    break
        if bval_path is not None and bvec_path is not None:
            return bval_path, bvec_path

    name = dwi_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else (name[:-4] if name.lower().endswith(".nii") else dwi_path.stem)
    parent = dwi_path.parent
    if bval_path is None:
        for cand_name in (f"{stem}.bval", f"{stem}.bvals", "bvals", "bval"):
            cand = parent / cand_name
            if cand.is_file():
                bval_path = cand
                break
    if bvec_path is None:
        for cand_name in (f"{stem}.bvec", f"{stem}.bvecs", "bvecs", "bvec"):
            cand = parent / cand_name
            if cand.is_file():
                bvec_path = cand
                break
    return bval_path, bvec_path


def _load_4d_nifti_scalar_fallback(slicer, image, name: str, bval_path: Optional[Path]):
    """Load a 3D representative volume (mean b0 or volume 0) from a 4D NIfTI while preserving exact RAS geometry.

    Avoids ``slicer.util.loadVolume`` on 4D left-handed (LAS) NIfTI files, where ITK flips the Y direction without
    adjusting the origin and shifts the volume in RAS space.
    """
    import numpy as np
    import vtk
    from vtk.util import numpy_support
    from shared.input_preview import B0_THRESHOLD, read_bvals

    count = int(image.shape[3])
    indices = [0]
    if bval_path is not None and bval_path.is_file():
        try:
            bvals = read_bvals(bval_path)
            if len(bvals) == count:
                b0 = [i for i, val in enumerate(bvals) if val <= B0_THRESHOLD]
                if b0:
                    indices = b0
        except Exception:
            pass
    total = None
    for index in indices:
        vol = np.asarray(image.dataobj[..., index], dtype=np.float32)
        total = vol.copy() if total is None else (total + vol)
    vol3d = np.ascontiguousarray((total / np.float32(len(indices))).transpose(2, 1, 0))
    nx, ny, nz = (int(size) for size in image.shape[:3])
    vtk_array = numpy_support.numpy_to_vtk(vol3d.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("ImageScalars")
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(nx, ny, nz)
    vtk_image.GetPointData().SetScalars(vtk_array)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
    if node is None:
        return None
    try:
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(np.asarray(image.affine, dtype=float)))
        node.SetAndObserveImageData(vtk_image)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:
            display.SetAutoWindowLevel(1)
    except Exception:
        slicer.mrmlScene.RemoveNode(node)
        raise
    return node


def load_dwi_result(slicer, path, name: str, candidate, manifest):
    """Load a 4D NIfTI DWI result as a ``vtkMRMLDiffusionWeightedVolumeNode`` with exact RAS coordinates.

    Returns the created MRML node, or ``None`` when ``path`` is not a readable 4D NIfTI (so callers can fall back to
    ``loadVolume`` for 3D files or dummy unit-test placeholders).
    """
    from shared.input_preview import (
        PreviewSource,
        build_dwi_volume_node,
        estimate_full_dwi_bytes,
        full_dwi_limit_bytes,
        gradient_table_unusable,
        read_full_dwi,
    )

    dwi_path = Path(path)
    try:
        import nibabel as nib

        image = nib.load(str(dwi_path))
        shape = tuple(int(size) for size in image.shape)
    except Exception:
        return None
    if len(shape) != 4:
        return None
    bval_path, bvec_path = dwi_gradient_paths(dwi_path, candidate, manifest)
    subject = str((manifest or {}).get("subject_id") or "")
    source = PreviewSource(
        kind="nifti",
        dwi=dwi_path,
        bval=bval_path,
        mask=None,
        subject_id=subject,
        origin=str(dwi_path.parent),
        bvec=bvec_path,
    )
    if (
        bval_path is not None
        and not gradient_table_unusable(source)
        and estimate_full_dwi_bytes(source) <= full_dwi_limit_bytes()
    ):
        try:
            full_dwi = read_full_dwi(source)
            node, _info = build_dwi_volume_node(slicer, full_dwi, name)
            return node
        except Exception:
            pass
    return _load_4d_nifti_scalar_fallback(slicer, image, name, bval_path)


def automatic_display_preset(slicer, node, candidate, *, hide_pial=True):
    """Only display properties change; voxel values, label IDs and mesh data do not."""
    annotate_automatic_result(node, candidate)
    role = candidate["display_role"]
    if role.startswith(("bundle:", "tractography")):
        return  # the fiber loader already applied the one native fiber style (shared.fiber_display)
    display = node.GetDisplayNode() if hasattr(node, "GetDisplayNode") else None
    if display is None:
        return

    def color_node(*identities):
        for identity in identities:
            if slicer.mrmlScene.GetNodeByID(identity) is not None:
                display.SetAndObserveColorNodeID(identity)
                break

    if role.startswith("scalar:"):
        metric = role.split(":", 1)[1]
        color_node("vtkMRMLColorTableNodeGrey" if metric in {"fa", "md", "ad", "rd", "fw_fa", "fw_md", "kfa", "corrected_kfa", "b0"}
                   else "vtkMRMLColorTableNodeRainbow")
        if metric == "b0":
            display.SetAutoWindowLevel(True)
            return
        display.SetAutoWindowLevel(False)
        maximum = 0.003 if metric in {"md", "ad", "rd", "fw_md"} else 3.0 if metric.removeprefix("corrected_") in {"mk", "ak", "rk", "mkt", "rtk"} else 1.0
        display.SetWindowLevel(maximum, maximum / 2)
    elif role.startswith("categorical"):
        return  # load_categorical_result already applied the one categorical colour source, on every load path
    elif role.startswith("surface:"):
        pial = ":pial:" in role
        display.SetColor(*( (0.84, 0.64, 0.51) if pial else (0.88, 0.88, 0.82) ))
        display.SetOpacity(0.3 if pial else 1.0)
        display.SetVisibility(not (pial and hide_pial))


def fit_slice_views_to_bounds(slicer, bounds):
    """Centre every slice view on a RAS box and zoom to it: the volume-free counterpart of resetSliceViews.

    The in-plane axes of a slice are the first two columns of its SliceToRAS rotation, so the extent of the box along
    an axis is the sum of |axis component| * box size. The field of view keeps the aspect ratio of the view.
    """
    for node in slicer.util.getNodesByClass("vtkMRMLSliceNode"):
        width, height = node.GetDimensions()[:2]
        if width < 1 or height < 1:
            continue
        matrix = node.GetSliceToRAS()
        extent = [sum(abs(matrix.GetElement(row, axis)) * (bounds[2 * row + 1] - bounds[2 * row]) for row in range(3))
                  for axis in (0, 1)]
        per_pixel = max(extent[0] / width, extent[1] / height)
        if per_pixel > 0:
            node.SetFieldOfView(per_pixel * width, per_pixel * height, node.GetFieldOfView()[2])
    center = [(bounds[index] + bounds[index + 1]) / 2 for index in (0, 2, 4)]
    slice_class = getattr(slicer, "vtkMRMLSliceNode", None)
    if slice_class is not None:
        slice_class.JumpAllSlices(slicer.mrmlScene, *center, slice_class.CenteredJumpSlice)


def show_automatic_results(slicer, entries, *, newly_loaded=None):
    """Initialize views once after new loads, not on every checkpoint reuse."""
    backgrounds, labels, segmentations = [], [], []
    new = entries if newly_loaded is None else newly_loaded
    new_ids = {identity for entry in new for identity in entry.get("node_ids", [])}
    for entry in entries:
        for identity in entry.get("node_ids", []):
            node = slicer.mrmlScene.GetNodeByID(identity)
            if node is None:
                continue
            role = entry.get("display_role", "")
            if role.startswith("scalar:") and node.IsA("vtkMRMLScalarVolumeNode"):
                rank = {"scalar:fa": 0, "scalar:fw_fa": 1, "scalar:md": 2,
                        "scalar:corrected_mk": 3, "scalar:mk": 4,
                        "scalar:ndi": 5, "scalar:odi": 6}.get(role, 10)
                backgrounds.append((rank, identity))
            elif role.startswith("categorical") and node.IsA("vtkMRMLLabelMapVolumeNode"):
                rank = 0 if entry["product"].startswith("segmentation.") else 2 if role == "categorical:mask" else 1
                labels.append((rank, identity))
            elif role.startswith("categorical") and node.IsA("vtkMRMLSegmentationNode") and identity in new_ids:
                segmentations.append(node)
    view = {}
    if backgrounds:
        view["background"] = sorted(backgrounds)[0][1]
    if labels:
        view["label"] = sorted(labels)[0][1]
        if sorted(labels)[0][0] == 0:
            # b2fix D4/B1: DDSeg owns the 2D overlay of an automatic load; DDParcel stays loaded but hidden. Only the node
            # display visibility is turned off, so one Results eye click shows it in 2D (Visibility2D stays on).
            for node in segmentations:
                node.SetDisplayVisibility(False)
    new_volumes = any(entry.get("display_role", "").startswith(("scalar:", "categorical")) for entry in new)
    if new_volumes and view and hasattr(slicer.util, "setSliceViewerLayers"):
        slicer.util.setSliceViewerLayers(**view, **({"labelOpacity": 0.35} if labels else {}))
        if hasattr(slicer.util, "resetSliceViews"):
            slicer.util.resetSliceViews()
        center_node = slicer.mrmlScene.GetNodeByID(view.get("background") or view.get("label"))
        slice_class = getattr(slicer, "vtkMRMLSliceNode", None)
        if center_node is not None and hasattr(center_node, "GetRASBounds") and slice_class is not None:
            bounds = [0.0] * 6
            center_node.GetRASBounds(bounds)
            center = [(bounds[index] + bounds[index + 1]) / 2 for index in (0, 2, 4)]
            # Fit/zoom alone may leave off-origin native data outside the
            # coronal/sagittal planes in a fresh scene.
            slice_class.JumpAllSlices(slicer.mrmlScene, *center, slice_class.CenteredJumpSlice)
    elif segmentations and not view:
        # b2fix D2: no volume layer to fit; fit and centre the slice views on the new segmentations themselves.
        boxes = []
        for node in segmentations:
            box = [0.0] * 6
            node.GetRASBounds(box)
            boxes.append(box)
        fit_slice_views_to_bounds(slicer, [(min if index % 2 == 0 else max)(box[index] for box in boxes) for index in range(6)])
    if any(entry.get("display_role", "").startswith(("surface:", "bundle:", "tractography")) for entry in new):
        if hasattr(slicer.util, "resetThreeDViews"):
            slicer.util.resetThreeDViews()
    return view
