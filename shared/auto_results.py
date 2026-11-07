"""Pure, current-manifest-only automatic result and child-run selection."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

from shared.result_scene import result_candidates, result_file_suffix, record_qc_state


DISPLAY_STREAMLINE_LIMIT = 30_000
AUTOMATIC_VTK_MAX_BYTES = 32 * 1024 * 1024
PER_PRODUCT_CAPS = {
    "microstructure.dki": 9, "microstructure.dti": 5,
    "microstructure": 4, "segmentation": 1, "parcellation": 1,
    "surface": 4, "tractography": 1, "saf": 1, "bundles": 6, "connectome": 4, "statistics": 2,
    "preprocessing": 2, "microstructure.dki": 9, "microstructure.mkcurve": 6,
}


def _read_record(path):
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _fresh_record(path, record, not_before):
    if not_before is None:
        return True
    timestamps = []
    for key in ("updated_at", "created_at", "started_at", "ended_at"):
        value = record.get(key)
        if value:
            try:
                timestamps.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
            except (ValueError, TypeError, OverflowError):
                pass
    # Checkpoints use second-resolution UTC metadata. Never use output/cache mtimes.
    if timestamps:
        return max(timestamps) >= math.floor(not_before)
    try:
        return path.stat().st_mtime >= not_before
    except OSError:
        return False


def _command_option(command, option):
    value = None
    for index, token in enumerate(command):
        if token == option and index + 1 < len(command):
            value = command[index + 1]
        elif isinstance(token, str) and token.startswith(option + "="):
            value = token.split("=", 1)[1]
    return value


def _logged_manifest(result):
    tail = result.get("stdout_tail") or result.get("stdout") or ""
    log = result.get("stdout_log")
    if not tail and log and Path(str(log)).is_absolute():
        try:
            with Path(log).open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 65536))
                tail = stream.read().decode("utf-8", errors="replace")
        except OSError:
            pass
    if not isinstance(tail, str):
        return None
    paths = re.findall(r'"run_manifest_path"\s*:\s*"([^"\n]+)"', tail)
    if paths:
        return paths[-1]
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line.startswith("/") and line.endswith("_orchestration_manifest.json"):
            return line
    return None


def _tracking_children(record):
    paths = []
    for child in record.get('child_runs', []):
        if not isinstance(child, dict) or not child.get('manifest_path') or child.get('status') == 'planned':
            continue
        candidate = Path(str(child['manifest_path'])).expanduser()
        if not candidate.is_absolute():
            continue
        child_record = _read_record(candidate)
        try:
            started = datetime.fromisoformat(child['started_at']).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        if (child_record.get('execute') is not False and isinstance(child_record.get('tasks'), list)
                and _fresh_record(candidate, child_record, started)
                and (not record.get('subject_id') or child_record.get('subject_id') == record['subject_id'])):
            paths.append(candidate.resolve())
    return tuple(dict.fromkeys(paths))


def automatic_manifest_paths(manifest_path: Path, *, not_before: float | None = None) -> tuple[Path, ...]:
    """Resolve only this run or actually recorded batch children, without scans.

    ``not_before`` is the controller's launch time in epoch seconds. Batch
    ``subjects`` and ``planned_commands`` are deliberately not evidence of runs.
    """
    path = Path(manifest_path).expanduser().resolve()
    record = _read_record(path)
    if not record or record.get("execute") is False or not _fresh_record(path, record, not_before):
        return ()
    if record.get('tractography_backends') and isinstance(record.get('child_runs'), list):
        return _tracking_children(record)
    if record.get("kind") != "slicerdsurfer_batch_run":
        if isinstance(record.get("tasks"), list):
            return (path,)
        return ()
    paths = []
    for result in record.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        command = result.get("command") or []
        if not isinstance(command, list):
            continue
        explicit = result.get("run_manifest_path") or result.get("manifest_path") or _logged_manifest(result)
        if explicit:
            child = Path(str(explicit)).expanduser()
        else:
            output = _command_option(command, "--output-root")
            subject = _command_option(command, "--subject-root") or result.get("subject_root")
            run = _command_option(command, "--run-name") or record.get("run_name") or "Test1"
            if not output and not subject:
                continue
            root = Path(str(output)).expanduser() if output else Path(str(subject)).expanduser() / "outputs"
            child = root / "00_run_metadata" / f"{str(run).strip().replace('/', '_') or 'csd_act'}_orchestration_manifest.json"
        if not child.is_absolute():
            continue
        child = child.resolve()
        child_record = _read_record(child)
        if (
            not child_record or child_record.get("execute") is False
            or not isinstance(child_record.get("tasks"), list)
            or not _fresh_record(child, child_record, not_before)
            or (result.get("subject_id") and child_record.get("subject_id") != result["subject_id"])
        ):
            continue
        if child_record.get('tractography_backends') and isinstance(child_record.get('child_runs'), list):
            paths.extend(path for path in _tracking_children(child_record) if path not in paths)
        elif child not in paths:
            paths.append(child)
    return tuple(paths)


def auto_manifest_paths(manifest_path: Path, not_before: float | None = None) -> tuple[Path, ...]:
    """Controller-friendly alias for current subject/batch manifest discovery."""
    return automatic_manifest_paths(manifest_path, not_before=not_before)


def uniform_streamline_indices(total: int, limit: int = DISPLAY_STREAMLINE_LIMIT) -> tuple[int, ...]:
    """Deterministic, endpoint-inclusive uniform display sample indices."""
    if limit < 1 or total < 0:
        raise ValueError("Streamline limit must be positive and total nonnegative")
    if total <= limit:
        return tuple(range(total))
    if limit == 1:
        return (total // 2,)
    return tuple(index * (total - 1) // (limit - 1) for index in range(limit))


def _role(candidate):
    product = candidate["product"]
    key = candidate["key"].lower().removesuffix("_path")
    path = Path(candidate["path"])
    suffix = result_file_suffix(path)
    if product == 'statistics.regions':
        if key == 'selected_regions' and suffix in {'.nii', '.nii.gz'}:
            return 'categorical:region_selection'
        if key == 'region_summary':
            return 'table:regions'
    if product.startswith("preprocessing.") and suffix in {".nii", ".nii.gz", ".mgz"}:
        if key in {"mask", "brain_mask", "brainmask", "final_mask", "brain_mask_final"}:
            return "categorical:mask"
        if key in {"b0", "mean_b0", "mean_b0_image", "nodif"}:
            return "scalar:b0"
    if product == "microstructure.dti" and suffix == ".nrrd" and key == "dec":
        return "vector:dec"  # sorts after the scalar roles, so it never displaces FA, MD, AD or RD
    if product.startswith("microstructure.") and suffix in {".nii", ".nii.gz", ".mgz"}:
        aliases = {"fa": "fa", "md": "md", "ad": "ad", "rd": "rd", "fwf": "fwf", "fw_fraction": "fwf",
                   "fw_fa": "fw_fa", "fw_tissue_fa": "fw_fa", "fw_md": "fw_md", "fw_tissue_md": "fw_md",
                   "ndi": "ndi", "noddi_ndi": "ndi", "odi": "odi", "noddi_odi": "odi",
                   "noddi_fwf": "isovf", "noddi_isovf": "isovf", "isovf": "isovf"}
        if key in aliases:
            return "scalar:" + aliases[key]
        kurtosis = key.removeprefix("dki_").removeprefix("mkcurve_")
        if kurtosis.removeprefix("corrected_") in {"mk", "ak", "rk", "kfa", "mkt", "rtk"}:
            return "scalar:" + kurtosis
    if product.startswith(("segmentation.", "parcellation.")) and suffix in {".nii", ".nii.gz", ".mgz"}:
        if key in {"segmentation", "labels", "label_map", "wmparc", "wmparc_labels", "parcellation"}:
            return "categorical"
    if product == "surface.ddsurfer" and suffix == ".stl":
        for hemisphere in ("lh", "rh"):
            if hemisphere in key or re.search(r"[_\.]" + hemisphere + r"[_\.]", path.name.lower()):
                if any(token in key or token in path.name.lower() for token in ("white", "wm")):
                    return "surface:white:" + hemisphere
                if "pial" in key or "pial" in path.name.lower():
                    return "surface:pial:" + hemisphere
    if product.startswith(("tractography.", "saf.")) and suffix in {".trk", ".tck", ".vtk", ".vtp"}:
        if not any(word in key for word in ("seed", "fod", "mask", "intermediate")):
            return "tractography"
    if product.startswith("bundles.") and suffix in {".trk", ".tck", ".vtk", ".vtp"}:
        name = path.name.lower()
        if re.search(r"(^|[^a-z])af([^a-z]|$)|arcuate", name):
            return "bundle:af"
        if re.search(r"(^|[^a-z])cst([^a-z]|$)|corticospinal", name):
            return "bundle:cst"
        if re.search(r"(^|[^a-z])cc([^a-z]|$)|corpus[_ -]?callos|forceps", name):
            return "bundle:cc"
        return "bundle:representative"
    if suffix in {".csv", ".tsv"}:
        if product.startswith("connectome.") and any(word in key for word in ("matrix", "nodes", "stat", "connections")):
            return "table:" + key
        if any(word in key for word in ("stat", "summary")):
            return "table:statistics"
    return None


def _qc_state(task):
    outputs = task.get("outputs") or {}
    value = task.get("qc_path") or next((value for key, value in outputs.items()
        if key == "qc" or key.endswith("_qc") or key.endswith("_qc_path")), None)
    record = _read_record(value) if value and Path(str(value)).is_absolute() else {}
    return record_qc_state(record)


def automatic_result_candidates(manifest: dict, *, per_product_caps: dict | None = None):
    """Return (selected, skipped, deferred); caps are per concrete product.

    Only absolute files declared by completed/reused tasks or their recorded
    bundle QC are eligible. No product folder is ever searched.
    """
    candidates, skipped = result_candidates(manifest, allow_bundle_directory_scan=False, include_preprocessing=True)
    caps = {**PER_PRODUCT_CAPS, **(per_product_caps or {})}
    tasks = {task.get("name"): task for task in manifest.get("tasks", []) or [] if isinstance(task, dict)}
    ranked = []
    deferred = []
    for candidate in candidates:
        role = _role(candidate)
        if role is None:
            deferred.append({**candidate, "reason": "Not a representative automatic product"})
            continue
        candidate = {**candidate, "display_role": role, "qc_state": _qc_state(tasks.get(candidate["product"], {})),
                     "human_review_state": "pending"}
        if candidate['product'] == 'tractography.regions' and candidate['key'] == 'display_preview':
            candidate['quantitative_source_path'] = tasks[candidate['product']]['outputs'].get('tractogram', candidate['path'])
        if role == 'categorical:region_selection':
            candidate['color_lut_path'] = manifest.get('atlas_lut_path')
            candidate['region_label_ids'] = manifest.get('selected_region_ids') or (manifest.get('request') or {}).get('selected_labels', [])
        if candidate["qc_state"] in {"failed", "partial_failed", "error"}:
            skipped.append({**candidate, "reason": "Recorded QC failed; automatic display excluded", "reason_category": "qc_failed"})
            continue
        path = Path(candidate["path"])
        if path.suffix.lower() in {".vtk", ".vtp"} and path.stat().st_size > AUTOMATIC_VTK_MAX_BYTES:
            deferred.append({**candidate, "reason": "Large VTK/VTP display deferred; use manual loading or a TRK/TCK preview"})
            continue
        format_rank = {".tck": 0, ".trk": 1, ".vtp": 2, ".vtk": 3}.get(path.suffix.lower(), 0)
        ranked.append((candidate["product"], role, format_rank, candidate["label"], candidate))
    selected, counts, role_counts = [], {}, {}
    named_bundle_products = {item[4]["product"] for item in ranked
                             if item[1] in {"bundle:af", "bundle:cst", "bundle:cc"}}
    for _, role, _, _, candidate in sorted(ranked, key=lambda item: item[:4]):
        product = candidate["product"]
        group = product.split(".", 1)[0]
        cap = max(0, int(caps.get(product, 3 if product == 'statistics.regions' else caps.get(group, 2))))
        if role == "bundle:representative" and product in named_bundle_products:
            deferred.append({**candidate, "reason": "Named anatomical representatives are available"})
            continue
        role_limit = 3 if role == "bundle:representative" else 2 if role.startswith("bundle:") else 1
        slot = (product, role)
        if counts.get(product, 0) >= cap or role_counts.get(slot, 0) >= role_limit:
            deferred.append({**candidate, "reason": "Per-product representative display limit"})
            continue
        selected.append(candidate)
        counts[product] = counts.get(product, 0) + 1
        role_counts[slot] = role_counts.get(slot, 0) + 1
    return selected, skipped, deferred
