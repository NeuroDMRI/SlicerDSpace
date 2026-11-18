"""Optional tissue/surface summaries and a self-contained regional report."""

import csv
from html import escape
from pathlib import Path

import nibabel as nib
import numpy as np

from shared.region_selection import region_identity


def write_table(path, rows, fields=None):
    fields = fields or list(rows[0] if rows else {})
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aligned_array(path, image):
    other = nib.load(str(path))
    if other.shape != image.shape or not np.allclose(other.affine, image.affine, rtol=0, atol=1e-4):
        raise ValueError("Regional input shape/affine differs from atlas: " + str(path))
    return np.asanyarray(other.dataobj)


def tissue_statistics(path, image, labels, regions):
    data = aligned_array(path, image)
    if not np.isfinite(data).all() or not np.isin(data, (0, 1, 2, 3)).all():
        raise ValueError("DDSeg tissue labels must be 0 background, 1 WM, 2 GM, 3 CSF")
    voxel_volume = abs(np.linalg.det(image.affine[:3, :3]))
    result = []
    for region in regions:
        values = data[labels == region["id"]]
        row = dict(label_id=region["id"], label_name=region["name"], roi_voxel_count=int(values.size))
        for code, name in ((0, "unclassified"), (1, "wm"), (2, "gm"), (3, "csf")):
            count = int(np.count_nonzero(values == code))
            row.update({name + "_voxels": count, name + "_volume_mm3": float(count * voxel_volume),
                        name + "_fraction": float(count / values.size) if values.size else None})
        result.append(row)
    return result


def _find_surface_file(outputs: dict, sub_dir_key: str, filename: str):
    if outputs.get(filename):
        cand = Path(outputs[filename])
        if cand.is_file():
            return cand
    if outputs.get(sub_dir_key):
        cand = Path(outputs[sub_dir_key]) / filename
        if cand.is_file():
            return cand
    if outputs.get("structural_root"):
        cand = Path(outputs["structural_root"]) / sub_dir_key / filename
        if cand.is_file():
            return cand
    return None


def discover_surfaces(task, regions):
    """Only exact cortex-name matches to recorded DK aparc annotations are linked."""
    outputs = task.get("outputs") or {}
    result = {}
    for hemi, side in (("lh", "left"), ("rh", "right")):
        annotation = _find_surface_file(outputs, "label", f"{hemi}.aparc.annot")
        morphs = {key: _find_surface_file(outputs, "surf", f"{hemi}.{key}") for key in ("thickness", "area", "curv")}
        if annotation is None or any(path is None for path in morphs.values()):
            continue
        vertex_labels, _, names = nib.freesurfer.read_annot(str(annotation))
        names = [name.decode("utf-8") for name in names]
        for region in regions:
            region_side, base, compartment = region_identity(region["name"])
            candidates = [i for i, name in enumerate(names) if region_identity(name)[1] == base]
            if (region_side == side and compartment == "ctx" and len(candidates) == 1
                    and np.any(vertex_labels == candidates[0])):
                result[str(region["id"])] = dict(
                    hemisphere=hemi, annotation=str(annotation.resolve()), annotation_name=names[candidates[0]],
                    annotation_index=candidates[0],
                    **{key: str(path.resolve()) for key, path in morphs.items()},
                )
    return result


def surface_statistics(surfaces, regions):
    loaded, result = {}, []
    for region in regions:
        spec = surfaces.get(str(region["id"]))
        if not spec:
            continue
        key = spec["annotation"]
        if key not in loaded:
            vertex_labels, _, _ = nib.freesurfer.read_annot(key)
            arrays = {name: nib.freesurfer.read_morph_data(spec[name]).astype(np.float64)
                      for name in ("thickness", "area", "curv")}
            if any(array.shape != vertex_labels.shape or not np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("Surface overlays must be finite and match annotation vertices")
            if np.any(arrays["thickness"] < 0) or np.any(arrays["area"] < 0):
                raise ValueError("Surface thickness and vertex area must be nonnegative")
            loaded[key] = vertex_labels, arrays
        vertex_labels, arrays = loaded[key]
        mask = vertex_labels == spec["annotation_index"]
        if not mask.any():
            raise ValueError("Surface region no longer has annotated vertices")
        row = dict(label_id=region["id"], label_name=region["name"], hemisphere=spec["hemisphere"],
                   surface_atlas="aparc", surface_region=spec["annotation_name"], vertex_count=int(mask.sum()),
                   area_mm2=float(arrays["area"][mask].sum()),
                   thickness_mean_mm=float(arrays["thickness"][mask].mean()),
                   thickness_median_mm=float(np.median(arrays["thickness"][mask])),
                   thickness_std_mm=float(arrays["thickness"][mask].std()),
                   curvature_mean_per_mm=float(arrays["curv"][mask].mean()))
        result.append(row)
    return result


def bilateral_statistics(regions, metric_rows, metric_keys, surface_rows=()):
    pairs, result = {}, []
    values = {row["id"]: {"volume_mm3": row["volume_mm3"]} for row in regions}
    for row in metric_rows:
        values[row["label_id"]].update({key: row[key + "_mean"] for key in metric_keys})
    for row in surface_rows:
        values[row["label_id"]].update({key: row[key] for key in ("area_mm2", "thickness_mean_mm")})
    for region in regions:
        side, base, compartment = region_identity(region["name"])
        if side:
            pairs.setdefault((base, compartment), {}).setdefault(side, []).append(region)
    for (base, compartment), sides in sorted(pairs.items(), key=lambda item: str(item[0])):
        if len(sides.get("left", [])) != 1 or len(sides.get("right", [])) != 1:
            continue
        left, right = sides["left"][0], sides["right"][0]
        for metric in values[left["id"]].keys() & values[right["id"]].keys():
            a, b = values[left["id"]][metric], values[right["id"]][metric]
            valid = a is not None and b is not None
            denom = abs(a) + abs(b) if valid else 0
            result.append(dict(region=base, compartment=compartment or "subcortical", metric=metric,
                               left_id=left["id"], right_id=right["id"], left_value=a, right_value=b,
                               left_minus_right=float(a - b) if valid else None,
                               asymmetry_index=float(2 * (a - b) / denom) if denom else None))
    return sorted(result, key=lambda row: (row["region"], row["compartment"], row["metric"]))


def write_report(directory, catalog, regions, metric_rows, metric_keys, tissue_rows, surface_rows,
                 bilateral_rows, tract_info, graph_info, warnings):
    """Local HTML only: no remote resources, inferred diagnoses or approximate numbers."""
    def table(rows, fields):
        if not rows:
            return "<p class='muted'>No matching results.</p>"
        headings = "".join("<th>" + escape(title) + "</th>" for _, title in fields)
        body = []
        for row in rows:
            cells = []
            for key, _ in fields:
                value = row.get(key)
                value = "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
                cells.append("<td>" + escape(value) + "</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<div class='scroll'><table><thead><tr>" + headings + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"

    sections = ["<h2>Selected regions</h2>" + table(regions, [("id", "Label"), ("name", "Region"), ("volume_mm3", "Volume (mm³)")])]
    for metric in metric_keys:
        sections.append("<h2>" + escape(metric) + "</h2>" + table(metric_rows, [
            ("label_name", "Region"), (metric + "_mean", "Mean"), (metric + "_median", "Median"),
            (metric + "_iqr", "IQR"), (metric + "_valid_voxel_count", "Valid voxels"),
            (metric + "_valid_fraction", "Valid fraction")]))
    if tissue_rows:
        sections.append("<h2>Tissue composition</h2>" + table(tissue_rows, [("label_name", "Region"),
            ("wm_fraction", "WM fraction"), ("gm_fraction", "GM fraction"), ("csf_fraction", "CSF fraction"),
            ("unclassified_fraction", "Unclassified fraction")]))
    if surface_rows:
        sections.append("<h2>Cortical morphology</h2><p class='muted'>Exact hemisphere/name matches to aparc surface annotations; vertex-mean thickness and summed vertex area.</p>" + table(surface_rows, [
            ("label_name", "Region"), ("vertex_count", "Vertices"), ("area_mm2", "Area (mm²)"),
            ("thickness_mean_mm", "Thickness (mm)"), ("curvature_mean_per_mm", "Curvature (mm⁻¹)")]))
    if bilateral_rows:
        sections.append("<h2>Left–right comparison</h2><p class='muted'>Asymmetry = 2(L − R)/(|L| + |R|); blank when unavailable or both values are zero.</p>" + table(bilateral_rows, [
            ("region", "Region"), ("metric", "Metric"), ("left_value", "Left"), ("right_value", "Right"),
            ("left_minus_right", "L − R"), ("asymmetry_index", "Asymmetry")]))
    if tract_info:
        sections.append("<h2>Selected streamlines</h2>" + table([tract_info], [("selection_rule", "Rule"),
            ("input_streamlines", "Source count"), ("selected_streamlines", "Selected count"),
            ("length_mean_mm", "Mean length (mm)"), ("length_std_mm", "Length SD (mm)")]) +
            "<p class='muted'>Geometric ROI selection. Streamline counts do not estimate axon counts.</p>")
    if graph_info:
        with Path(graph_info["connections"]).open(encoding="utf-8", newline="") as stream:
            edges = list(csv.DictReader(stream))
        metric = graph_info["metric"]
        edges.sort(key=lambda row: (-float(row[metric]), int(row["selected_label"]), int(row["partner_label"])))
        sections.append("<h2>Regional connectivity</h2><p class='muted'>Top 20 nonzero node connections; original " +
                        escape(metric) + " weights. Complete edges are retained in CSV.</p>" + table(edges[:20], [
                            ("selected_name", "Region"), ("partner_name", "Partner"), (metric, "Weight")]))
    if warnings:
        sections.append("<h2>Coverage notes</h2><ul>" + "".join("<li>" + escape(note) + "</li>" for note in warnings) + "</ul>")
    links = " · ".join(f'<a href="{escape(path.name, quote=True)}">{escape(path.name)}</a>'
                       for path in sorted(Path(directory).glob("*.csv")))
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SlicerDSpace regional analysis</title><style>
body{font:15px/1.6 Georgia,serif;color:#24343d;background:#f4f5f3;margin:0;padding:28px}
main{max-width:1050px;margin:auto;background:white;padding:32px;border:1px solid #dde3df}
h1{font-size:29px;margin:0}h2{font-size:20px;margin:30px 0 12px}.muted{color:#63747b;font-size:13px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font:13px/1.5 sans-serif}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid #e2e6e4}th{background:#edf3f2}a{color:#275b6b}
@media(max-width:600px){body{padding:8px}main{padding:16px}}
</style><main><h1>Regional analysis</h1>"""
    html += "<p class='muted'>SlicerDSpace · " + escape(catalog["subject_id"]) + " · " + escape(catalog["run_name"]) + "</p>"
    html += "<p class='muted'>Atlas: " + escape(catalog["atlas"]) + "</p>" + "".join(sections)
    html += "<h2>Data exports</h2><p>" + links + "</p></main></html>"
    path = Path(directory) / "regional_report.html"
    path.write_text(html, encoding="utf-8")
    return path
