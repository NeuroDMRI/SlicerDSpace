#!/usr/bin/env python3
"""Independently check regional postprocessing against its frozen parent files."""

import argparse
import csv
import itertools
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.region_analysis import inspect_run, native_streamlines, read_json, sha256, write_json


def validate(path):
    start = time.monotonic()
    manifest = read_json(path)
    if manifest.get("status") != "completed":
        raise ValueError("Only a completed regional job can be accepted")
    for category in ("source_sha256", "artifact_sha256"):
        for source, digest in manifest[category].items():
            if sha256(source) != digest:
                raise ValueError("Changed file: " + source)
    request = manifest["request"]
    catalog = inspect_run(
        request["source_manifest"], atlas_override=request.get("atlas_override"), lut_path=request.get("lut_path")
    )
    selected = manifest.get("selected_region_ids", sorted(set(request["selected_labels"])))
    folder = Path(path).parent
    atlas = nib.load(catalog["atlas"])
    labels = np.asarray(atlas.dataobj)
    actual_mask = nib.load(folder / "selected_regions.nii.gz")
    np.testing.assert_array_equal(actual_mask.affine, atlas.affine)
    np.testing.assert_array_equal(np.asarray(actual_mask.dataobj), np.where(np.isin(labels, selected), labels, 0))
    checks = {"source_and_artifact_hashes": True, "selected_label_map_exact": True}
    statistics = []
    if request.get("metrics"):
        rows = read_json(folder / "selected_microstructure.json")["rows"]
        if [row["label_id"] for row in rows] != selected:
            raise AssertionError("Selected-region statistics order mismatch")
        brain = np.asarray(nib.load(catalog["brain_mask"]).dataobj) > 0
        for key in request["metrics"]:
            spec = catalog["metrics"][key]
            data = nib.load(spec["path"]).get_fdata(dtype=np.float64)
            valid = np.isfinite(data) & brain
            if spec["valid_mask"]:
                valid &= np.asarray(nib.load(spec["valid_mask"]).dataobj) > 0
            for row in rows:
                mask = labels == row["label_id"]
                values = data[mask & valid]
                if row[key + "_valid_voxel_count"] != len(values):
                    raise AssertionError("Valid voxel count differs")
                if row[key + "_excluded_voxel_count"] != int(mask.sum()) - len(values):
                    raise AssertionError("Excluded voxel count differs")
                for statistic, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max),
                                            ("median", np.median), ("q25", lambda x: np.percentile(x, 25)),
                                            ("q75", lambda x: np.percentile(x, 75)),
                                            ("iqr", lambda x: np.subtract(*np.percentile(x, (75, 25))))):
                    if key + "_" + statistic not in row:
                        continue
                    actual = row[key + "_" + statistic]
                    if len(values):
                        if not np.isclose(actual, function(values), rtol=2e-6, atol=1e-9):
                            raise AssertionError(f"{key} {statistic} differs")
                    elif actual is not None:
                        raise AssertionError("An invalid estimate was filled instead of null")
                statistics.append(
                    dict(label_id=row["label_id"], metric=key, valid_voxels=len(values), mean=row[key + "_mean"])
                )
        checks["independent_boolean_roi_statistics"] = True
    selected_count = None
    maximum_coordinate_error_mm = 0.0
    if request.get("tractogram"):
        source = catalog["tractograms"][request["tractogram"]]
        info = read_json(folder / "selected_tractography.json")
        with Path(info["source_indices"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        indices = [int(row["source_index"]) for row in rows]
        if indices != sorted(set(indices)):
            raise AssertionError("Source indices are not unique and ordered")
        if [int(row["selected_index"]) for row in rows] != list(range(len(rows))):
            raise AssertionError("Selected indices are not consecutive")
        actual = iter(native_streamlines(info["output"])) if info["output"] else iter(())
        wanted = iter(indices)
        current = next(wanted, None)
        input_count = 0
        sentinel = object()
        inverse = np.linalg.inv(atlas.affine)
        expected_indices = []
        required = set(request.get("include_labels", []))
        excluded = set(request.get("exclude_labels", []))
        mode = request.get("tract_rule", "through_any")
        for index, line in enumerate(native_streamlines(source["path"])):
            input_count += 1
            if mode.startswith("endpoint") and len(line) >= 2:
                ends = np.floor(nib.affines.apply_affine(inverse, line[[0, -1]]) + 0.5).astype(int)
                end_labels = [int(labels[tuple(point)]) if np.all(point >= 0) and np.all(point < labels.shape) else 0
                              for point in ends]
                a = request["selected_labels"]
                b = request.get("target_labels", [])
                if mode == "endpoint_pair":
                    accepted = (end_labels[0] in a and end_labels[1] in b) or (end_labels[1] in a and end_labels[0] in b)
                else:
                    hits = [value in a for value in end_labels]
                    accepted = all(hits) if mode == "endpoint_both" else any(hits)
                if accepted and (required or excluded):
                    from dipy.tracking.utils import target_line_based

                    accepted = all(list(target_line_based([line], atlas.affine, labels == label)) for label in required)
                    if accepted and excluded:
                        accepted = not list(target_line_based([line], atlas.affine, np.isin(labels, list(excluded))))
                if accepted:
                    expected_indices.append(index)
            if index == current:
                result = next(actual, sentinel)
                if result is sentinel or result.shape != line.shape:
                    raise AssertionError("Selected geometry is missing or clipped")
                error = float(np.max(np.abs(line - result)))
                maximum_coordinate_error_mm = max(maximum_coordinate_error_mm, error)
                if error != 0:
                    raise AssertionError("Selected coordinates differ from source")
                current = next(wanted, None)
        if current is not None or next(actual, sentinel) is not sentinel:
            raise AssertionError("Unexpected/missing selected geometry")
        if input_count != info["input_streamlines"] or len(rows) != info["selected_streamlines"]:
            raise AssertionError("Tractogram counts differ")
        if mode.startswith("endpoint") and indices != expected_indices:
            raise AssertionError("Endpoint selection membership differs from independent lookup")
        if mode.startswith("endpoint"):
            checks["endpoint_selection_membership"] = True
        if source.get("weights"):
            weights = np.loadtxt(source["weights"], ndmin=1)
            np.testing.assert_array_equal([float(row["sift2_weight"]) for row in rows], weights[indices])
            checks["selected_sift2_weights_exact"] = True
        selected_count = len(rows)
        checks["all_selected_geometry_and_source_indices_exact"] = True
    if request.get("tissue"):
        data = np.asarray(nib.load(catalog["tissue"]["path"]).dataobj)
        with (folder / "selected_tissue_statistics.csv").open(newline="") as stream:
            tissue_rows = list(csv.DictReader(stream))
        if [int(row["label_id"]) for row in tissue_rows] != selected:
            raise AssertionError("Tissue region IDs differ")
        for row in tissue_rows:
            mask = labels == int(row["label_id"])
            for code, name in enumerate(("unclassified", "wm", "gm", "csf")):
                count = int(np.count_nonzero(data[mask] == code))
                if int(row[name + "_voxels"]) != count or not np.isclose(float(row[name + "_fraction"]), count / mask.sum()):
                    raise AssertionError("Tissue composition differs")
        checks["independent_tissue_counts"] = True
    if request.get("surface"):
        with (folder / "selected_surface_statistics.csv").open(newline="") as stream:
            surface_rows = list(csv.DictReader(stream))
        expected_labels = [label for label in selected if str(label) in catalog["surfaces"]]
        if [int(row["label_id"]) for row in surface_rows] != expected_labels:
            raise AssertionError("Surface annotation mapping differs")
        for row in surface_rows:
            spec = catalog["surfaces"][row["label_id"]]
            vertex_labels, _, _ = nib.freesurfer.read_annot(spec["annotation"])
            mask = vertex_labels == spec["annotation_index"]
            thick = nib.freesurfer.read_morph_data(spec["thickness"]).astype(float)[mask]
            area = nib.freesurfer.read_morph_data(spec["area"]).astype(float)[mask]
            curv = nib.freesurfer.read_morph_data(spec["curv"]).astype(float)[mask]
            if int(row["vertex_count"]) != int(mask.sum()):
                raise AssertionError("Surface vertex count differs")
            for key, value in (("area_mm2", area.sum()), ("thickness_mean_mm", thick.mean()),
                               ("thickness_median_mm", np.median(thick)), ("thickness_std_mm", thick.std()),
                               ("curvature_mean_per_mm", curv.mean())):
                if not np.isclose(float(row[key]), value, rtol=1e-12, atol=1e-12):
                    raise AssertionError("Surface statistic differs: " + key)
        checks["independent_surface_arrays"] = True
    if request.get("connectome"):
        source = catalog["connectomes"][request["connectome"]]
        with Path(source["matrix"]).open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if header[:2] == ["source_node", "target_node"]:
                original = {tuple(sorted((int(row[0]), int(row[1])))): float(row[2]) for row in reader}
            else:
                dense = np.loadtxt(source["matrix"], delimiter=",", ndmin=2)
                with Path(source["nodes"]).open(newline="") as nodes:
                    node_labels = sorted(int(row["label"]) for row in csv.DictReader(nodes, delimiter="\t"))
                lookup = (
                    {label: i for i, label in enumerate(node_labels)}
                    if len(dense) == len(node_labels)
                    else {label: label - 1 for label in node_labels}
                )
                original = {
                    tuple(sorted((a, b))): float(dense[lookup[a], lookup[b]]) for a in selected for b in node_labels
                }
        with (folder / "selected_connectome_matrix.csv").open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            assert [int(value) for value in header[1:]] == selected
            for row, label in itertools.zip_longest(reader, selected):
                if row is None or label is None or int(row[0]) != label:
                    raise AssertionError("Selected matrix labels/order differ")
                if len(row[1:]) != len(selected):
                    raise AssertionError("Selected matrix row length differs")
                for partner, value in zip(selected, row[1:]):
                    if float(value) != original.get(tuple(sorted((label, partner))), 0.0):
                        raise AssertionError("A connectivity weight changed")
        with (folder / "selected_region_connections.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                a, b = int(row["selected_label"]), int(row["partner_label"])
                value = float(list(row.values())[-1])
                if a not in selected or value != original.get(tuple(sorted((a, b))), 0.0):
                    raise AssertionError("A regional connection changed")
        checks["induced_matrix_and_connection_weights_exact"] = True
    return dict(
        status="PASS",
        checks=checks,
        region_ids=selected,
        statistics=statistics,
        selected_streamlines=selected_count,
        maximum_coordinate_error_mm=maximum_coordinate_error_mm,
        postprocessing_runtime_seconds=manifest["runtime_seconds"],
        validation_runtime_seconds=time.monotonic() - start,
        scope="Postprocessing consistency only; no new fit, anatomical or cross-platform acceptance",
        human_review="pending",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.report, validate(args.manifest))
    print("REGIONAL_POSTPROCESSING_VALIDATION_PASS: " + str(args.report), flush=True)


if __name__ == "__main__":
    main()
