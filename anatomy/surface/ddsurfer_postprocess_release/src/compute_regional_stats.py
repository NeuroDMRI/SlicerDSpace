#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import datetime

import nibabel as nib
import numpy as np


EXCLUDED_ROI_NAMES = {
    "unknown",
    "???",
    "corpuscallosum",
    "freesurfer_defined_medial_wall",
}


def parse_ctab_file(filepath):
    """Parse a FreeSurfer-style CTAB/LUT file."""
    ctab = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CTAB file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                label_id = int(parts[0])
                r_val = int(parts[-4])
                g_val = int(parts[-3])
                b_val = int(parts[-2])
                a_val = int(parts[-1])
                struct_name = " ".join(parts[1:-4])
                ctab[label_id] = (struct_name, r_val, g_val, b_val, a_val)
            except ValueError:
                continue

    if not ctab:
        print(f"WARN: No entries parsed from CTAB file: {filepath}")
    return ctab


def load_morph_data_if_exists(file_path, n_vertices_expected=None, filename_for_log="Unknown"):
    """Load FreeSurfer morph data if present and valid."""
    if file_path and os.path.exists(file_path):
        try:
            data = nib.freesurfer.io.read_morph_data(file_path)
            if n_vertices_expected is not None and len(data) != n_vertices_expected:
                print(
                    f"WARN: {filename_for_log} vertex count mismatch "
                    f"({len(data)} != {n_vertices_expected}). Skipping."
                )
                return None
            print(f"INFO: Loaded {filename_for_log}: {file_path}")
            return data
        except Exception as exc:
            print(f"WARN: Failed to load {filename_for_log} ({file_path}): {exc}")
            return None
    elif file_path:
        print(f"WARN: File not found for {filename_for_log}: {file_path}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute region-wise surface stats from a FreeSurfer annotation."
    )
    parser.add_argument("--annot_file", required=True, help="Input .annot file.")
    parser.add_argument("--ctab_file", required=True, help="CTAB file matching the annotation.")
    parser.add_argument("--hemi", required=True, choices=["lh", "rh"], help="Hemisphere.")
    parser.add_argument("--out_stats_file", required=True, help="Output stats file path.")

    parser.add_argument("--thickness", help="Thickness overlay (.thickness).")
    parser.add_argument("--area", help="Area overlay (.area or .area.pial).")
    parser.add_argument("--curv", help="Mean curvature overlay (.curv).")
    parser.add_argument("--sulc", help="Sulcal depth overlay (.sulc).")
    parser.add_argument("--gauscurv", help="Gaussian curvature overlay (.K).")
    parser.add_argument(
        "--dti_metrics",
        nargs="*",
        default=[],
        help="Custom metrics in the format 'MetricName:/path/to/metric.mgh'.",
    )

    args = parser.parse_args()

    script_name = os.path.basename(__file__)
    print(f"--- [{script_name}] Regional surface stats ({datetime.now():%Y-%m-%d %H:%M:%S}) ---")
    print(f"Annotation: {args.annot_file}")
    print(f"CTAB:       {args.ctab_file}")
    print(f"Hemi:       {args.hemi}")

    if not os.path.exists(args.annot_file):
        print(f"ERROR: Annotation file not found: {args.annot_file}")
        sys.exit(1)

    try:
        vertex_label_indices, annot_ctab, struct_names_bytes = nib.freesurfer.io.read_annot(
            args.annot_file, orig_ids=False
        )
        struct_names = [name.decode("utf-8") for name in struct_names_bytes]
        n_vertices = len(vertex_label_indices)
        print(f"INFO: Loaded labels for {n_vertices} vertices")

        ctab_data = parse_ctab_file(args.ctab_file)
        if not ctab_data:
            print(f"ERROR: Failed to load CTAB data from {args.ctab_file}")
            sys.exit(1)
        print(f"INFO: Loaded {len(ctab_data)} CTAB entries")

        data_overlays = {
            "ThickAvg": load_morph_data_if_exists(args.thickness, n_vertices, "Thickness"),
            "SurfArea_per_vertex": load_morph_data_if_exists(args.area, n_vertices, "Area"),
            "MeanCurv": load_morph_data_if_exists(args.curv, n_vertices, "MeanCurv"),
            "MeanSulc": load_morph_data_if_exists(args.sulc, n_vertices, "Sulc"),
            "GausCurv": load_morph_data_if_exists(args.gauscurv, n_vertices, "GausCurv"),
        }

        custom_metrics_col_names = []
        for item in args.dti_metrics:
            try:
                name, path = item.split(":", 1)
                metric_data = load_morph_data_if_exists(path, n_vertices, name)
                if metric_data is not None:
                    col_name = f"Mean{name}"
                    data_overlays[col_name] = metric_data
                    custom_metrics_col_names.append(col_name)
            except ValueError:
                print(f"WARN: Invalid metric format '{item}'. Skipping.")

        header_parts = ["StructName", "NumVert", "SurfArea", "GrayVol"]
        if data_overlays.get("ThickAvg") is not None:
            header_parts.extend(["ThickAvg", "ThickStd"])
        if data_overlays.get("MeanCurv") is not None:
            header_parts.append("MeanCurv")
        if data_overlays.get("GausCurv") is not None:
            header_parts.append("GausCurv")
        header_parts.extend(custom_metrics_col_names)

        results = []
        processed_roi_names = set()
        missing_ctab_ids = []

        # ``read_annot(..., orig_ids=False)`` maps packed annotation values to
        # color-table row indices.  This is important for legacy atlases such
        # as PALS, whose table may contain multiple names with the same RGB
        # (and therefore the same packed ID).  A name-keyed CTAB dictionary
        # loses those duplicates, while row indices preserve nibabel's actual
        # vertex-to-structure assignment.
        roi_indices = sorted(int(value) for value in np.unique(vertex_label_indices) if int(value) >= 0)
        for roi_index in roi_indices:
            if roi_index >= len(struct_names) or roi_index >= len(annot_ctab):
                raise RuntimeError(f"Annotation references invalid color-table row {roi_index}.")
            roi_name = struct_names[roi_index]
            if roi_name in processed_roi_names:
                continue
            if roi_name.strip().lower() in EXCLUDED_ROI_NAMES:
                continue

            packed_label_id = int(annot_ctab[roi_index, 4])
            if packed_label_id not in ctab_data:
                missing_ctab_ids.append((roi_name, packed_label_id))
                continue

            roi_vertices_mask = vertex_label_indices == roi_index
            roi_vertex_indices = np.where(roi_vertices_mask)[0]
            num_vert_in_roi = len(roi_vertex_indices)
            if num_vert_in_roi == 0:
                continue

            row_dict = {"StructName": roi_name, "NumVert": num_vert_in_roi}

            surf_area_roi_total = 0.0
            if data_overlays.get("SurfArea_per_vertex") is not None:
                surf_area_roi_total = np.sum(data_overlays["SurfArea_per_vertex"][roi_vertex_indices])
            row_dict["SurfArea"] = f"{surf_area_roi_total:.1f}"

            thick_avg_roi = np.nan
            if data_overlays.get("ThickAvg") is not None:
                roi_thickness_values = data_overlays["ThickAvg"][roi_vertex_indices]
                if len(roi_thickness_values) > 0:
                    thick_avg_roi = np.nanmean(roi_thickness_values)
                    thick_std_roi = np.nanstd(roi_thickness_values)
                    row_dict["ThickAvg"] = f"{thick_avg_roi:.3f}"
                    row_dict["ThickStd"] = f"{thick_std_roi:.3f}"

            gray_vol_roi_approx = 0.0
            if not np.isnan(thick_avg_roi) and surf_area_roi_total > 0:
                gray_vol_roi_approx = thick_avg_roi * surf_area_roi_total
            row_dict["GrayVol"] = f"{gray_vol_roi_approx:.1f}"

            if data_overlays.get("MeanCurv") is not None:
                row_dict["MeanCurv"] = f"{np.nanmean(data_overlays['MeanCurv'][roi_vertex_indices]):.3f}"
            if data_overlays.get("GausCurv") is not None:
                row_dict["GausCurv"] = f"{np.nanmean(data_overlays['GausCurv'][roi_vertex_indices]):.3f}"

            for metric_col_name in custom_metrics_col_names:
                if data_overlays.get(metric_col_name) is not None:
                    row_dict[metric_col_name] = (
                        f"{np.nanmean(data_overlays[metric_col_name][roi_vertex_indices]):.4f}"
                    )

            results.append(row_dict)
            processed_roi_names.add(roi_name)

        if missing_ctab_ids:
            missing_text = ", ".join(
                f"{name}={label_id}" for name, label_id in sorted(set(missing_ctab_ids))
            )
            raise RuntimeError(
                "CTAB does not contain packed annotation IDs for: " + missing_text
            )
        if not results:
            raise RuntimeError(
                "Regional statistics produced no non-empty ROI rows; annotation/CTAB IDs do not match."
            )

        os.makedirs(os.path.dirname(args.out_stats_file), exist_ok=True)
        with open(args.out_stats_file, "w", encoding="utf-8") as f:
            f.write(f"# Measure \"{os.path.basename(args.annot_file)}\", {args.hemi} hemisphere\n")
            f.write(f"# Annotation File: {args.annot_file}\n")
            f.write(f"# ColorTable File: {args.ctab_file}\n")
            f.write(f"# Generated by: {script_name} on {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            subjects_dir_env = os.environ.get("SUBJECTS_DIR", "unknown_subjects_dir")
            subject_id_env = os.environ.get("SUBJECT_ID", "unknown_subject_id")
            f.write(f"# SUBJECTS_DIR {subjects_dir_env}\n")
            f.write(f"# SUBJECTID {subject_id_env}\n")
            f.write(f"# Hemisphere {args.hemi}\n")

            formatted_header_cols = []
            for col in header_parts:
                if col == "StructName":
                    formatted_header_cols.append(col)
                else:
                    formatted_header_cols.append(f"{args.hemi}.{col}")

            f.write(f"# ColHeaders {' '.join(formatted_header_cols)}\n")

            for row_data in results:
                line_values = [str(row_data.get(col_key, "0.000")) for col_key in header_parts]
                f.write(" ".join(line_values) + "\n")

        print(f"INFO: Regional stats saved to: {args.out_stats_file}")
        print(f"INFO: Regional stats rows: {len(results)}")
        print(f"--- [{script_name}] Done ---")

    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to compute regional stats: {exc}")
        raise


if __name__ == "__main__":
    main()
