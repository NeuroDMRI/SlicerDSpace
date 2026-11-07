"""Refresh QC for a real completed MRtrix/FSL run without rerunning eddy."""

import hashlib
import json
import os
from pathlib import Path
import traceback
from dataclasses import replace

import qt
import slicer

ROOT = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def start():
    record = {"kind": "real_preprocessing_cache_refresh", "passed": False}
    try:
        slicer.util.selectModule("SlicerDSpace")
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing
        from preprocessing.pipeline import (
            _planned_paths, _preprocessing_cache_key, inspect_preprocessing_backend,
        )
        from shared.pipeline import find_subject_root

        config = json.loads(Path(os.environ["SLICERDSPACE_TASKS_CONFIG"]).read_text())
        case = next(c for c in config["cases"] if c["name"] == "fsl_preprocessing")
        deps = config["external_dependencies"]
        settings = DwiPreprocessingConfig(
            subject_id=case["subject_id"], input_root=Path(case["subject_root"]),
            output_dir=Path(case["output_root"]) / "01_preproc",
            execute=True, force=False,
            mrtrix_bin_dir=Path(deps["mrtrix_bin_dir"]),
            fsl_bin_dir=Path(deps["preprocessing_fsl_bin_dir"]),
            reverse_phase_b0=Path(case["reverse_b0"]),
            phase_encoding_direction=case["pe"], readout_time=case["readout"],
            eddy_threads=4, eddy_second_level_model="linear",
        )
        raw = find_subject_root(settings.input_root, settings.subject_id,
                                mrtrix_bin_dir=settings.mrtrix_bin_dir)
        paths = _planned_paths(settings.output_dir, settings.subject_id)
        previous = json.loads(paths["manifest"].read_text())
        assert previous["status"] in {"completed", "cached"}
        assert previous["cache_key"] == _preprocessing_cache_key(
            raw, settings, inspect_preprocessing_backend(settings)
        ), "Refuse a cache miss: this smoke must never launch eddy"
        products = [paths[key] for key in ("final_dwi", "final_bval", "final_bvec", "final_mask")]
        before = {str(path): digest(path) for path in products}
        before_mtime = {str(path): path.stat().st_mtime_ns for path in products}
        from shared import ProductOrchestratorConfig, plan_subject_products
        plan_config = ProductOrchestratorConfig(
            subject_id=case["subject_id"], subject_root=Path(case["subject_root"]),
            output_root=Path(case["output_root"]), targets=("preprocessing",),
            execute=False,
            preprocessing_mrtrix_bin_dir=settings.mrtrix_bin_dir,
            preprocessing_fsl_bin_dir=settings.fsl_bin_dir,
            preprocessing_reverse_phase_b0=settings.reverse_phase_b0,
            preprocessing_phase_encoding_direction=settings.phase_encoding_direction,
            preprocessing_readout_time=settings.readout_time,
            preprocessing_eddy_threads=settings.eddy_threads,
            preprocessing_eddy_second_level_model=settings.eddy_second_level_model,
        )
        plans = {
            "matching": plan_subject_products(plan_config, write=False).items[0].status,
            "changed_gibbs": plan_subject_products(replace(plan_config, preprocessing_dipy_run_gibbs=False), write=False).items[0].status,
            "changed_parameters": plan_subject_products(replace(plan_config, preprocessing_eddy_second_level_model="none"), write=False).items[0].status,
            "forced": plan_subject_products(replace(plan_config, preprocessing_force=True), write=False).items[0].status,
        }
        assert plans == {"matching": "skipped", "changed_gibbs": "will_run", "changed_parameters": "will_run", "forced": "will_run"}, plans
        (ROOT / "preprocessing_qc_before.json").write_bytes(paths["qc"].read_bytes())
        result = run_dwi_preprocessing(settings, log=lambda line: print(line, flush=True))
        assert result.status == "cached", result.status
        after = {str(path): digest(path) for path in products}
        assert before == after
        assert before_mtime == {str(path): path.stat().st_mtime_ns for path in products}
        qc = json.loads(result.qc_path.read_text())
        assert qc["warnings"] == ["Bias-field correction was not applied by the selected backend."]
        assert qc["backend_details"]["denoise"]["status"] == "completed"
        assert qc["backend_details"]["susceptibility_distortion_correction"]["status"] == "completed"
        assert qc["mask_generation"]["method"] == "dwi2mask_corrected_dwi"
        (ROOT / "preprocessing_qc_after.json").write_bytes(result.qc_path.read_bytes())
        record.update(passed=True, status=result.status, product_sha256=after,
                      product_mtime_unchanged=True, qc=str(result.qc_path), warnings=qc["warnings"], plans=plans)
        slicer.util.mainWindow().grab().save(str(ROOT / "cache_refresh.png"))
    except Exception as exc:
        traceback.print_exc()
        record["error"] = str(exc)
    (ROOT / "preprocessing_cache.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PREPROCESSING_CACHE=" + json.dumps(record), flush=True)
    slicer.app.exit(0 if record["passed"] else 2)


qt.QTimer.singleShot(5000, start)
