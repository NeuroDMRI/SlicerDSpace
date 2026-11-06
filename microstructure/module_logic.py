"""Microstructure workflow logic glue for SlicerDSpace."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shared import ensure_native_dti_bundle, find_subject_root, resolve_preprocessed_input_bundle

MICROSTRUCTURE_PYTHON_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("nibabel>=5.0", "nibabel"),
    ("dipy>=1.10,<1.13", "dipy"),
    ("dmri-amico", "amico"),
)


class MicrostructureLogicMixin:
    """Logic entrypoints for the microstructure workflow."""

    def setupMicrostructureRequirements(self) -> None:
        self._ensure_python_packages(MICROSTRUCTURE_PYTHON_REQUIREMENTS)

    def runMicrostructurePipeline(
        self,
        *,
        subject_id: str,
        input_root: Path,
        output_dir: Optional[Path],
        roi_label_map: Optional[Path] = None,
        fit_method: str = "wls",
        run_free_water: bool = False,
        run_noddi: bool = False,
        run_dki: bool = False,
        dki_fit_method: str = "MKCURVE",
        dki_matlab_executable: Optional[Path] = None,
        dki_mkcurve_root: Optional[Path] = None,
        force_recompute: bool = False,
    ) -> None:
        from microstructure import MicrostructureConfig, run_microstructure

        input_root = input_root.expanduser().resolve()
        raw_bundle = find_subject_root(input_root, subject_id)
        resolved_output_dir = self._resolve_optional_output_dir(
            output_dir,
            subject_root=raw_bundle.subject_root,
            task_name="microstructure",
        )
        analysis_input_bundle = resolve_preprocessed_input_bundle(
            raw_bundle,
            output_root=resolved_output_dir.parent if resolved_output_dir.name == "02_microstructure" else None,
        )
        native_bundle = ensure_native_dti_bundle(
            analysis_input_bundle,
            fit_method=fit_method,
            reuse=True,
            log=self.log,
        )
        config = MicrostructureConfig(
            subject_id=subject_id,
            raw_input_bundle=analysis_input_bundle,
            native_bundle=native_bundle,
            output_dir=resolved_output_dir,
            roi_label_map=roi_label_map.expanduser().resolve() if roi_label_map is not None else None,
            run_free_water=bool(run_free_water),
            run_noddi=bool(run_noddi),
            run_dki=bool(run_dki),
            dki_fit_method=dki_fit_method,
            dki_matlab_executable=dki_matlab_executable,
            dki_mkcurve_root=dki_mkcurve_root,
            force_recompute=bool(force_recompute),
        )
        result = run_microstructure(config, log=self.log)
        self.log(f"[Microstructure] FA: {result.fa_path}")
        self.log(f"[Microstructure] MD: {result.md_path}")
        self.log(f"[Microstructure] AD: {result.ad_path}")
        self.log(f"[Microstructure] RD: {result.rd_path}")
        self.log(f"[Microstructure] L1: {result.l1_path}")
        self.log(f"[Microstructure] L2: {result.l2_path}")
        self.log(f"[Microstructure] L3: {result.l3_path}")
        self.log(f"[Microstructure] Trace: {result.trace_path}")
        self._load_scalar_volume_node(result.fa_path, f"{subject_id}_FA")
        self._load_scalar_volume_node(result.md_path, f"{subject_id}_MD")
        self._load_scalar_volume_node(result.ad_path, f"{subject_id}_AD")
        self._load_scalar_volume_node(result.rd_path, f"{subject_id}_RD")
        self._load_scalar_volume_node(result.l1_path, f"{subject_id}_L1")
        self._load_scalar_volume_node(result.l2_path, f"{subject_id}_L2")
        self._load_scalar_volume_node(result.l3_path, f"{subject_id}_L3")
        self._load_scalar_volume_node(result.trace_path, f"{subject_id}_Trace")
        if run_free_water and result.fwf_path.is_file():
            self.log(f"[Microstructure] Free Water Fraction: {result.fwf_path}")
            self._load_scalar_volume_node(result.fwf_path, f"{subject_id}_FWF")
        if run_free_water and result.fw_tissue_fa_path.is_file():
            self.log(f"[Microstructure] Free Water Tissue FA: {result.fw_tissue_fa_path}")
            self._load_scalar_volume_node(result.fw_tissue_fa_path, f"{subject_id}_FW_TissueFA")
        if run_free_water and result.fw_tissue_md_path.is_file():
            self.log(f"[Microstructure] Free Water Tissue MD: {result.fw_tissue_md_path}")
            self._load_scalar_volume_node(result.fw_tissue_md_path, f"{subject_id}_FW_TissueMD")
        if run_noddi and result.noddi_ndi_path.is_file():
            self.log(f"[Microstructure] NODDI NDI: {result.noddi_ndi_path}")
            self._load_scalar_volume_node(result.noddi_ndi_path, f"{subject_id}_NODDI_NDI")
        if run_noddi and result.noddi_odi_path.is_file():
            self.log(f"[Microstructure] NODDI ODI: {result.noddi_odi_path}")
            self._load_scalar_volume_node(result.noddi_odi_path, f"{subject_id}_NODDI_ODI")
        if run_noddi and result.noddi_fwf_path.is_file():
            self.log(f"[Microstructure] NODDI FWF: {result.noddi_fwf_path}")
            self._load_scalar_volume_node(result.noddi_fwf_path, f"{subject_id}_NODDI_FWF")
        if result.dki_result is not None:
            from microstructure.dki import DKI_METRICS
            for key in DKI_METRICS:
                self._load_scalar_volume_node(result.dki_result.outputs[key], f"{subject_id}_DKI_{key.upper()}")
        if result.roi_stats_csv_path is not None:
            self.log(f"[Microstructure] ROI-wise stats CSV: {result.roi_stats_csv_path}")
            self._load_table_node(result.roi_stats_csv_path, f"{subject_id}_ROIWise_Microstructure")
        else:
            self.log("[Microstructure] ROI-wise statistics were skipped because no ROI label map was available.")
