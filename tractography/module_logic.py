"""Tractography workflow logic glue for SlicerDSpace."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from shared import default_task_output_dir_for_dependency
from tractography.act import ACTTrackingConfig, ACTTrackingResult, run_act_tracking
from tractography.act_prep import ACTPrepConfig, ACTPrepResult, run_act_prep
from tractography.bundle_parcellation import (
    BundleParcellationConfig,
    BundleParcellationResult,
    run_bundle_parcellation,
)
from tractography.csd import CSDActConfig, CSDActResult, run_csd_act
from tractography.dti import DTITrackingConfig, DTITrackingResult, run_dti_tracking
from tractography.saf import SAFConfig, SAFResult, run_saf
from tractography.tractcloud import TractCloudConfig, TractCloudResult, run_tractcloud_parcellation
from tractography.wma import WMAOrgAtlasConfig, WMAOrgAtlasResult, run_wma_org_atlas
from tractography.wmql import WMQLConfig, WMQLResult, run_wmql_bundle_segmentation


class TractographyLogicMixin:
    """Logic entrypoints for tractography-related workflows."""

    def runDTITrackingPipeline(
        self,
        *,
        subject_id: str,
        input_root: Path,
        output_dir: Optional[Path] = None,
        fit_method: str = "wls",
        tracking_method: str = "deterministic",
        output_format: str = "trk",
        step_size_mm: float = 0.5,
        max_angle_deg: float = 30.0,
        seed_density: int = 1,
        max_seed_voxels: Optional[int] = None,
        min_fa: float = 0.2,
        min_length_mm: float = 10.0,
        max_length_mm: float = 250.0,
        force_recompute: bool = False,
        execute: bool = False,
    ) -> DTITrackingResult:
        config = DTITrackingConfig(
            subject_id=subject_id,
            input_root=input_root,
            output_dir=output_dir,
            fit_method=fit_method,
            tracking_method=tracking_method,
            output_format=output_format,
            step_size_mm=step_size_mm,
            max_angle_deg=max_angle_deg,
            seed_density=seed_density,
            max_seed_voxels=max_seed_voxels,
            min_fa=min_fa,
            min_length_mm=min_length_mm,
            max_length_mm=max_length_mm,
            force_recompute=force_recompute,
            execute=execute,
        )
        return run_dti_tracking(config, log=self.log)

    def runCSDActPipeline(
        self,
        *,
        subject_id: str,
        dwi_path: Path,
        bval_path: Path,
        bvec_path: Path,
        mask_path: Path,
        five_tt_path: Path,
        gmwmi_path: Path,
        output_dir: Path,
        algorithm: str = "iFOD2",
        select_streamlines: int = 1_000_000,
        step_size_mm: Optional[float] = None,
        cutoff: Optional[float] = None,
        use_sift2: bool = True,
        nthreads: Optional[int] = None,
        execute: bool = False,
    ) -> CSDActResult:
        config = CSDActConfig(
            subject_id=subject_id,
            dwi_path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            five_tt_path=five_tt_path,
            gmwmi_path=gmwmi_path,
            output_dir=output_dir,
            algorithm=algorithm,
            select_streamlines=select_streamlines,
            step_size_mm=step_size_mm,
            cutoff=cutoff,
            use_sift2=use_sift2,
            # Off (None) by default: MRtrix3 keeps using its own default
            # thread count. See CSDActConfig.nthreads for why pinning this
            # does not, by itself, make tckgen bit-reproducible.
            nthreads=nthreads,
            execute=execute,
        )
        result = run_csd_act(config, log=self.log)
        self.log(f"[CSD/ACT] Manifest: {result.manifest_path}")
        self.log(f"[CSD/ACT] QC: {result.qc_path}")
        return result

    def runACTTrackingPipeline(
        self,
        *,
        subject_id: str,
        parcellation_path: Path,
        output_dir: Optional[Path] = None,
        tractography_path: Optional[Path] = None,
        dwi_path: Optional[Path] = None,
        bval_path: Optional[Path] = None,
        bvec_path: Optional[Path] = None,
        mask_path: Optional[Path] = None,
        five_tt_path: Optional[Path] = None,
        gmwmi_path: Optional[Path] = None,
        wm_fod_path: Optional[Path] = None,
        algorithm: str = "iFOD2",
        seed_count: int = 1_000_000,
        sift2: bool = True,
        execute: bool = False,
    ) -> ACTTrackingResult:
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else default_task_output_dir_for_dependency(parcellation_path, "tractography_act")
        )
        config = ACTTrackingConfig(
            subject_id=subject_id,
            parcellation_path=parcellation_path,
            output_dir=resolved_output_dir,
            tractography_path=tractography_path,
            dwi_path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            five_tt_path=five_tt_path,
            gmwmi_path=gmwmi_path,
            wm_fod_path=wm_fod_path,
            algorithm=algorithm,
            seed_count=seed_count,
            sift2=sift2,
            execute=execute,
        )
        return run_act_tracking(config, log=self.log)

    def runACTPrepPipeline(
        self,
        *,
        subject_id: str,
        segmentation_path: Path,
        output_dir: Optional[Path] = None,
        wm_label: int = 1,
        gm_label: int = 2,
        csf_label: int = 3,
        execute: bool = False,
        parcellation_path: Optional[Path] = None,
        allow_ddseg_only: bool = False,
    ) -> ACTPrepResult:
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else default_task_output_dir_for_dependency(segmentation_path, "act_prep")
        )
        config = ACTPrepConfig(
            subject_id=subject_id,
            segmentation_path=segmentation_path,
            output_dir=resolved_output_dir,
            wm_label=wm_label,
            gm_label=gm_label,
            csf_label=csf_label,
            execute=execute,
            parcellation_path=parcellation_path,
            allow_ddseg_only=allow_ddseg_only,
        )
        result = run_act_prep(config, log=self.log)
        self.log(f"[ACT Prep] 5TT: {result.five_tt_path}")
        self.log(f"[ACT Prep] GMWMI: {result.gmwmi_path}")
        return result

    def runWMAOrgAtlasPipeline(
        self,
        *,
        subject_id: str,
        input_tractography: Path,
        atlas_dir: Path,
        output_dir: Optional[Path] = None,
        slicer_executable: Path,
        wma_root: Optional[Path] = None,
        transform_path: Optional[Path] = None,
        registration_mode: str = "rig",
        num_threads: int = 1,
        use_virtual_x: bool = False,
        export_diffusion_measures: bool = False,
        fiber_tract_measurements_cli: Optional[Path] = None,
        clean_files: int = 0,
        execute: bool = False,
    ) -> WMAOrgAtlasResult:
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else default_task_output_dir_for_dependency(input_tractography, "wma_org_atlas")
        )
        config = WMAOrgAtlasConfig(
            subject_id=subject_id,
            input_tractography=input_tractography,
            atlas_dir=atlas_dir,
            output_dir=resolved_output_dir,
            slicer_executable=slicer_executable,
            wma_root=wma_root,
            transform_path=transform_path,
            registration_mode=registration_mode,
            num_threads=num_threads,
            use_virtual_x=use_virtual_x,
            export_diffusion_measures=export_diffusion_measures,
            fiber_tract_measurements_cli=fiber_tract_measurements_cli,
            clean_files=clean_files,
            execute=execute,
        )
        return run_wma_org_atlas(config, log=self.log)

    def runTractCloudParcellationPipeline(
        self,
        *,
        subject_id: str,
        input_tractography: Path,
        output_dir: Optional[Path] = None,
        tractcloud_root: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        include_other: bool = False,
        device: str = "cpu",
        batch_size: int = 2048,
        data_dir: Optional[Path] = None,
        create_mrb: bool = False,
        execute: bool = False,
    ) -> TractCloudResult:
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else default_task_output_dir_for_dependency(input_tractography, "tractcloud")
        )
        config = TractCloudConfig(
            subject_id=subject_id,
            input_tractography=input_tractography,
            output_dir=resolved_output_dir,
            tractcloud_root=tractcloud_root,
            python_executable=python_executable,
            include_other=include_other,
            device=device,
            batch_size=batch_size,
            data_dir=data_dir,
            create_mrb=create_mrb,
            execute=execute,
        )
        return run_tractcloud_parcellation(config, log=self.log)

    def runWMQLBundleSegmentationPipeline(
        self,
        *,
        subject_id: str,
        input_tractography: Path,
        atlas_path: Path,
        output_dir: Optional[Path] = None,
        query_path: Optional[Path] = None,
        query_selection: tuple[str, ...] = (),
        tract_querier_root: Optional[Path] = None,
        tract_querier_executable: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        include_dirs: tuple[Path, ...] = (),
        threshold: float = 2.0,
        length_threshold: float = 0.0,
        execute: bool = False,
    ) -> WMQLResult:
        resolved_output_dir = (
            output_dir
            if output_dir is not None
            else default_task_output_dir_for_dependency(input_tractography, "wmql")
        )
        config = WMQLConfig(
            subject_id=subject_id,
            input_tractography=input_tractography,
            atlas_path=atlas_path,
            output_dir=resolved_output_dir,
            query_path=query_path,
            query_selection=query_selection,
            tract_querier_root=tract_querier_root,
            tract_querier_executable=tract_querier_executable,
            python_executable=python_executable,
            include_dirs=include_dirs,
            threshold=threshold,
            length_threshold=length_threshold,
            execute=execute,
        )
        result = run_wmql_bundle_segmentation(config, log=self.log)
        self.log(f"[WMQL] Manifest: {result.manifest_path}")
        self.log(f"[WMQL] QC: {result.qc_path}")
        return result

    def runSAFPipeline(
        self,
        *,
        subject_id: str,
        input_root: Path,
        fs_dir: Path,
        fod_path: Path,
        output_dir: Optional[Path] = None,
        fa_path: Optional[Path] = None,
        pregenerated_tractogram: Optional[Path] = None,
        streamline_weights: Optional[Path] = None,
        tckgen_options: str = "",
        mrtrix_bin_dir: Optional[Path] = None,
        mrtrix_prefix: str = "",
        notrunc: bool = False,
        fast_intersection: bool = True,
        gwg_hard: bool = False,
        pialfilter: bool = False,
        remesh_seeds: bool | float = False,
        maxlen: Optional[float] = 90.0,
        scalar2surf: tuple[Path, ...] = (),
        force_recompute: bool = False,
        diagnostics: bool = False,
        execute: bool = False,
    ) -> SAFResult:
        config = SAFConfig(
            subject_id=subject_id,
            input_root=input_root,
            fs_dir=fs_dir,
            fod_path=fod_path,
            output_dir=output_dir,
            fa_path=fa_path,
            pregenerated_tractogram=pregenerated_tractogram,
            streamline_weights=streamline_weights,
            tckgen_options=tckgen_options,
            mrtrix_bin_dir=mrtrix_bin_dir,
            mrtrix_prefix=mrtrix_prefix,
            notrunc=notrunc,
            fast_intersection=fast_intersection,
            gwg_hard=gwg_hard,
            pialfilter=pialfilter,
            remesh_seeds=remesh_seeds,
            maxlen=maxlen,
            scalar2surf=scalar2surf,
            force_recompute=force_recompute,
            diagnostics=diagnostics,
            execute=execute,
        )
        result = run_saf(config, log=self.log)
        self.log(f"[SAF] Manifest: {result.manifest_path}")
        self.log(f"[SAF] QC: {result.qc_path}")
        self.log(f"[SAF] Tractogram: {result.saf_tractogram}")
        load_path = result.saf_vtk_path if result.saf_vtk_path and result.saf_vtk_path.is_file() else result.saf_tractogram
        if load_path.is_file():
            self._load_fiber_bundle_node(load_path, f"{subject_id}_SAF")
        return result

    def runBundleParcellationPipeline(
        self,
        *,
        config: BundleParcellationConfig,
    ) -> BundleParcellationResult:
        return run_bundle_parcellation(config, log=self.log)
