"""Regression tests for correctness, cache-reuse, locking, NRRD topup, launcher UTF-8, and UKF AVX2 fixes."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import nibabel as nib
from nibabel.nifti1 import Nifti1Extension
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _save_extended_scaled_nifti(
    path: Path,
    raw_int: np.ndarray,
    *,
    slope: float = 0.5,
    inter: float = 10.0,
    extension_bytes: int = 160,
    endianness: str = "<",
    nifti2: bool = False,
) -> Path:
    if nifti2:
        hdr = nib.Nifti2Header(endianness=endianness)
        img = nib.Nifti2Image(raw_int, np.eye(4), header=hdr)
    else:
        hdr = nib.Nifti1Header(endianness=endianness)
        img = nib.Nifti1Image(raw_int, np.eye(4), header=hdr)
    img.header.set_data_dtype(np.dtype(raw_int.dtype).newbyteorder(endianness))
    img.header.set_slope_inter(slope, inter)
    if extension_bytes > 0 and not nifti2:
        img.header.extensions.append(Nifti1Extension("comment", b"E" * extension_bytes))
    nib.save(img, str(path))
    return path


class NiftiExtendedHeaderAndScalingTests(unittest.TestCase):
    """[Item 1] NODDI, DTI, brain_mask, and diffusion_io preserve vox_offset > 352 and scl_slope/scl_inter."""

    def test_noddi_fast_read_matches_nibabel_cached_on_off_and_pigz(self) -> None:
        from microstructure import noddi_accel
        from shared.diffusion_io import ensure_cached_uncompressed_nifti

        rng = np.random.default_rng(123)
        raw_int = rng.integers(300, 1200, size=(6, 7, 5, 6), dtype=np.int16)
        mask = np.zeros((6, 7, 5), dtype=np.uint8)
        mask[1:5, 1:6, 1:4] = 1
        scheme = types.SimpleNamespace(nS=6, b0_count=2, b0_idx=[0, 3])

        for label, kwargs in (
            ("standard_unscaled", dict(slope=1.0, inter=0.0, extension_bytes=0, endianness="<", nifti2=False)),
            ("ext_scaled_le", dict(slope=0.5, inter=10.0, extension_bytes=160, endianness="<", nifti2=False)),
            ("ext_scaled_be", dict(slope=0.25, inter=-15.0, extension_bytes=160, endianness=">", nifti2=False)),
            ("nifti2_scaled", dict(slope=0.5, inter=10.0, extension_bytes=0, endianness="<", nifti2=True)),
        ):
            with self.subTest(case=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                inp = root / "input"
                inp.mkdir()
                dwi_path = _save_extended_scaled_nifti(inp / "dwi.nii.gz", raw_int, **kwargs)
                mask_path = inp / "mask.nii.gz"
                nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))

                ref_img = nib.load(str(dwi_path))
                if kwargs["extension_bytes"] > 0:
                    self.assertGreater(int(ref_img.dataobj.offset), 352)
                ref_4d = ref_img.get_fdata(dtype=np.float32)
                ref_y = ref_4d[mask == 1].copy()
                mean_b0 = np.mean(ref_y[:, [0, 3]], axis=1, dtype=np.float32)
                ref_y = (ref_y * (np.float32(1.0) / mean_b0)[:, None]).astype(np.float64)
                ref_y[ref_y < 0.0] = 0.0

                # 1) Cached ON
                with mock.patch.dict(
                    os.environ,
                    {"SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES": "1"},
                    clear=False,
                ):
                    os.environ.pop("SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE", None)
                    cached = ensure_cached_uncompressed_nifti(dwi_path)
                    self.assertIsNotNone(cached)
                    _, _, _, y_cached = noddi_accel._load_masked_normalized_dwi(
                        dwi_path=dwi_path, mask_path=mask_path, scheme=scheme
                    )
                np.testing.assert_allclose(y_cached, ref_y, rtol=0.0, atol=1e-6)

                # 2) Cached OFF (nibabel fallback)
                with mock.patch.dict(
                    os.environ,
                    {"SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE": "1"},
                    clear=False,
                ), mock.patch.object(noddi_accel.shutil, "which", return_value=None):
                    _, _, _, y_uncached = noddi_accel._load_masked_normalized_dwi(
                        dwi_path=dwi_path, mask_path=mask_path, scheme=scheme
                    )
                np.testing.assert_allclose(y_uncached, ref_y, rtol=0.0, atol=1e-6)
                np.testing.assert_allclose(y_cached, y_uncached, rtol=0.0, atol=1e-6)

                # 3) Simulated pigz/gzip streaming path (Linux/macOS code path)
                raw_gunzipped = gzip.decompress(dwi_path.read_bytes())
                fake_proc = types.SimpleNamespace(stdout=raw_gunzipped, returncode=0)
                with mock.patch.dict(
                    os.environ,
                    {"SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE": "1"},
                    clear=False,
                ), mock.patch.object(noddi_accel.shutil, "which", return_value="/usr/bin/pigz"), mock.patch.object(
                    noddi_accel.subprocess, "run", return_value=fake_proc
                ):
                    _, _, _, y_pigz = noddi_accel._load_masked_normalized_dwi(
                        dwi_path=dwi_path, mask_path=mask_path, scheme=scheme
                    )
                np.testing.assert_allclose(y_pigz, ref_y, rtol=0.0, atol=1e-6)

    def test_dti_fast_read_matches_nibabel_cached_on_off_and_pigz(self) -> None:
        from shared.pipeline import RawInputBundle
        from tractography import dti as dti_mod

        rng = np.random.default_rng(77)
        raw_int = rng.integers(250, 1100, size=(5, 5, 5, 7), dtype=np.int16)
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        bvals = np.array([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        bvecs = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.7071, 0.7071, 0.0],
                [0.7071, 0.0, 0.7071],
                [0.0, 0.7071, 0.7071],
            ]
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            inp.mkdir()
            dwi_path = _save_extended_scaled_nifti(
                inp / "dwi.nii.gz", raw_int, slope=0.5, inter=10.0, extension_bytes=160
            )
            mask_path = inp / "mask.nii.gz"
            bval_path = inp / "dwi.bval"
            bvec_path = inp / "dwi.bvec"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_path))
            np.savetxt(bval_path, bvals[None, :], fmt="%.1f")
            np.savetxt(bvec_path, bvecs.T, fmt="%.6f")

            ref_data = nib.load(str(dwi_path)).get_fdata(dtype=np.float32)[mask == 1]
            raw = RawInputBundle(
                subject_root=root,
                subject_id="sub01",
                input_dir=inp,
                dwi_path=dwi_path,
                bval_path=bval_path,
                bvec_path=bvec_path,
                mask_path=mask_path,
            )
            cfg = dti_mod.DTITrackingConfig(
                subject_id="sub01",
                input_root=root,
                output_dir=root / "out",
                input_bundle=raw,
                execute=True,
            )

            captured: dict[str, np.ndarray] = {}

            class _StopAfterCapture(Exception):
                pass

            def _capture_fit(data_in_mask, *args, **kwargs):
                captured["arr"] = np.array(data_in_mask, copy=True)
                raise _StopAfterCapture()

            # 1) Cached ON
            with mock.patch.dict(os.environ, {"SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES": "1"}, clear=False):
                os.environ.pop("SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE", None)
                with mock.patch.object(dti_mod, "_fit_dti_fa_and_peaks", side_effect=_capture_fit):
                    with self.assertRaises(_StopAfterCapture):
                        dti_mod._run_dipy_dti_tracking(cfg, streamlines_path=root / "out" / "t.trk", log=None)
            cached_arr = captured.pop("arr")
            np.testing.assert_allclose(cached_arr, ref_data, rtol=0.0, atol=1e-6)

            # 2) Simulated pigz/gzip path (Cached OFF)
            raw_gunzipped = gzip.decompress(dwi_path.read_bytes())
            fake_proc = types.SimpleNamespace(stdout=raw_gunzipped, returncode=0)
            with mock.patch.dict(os.environ, {"SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE": "1"}, clear=False):
                with mock.patch("shutil.which", return_value="/usr/bin/pigz"), mock.patch(
                    "subprocess.run", return_value=fake_proc
                ), mock.patch.object(dti_mod, "_fit_dti_fa_and_peaks", side_effect=_capture_fit):
                    with self.assertRaises(_StopAfterCapture):
                        dti_mod._run_dipy_dti_tracking(cfg, streamlines_path=root / "out" / "t.trk", log=None)
            pigz_arr = captured.pop("arr")
            np.testing.assert_allclose(pigz_arr, ref_data, rtol=0.0, atol=1e-6)

            # 3) Fallback path (Cached OFF, no pigz)
            with mock.patch.dict(os.environ, {"SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE": "1"}, clear=False):
                with mock.patch("shutil.which", return_value=None), mock.patch.object(
                    dti_mod, "_fit_dti_fa_and_peaks", side_effect=_capture_fit
                ):
                    with self.assertRaises(_StopAfterCapture):
                        dti_mod._run_dipy_dti_tracking(cfg, streamlines_path=root / "out" / "t.trk", log=None)
            fallback_arr = captured.pop("arr")
            np.testing.assert_allclose(fallback_arr, ref_data, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(cached_arr, fallback_arr, rtol=0.0, atol=1e-6)

    def test_brain_mask_single_pass_gzip_respects_extension_and_scaling(self) -> None:
        from shared.brain_mask import _accumulate_b0_single_pass_gzip

        raw_int = np.arange(3 * 4 * 5 * 4, dtype=np.int16).reshape(3, 4, 5, 4) + 100
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "input"
            inp.mkdir()
            dwi_path = _save_extended_scaled_nifti(
                inp / "dwi.nii.gz", raw_int, slope=0.5, inter=10.0, extension_bytes=160
            )
            img = nib.load(str(dwi_path))
            ref = img.get_fdata(dtype=np.float32)
            expected = ref[..., 0].astype(np.float64) + ref[..., 2].astype(np.float64)
            with mock.patch.dict(os.environ, {"SLICERDSPACE_DISABLE_UNCOMPRESSED_CACHE": "1"}, clear=False):
                got = _accumulate_b0_single_pass_gzip(img, dwi_path, [0, 2])
            self.assertIsNotNone(got)
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-5)


class CacheReuseCorrectnessTests(unittest.TestCase):
    """[Item 2] Preprocessing reverse-PE sidecars, MK-Curve PORT_VERSION, and SIFT2 connectome reuse."""

    def test_2a_preprocessing_cache_key_invalidates_on_reverse_pe_bval_bvec_and_nhdr_data(self) -> None:
        import nrrd
        from preprocessing.pipeline import (
            DwiPreprocessingConfig,
            _preprocessing_cache_key,
            preprocessing_cache_key_matches,
        )
        from shared.pipeline import RawInputBundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            inp.mkdir()
            dwi = inp / "dwi.nii.gz"
            bval = inp / "dwi.bval"
            bvec = inp / "dwi.bvec"
            mask = inp / "mask.nii.gz"
            rev = inp / "rev_dwi.nii.gz"
            rev_bval = inp / "rev_dwi.bval"
            rev_bvec = inp / "rev_dwi.bvec"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 4), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(mask))
            bval.write_text("0 1000 1000 1000\n", encoding="utf-8")
            bvec.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 3), dtype=np.float32), np.eye(4)), str(rev))

            raw = RawInputBundle(
                subject_root=root,
                subject_id="sub01",
                input_dir=inp,
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
            )
            cfg = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=root,
                output_dir=root / "out",
                reverse_phase_b0=rev,
                phase_encoding_direction="j-",
                readout_time=0.05,
            )
            fake_backend = {"execution_backend": "mrtrix", "commands": {}}
            fake_mask_backend = {"backend": "provided"}

            # Without sidecars: key stays backwards-compatible (no reverse_phase_bval key)
            k_no_sidecar = _preprocessing_cache_key(raw, cfg, fake_backend, fake_mask_backend)
            self.assertNotIn("reverse_phase_bval", k_no_sidecar["inputs"])

            # Adding rev_dwi.bval invalidates cache
            rev_bval.write_text("0 1000 1000\n", encoding="utf-8")
            k_bval1 = _preprocessing_cache_key(raw, cfg, fake_backend, fake_mask_backend)
            self.assertIn("reverse_phase_bval", k_bval1["inputs"])
            self.assertFalse(preprocessing_cache_key_matches(k_no_sidecar, k_bval1))

            # Modifying rev_dwi.bval invalidates cache
            rev_bval.write_text("1000 1000 0\n", encoding="utf-8")
            k_bval2 = _preprocessing_cache_key(raw, cfg, fake_backend, fake_mask_backend)
            self.assertFalse(preprocessing_cache_key_matches(k_bval1, k_bval2))

            # Adding and modifying rev_dwi.bvec invalidates cache
            rev_bvec.write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")
            k_bvec1 = _preprocessing_cache_key(raw, cfg, fake_backend, fake_mask_backend)
            self.assertFalse(preprocessing_cache_key_matches(k_bval2, k_bvec1))
            rev_bvec.write_text("0 0 1\n0 1 0\n0 0 0\n", encoding="utf-8")
            k_bvec2 = _preprocessing_cache_key(raw, cfg, fake_backend, fake_mask_backend)
            self.assertFalse(preprocessing_cache_key_matches(k_bvec1, k_bvec2))

            # Detached .nhdr data file modification invalidates cache
            rev_nhdr = inp / "rev_b0.nhdr"
            nrrd.write(
                str(rev_nhdr),
                np.ones((4, 4, 4), dtype=np.float32),
                {"space": "right-anterior-superior", "space directions": np.eye(3), "space origin": np.zeros(3)},
                detached_header=True,
            )
            cfg_nhdr = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=root,
                output_dir=root / "out",
                reverse_phase_b0=rev_nhdr,
                phase_encoding_direction="j-",
                readout_time=0.05,
            )
            k_nhdr1 = _preprocessing_cache_key(raw, cfg_nhdr, fake_backend, fake_mask_backend)
            self.assertIn("reverse_phase_data", k_nhdr1["inputs"])
            raw_data_file = Path(k_nhdr1["inputs"]["reverse_phase_data"]["path"])
            raw_data_file.write_bytes(b"\x00" * raw_data_file.stat().st_size)
            k_nhdr2 = _preprocessing_cache_key(raw, cfg_nhdr, fake_backend, fake_mask_backend)
            self.assertFalse(preprocessing_cache_key_matches(k_nhdr1, k_nhdr2))

    def test_2b_mkcurve_port_version_invalidates_pre_fix_v2_cache(self) -> None:
        from microstructure.dki import _mkcurve_backend_matches
        from microstructure.mkcurve import PORT_VERSION, inspect_mkcurve_backend

        self.assertGreaterEqual(PORT_VERSION, 3)
        current = inspect_mkcurve_backend()
        pre_fix_v2 = dict(current, port_version=2)
        self.assertFalse(_mkcurve_backend_matches(pre_fix_v2, current))
        # Current v3 still ignores pure .py source churn
        same_v3_diff_py = dict(
            current,
            source_sha256={**current["source_sha256"], "mkcurve_tensor.py": "0" * 64},
        )
        self.assertTrue(_mkcurve_backend_matches(same_v3_diff_py, current))

    def test_2c_sift2_stale_weights_rejected_when_disabled_or_tractogram_changed(self) -> None:
        from shared.orchestrator import _sift2_weights_current
        from tractography.csd import _file_info as _file_record

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tck = root / "sub01_csd_act_ifod2.tck"
            weights = root / "sub01_csd_act_ifod2_sift2_weights.txt"
            manifest = root / "csd_act_manifest.json"
            qc = root / "csd_act_qc.json"
            tck.write_bytes(b"mrtrix tracks v1\n")
            weights.write_text("1.0\n2.0\n", encoding="utf-8")

            manifest_payload = {
                "status": "completed",
                "parameters": {"use_sift2": True},
                "outputs": {"sift2_weights": weights.as_posix()},
            }
            qc_payload = {
                "status": "completed",
                "scientific_output_summary": {
                    "tck": {"file": _file_record(tck), "streamline_count": 2},
                    "sift2": {
                        "file": _file_record(weights),
                        "weight_count": 2,
                        "all_finite": True,
                        "all_nonnegative": True,
                    },
                },
            }
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            qc.write_text(json.dumps(qc_payload), encoding="utf-8")

            # Valid matching SIFT2 run
            self.assertTrue(_sift2_weights_current(weights, tck, manifest, qc, use_sift2=True))

            # Turning SIFT2 OFF immediately rejects stale weights on disk
            self.assertFalse(_sift2_weights_current(weights, tck, manifest, qc, use_sift2=False))

            # Recomputing a new tractogram (even with same streamline count) invalidates old weights
            tck.write_bytes(b"mrtrix tracks v2 - totally different streamlines!\n")
            self.assertFalse(_sift2_weights_current(weights, tck, manifest, qc, use_sift2=True))

            # Manifest built with use_sift2=False rejects stale weights even if caller passes use_sift2=True
            tck.write_bytes(b"mrtrix tracks v1\n")
            manifest_payload["parameters"]["use_sift2"] = False
            manifest_payload["outputs"]["sift2_weights"] = None
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            self.assertFalse(_sift2_weights_current(weights, tck, manifest, qc, use_sift2=True))

    def test_2c_orchestrator_does_not_pair_stale_sift2_weights_when_sift2_disabled(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import (
            ProductOrchestratorConfig,
            _run_stage_dirs,
            run_subject_products,
        )
        from tractography.csd import _file_info as _file_record

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            inp.mkdir()
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 4), dtype=np.float32), np.eye(4)), str(inp / "data.nii.gz"))
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(inp / "nodif_brain_mask.nii.gz"))
            (inp / "bvals").write_text("0 1000 1000 1000\n", encoding="utf-8")
            (inp / "bvecs").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")

            layout = subject_layout(root)
            parcel_dir = layout.anatomy_input_dir("ddparcel")
            parcel_dir.mkdir(parents=True, exist_ok=True)
            parcel = parcel_dir / "sub01-DDSurfer-wmparc.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4)), str(parcel))

            dirs = _run_stage_dirs(layout, "run1", root)
            dirs["tractogram_dir"].mkdir(parents=True, exist_ok=True)
            dirs["csd_dir"].mkdir(parents=True, exist_ok=True)
            tck = dirs["tractogram_dir"] / "sub01_csd_act_ifod2.tck"
            stale_weights = dirs["tractogram_dir"] / "sub01_csd_act_ifod2_sift2_weights.txt"
            tck.write_bytes(b"new tractogram without sift2\n")
            stale_weights.write_text("1.0\n", encoding="utf-8")

            # Active CSD manifest/QC explicitly ran with use_sift2=False, while stale weights remain on disk!
            (dirs["csd_dir"] / "csd_act_manifest.json").write_text(
                json.dumps({"status": "completed", "parameters": {"use_sift2": False}, "outputs": {"sift2_weights": None}}),
                encoding="utf-8",
            )
            (dirs["csd_dir"] / "csd_act_qc.json").write_text(
                json.dumps({"status": "completed", "scientific_output_summary": {"tck": {"file": _file_record(tck), "streamline_count": 1}}}),
                encoding="utf-8",
            )

            seen_metrics: list[str] = []

            def _fake_connectome(cfg, *, log=None):
                seen_metrics.append(cfg.metric)
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{cfg.subject_id}-connectome-{cfg.metric}"
                mat = cfg.output_dir / f"{stem}.csv"
                nodes = cfg.output_dir / f"{cfg.subject_id}-connectome-nodes.tsv"
                qc_p = cfg.output_dir / f"{stem}_qc.json"
                man_p = cfg.output_dir / f"{stem}_manifest.json"
                mat.write_text("source_node,target_node,streamline_count\n1,2,1\n", encoding="utf-8")
                nodes.write_text("node_id\tlabel\tname\n1\t1\tA\n2\t2\tB\n", encoding="utf-8")
                qc_p.write_text(json.dumps({"status": "completed", "node_count": 2, "edge_count": 1}), encoding="utf-8")
                man_p.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
                return types.SimpleNamespace(
                    status="completed",
                    qc_path=qc_p,
                    manifest_path=man_p,
                    matrix_path=mat,
                    nodes_path=nodes,
                    node_parcellation_path=None,
                )

            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                run_name="run1",
                tractography_backend="csd_act",
                csd_act_parameters='{"use_sift2": false}',
                connectome_backend="python",
                closure_applied=True,
                execute=True,
                resume=False,
            )
            with mock.patch("connectome.run_connectome", side_effect=_fake_connectome), mock.patch(
                "shared.orchestrator._resolve_parcellation", return_value=parcel
            ):
                res = run_subject_products(cfg)
            self.assertEqual(seen_metrics, ["streamline_count"])
            self.assertNotIn("connectome.sift2_weighted", [t.name for t in res.tasks])


class WorkspaceLockTests(unittest.TestCase):
    """[Item 3] Cross-process per-subject workspace lock contention, stale recovery, inheritance, and isolation."""

    def test_cross_process_contention_and_preserves_owner_manifest(self) -> None:
        from shared.workspace_lock import LOCK_FILENAME

        with tempfile.TemporaryDirectory() as td:
            lock_dir = Path(td) / "outputs" / "00_run_metadata"
            ready_file = Path(td) / "ready.flag"
            stop_file = Path(td) / "stop.flag"
            holder_code = (
                "import sys, time\n"
                "from pathlib import Path\n"
                "from shared.workspace_lock import workspace_lock\n"
                "lock_dir, ready, stop = map(Path, sys.argv[1:4])\n"
                "with workspace_lock(lock_dir, subject_id='sub01', run_name='runA'):\n"
                "    ready.write_text('ok', encoding='utf-8')\n"
                "    deadline = time.monotonic() + 20.0\n"
                "    while not stop.exists() and time.monotonic() < deadline:\n"
                "        time.sleep(0.05)\n"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")])
            env.pop("SLICERDSPACE_WORKSPACE_LOCKS", None)
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_code, str(lock_dir), str(ready_file), str(stop_file)],
                env=env,
            )
            try:
                deadline = time.monotonic() + 10.0
                while not ready_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready_file.exists(), "Holder process failed to acquire lock")

                # Contender process must fail cleanly with owner PID, run_name, started_at, status!
                contender_code = (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "from shared.workspace_lock import WorkspaceLockError, workspace_lock\n"
                    "try:\n"
                    "    with workspace_lock(Path(sys.argv[1]), subject_id='sub01', run_name='runB'):\n"
                    "        print('ACQUIRED')\n"
                    "        sys.exit(0)\n"
                    "except WorkspaceLockError as exc:\n"
                    "    print('LOCKED:', exc)\n"
                    "    sys.exit(42)\n"
                )
                contender = subprocess.run(
                    [sys.executable, "-c", contender_code, str(lock_dir)],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=15,
                )
                self.assertEqual(contender.returncode, 42, contender.stdout + contender.stderr)
                self.assertIn(f"PID={holder.pid}", contender.stdout)
                self.assertIn("run_name=runA", contender.stdout)
                self.assertIn("status=running", contender.stdout)
                self.assertIn("started_at=", contender.stdout)
                # Lock file still intact with holder's PID
                meta = json.loads((lock_dir / LOCK_FILENAME).read_text(encoding="utf-8"))
                self.assertEqual(meta["pid"], holder.pid)
            finally:
                stop_file.write_text("stop", encoding="utf-8")
                holder.wait(timeout=10)
            self.assertFalse((lock_dir / LOCK_FILENAME).exists())

    def test_stale_lock_reclaimed_automatically(self) -> None:
        import socket
        from shared.workspace_lock import LOCK_FILENAME, workspace_lock

        # Spawn a tiny subprocess and wait for it to exit so we have a guaranteed-dead PID.
        dead_proc = subprocess.run([sys.executable, "-c", "pass"], check=True)
        # Re-obtain a dead PID via Popen
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        dead_pid = p.pid
        p.wait(timeout=10)

        with tempfile.TemporaryDirectory() as td:
            lock_dir = Path(td) / "outputs" / "00_run_metadata"
            lock_dir.mkdir(parents=True)
            (lock_dir / LOCK_FILENAME).write_text(
                json.dumps(
                    {
                        "pid": dead_pid,
                        "hostname": socket.gethostname(),
                        "started_at": "2020-01-01T00:00:00+00:00",
                        "run_name": "crashed_run",
                        "subject_id": "sub01",
                        "status": "running",
                        "token": "dead-token",
                    }
                ),
                encoding="utf-8",
            )
            with workspace_lock(lock_dir, subject_id="sub01", run_name="fresh_run") as lp:
                meta = json.loads(lp.read_text(encoding="utf-8"))
                self.assertEqual(meta["pid"], os.getpid())
                self.assertEqual(meta["run_name"], "fresh_run")
            self.assertFalse((lock_dir / LOCK_FILENAME).exists())

    def test_child_subprocess_inherits_lock_and_unrelated_subjects_run_concurrently(self) -> None:
        from shared.workspace_lock import LOCK_FILENAME, WorkspaceLockError, workspace_lock

        with tempfile.TemporaryDirectory() as td:
            lock_a = Path(td) / "subA" / "outputs" / "00_run_metadata"
            lock_b = Path(td) / "subB" / "outputs" / "00_run_metadata"
            with workspace_lock(lock_a, subject_id="subA", run_name="r1"):
                # 1) Same-thread re-entrant acquire succeeds
                with workspace_lock(lock_a, subject_id="subA", run_name="r1_child"):
                    self.assertTrue((lock_a / LOCK_FILENAME).is_file())
                self.assertTrue((lock_a / LOCK_FILENAME).is_file())

                # 2) Different thread in same process targeting SAME subject fails cleanly
                thread_err: list[BaseException] = []

                def _other_thread():
                    try:
                        with workspace_lock(lock_a, subject_id="subA", run_name="r2"):
                            pass
                    except BaseException as exc:
                        thread_err.append(exc)

                t = threading.Thread(target=_other_thread)
                t.start()
                t.join(timeout=5)
                self.assertEqual(len(thread_err), 1)
                self.assertIsInstance(thread_err[0], WorkspaceLockError)

                # 3) Child subprocess spawned by lock owner inherits lock without deadlocking
                child_code = (
                    "import sys\n"
                    "from pathlib import Path\n"
                    "from shared.workspace_lock import workspace_lock\n"
                    "with workspace_lock(Path(sys.argv[1]), subject_id='subA', run_name='child'):\n"
                    "    print('CHILD_OK')\n"
                )
                child = subprocess.run(
                    [sys.executable, "-c", child_code, str(lock_a)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(child.returncode, 0, child.stdout + child.stderr)
                self.assertIn("CHILD_OK", child.stdout)
                # Child exit must NOT delete parent's lock!
                self.assertTrue((lock_a / LOCK_FILENAME).is_file())

                # 4) Unrelated subject B runs concurrently with zero contention
                with workspace_lock(lock_b, subject_id="subB", run_name="r1"):
                    self.assertTrue((lock_b / LOCK_FILENAME).is_file())

    def test_orchestrator_lock_rejection_does_not_clobber_active_manifest(self) -> None:
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        from shared.workspace_lock import WorkspaceLockError, workspace_lock

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "sub01"
            root.mkdir()
            layout = subject_layout(root)
            layout.run_metadata.mkdir(parents=True)
            manifest = layout.run_metadata / "run1_orchestration_manifest.json"
            manifest.write_text(json.dumps({"status": "running", "owner": "processA"}), encoding="utf-8")

            # Simulate another process holding the lock (use current PID + fake non-matching token on disk
            # in a separate thread via workspace_lock so run_subject_products hits WorkspaceLockError).
            ready = threading.Event()
            release = threading.Event()

            def _hold():
                with workspace_lock(layout.run_metadata, subject_id="sub01", run_name="run1"):
                    ready.set()
                    release.wait(timeout=10.0)

            holder = threading.Thread(target=_hold)
            holder.start()
            self.assertTrue(ready.wait(timeout=5.0))
            try:
                cfg = ProductOrchestratorConfig(
                    subject_id="sub01",
                    subject_root=root,
                    targets=("validation",),
                    run_name="run1",
                    execute=True,
                )
                with self.assertRaises(WorkspaceLockError):
                    run_subject_products(cfg)
                # Active manifest MUST remain completely untouched!
                self.assertEqual(json.loads(manifest.read_text(encoding="utf-8")), {"status": "running", "owner": "processA"})
            finally:
                release.set()
                holder.join(timeout=5.0)


class NrrdTopupCommandGenerationTests(unittest.TestCase):
    """[Item 4] Topup command generation accepts valid 3D/4D .nrrd and .nhdr reverse-PE inputs."""

    def test_build_commands_accepts_valid_nrrd_and_nhdr_inputs(self) -> None:
        import nrrd
        from preprocessing.pipeline import DwiPreprocessingConfig, _build_commands, _planned_paths
        from shared.pipeline import RawInputBundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "input"
            inp.mkdir()
            dwi = inp / "dwi.nii.gz"
            bval = inp / "dwi.bval"
            bvec = inp / "dwi.bvec"
            mask = inp / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 4), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(mask))
            bval.write_text("0 1000 1000 1000\n", encoding="utf-8")
            bvec.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")

            raw = RawInputBundle(
                subject_root=root,
                subject_id="sub01",
                input_dir=inp,
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
            )
            fake_backend = {
                "execution_backend": "mrtrix",
                "commands": {
                    "mrconvert": "mrconvert",
                    "dwiextract": "dwiextract",
                    "mrcat": "mrcat",
                    "dwifslpreproc": "dwifslpreproc",
                },
            }
            paths = _planned_paths(root / "out", "sub01")

            for ext, detached, shape, hdr_extra in (
                (".nrrd", False, (4, 4, 4), {"space directions": np.eye(3)}),
                (".nhdr", True, (4, 4, 4), {"space directions": np.eye(3)}),
                (
                    ".nrrd",
                    False,
                    (4, 4, 4, 2),
                    {
                        "space directions": np.vstack([np.eye(3), [np.nan, np.nan, np.nan]]),
                        "kinds": ["domain", "domain", "domain", "list"],
                    },
                ),
            ):
                with self.subTest(ext=ext, shape=shape):
                    rev = inp / f"rev_{len(shape)}d{ext}"
                    nrrd.write(
                        str(rev),
                        np.ones(shape, dtype=np.float32),
                        {"space": "right-anterior-superior", "space origin": np.zeros(3), **hdr_extra},
                        detached_header=detached,
                    )
                    cfg = DwiPreprocessingConfig(
                        subject_id="sub01",
                        input_root=root,
                        output_dir=root / "out",
                        reverse_phase_b0=rev,
                        phase_encoding_direction="j-",
                        readout_time=0.05,
                    )
                    cmds = _build_commands(raw, cfg, paths, fake_backend)
                    prepare_cmds = [c for c in cmds if c and c[0] == "prepare-reverse-b0"]
                    self.assertEqual(len(prepare_cmds), 1)
                    self.assertEqual(Path(prepare_cmds[0][1]), rev.resolve())


class WindowsNonAsciiLauncherTests(unittest.TestCase):
    """[Item 5] Windows CTKAppLauncher non-ASCII/Chinese path preservation via pure-ASCII UTF-8 trampoline."""

    def test_windows_trampoline_keeps_ascii_unchanged_and_wraps_non_ascii(self) -> None:
        from shared.external_process import normalize_slicer_python_launch

        with mock.patch("shared.external_process.sys.platform", "win32"):
            ascii_args = ["--launch", "PythonSlicer", "-u", r"C:\work\script.py", "--subject", "sub01"]
            prog, out_args = normalize_slicer_python_launch(r"C:\Slicer\Slicer.exe", ascii_args)
            self.assertEqual((prog, out_args), (r"C:\Slicer\Slicer.exe", ascii_args))

            zh_args = ["--launch", "PythonSlicer", "-u", r"C:\测试路径\script.py", "--root", r"C:\数据\患者_αβγ"]
            prog2, out_args2 = normalize_slicer_python_launch(r"C:\Slicer\Slicer.exe", zh_args)
            self.assertEqual(prog2, r"C:\Slicer\Slicer.exe")
            self.assertEqual(out_args2[:4], ["--launch", "PythonSlicer", "-u", "-c"])
            self.assertTrue(all(a.isascii() for a in out_args2))
            self.assertNotIn('"', out_args2[4])

            # Direct PythonSlicer.exe with non-ASCII args also wrapped
            prog3, out_args3 = normalize_slicer_python_launch(
                r"C:\Slicer\bin\PythonSlicer.exe", ["-u", r"C:\测试路径\script.py", r"C:\数据\文件.txt"]
            )
            self.assertEqual(out_args3[:2], ["-u", "-c"])
            self.assertTrue(all(a.isascii() for a in out_args3))

    @unittest.skipUnless(sys.platform == "win32", "Live Windows Slicer launcher test")
    def test_live_windows_slicer_launcher_with_chinese_path(self) -> None:
        from shared.external_process import normalize_slicer_python_launch

        slicer_root = Path(sys.executable).resolve().parent.parent
        slicer_exe = slicer_root / "Slicer.exe"
        py_slicer = slicer_root / "bin" / "PythonSlicer.exe"
        if not (slicer_exe.is_file() and py_slicer.is_file()):
            self.skipTest("Slicer.exe / PythonSlicer.exe not found next to interpreter")

        test_dir = Path(tempfile.gettempdir()) / f"测试路径_dspace_test_{os.getpid()}"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            script = test_dir / "脚本_echo.py"
            target = test_dir / "数据_测试_sub01.txt"
            target.write_text("ok", encoding="utf-8")
            script.write_text(
                "import json, sys; from pathlib import Path\n"
                "print(json.dumps({'argv0': sys.argv[0], 'arg1': sys.argv[1], "
                "'exists': Path(sys.argv[1]).exists(), 'file_exists': Path(__file__).exists()}, ensure_ascii=False))\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"

            for launcher, raw_args in (
                (str(slicer_exe), ["--launch", "PythonSlicer", "-u", str(script), str(target)]),
                (str(py_slicer), ["-u", str(script), str(target)]),
            ):
                prog, norm_args = normalize_slicer_python_launch(launcher, raw_args)
                proc = subprocess.run(
                    [prog, *norm_args],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=30,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
                self.assertEqual(payload["argv0"], str(script))
                self.assertEqual(payload["arg1"], str(target))
                self.assertTrue(payload["exists"])
                self.assertTrue(payload["file_exists"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class UkfAvx2CapabilityTests(unittest.TestCase):
    """[Item 6] Standalone UKF AVX2 CPU check skips incompatible binaries and reports clear diagnostics."""

    def test_standalone_avx2_fallback_and_diagnostic_error(self) -> None:
        from tractography import ukf

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            standalone = root / "bundled-runtimes" / "ukf-windows-x64" / "bin" / "UKFTractography.exe"
            standalone.parent.mkdir(parents=True)
            standalone.write_bytes(b"MZ")
            (standalone.parent.parent / "runtime.json").write_text(
                json.dumps({"kind": "ukf-standalone", "platform": "windows-x64", "arch": "x86_64", "requires_avx2": True}),
                encoding="utf-8",
            )
            self.assertTrue(ukf._standalone_requires_avx2(standalone))

            # Arm64 standalone does NOT require AVX2
            arm64_exe = root / "bundled-runtimes" / "ukf-darwin-arm64" / "bin" / "UKFTractography"
            arm64_exe.parent.mkdir(parents=True)
            arm64_exe.write_bytes(b"#!/bin/sh\n")
            self.assertFalse(ukf._standalone_requires_avx2(arm64_exe))

            # Fallback to Slicer cli-modules when CPU lacks AVX2
            fallback_cli = root / "lib" / "Slicer-5.10" / "cli-modules" / "UKFTractography.exe"
            fallback_cli.parent.mkdir(parents=True)
            fallback_cli.write_bytes(b"MZ")

            env_clean = {k: v for k, v in os.environ.items() if k not in {"SLICERDSPACE_UKF_EXECUTABLE", "UKF_GPU_EXECUTABLE"}}
            with mock.patch.dict(os.environ, env_clean, clear=True), mock.patch.object(
                ukf, "_extension_search_roots", return_value=[root]
            ), mock.patch.object(ukf.shutil, "which", return_value=None), mock.patch.object(
                ukf, "ukf_vtk_mismatch", return_value=None
            ), mock.patch(
                "shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("no gui")
            ):
                # With AVX2 available: picks standalone (after removing arm64 so only windows-x64 exists)
                arm64_exe.unlink()
                with mock.patch.object(ukf, "_host_supports_avx2", return_value=True):
                    self.assertEqual(ukf.resolve_executable(), standalone.resolve())

                # Without AVX2: skips AVX2 standalone and falls back cleanly to cli-modules!
                with mock.patch.object(ukf, "_host_supports_avx2", return_value=False):
                    self.assertEqual(ukf.resolve_executable(), fallback_cli.resolve())

                # Without AVX2 AND without fallback: raises explicit AVX2 diagnostic RuntimeError!
                fallback_cli.unlink()
                with mock.patch.object(ukf, "_host_supports_avx2", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "AVX2"):
                        ukf.resolve_executable()


if __name__ == "__main__":
    unittest.main()
