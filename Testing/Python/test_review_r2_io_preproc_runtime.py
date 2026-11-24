"""Round-2 code-review regression tests (Group B: Items 3, 4, 5, 8).

Covers:
- Item 3 (P1): Windows non-ASCII/Chinese paths inside inner DTI/Teem/CLI subprocesses + UKF AVX2 checks
  across explicit config, env vars, PATH, and auto-discovery.
- Item 4 (P1): Atomic overwrite input import validation (preserves old mask/DWI/JSON/manifest on failure)
  and stale sidecar JSON / mask cleanup on commit.
- Item 5 (P1): Pre-merge Topup spatial geometry validation (origin, spacing, rotation, dimensions) BEFORE
  MRtrix ``mrcat`` and FSL-native Topup, plus PhaseEncodingDirection & TotalReadoutTime validation.
- Item 8 (P2): DTI & DKI low-available-RAM worker capping (128 MiB -> 1 worker without breaking high-RAM
  multi-core throughput) and ``SUPPORT_MATRIX.md`` / ``SUPPORT_MATRIX.zh-CN.md`` CPU-only sync.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_tiny_dwi_subject(
    subject_root: Path,
    *,
    subject_id: str = "sub01",
    shape: tuple[int, int, int] = (4, 4, 4),
    affine: np.ndarray | None = None,
    fill_scale: float = 500.0,
    sidecar: dict | None = None,
    with_mask: bool = True,
    mask_value: int = 1,
) -> Path:
    inp = subject_root / "input"
    inp.mkdir(parents=True, exist_ok=True)
    aff = np.diag([-2.0, -2.0, 2.0, 1.0]) if affine is None else np.asarray(affine, dtype=float)
    dirs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    dirs[1:] /= np.linalg.norm(dirs[1:], axis=1, keepdims=True)
    bvals = np.array([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=float)
    tensor = np.diag([1.5e-3, 0.5e-3, 0.4e-3])
    sig = fill_scale * np.exp(-bvals * np.einsum("ni,ij,nj->n", dirs, tensor, dirs))
    data = np.broadcast_to(sig, (*shape, len(bvals))).astype(np.float32).copy()
    nib.save(nib.Nifti1Image(data, aff), str(inp / "data.nii.gz"))
    np.savetxt(inp / "bvals", bvals[None, :], fmt="%.1f")
    np.savetxt(inp / "bvecs", dirs.T, fmt="%.6f")
    if with_mask:
        mask = np.full(shape, mask_value, dtype=np.uint8)
        nib.save(nib.Nifti1Image(mask, aff), str(inp / "nodif_brain_mask.nii.gz"))
    if sidecar is not None:
        (inp / "dwi.json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return subject_root


# --------------------------------------------------------------------------- Item 3 (P1)


class Item3WindowsChinesePathAndUkfAvx2Tests(unittest.TestCase):
    """[Item 3] Windows non-ASCII paths in inner Teem/CLI subprocesses + UKF AVX2 checks on all selection paths."""

    def test_run_cli_wraps_windows_pythonslicer_non_ascii_args(self) -> None:
        from shared import pipeline

        captured: list[list[str]] = []

        class _FakeProc:
            stdout = None

            def poll(self):
                return 0

            def wait(self):
                return 0

        def _fake_popen(cmd, **kwargs):
            captured.append(list(cmd))
            return _FakeProc()

        zh_cmd = [
            r"C:\Slicer\bin\PythonSlicer.exe",
            r"C:\repo\shared\teem_dti.py",
            "estimate",
            r"D:\测试被试\cache\DWI.nhdr",
            r"D:\测试被试\cache\DTI.nrrd",
            r"D:\测试被试\cache\b0.nrrd",
        ]
        with (
            mock.patch("shared.external_process.sys.platform", "win32"),
            mock.patch("shared.pipeline.subprocess.Popen", side_effect=_fake_popen),
            mock.patch("shared.line_stream.stream_process_lines", return_value=0),
        ):
            pipeline._run_cli(zh_cmd)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], r"C:\Slicer\bin\PythonSlicer.exe")
        self.assertEqual(captured[0][1], "-c")
        self.assertTrue(all(token.isascii() for token in captured[0]))

    def test_windows_ascii_temp_dir_always_yields_ascii_path(self) -> None:
        from shared.external_process import needs_windows_ascii_staging, windows_ascii_temp_dir

        with mock.patch("shared.external_process.sys.platform", "win32"):
            self.assertTrue(needs_windows_ascii_staging(Path(r"C:\数据\测试\dwi.nhdr")))
            self.assertFalse(needs_windows_ascii_staging(Path(r"C:\data\sub01\dwi.nhdr")))
        with mock.patch("shared.external_process.sys.platform", "linux"):
            self.assertFalse(needs_windows_ascii_staging(Path("/data/测试/dwi.nhdr")))

        with windows_ascii_temp_dir(prefix="r2_ascii_") as stage_dir:
            self.assertTrue(stage_dir.is_dir())
            self.assertTrue(str(stage_dir).isascii())
            probe = stage_dir / "probe.txt"
            probe.write_text("ok", encoding="utf-8")
            self.assertTrue(probe.is_file())
        self.assertFalse(stage_dir.exists())

    @unittest.skipUnless(sys.platform == "win32", "Live Windows Teem/PythonSlicer Chinese-path regression test")
    def test_live_windows_native_dti_teem_succeeds_in_chinese_directory(self) -> None:
        import shared.pipeline as sp

        try:
            import vtkTeem  # noqa: F401
        except ImportError:
            self.skipTest("vtkTeem is unavailable outside Slicer Python")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "测试被试_中文路径_DTI"
            _write_tiny_dwi_subject(root, subject_id="sub_zh")
            inp = root / "input"
            raw = sp.RawInputBundle(
                subject_root=root,
                subject_id="sub_zh",
                input_dir=inp,
                dwi_path=inp / "data.nii.gz",
                bval_path=inp / "bvals",
                bvec_path=inp / "bvecs",
                mask_path=inp / "nodif_brain_mask.nii.gz",
            )
            with (
                mock.patch.object(sp, "slicer", object()),
                mock.patch.object(sp, "SLICER_IMPORT_ERROR", None),
            ):
                bundle = sp.ensure_native_dti_bundle(raw, fit_method="wls", reuse=False)

            self.assertTrue(bundle.dti_nrrd_path.is_file())
            self.assertTrue(bundle.b0_nrrd_path.is_file())
            self.assertTrue(bundle.fa_nii_path.is_file())
            self.assertTrue(bundle.dec_nrrd_path.is_file())
            fa = nib.load(str(bundle.fa_nii_path)).get_fdata()
            # Analytical FA for diag(1.5e-3, 0.5e-3, 0.4e-3) is ~0.646.
            self.assertGreater(float(fa.mean()), 0.55)
            self.assertLess(float(fa.mean()), 0.75)

            # Also verify fallback non-NIfTI-backed NRRD direct estimate_file + scalar_files in Chinese path.
            from shared.teem_dti import dec_file, estimate_file, scalar_files

            direct_out = root / "直接调用_输出"
            direct_out.mkdir()
            dti_out = direct_out / "张量.nrrd"
            b0_out = direct_out / "基线.nrrd"
            fa_out = direct_out / "各向异性.nrrd"
            dec_out = direct_out / "彩色FA.nrrd"
            rec = estimate_file(bundle.dwi_nhdr_path, dti_out, b0_out, method="WLS", log=None)
            self.assertEqual(rec["failed_voxels"], 0)
            self.assertTrue(dti_out.is_file())
            self.assertTrue(b0_out.is_file())
            scalar_files(dti_out, [("FractionalAnisotropy", fa_out)], log=None)
            self.assertTrue(fa_out.is_file())
            dec_file(dti_out, fa_out, dec_out, log=None)
            self.assertTrue(dec_out.is_file())

    def test_ukf_avx2_check_enforced_for_explicit_env_and_path_selection(self) -> None:
        from tractography import ukf

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 1) Bundled AVX2 standalone binary with runtime.json
            avx2_exe = root / "bundled-runtimes" / "ukf-windows-x64" / "bin" / "UKFTractography.exe"
            avx2_exe.parent.mkdir(parents=True)
            avx2_exe.write_bytes(b"MZ")
            (avx2_exe.parent.parent / "runtime.json").write_text(
                json.dumps({"kind": "ukf-standalone", "arch": "x86_64", "requires_avx2": True}),
                encoding="utf-8",
            )

            # 2) Flat copy with sibling runtime.json (e.g., user placed standalone on PATH or custom folder)
            custom_avx2 = root / "custom_bin" / "UKFTractography.exe"
            custom_avx2.parent.mkdir(parents=True)
            custom_avx2.write_bytes(b"MZ")
            (custom_avx2.parent / "runtime.json").write_text(
                json.dumps({"kind": "ukf-standalone", "requires_avx2": True}),
                encoding="utf-8",
            )

            # 3) Non-AVX2 Slicer cli-modules fallback
            fallback_cli = root / "lib" / "Slicer-5.10" / "cli-modules" / "UKFTractography.exe"
            fallback_cli.parent.mkdir(parents=True)
            fallback_cli.write_bytes(b"MZ")

            env_clean = {
                k: v for k, v in os.environ.items()
                if k not in {"SLICERDSPACE_UKF_EXECUTABLE", "UKF_GPU_EXECUTABLE"}
            }

            with (
                mock.patch.object(ukf, "_host_supports_avx2", return_value=False),
                mock.patch.object(ukf, "ukf_vtk_mismatch", return_value=None),
            ):
                # (a) Explicit config / --ukf-executable must reject AVX2 binary instead of bypassing!
                with mock.patch.dict(os.environ, env_clean, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "AVX2"):
                        ukf.resolve_executable(explicit=str(avx2_exe))
                    with self.assertRaisesRegex(RuntimeError, "AVX2"):
                        ukf.resolve_executable(explicit=str(custom_avx2))
                    # Non-AVX2 explicit binary still succeeds cleanly
                    self.assertEqual(ukf.resolve_executable(explicit=str(fallback_cli)), fallback_cli.resolve())

                # (b) Environment variable SLICERDSPACE_UKF_EXECUTABLE must also reject AVX2 binary!
                with mock.patch.dict(
                    os.environ,
                    {**env_clean, "SLICERDSPACE_UKF_EXECUTABLE": str(avx2_exe)},
                    clear=True,
                ):
                    with self.assertRaisesRegex(RuntimeError, "AVX2"):
                        ukf.resolve_executable()

                # (c) PATH discovery (shutil.which) pointing to an AVX2 binary must NOT bypass AVX2 check;
                #     it falls back cleanly to compatible cli-modules when present, or raises AVX2 error when absent.
                with (
                    mock.patch.dict(os.environ, env_clean, clear=True),
                    mock.patch.object(ukf, "_extension_search_roots", return_value=[root]),
                    mock.patch.object(
                        ukf.shutil,
                        "which",
                        side_effect=lambda name: str(custom_avx2) if name == "UKFTractography" else None,
                    ),
                    mock.patch("shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("no gui")),
                ):
                    self.assertEqual(ukf.resolve_executable(), fallback_cli.resolve())

                    fallback_cli.unlink()
                    with self.assertRaisesRegex(RuntimeError, "AVX2"):
                        ukf.resolve_executable()

            # On AVX2-capable CPU, explicit and PATH selections still return the standalone binary.
            with mock.patch.object(ukf, "_host_supports_avx2", return_value=True):
                self.assertEqual(ukf.resolve_executable(explicit=str(avx2_exe)), avx2_exe.resolve())
                self.assertEqual(ukf.resolve_executable(explicit=str(custom_avx2)), custom_avx2.resolve())


# --------------------------------------------------------------------------- Item 4 (P1)


class Item4OverwriteImportAtomicityAndStaleCleanupTests(unittest.TestCase):
    """[Item 4] Overwrite import validates fully before touching existing files and clears stale JSON/mask."""

    def _write_source_nifti(
        self,
        folder: Path,
        stem: str,
        *,
        fill: float,
        bval_text: str = "0 1000\n",
        bvec_text: str = "0 1\n0 0\n0 0\n",
        json_payload: dict | None = None,
        mask_shape: tuple[int, int, int] | None = None,
    ) -> tuple[Path, Path | None]:
        folder.mkdir(parents=True, exist_ok=True)
        dwi = folder / f"{stem}.nii.gz"
        nib.save(nib.Nifti1Image(np.full((3, 3, 3, 2), fill, dtype=np.float32), np.eye(4)), str(dwi))
        (folder / f"{stem}.bval").write_text(bval_text, encoding="utf-8")
        (folder / f"{stem}.bvec").write_text(bvec_text, encoding="utf-8")
        if json_payload is not None:
            (folder / f"{stem}.json").write_text(json.dumps(json_payload) + "\n", encoding="utf-8")
        mask_path: Path | None = None
        if mask_shape is not None:
            mask_path = folder / f"{stem}_mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones(mask_shape, dtype=np.uint8), np.eye(4)), str(mask_path))
        return dwi, mask_path

    def test_overwrite_without_json_clears_stale_sidecar_and_mask(self) -> None:
        from shared.input_import import InputImportConfig, prepare_inputs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject_root = root / "sub01"
            dwi1, mask1 = self._write_source_nifti(
                root / "scan1",
                "dwi1",
                fill=10.0,
                json_payload={
                    "PhaseEncodingDirection": "j-",
                    "TotalReadoutTime": 0.052,
                    "EchoTime": 0.089,
                },
                mask_shape=(3, 3, 3),
            )
            first = prepare_inputs(
                InputImportConfig(
                    source=dwi1,
                    subject_root=subject_root,
                    subject_id="sub01",
                    source_kind="nifti",
                    mask_path=mask1,
                    force=True,
                )
            )
            self.assertEqual(first["status"], "completed")
            self.assertTrue((subject_root / "input" / "dwi.json").is_file())
            self.assertTrue((subject_root / "input" / "nodif_brain_mask.nii.gz").is_file())

            # Also plant a legacy data.json to confirm stale aliases are cleaned up on overwrite.
            (subject_root / "input" / "data.json").write_text('{"EchoTime": 0.099}\n', encoding="utf-8")

            # Overwrite with a new DWI that has NO sidecar .json and NO mask.
            dwi2, _ = self._write_source_nifti(root / "scan2", "dwi2", fill=25.0, json_payload=None, mask_shape=None)
            second = prepare_inputs(
                InputImportConfig(
                    source=dwi2,
                    subject_root=subject_root,
                    subject_id="sub01",
                    source_kind="nifti",
                    force=True,
                )
            )
            self.assertEqual(second["status"], "completed")
            self.assertFalse((subject_root / "input" / "dwi.json").exists())
            self.assertFalse((subject_root / "input" / "data.json").exists())
            self.assertFalse((subject_root / "input" / "nodif_brain_mask.nii.gz").exists())
            self.assertIsNone(second["outputs"]["metadata"])
            self.assertIsNone(second["outputs"]["mask"])
            np.testing.assert_array_equal(
                np.asarray(nib.load(str(subject_root / "input" / "data.nii.gz")).dataobj),
                np.full((3, 3, 3, 2), 25.0, dtype=np.float32),
            )

    def test_failed_overwrite_bvec_or_mask_validation_preserves_old_mask_dwi_json_and_manifest(self) -> None:
        from shared.input_import import InputImportConfig, prepare_inputs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject_root = root / "sub01"
            dwi1, mask1 = self._write_source_nifti(
                root / "scan1",
                "dwi1",
                fill=11.0,
                json_payload={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
                mask_shape=(3, 3, 3),
            )
            prepare_inputs(
                InputImportConfig(
                    source=dwi1,
                    subject_root=subject_root,
                    subject_id="sub01",
                    source_kind="nifti",
                    mask_path=mask1,
                    force=True,
                )
            )

            input_dir = subject_root / "input"
            before_files = {
                p.name: p.read_bytes()
                for p in sorted(input_dir.iterdir())
                if p.is_file()
            }
            self.assertIn("nodif_brain_mask.nii.gz", before_files)
            self.assertIn("dwi.json", before_files)
            self.assertIn("input_import_manifest.json", before_files)

            # (a) Overwrite with bad bvec (wrong column count): MUST raise and leave EVERY existing file intact!
            bad_bvec_dwi, _ = self._write_source_nifti(
                root / "scan_bad_bvec",
                "dwi_bad",
                fill=99.0,
                bval_text="0 1000\n",
                bvec_text="0 1 0\n0 0 1\n0 0 0\n",  # 3 cols vs 2 volumes!
            )
            with self.assertRaisesRegex(ValueError, "Gradient count/shape"):
                prepare_inputs(
                    InputImportConfig(
                        source=bad_bvec_dwi,
                        subject_root=subject_root,
                        subject_id="sub01",
                        source_kind="nifti",
                        force=True,
                    )
                )
            after_bad_bvec = {
                p.name: p.read_bytes()
                for p in sorted(input_dir.iterdir())
                if p.is_file()
            }
            self.assertEqual(after_bad_bvec, before_files)

            # (b) Overwrite with mismatched mask shape: MUST raise and leave EVERY existing file intact!
            good_dwi, bad_mask = self._write_source_nifti(
                root / "scan_bad_mask",
                "dwi_ok",
                fill=42.0,
                mask_shape=(4, 4, 4),  # does not match (3, 3, 3)!
            )
            with self.assertRaisesRegex(ValueError, "Brain mask shape"):
                prepare_inputs(
                    InputImportConfig(
                        source=good_dwi,
                        subject_root=subject_root,
                        subject_id="sub01",
                        source_kind="nifti",
                        mask_path=bad_mask,
                        force=True,
                    )
                )
            after_bad_mask = {
                p.name: p.read_bytes()
                for p in sorted(input_dir.iterdir())
                if p.is_file()
            }
            self.assertEqual(after_bad_mask, before_files)

    def test_failed_converter_overwrite_preserves_existing_workspace_and_success_clears_stale_json(self) -> None:
        from shared.input_import import InputImportConfig, prepare_inputs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject_root = root / "sub01"
            dwi1, mask1 = self._write_source_nifti(
                root / "scan1",
                "dwi1",
                fill=7.0,
                json_payload={"PhaseEncodingDirection": "j-", "EchoTime": 0.088},
                mask_shape=(3, 3, 3),
            )
            prepare_inputs(
                InputImportConfig(
                    source=dwi1,
                    subject_root=subject_root,
                    subject_id="sub01",
                    source_kind="nifti",
                    mask_path=mask1,
                    force=True,
                )
            )
            input_dir = subject_root / "input"
            before_files = {p.name: p.read_bytes() for p in sorted(input_dir.iterdir()) if p.is_file()}

            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            (dicom_dir / "IM0001.dcm").write_bytes(b"\0" * 128 + b"DICM" + b"\x01" * 16)

            cfg = InputImportConfig(
                source=dicom_dir,
                subject_root=subject_root,
                subject_id="sub01",
                source_kind="dicom",
                backend="dcm2niix",
                dcm2niix_executable=Path("/fake/dcm2niix"),
                force=True,
            )

            # Failing dcm2niix leaves old files completely untouched
            with mock.patch("shared.input_import._run", side_effect=RuntimeError("converter failed")):
                with self.assertRaisesRegex(RuntimeError, "converter failed"):
                    prepare_inputs(cfg)
            after_fail = {p.name: p.read_bytes() for p in sorted(input_dir.iterdir()) if p.is_file()}
            self.assertEqual(after_fail, before_files)

            # Succeeding dcm2niix that emits no JSON removes stale dwi.json and stale mask atomically
            def _fake_dcm2niix_no_json(command, **_kwargs):
                out_dir = Path(command[command.index("-o") + 1])
                nib.save(nib.Nifti1Image(np.full((3, 3, 3, 2), 88.0, dtype=np.float32), np.eye(4)), str(out_dir / "data.nii.gz"))
                (out_dir / "data.bval").write_text("0 1000\n", encoding="utf-8")
                (out_dir / "data.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
                return {"returncode": 0, "command": command}

            with mock.patch("shared.input_import._run", side_effect=_fake_dcm2niix_no_json):
                manifest = prepare_inputs(cfg)
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse((input_dir / "dwi.json").exists())
            self.assertFalse((input_dir / "nodif_brain_mask.nii.gz").exists())
            self.assertIsNone(manifest["outputs"]["metadata"])


# --------------------------------------------------------------------------- Item 5 (P1)


class Item5TopupPreMergeGeometryAndPeValidationTests(unittest.TestCase):
    """[Item 5] Forward & reverse b0 spatial grid and PE/readout metadata validated BEFORE mrcat/Topup."""

    def test_10mm_origin_mismatch_is_rejected_before_mrcat_for_mrtrix_and_native_backends(self) -> None:
        from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject = _write_tiny_dwi_subject(
                root / "sub01",
                shape=(5, 5, 5),
                affine=np.diag([-2.0, 2.0, 2.0, 1.0]),
                sidecar={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
            )
            # Reverse b0 with IDENTICAL dimensions/spacing/orientation, but origin shifted by 10 mm!
            shifted_affine = np.diag([-2.0, 2.0, 2.0, 1.0])
            shifted_affine[0, 3] = 10.0
            rev_b0 = root / "reverse_shifted_10mm.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), shifted_affine), str(rev_b0))

            for exec_backend in ("legacy", "native"):
                with self.subTest(execution_backend=exec_backend):
                    cfg = DwiPreprocessingConfig(
                        subject_id="sub01",
                        input_root=subject,
                        output_dir=root / f"out_{exec_backend}",
                        execute=True,
                        reverse_phase_b0=rev_b0,
                        phase_encoding_direction="AP",
                        readout_time=0.05,
                        execution_backend=exec_backend,
                    )
                    with mock.patch("preprocessing.pipeline._run_command") as mock_cmd:
                        with self.assertRaisesRegex(ValueError, r"origin differs by 10\.000 mm"):
                            run_dwi_preprocessing(cfg)
                        mock_cmd.assert_not_called()

    def test_spacing_rotation_and_dimension_mismatches_rejected_before_merge(self) -> None:
        from preprocessing.pipeline import validate_reverse_b0_spatial_grid

        base_aff = np.diag([2.0, 2.0, 2.0, 1.0])
        # Exact match passes
        validate_reverse_b0_spatial_grid((6, 6, 6, 7), base_aff, (6, 6, 6), base_aff)

        # Dimension mismatch
        with self.assertRaisesRegex(ValueError, "3D dimensions differ"):
            validate_reverse_b0_spatial_grid((6, 6, 6, 7), base_aff, (6, 6, 8), base_aff)

        # Voxel size mismatch
        bad_spacing = np.diag([2.0, 2.5, 2.0, 1.0])
        with self.assertRaisesRegex(ValueError, "voxel sizes differ"):
            validate_reverse_b0_spatial_grid((6, 6, 6, 7), base_aff, (6, 6, 6), bad_spacing)

        # Orientation / rotation mismatch
        theta = np.deg2rad(5.0)
        rot = np.eye(4)
        rot[0, 0] = 2.0 * np.cos(theta)
        rot[0, 1] = -2.0 * np.sin(theta)
        rot[1, 0] = 2.0 * np.sin(theta)
        rot[1, 1] = 2.0 * np.cos(theta)
        rot[2, 2] = 2.0
        with self.assertRaisesRegex(ValueError, "orientation/rotation matrix differs"):
            validate_reverse_b0_spatial_grid((6, 6, 6, 7), base_aff, (6, 6, 6), rot)

        # 10 mm origin shift
        shifted = base_aff.copy()
        shifted[1, 3] = 10.0
        with self.assertRaisesRegex(ValueError, r"origin differs by 10\.000 mm"):
            validate_reverse_b0_spatial_grid((6, 6, 6, 7), base_aff, (6, 6, 6), shifted)

    def test_nrrd_reverse_b0_10mm_origin_shift_rejected_before_merge(self) -> None:
        import nrrd
        from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject = _write_tiny_dwi_subject(
                root / "sub01",
                shape=(4, 4, 4),
                affine=np.eye(4),
                sidecar={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
            )
            rev_nrrd = root / "rev_shifted.nrrd"
            nrrd.write(
                str(rev_nrrd),
                np.ones((4, 4, 4), dtype=np.float32),
                {
                    "space": "right-anterior-superior",
                    "space directions": np.eye(3),
                    "space origin": np.array([10.0, 0.0, 0.0]),
                },
            )
            cfg = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=subject,
                output_dir=root / "out",
                execute=True,
                reverse_phase_b0=rev_nrrd,
                phase_encoding_direction="AP",
                readout_time=0.05,
            )
            with self.assertRaisesRegex(ValueError, r"origin differs by 10\.000 mm"):
                run_dwi_preprocessing(cfg)

    def test_bad_pe_direction_non_opposite_reverse_sidecar_and_bad_readout_rejected(self) -> None:
        from preprocessing.fsl_native import validate_b0_pair_headers, validate_topup_pe_metadata
        from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing

        # Direct FSL-native validator checks
        with self.assertRaisesRegex(ValueError, "Unsupported anatomical phase encoding"):
            validate_topup_pe_metadata("XY", 0.05)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            validate_topup_pe_metadata("AP", -0.05)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            validate_topup_pe_metadata("AP", float("nan"))
        with self.assertRaisesRegex(ValueError, "not opposite"):
            validate_topup_pe_metadata("AP", 0.05, reverse_phase_encoding="AP")
        with self.assertRaisesRegex(ValueError, "not opposite"):
            validate_topup_pe_metadata("AP", 0.05, reverse_phase_encoding="LR")
        self.assertEqual(validate_topup_pe_metadata("AP", 0.05, reverse_phase_encoding="PA"), ((0, -1, 0), 0.05))

        hdr_fwd = {"size": [4, 4, 4, 1], "spacing": [2.0, 2.0, 2.0, 1.0], "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]}
        hdr_rev = {"size": [4, 4, 4, 1], "spacing": [2.0, 2.0, 2.0, 1.0], "transform": [[1, 0, 0, 10.0], [0, 1, 0, 0], [0, 0, 1, 0]]}
        with self.assertRaisesRegex(ValueError, r"origin differs by 10\.000 mm"):
            validate_b0_pair_headers(hdr_fwd, hdr_rev)

        # End-to-end preprocessing rejection of non-positive readout_time & non-opposite reverse sidecar
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject = _write_tiny_dwi_subject(
                root / "sub01",
                shape=(4, 4, 4),
                affine=np.eye(4),
                sidecar={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05},
            )
            rev = root / "rev_b0.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), str(rev))

            # Non-positive / non-finite readout time
            for bad_ro in (-0.05, 0.0, float("nan"), float("inf")):
                with self.subTest(bad_readout=bad_ro):
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        run_dwi_preprocessing(
                            DwiPreprocessingConfig(
                                subject_id="sub01",
                                input_root=subject,
                                output_dir=root / "out_ro",
                                execute=True,
                                reverse_phase_b0=rev,
                                phase_encoding_direction="AP",
                                readout_time=bad_ro,
                            )
                        )

            # Reverse sidecar declaring SAME direction (j- == AP, not PA!)
            (root / "rev_b0.json").write_text(
                json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not opposite"):
                run_dwi_preprocessing(
                    DwiPreprocessingConfig(
                        subject_id="sub01",
                        input_root=subject,
                        output_dir=root / "out_same_pe",
                        execute=True,
                        reverse_phase_b0=rev,
                        phase_encoding_direction="AP",
                        readout_time=0.05,
                    )
                )

            # Reverse sidecar with invalid TotalReadoutTime
            (root / "rev_b0.json").write_text(
                json.dumps({"PhaseEncodingDirection": "j", "TotalReadoutTime": -0.02}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "TotalReadoutTime must be positive and finite"):
                run_dwi_preprocessing(
                    DwiPreprocessingConfig(
                        subject_id="sub01",
                        input_root=subject,
                        output_dir=root / "out_bad_rev_ro",
                        execute=True,
                        reverse_phase_b0=rev,
                        phase_encoding_direction="AP",
                        readout_time=0.05,
                    )
                )

    def test_fsl_native_run_fsl_correction_validates_forward_and_reverse_b0_before_topup(self) -> None:
        from preprocessing.fsl_native import run_fsl_correction

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            scratch = root / "scratch"
            src = root / "dwi.nii.gz"
            bvec = root / "bvec"
            bval = root / "bval"
            fwd_b0 = root / "fwd_b0.nii.gz"
            rev_b0 = root / "rev_b0.nii.gz"
            topup_cfg = root / "b02b0.cnf"
            for p in (src, bvec, bval, fwd_b0, rev_b0, topup_cfg):
                p.write_text("x", encoding="utf-8")
            commands = {
                name: str(root / "bin" / name)
                for name in ("mrinfo", "mrconvert", "dwi2mask", "maskfilter", "mrcat", "mrtransform", "mrcalc", "eddy", "topup", "applytopup")
            }
            calls: list[str] = []

            def _sim(cmd):
                tool = Path(cmd[0]).name
                calls.append(tool)
                if tool == "mrconvert":
                    Path(cmd[2]).write_text("ok", encoding="utf-8")
                elif tool == "mrinfo":
                    target = Path(cmd[1])
                    out_json = Path(cmd[cmd.index("-json_all") + 1])
                    origin_x = 10.0 if target == rev_b0 else 0.0
                    out_json.write_text(
                        json.dumps({
                            "size": [4, 4, 4, 3 if target.name == "dwi.mif" else 1],
                            "strides": [1, 2, 3, 4],
                            "spacing": [2.0, 2.0, 2.0, 1.0],
                            "transform": [[1, 0, 0, origin_x], [0, 1, 0, 0], [0, 0, 1, 0]],
                            "keyval": {"dw_scheme": [[0, 0, 0, 0], [1, 0, 0, 1000], [0, 1, 0, 1000]]},
                        }),
                        encoding="utf-8",
                    )
                return ""

            with self.assertRaisesRegex(ValueError, r"origin differs by 10\.000 mm"):
                run_fsl_correction(
                    input_dwi=src,
                    input_bvec=bvec,
                    input_bval=bval,
                    output_dwi=root / "out.nii.gz",
                    output_bvec=root / "out.bvec",
                    output_bval=root / "out.bval",
                    scratch_dir=scratch,
                    phase_encoding="AP",
                    readout_time=0.05,
                    se_epi_pair=None,
                    forward_b0=fwd_b0,
                    reverse_b0=rev_b0,
                    topup_config=topup_cfg,
                    commands=commands,
                    eddy_options=("--nthr=1",),
                    run_stage=_sim,
                )
            self.assertNotIn("mrcat", calls)
            self.assertNotIn("topup", calls)


# --------------------------------------------------------------------------- Item 8 (P2)


class Item8DtiDkiRamBudgetAndDocsTests(unittest.TestCase):
    """[Item 8] DTI & DKI cap concurrency under low available RAM (128 MiB -> 1 worker) and docs match CPU-only."""

    def test_dti_fit_workers_caps_to_one_under_128_mib_and_scales_on_high_ram(self) -> None:
        from shared.hardware_adapt import GIB, MIB
        from tractography.dti import _resolve_dti_fit_workers

        # Under 128 MiB available RAM, DTI MUST throttle down to 1 worker thread (NOT 16!)
        workers_low, chunk_low = _resolve_dti_fit_workers(
            24,
            n_vox=200_000,
            n_gradients=288,
            is_wls=True,
            available_ram_bytes=128 * MIB,
        )
        self.assertEqual(workers_low, 1)
        self.assertLess(chunk_low, 32768)

        workers_low_clin, _ = _resolve_dti_fit_workers(
            16,
            n_vox=100_000,
            n_gradients=30,
            is_wls=True,
            available_ram_bytes=128 * MIB,
        )
        self.assertEqual(workers_low_clin, 1)

        # Under ample host RAM (32 GiB available), full multi-core performance (16 workers) is preserved!
        workers_high, chunk_high = _resolve_dti_fit_workers(
            24,
            n_vox=500_000,
            n_gradients=288,
            is_wls=True,
            available_ram_bytes=32 * GIB,
        )
        self.assertEqual(workers_high, 16)
        self.assertEqual(chunk_high, 32768)

    def test_dti_fit_fa_and_peaks_does_not_open_16_thread_pool_under_128_mib_ram(self) -> None:
        from dipy.core.gradients import gradient_table
        from dipy.data import default_sphere
        from shared.hardware_adapt import MIB
        from tractography import dti

        dirs = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
            dtype=float,
        )
        dirs[1:] /= np.linalg.norm(dirs[1:], axis=1, keepdims=True)
        bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=float)
        gtab = gradient_table(bvals, bvecs=dirs)
        mask = np.ones((40, 40, 41), dtype=bool)  # 65,600 voxels (>= 65,536 parallel threshold)
        data = np.broadcast_to(np.array([500, 200, 350, 360, 250, 260, 340], dtype=np.float32), (int(mask.sum()), 7)).copy()

        with (
            mock.patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=128 * MIB),
            mock.patch("shared.numeric_threads.reserved_stage_cpus", return_value=None),
            mock.patch("os.cpu_count", return_value=24),
            mock.patch("concurrent.futures.ThreadPoolExecutor") as pool_cls,
        ):
            fa, tracking_mask, peaks, dev = dti._fit_dti_fa_and_peaks(
                data,
                mask,
                gtab,
                default_sphere,
                fit_method="WLS",
                min_fa=0.2,
                max_angle_deg=30.0,
                device="auto",
            )
            pool_cls.assert_not_called()
        self.assertEqual(dev, "cpu")
        self.assertTrue(np.isfinite(fa).all())
        self.assertTrue(bool(tracking_mask.any()))

    def test_dki_and_adaptive_spawn_workers_throttle_to_one_under_low_available_ram(self) -> None:
        from microstructure import dki
        from shared.hardware_adapt import MIB, adaptive_spawn_workers

        # Low available RAM (128 MiB = 0.125 GiB) MUST override high total physical RAM (32 / 64 GiB) -> 1 worker!
        for plat, tot in (("win32", 32.0), ("darwin", 16.0), ("linux", 64.0)):
            with self.subTest(platform=plat, total_ram_gib=tot):
                self.assertEqual(
                    adaptive_spawn_workers(
                        16,
                        per_worker_ram_gib=1.0,
                        reserve_ram_gib=4.0,
                        max_workers=4,
                        platform_name=plat,
                        total_ram_gib=tot,
                        available_ram_gib=0.125,
                    ),
                    1,
                )

        # High available RAM preserves multi-worker scaling on all platforms.
        self.assertEqual(
            adaptive_spawn_workers(
                16,
                per_worker_ram_gib=1.0,
                reserve_ram_gib=4.0,
                max_workers=4,
                platform_name="win32",
                total_ram_gib=32.0,
                available_ram_gib=24.0,
            ),
            4,
        )
        self.assertEqual(
            adaptive_spawn_workers(
                16,
                max_workers=8,
                platform_name="linux",
                total_ram_gib=256.0,
                available_ram_gib=200.0,
            ),
            16,
        )

        # DKI spawn & fork resolvers both throttle to 1 under 128 MiB available RAM
        fake_cfg = types.SimpleNamespace(num_threads=None)
        fake_main = types.SimpleNamespace(__file__=__file__)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch.object(dki.sys, "platform", "win32"),
                mock.patch.object(dki.sys, "executable", r"C:\Slicer\bin\python-real.exe"),
                mock.patch.dict(sys.modules, {"__main__": fake_main}),
                mock.patch("os.cpu_count", return_value=24),
                mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=32.0),
                mock.patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=128 * MIB),
            ):
                self.assertEqual(dki._resolve_dki_spawn_workers(fake_cfg), 1)

            with (
                mock.patch.object(dki.sys, "platform", "linux"),
                mock.patch("microstructure.dki.os.sched_getaffinity", return_value=set(range(64)), create=True),
                mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=128.0),
                mock.patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=128 * MIB),
            ):
                self.assertEqual(dki._resolve_dki_fork_workers(fake_cfg), 1)

    def test_support_matrix_docs_list_dti_and_dki_as_cpu_only(self) -> None:
        en_text = (REPO_ROOT / "docs" / "setup" / "SUPPORT_MATRIX.md").read_text(encoding="utf-8")
        zh_text = (REPO_ROOT / "docs" / "setup" / "SUPPORT_MATRIX.zh-CN.md").read_text(encoding="utf-8")

        dki_en = next(line for line in en_text.splitlines() if line.startswith("| **Microstructure** (`DKI`)"))
        dti_en = next(line for line in en_text.splitlines() if line.startswith("| **Tractography** (`DTI Det`)"))
        self.assertIn("CPU only", dki_en)
        self.assertNotIn("CUDA", dki_en)
        self.assertIn("CPU only", dti_en)
        self.assertNotIn("CUDA", dti_en)

        dki_zh = next(line for line in zh_text.splitlines() if line.startswith("| **微结构**（`DKI`）"))
        dti_zh = next(line for line in zh_text.splitlines() if line.startswith("| **纤维追踪**（`DTI` 确定性追踪）"))
        self.assertIn("仅 CPU", dki_zh)
        self.assertNotIn("CUDA", dki_zh)
        self.assertIn("仅 CPU", dti_zh)
        self.assertNotIn("CUDA", dti_zh)


if __name__ == "__main__":
    unittest.main()
