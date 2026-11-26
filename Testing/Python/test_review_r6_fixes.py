"""Round-6 surgical regression tests covering all review findings on 4d5043b:
1. P1: Gibbs (GIBBS_CONTRACT_VERSION=2) & DDParcel (DDPARCEL_CONTRACT_VERSION=2) invalidate stale pre-fix caches.
2. P1: Windows CUDA Torch auto-install passes valid versioned CUDA backends (never bare 'cu') to SlicerPyTorch/ltt,
       uninstalls CPU torch before reinstalling when needed, and standalone external Python never returns
       'completed' while retaining CPU torch.
3. P1: Linux DKI spawn workers never bypass max_workers or available RAM budget when available RAM > 48 GiB.
4. P1: UKF --xml probe timeout fails closed (reserves 1 GPU slot on retry) + DDSurfer clamps --max-gpu-gib to
       SLICERDSPACE_MAX_VRAM_GIB / SLICERDSPACE_MAX_VRAM_GB.
5. P2: Batch subject logs for 'Alpha beta' and 'Alpha_beta' never collide + WDDM partial per-process VRAM
       omits total owned_vram_mib instead of presenting partial sum as total.
6. Test isolation: Linux DKI fork test stays isolated even when torch is already imported.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None


class GibbsAndDDParcelCacheInvalidationTests(unittest.TestCase):
    """Finding 1 (P1): Stale pre-fix Gibbs and DDParcel caches must invalidate and recompute."""

    def test_gibbs_contract_version_rejects_stale_pre_fix_gibbs_cache_and_reuses_v2(self) -> None:
        from preprocessing.pipeline import (
            DwiPreprocessingConfig,
            GIBBS_CONTRACT_VERSION,
            _preprocessing_cache_key,
            preprocessing_cache_key_matches,
            preprocessing_gibbs_current,
        )
        from shared.pipeline import RawInputBundle, current_preprocessed_outputs

        self.assertGreaterEqual(GIBBS_CONTRACT_VERSION, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inp = root / "input"
            inp.mkdir()
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (inp / name).write_bytes(b"raw-" + name.encode())
            (inp / "dwi.json").write_text(
                json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}),
                encoding="utf-8",
            )
            raw = RawInputBundle(
                subject_root=root,
                subject_id="sub01",
                input_dir=inp,
                dwi_path=inp / "data.nii.gz",
                bval_path=inp / "bvals",
                bvec_path=inp / "bvecs",
                mask_path=inp / "nodif_brain_mask.nii.gz",
            )
            fake_backend = {"execution_backend": "legacy", "commands": {}}
            fake_mask = {"backend": "provided"}
            cfg_gibbs = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=root,
                dipy_run_gibbs=True,
                phase_encoding_direction="AP",
                readout_time=0.05,
                brain_mask_backend="provided",
            )
            cfg_no_gibbs = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=root,
                dipy_run_gibbs=False,
                phase_encoding_direction="AP",
                readout_time=0.05,
                brain_mask_backend="provided",
            )

            new_key = _preprocessing_cache_key(raw, cfg_gibbs, fake_backend, fake_mask)
            self.assertEqual(new_key["parameters"]["gibbs_contract_version"], GIBBS_CONTRACT_VERSION)

            # Pre-fix Gibbs cache key had dipy_run_gibbs=True without gibbs_contract_version (implicit v1)
            stale_key = json.loads(json.dumps(new_key))
            stale_key["parameters"].pop("gibbs_contract_version")
            self.assertFalse(preprocessing_cache_key_matches(stale_key, new_key))
            self.assertFalse(preprocessing_gibbs_current({"status": "completed", "cache_key": stale_key}))

            # Explicit v1 is also rejected; v2 matches cleanly
            stale_v1 = json.loads(json.dumps(new_key))
            stale_v1["parameters"]["gibbs_contract_version"] = 1
            self.assertFalse(preprocessing_cache_key_matches(stale_v1, new_key))
            self.assertFalse(preprocessing_gibbs_current({"status": "completed", "cache_key": stale_v1}))
            self.assertTrue(preprocessing_cache_key_matches(new_key, new_key))
            self.assertTrue(preprocessing_gibbs_current({"status": "completed", "cache_key": new_key}))

            # Non-Gibbs runs omit gibbs_contract_version and remain unaffected
            no_gibbs_key = _preprocessing_cache_key(raw, cfg_no_gibbs, fake_backend, fake_mask)
            self.assertNotIn("gibbs_contract_version", no_gibbs_key["parameters"])
            self.assertTrue(preprocessing_cache_key_matches(no_gibbs_key, no_gibbs_key))
            self.assertTrue(preprocessing_gibbs_current({"status": "completed", "cache_key": no_gibbs_key}))

            # Downstream current_preprocessed_outputs also rejects stale pre-fix Gibbs manifests
            preproc_dir = root / "outputs" / "01_preproc"
            preproc_dir.mkdir(parents=True)
            for suffix in ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec", "brain_mask_final.nii.gz"):
                (preproc_dir / f"sub01_{suffix}").write_bytes(b"preproc")
            manifest_path = preproc_dir / "sub01_preproc_manifest.json"

            manifest_path.write_text(json.dumps({"status": "completed", "cache_key": stale_key}), encoding="utf-8")
            self.assertIsNone(current_preprocessed_outputs(root, None, "sub01"))

            manifest_path.write_text(json.dumps({"status": "completed", "cache_key": new_key}), encoding="utf-8")
            self.assertIsNotNone(current_preprocessed_outputs(root, None, "sub01"))

    def test_newly_exported_user_mask_overrides_preprocessed_mask_automatically(self) -> None:
        from preprocessing.pipeline import DwiPreprocessingConfig, _preprocessing_cache_key
        from shared.pipeline import RawInputBundle, current_preprocessed_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            inp = root / "input"
            inp.mkdir()
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (inp / name).write_bytes(b"raw-" + name.encode())
            preproc_dir = root / "preprocessed"
            preproc_dir.mkdir(parents=True)
            auto_mask = preproc_dir / "nodif_brain_mask.nii.gz"
            auto_mask.write_bytes(b"auto-mask")
            auto_mask.with_name("nodif_brain_mask.manifest.json").write_text("{}", encoding="utf-8")
            raw = RawInputBundle(
                subject_root=root,
                subject_id="sub01",
                input_dir=inp,
                dwi_path=inp / "data.nii.gz",
                bval_path=inp / "bvals",
                bvec_path=inp / "bvecs",
                mask_path=auto_mask,
            )
            cfg = DwiPreprocessingConfig(
                subject_id="sub01",
                input_root=root,
                dipy_run_gibbs=True,
                phase_encoding_direction="AP",
                readout_time=0.05,
                brain_mask_backend="pnl_cnn",
            )
            cache_key = _preprocessing_cache_key(
                raw, cfg, {"execution_backend": "legacy", "commands": {}}, {"backend": "pnl_cnn"}
            )
            for suffix in ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec", "brain_mask_final.nii.gz"):
                (preproc_dir / f"sub01_{suffix}").write_bytes(b"preproc")
            (preproc_dir / "sub01_preproc_manifest.json").write_text(
                json.dumps({"status": "completed", "cache_key": cache_key}),
                encoding="utf-8",
            )

            before = current_preprocessed_outputs(root, None, "sub01")
            self.assertIsNotNone(before)
            self.assertEqual(before.mask_path, preproc_dir / "sub01_brain_mask_final.nii.gz")

            # Exporting an edited mask to input/nodif_brain_mask.nii.gz must be picked up automatically
            # while still reusing sub01_dwi_final.nii.gz (even across repeated edits).
            edited_mask = inp / "nodif_brain_mask.nii.gz"
            edited_mask.write_bytes(b"edited-user-mask")
            after = current_preprocessed_outputs(root, None, "sub01")
            self.assertIsNotNone(after)
            self.assertEqual(after.dwi_path, preproc_dir / "sub01_dwi_final.nii.gz")
            self.assertEqual(after.mask_path, edited_mask)

            edited_mask.write_bytes(b"edited-user-mask-v2")
            after_v2 = current_preprocessed_outputs(root, None, "sub01")
            self.assertIsNotNone(after_v2)
            self.assertEqual(after_v2.dwi_path, preproc_dir / "sub01_dwi_final.nii.gz")
            self.assertEqual(after_v2.mask_path, edited_mask)

    @unittest.skipUnless(HAS_NIBABEL, "nibabel unavailable")
    def test_ddparcel_contract_version_2_invalidates_stale_v1_and_unversioned_receipts(self) -> None:
        import nibabel as nib
        from anatomy.parcellation.pipeline import (
            DDPARCEL_CONTRACT_VERSION,
            DDParcelRegisteredConfig,
            _registered_signature_matches,
            run_ddparcel_from_registered_bundle,
        )
        from shared import pipeline as shared_pipeline
        from shared.orchestrator import _parcellation_complete, _parcellation_outputs
        from test_shared_preparation import inputs as make_raw_inputs

        self.assertGreaterEqual(DDPARCEL_CONTRACT_VERSION, 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            raw = make_raw_inputs(root)
            bundle = shared_pipeline._build_registered_bundle(raw, "wls", root / "atlas.nii.gz")
            bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
            for name in bundle.__dataclass_fields__:
                if name.endswith("_path") and name != "manifest_path":
                    getattr(bundle, name).write_bytes(b"registered-" + name.encode())
            source_sigs = {
                key: shared_pipeline._cache_file_signature(getattr(raw, key + "_path"))
                for key in ("dwi", "bval", "bvec", "mask")
            }
            bundle.manifest_path.write_text(
                json.dumps({"native_bundle": {"manifest": {"source_files": source_sigs}}}),
                encoding="utf-8",
            )
            weights = root / "weights"
            weights.mkdir()
            (weights / "model.pkl").write_bytes(b"weights")
            out = _parcellation_outputs(root, "x")["wmparc"].parent
            out.mkdir(parents=True, exist_ok=True)
            cfg = DDParcelRegisteredConfig("x", bundle, out, raw.mask_path, weights, device="cpu")
            affine = np.eye(4)

            def _save_nii(path: Path, data: np.ndarray) -> None:
                nib.save(nib.Nifti1Image(data, affine), str(path))

            pred_label = {"val": 1001}

            def _norm(src, msk, dst, flip):
                _save_nii(dst, np.ones((4, 4, 4), dtype=np.float32))

            def _pred(folder, sid, wdir, log, **kw):
                arr = np.full((4, 4, 4), pred_label["val"], dtype=np.int32)
                nib.save(nib.MGHImage(arr, affine), str(folder / f"{sid}-DDSurfer-wmparc-Reg.mgz"))

            def _resample(cmd, log=None):
                arr = np.full((4, 4, 4), pred_label["val"], dtype=np.int32)
                _save_nii(Path(cmd[-1]), arr)

            with (
                patch("anatomy.parcellation.pipeline._run_normalize", side_effect=_norm),
                patch("anatomy.parcellation.pipeline._run_ddsurfer_pred", side_effect=_pred) as mock_pred,
                patch("anatomy.parcellation.pipeline.resolve_cli_executable", return_value="resample"),
                patch("anatomy.parcellation.pipeline._run_cli", side_effect=_resample),
                patch("anatomy.parcellation.qc.validate_outputs"),
            ):
                # 1. Fresh run produces v2 receipt with FP32 label 1001
                run_ddparcel_from_registered_bundle(cfg)
                self.assertEqual(mock_pred.call_count, 1)
                receipt_path = out / "ddparcel_preparation.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(receipt["contract_version"], DDPARCEL_CONTRACT_VERSION)

                # Fresh v2 hits cache
                run_ddparcel_from_registered_bundle(cfg)
                self.assertEqual(mock_pred.call_count, 1)

                # 2. Simulate stale pre-fix FP16 v1 output (label 2) and v1 receipt
                stale_receipt = dict(receipt, contract_version=1)
                self.assertFalse(_registered_signature_matches(stale_receipt, receipt))
                unversioned_receipt = {k: v for k, v in receipt.items() if k != "contract_version"}
                self.assertFalse(_registered_signature_matches(unversioned_receipt, receipt))

                wmparc_path = out / "x-DDSurfer-wmparc.nii.gz"
                _save_nii(wmparc_path, np.full((4, 4, 4), 2, dtype=np.int32))
                receipt_path.write_text(json.dumps(stale_receipt), encoding="utf-8")

                # Orchestrator _parcellation_complete MUST reject stale v1 receipt
                with (
                    patch("shared.orchestrator._resolve_analysis_inputs", return_value={
                        key: getattr(raw, key + "_path") for key in ("dwi", "bval", "bvec", "mask")
                    }),
                    patch("shared.orchestrator._tensor_bundle_key_current", return_value=True),
                    patch("shared.scalar_qc.scalar_qc_current", return_value=True),
                ):
                    self.assertFalse(_parcellation_complete(root, "x"))

                # Normal resume run MUST miss cache and recompute label 1001
                run_ddparcel_from_registered_bundle(cfg)
                self.assertEqual(mock_pred.call_count, 2)
                recomputed_data = np.asarray(nib.load(str(wmparc_path)).dataobj)
                self.assertEqual(int(recomputed_data[0, 0, 0]), 1001)
                refreshed = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertEqual(refreshed["contract_version"], DDPARCEL_CONTRACT_VERSION)

                # Orchestrator _parcellation_complete accepts v2 receipt cleanly
                with (
                    patch("shared.orchestrator._resolve_analysis_inputs", return_value={
                        key: getattr(raw, key + "_path") for key in ("dwi", "bval", "bvec", "mask")
                    }),
                    patch("shared.orchestrator._tensor_bundle_key_current", return_value=True),
                    patch("shared.scalar_qc.scalar_qc_current", return_value=True),
                ):
                    self.assertTrue(_parcellation_complete(root, "x"))


class WindowsCudaTorchAutoInstallTests(unittest.TestCase):
    """Finding 2 (P1): Real SlicerPyTorch/light-the-torch backend contract + standalone Python CUDA guard."""

    _LTT_CUDA_TOKEN_RE = re.compile(r"^cu(?:da)?\d+(?:\.\d+)?$", re.IGNORECASE)

    def test_pytorch_utils_uses_plural_backends_never_bare_cu_and_uninstalls_cpu_torch(self) -> None:
        from shared import python_setup

        state = {"torch_ver": "2.15.0+cpu"}
        events: list[tuple] = []

        class _FakeCB:
            def __init__(self, spec: str):
                self.local_specifier = spec

            def __str__(self) -> str:
                return self.local_specifier

        class _UpstreamPyTorchUtilsLogic:
            """Matches fepegar/SlicerPyTorch PyTorchUtilsLogic exact method names and signatures."""

            @staticmethod
            def getCompatibleComputationBackends(forceComputationBackend=None, torchVersionRequirement=None):
                events.append(("detect", torchVersionRequirement))
                return [_FakeCB("cpu"), _FakeCB("cu118"), _FakeCB("cu124"), _FakeCB("cu126"), _FakeCB("cu128")]

            def uninstallTorch(self, askConfirmation=False, forceComputationBackend=None):
                events.append(("uninstall",))
                state["torch_ver"] = None

            def installTorch(
                self,
                askConfirmation=False,
                forceComputationBackend=None,
                torchVersionRequirement=None,
                torchvisionVersionRequirement=None,
            ):
                # Enforce Slicer/light-the-torch _cb.py ComputationBackend.from_str() contract!
                self.assert_valid_ltt_backend(forceComputationBackend)
                events.append(("install", askConfirmation, forceComputationBackend, torchVersionRequirement))
                state["torch_ver"] = "2.6.0+cu128"
                return SimpleNamespace(__version__=state["torch_ver"])

            @staticmethod
            def assert_valid_ltt_backend(spec: str | None) -> None:
                if not spec:
                    raise AssertionError("Expected explicit versioned CUDA backend")
                for token in str(spec).split(","):
                    tok = token.strip()
                    if not WindowsCudaTorchAutoInstallTests._LTT_CUDA_TOKEN_RE.match(tok):
                        raise ValueError(f"Unable to parse {tok} into a computation backend")

        def _fake_meta_version(name: str) -> str:
            if name == "torch":
                if state["torch_ver"] is None:
                    raise python_setup.importlib.metadata.PackageNotFoundError("torch")
                return state["torch_ver"]
            return "0.21.0"

        helper = SimpleNamespace(PyTorchUtilsLogic=lambda: _UpstreamPyTorchUtilsLogic())
        with (
            patch.dict("sys.modules", {"PyTorchUtils": helper}),
            patch.object(python_setup, "_query_nvidia_smi_driver_and_gpus", return_value=((560, 94), False)),
            patch.object(python_setup, "requirement_satisfied", side_effect=lambda req: state["torch_ver"] is not None),
            patch.object(python_setup.importlib.metadata, "version", side_effect=_fake_meta_version),
            patch.object(
                python_setup,
                "installed_versions",
                side_effect=lambda: ({"torch": state["torch_ver"]} if state["torch_ver"] else {}),
            ),
            patch.object(python_setup, "bundled_shortfalls", return_value=[]),
        ):
            rec = python_setup.install_requirements(
                (),
                loaded=("torch",),
                torch_minimum="2.3.1",
                require_cuda_torch=True,
                replace_cpu_torch=True,
            )

        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["backend"], "PyTorch_extension")
        self.assertEqual(events[0], ("detect", ">=2.3.1"))
        self.assertEqual(events[1], ("uninstall",))
        self.assertEqual(events[2], ("install", False, "cu118,cu124,cu126,cu128", ">=2.3.1"))

    def test_pytorch_utils_fails_cleanly_without_passing_bare_cu_when_no_cuda_driver(self) -> None:
        from shared import python_setup

        install_called = {"n": 0}

        class _CpuOnlyDriverLogic:
            @staticmethod
            def getCompatibleComputationBackends(forceComputationBackend=None, torchVersionRequirement=None):
                return ["cpu"]

            def installTorch(self, askConfirmation=False, forceComputationBackend=None, torchVersionRequirement=None):
                install_called["n"] += 1
                return SimpleNamespace(__version__="2.15.0+cpu")

        helper = SimpleNamespace(PyTorchUtilsLogic=lambda: _CpuOnlyDriverLogic())
        with (
            patch.dict("sys.modules", {"PyTorchUtils": helper}),
            patch.object(python_setup, "_query_nvidia_smi_driver_and_gpus", return_value=(None, False)),
            patch.object(python_setup, "requirement_satisfied", return_value=True),
            patch.object(python_setup.importlib.metadata, "version", return_value="2.15.0+cpu"),
            patch.object(python_setup, "installed_versions", return_value={"torch": "2.15.0+cpu"}),
            patch.object(python_setup, "bundled_shortfalls", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "No CUDA-compatible NVIDIA"),
        ):
            python_setup.install_requirements(
                (),
                torch_minimum="2.3.1",
                require_cuda_torch=True,
                replace_cpu_torch=True,
            )
        self.assertEqual(install_called["n"], 0)

    def test_standalone_python_env_never_returns_completed_with_cpu_torch(self) -> None:
        from shared import python_setup

        # Case A: pip runs (using CUDA index) but CPU-only torch remains -> MUST raise, never return completed
        pip_cmds: list[list[str]] = []

        def _fake_pip_noop(cmd, **kwargs):
            pip_cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        with (
            patch.object(python_setup, "_detect_host_cuda_backends", return_value=["cu118", "cu124", "cu126", "cu128"]),
            patch.object(python_setup, "requirement_satisfied", return_value=True),
            patch.object(python_setup.importlib.metadata, "version", return_value="2.15.0+cpu"),
            patch.object(python_setup, "installed_versions", return_value={"torch": "2.15.0+cpu"}),
            patch.object(python_setup, "bundled_shortfalls", return_value=[]),
            patch.object(python_setup.subprocess, "run", side_effect=_fake_pip_noop),
            self.assertRaisesRegex(RuntimeError, "CPU-only|CUDA"),
        ):
            python_setup.install_requirements(
                ("torch>=2.0.0",),
                protected=False,
                torch_minimum=None,
                require_cuda_torch=True,
                replace_cpu_torch=True,
            )
        self.assertEqual(len(pip_cmds), 1)
        self.assertIn("--force-reinstall", pip_cmds[0])
        self.assertIn("https://download.pytorch.org/whl/cu128", pip_cmds[0])

        # Case B: standalone pip installs CUDA wheel -> succeeds cleanly
        state = {"ver": "2.15.0+cpu"}

        def _fake_pip_cuda(cmd, **kwargs):
            state["ver"] = "2.6.0+cu128"
            return subprocess.CompletedProcess(cmd, 0)

        with (
            patch.object(python_setup, "_detect_host_cuda_backends", return_value=["cu126", "cu128"]),
            patch.object(python_setup, "requirement_satisfied", return_value=True),
            patch.object(python_setup.importlib.metadata, "version", side_effect=lambda _n: state["ver"]),
            patch.object(python_setup, "installed_versions", side_effect=lambda: {"torch": state["ver"]}),
            patch.object(python_setup, "bundled_shortfalls", return_value=[]),
            patch.object(python_setup.subprocess, "run", side_effect=_fake_pip_cuda),
        ):
            rec = python_setup.install_requirements(
                ("torch>=2.0.0",),
                protected=False,
                torch_minimum=None,
                require_cuda_torch=True,
                replace_cpu_torch=True,
            )
        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["backend"], "python_subprocess_cuda_wheel")


class LinuxDkiSpawnMemoryCapTests(unittest.TestCase):
    """Finding 3 (P1): Linux DKI spawn workers must respect max_workers and RAM budget above 48 GiB."""

    def test_128_core_linux_server_dki_spawn_does_not_jump_from_16_to_96_at_49_gib(self) -> None:
        from microstructure import dki
        from shared.hardware_adapt import GIB, adaptive_spawn_workers

        fake_cfg = SimpleNamespace(num_threads=96)
        fake_main = SimpleNamespace(__file__=__file__)

        with (
            patch.object(dki.sys, "platform", "linux"),
            patch.object(dki.sys, "executable", "/usr/bin/python3"),
            patch.dict(sys.modules, {"__main__": fake_main}),
            patch("microstructure.dki.os.sched_getaffinity", return_value=set(range(128)), create=True),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=256.0),
        ):
            with patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=48 * GIB):
                w48 = dki._resolve_dki_spawn_workers(fake_cfg)
            with patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=49 * GIB):
                w49 = dki._resolve_dki_spawn_workers(fake_cfg)
            with patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=10 * GIB):
                w10 = dki._resolve_dki_spawn_workers(fake_cfg)

        self.assertEqual(w48, 16)
        self.assertEqual(w49, 16)
        self.assertEqual(w10, 10)

        # Even direct adaptive_spawn_workers on Linux never exceeds available_ram_gib / per_worker_ram_gib
        self.assertEqual(
            adaptive_spawn_workers(
                96,
                per_worker_ram_gib=1.0,
                reserve_ram_gib=4.0,
                max_workers=16,
                platform_name="linux",
                total_ram_gib=256.0,
                available_ram_gib=49.0,
                enforce_on_linux=False,
            ),
            49,
        )
        self.assertEqual(
            adaptive_spawn_workers(
                96,
                per_worker_ram_gib=1.0,
                reserve_ram_gib=4.0,
                max_workers=16,
                platform_name="linux",
                total_ram_gib=256.0,
                available_ram_gib=49.0,
                enforce_on_linux=True,
            ),
            16,
        )


class UkfProbeRetryAndDdsurferVramCapTests(unittest.TestCase):
    """Finding 4 (P1): UKF --xml probe timeout reserves GPU slot + DDSurfer clamps --max-gpu-gib."""

    def test_ukf_xml_probe_timeout_then_success_still_reserves_one_gpu_slot(self) -> None:
        from shared import cohort_schedule
        from tractography import ukf

        gpu_schema = ET.fromstring(
            "<executable><parameters>"
            "<boolean><name>useGPU</name><longflag>useGPU</longflag></boolean>"
            "<integer><name>gpuDevice</name><longflag>gpuDevice</longflag></integer>"
            "<integer><name>gpuPrecision</name><longflag>gpuPrecision</longflag></integer>"
            "</parameters></executable>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda_exe = root / "ukf-gpu"
            cuda_exe.write_bytes(b"#!/bin/sh\n")
            sched = cohort_schedule.create_schedule(
                root / "sched", cpu_budget=4, memory_gb=8, stage_cpus=2, stage_memory_gb=2, gpu_slots=1
            )
            raw = SimpleNamespace(subject_id="sub01")
            probe_results = [
                subprocess.TimeoutExpired([str(cuda_exe), "--xml"], 30),
                (gpu_schema, "<executable/>", 0),
            ]

            def _flaky_schema(_binary, _env):
                item = probe_results.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            with (
                patch.dict(os.environ, {cohort_schedule.ENV: str(sched)}, clear=False),
                patch.object(ukf, "resolve_executable", return_value=cuda_exe),
                patch.object(ukf, "inspect_schema", side_effect=_flaky_schema),
                patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True),
                patch("shared.pipeline.build_cli_environment", return_value={}),
                patch.object(ukf, "signature", return_value={"v": 1}),
                patch.object(ukf, "current", return_value=False),
                patch.object(ukf, "build_command", return_value=[str(cuda_exe), "--useGPU"]),
                patch.object(ukf, "_validated_inputs", side_effect=RuntimeError("stop-after-schema")),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-schema"):
                    ukf.run_ukf(raw, root / "out", device="cuda:0")

            summary = cohort_schedule.schedule_summary(sched)
            self.assertEqual(summary["peak_declared_reservations"]["gpu"], 1)
            self.assertEqual(probe_results, [])

    def test_ddsurfer_clamps_max_gpu_gib_to_global_vram_cap(self) -> None:
        from anatomy.surface.pipeline import SurfaceConfig, _raw_command
        from anatomy.surface.prediction import PredictionStageConfig, predict_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = SimpleNamespace(**{name: root / name for name in ("dwi_path", "bval_path", "bvec_path", "mask_path")})
            for path in vars(bundle).values():
                path.touch()
            resources = dict(checkpoint_root="/weights", template_dir="/templates", registration_atlas="/atlas")

            for env_key in ("SLICERDSPACE_MAX_VRAM_GIB", "SLICERDSPACE_MAX_VRAM_GB"):
                captured_cmd: list[str] = []
                cfg = PredictionStageConfig(
                    subject_id="sub01",
                    hemisphere="left",
                    predict_mode="all",
                    device="cuda:0",
                    step_size=0.02,
                    n_svf=2,
                    n_res=3,
                    template_dir=root,
                    checkpoint_root=root,
                    input_root=root,
                    output_root=root / "out",
                    max_gpu_gib=24.0,
                )
                with (
                    patch.dict(os.environ, {env_key: "8"}, clear=False),
                    patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                    patch("anatomy.surface.prediction.run_backend", side_effect=lambda cmd, **kw: captured_cmd.extend(map(str, cmd))),
                ):
                    other_key = "SLICERDSPACE_MAX_VRAM_GB" if env_key.endswith("GIB") else "SLICERDSPACE_MAX_VRAM_GIB"
                    os.environ.pop(other_key, None)
                    predict_subject(cfg)
                    raw_cmd, _ = _raw_command(
                        SurfaceConfig("sub01", root, root, input_bundle=bundle, slicer_home=root, max_gpu_gib=24.0),
                        resources,
                        root,
                    )
                    smaller_cmd, _ = _raw_command(
                        SurfaceConfig("sub01", root, root, input_bundle=bundle, slicer_home=root, max_gpu_gib=6.0),
                        resources,
                        root,
                    )

                self.assertEqual(captured_cmd[captured_cmd.index("--max-gpu-gib") + 1], "8.0")
                self.assertEqual(raw_cmd[raw_cmd.index("--max-gpu-gib") + 1], "8.0")
                self.assertEqual(smaller_cmd[smaller_cmd.index("--max-gpu-gib") + 1], "6.0")


class BatchLogCollisionAndWddmPartialVramTests(unittest.TestCase):
    """Finding 5 (P2): Collision-free batch subject logs + WDDM partial per-process VRAM reporting."""

    def test_alpha_space_beta_and_alpha_underscore_beta_do_not_collide(self) -> None:
        from scripts.run_batch_pipeline import BatchSubject, _assign_subject_log_stems, _run_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            log_dir = root / "logs"
            s1 = BatchSubject("Alpha beta", root / "Alpha beta", "glob", "subject_root")
            s2 = BatchSubject("Alpha_beta", root / "Alpha_beta", "glob", "subject_root")
            s3 = BatchSubject("sub-01", root / "sub-01", "glob", "subject_root")
            s4 = BatchSubject("SubA", root / "SubA", "glob", "subject_root")
            s5 = BatchSubject("suba", root / "suba", "glob", "subject_root")

            stems = _assign_subject_log_stems([s1, s2, s3, s4, s5])
            self.assertEqual(stems["sub-01"], "sub-01")
            self.assertEqual(stems["Alpha_beta"], "Alpha_beta")
            self.assertNotEqual(stems["Alpha beta"].casefold(), stems["Alpha_beta"].casefold())
            self.assertNotEqual(stems["SubA"].casefold(), stems["suba"].casefold())
            self.assertEqual(len({v.casefold() for v in stems.values()}), 5)

            # Direct _run_subject calls for both subjects must keep both logs intact
            cmd1 = [sys.executable, "-c", "print('from Alpha beta')"]
            cmd2 = [sys.executable, "-c", "print('from Alpha_beta')"]
            r1 = _run_subject(cmd1, s1, log_dir, relay=None)
            r2 = _run_subject(cmd2, s2, log_dir, relay=None)

            self.assertNotEqual(r1.stdout_log, r2.stdout_log)
            self.assertNotEqual(r1.stderr_log, r2.stderr_log)
            self.assertIn("from Alpha beta", r1.stdout_log.read_text(encoding="utf-8"))
            self.assertIn("from Alpha_beta", r2.stdout_log.read_text(encoding="utf-8"))

    def test_wddm_partial_per_process_vram_omits_total_and_records_partial_lower_bound(self) -> None:
        from shared.cohort_telemetry import CohortTelemetry, gpu_sample

        gpu_ok = subprocess.CompletedProcess([], 0, "GPU-win-0, 72, 6144\nGPU-win-1, 45, 3072\n")
        apps_partial = subprocess.CompletedProcess([], 0, "1001, 1024, GPU-win-0\n1002, [N/A], GPU-win-1\n")

        with patch("shared.cohort_telemetry.subprocess.run", side_effect=[gpu_ok, apps_partial]):
            sample = gpu_sample({1001, 1002}, executable="nvidia-smi")

        self.assertTrue(sample["available"])
        self.assertIsNone(sample["owned_vram_mib"])
        self.assertEqual(sample["owned_vram_partial_mib"], 1024.0)
        self.assertFalse(sample["owned_vram_complete"])
        self.assertIn("1/2", sample["owned_vram_note"])
        self.assertIn("1024.0", sample["owned_vram_note"])
        self.assertEqual(len(sample["devices"]), 2)
        self.assertEqual(sample["devices"][0]["used_mib"], 6144.0)

        # Complete per-process sample still sets owned_vram_mib to exact sum
        apps_complete = subprocess.CompletedProcess([], 0, "1001, 1024, GPU-win-0\n1002, 2048, GPU-win-1\n")
        with patch("shared.cohort_telemetry.subprocess.run", side_effect=[gpu_ok, apps_complete]):
            complete = gpu_sample({1001, 1002}, executable="nvidia-smi")
        self.assertEqual(complete["owned_vram_mib"], 3072.0)
        self.assertTrue(complete["owned_vram_complete"])

        # CohortTelemetry must NOT treat a partial lower bound as peak_vram_mib
        with tempfile.TemporaryDirectory() as td:
            tel = CohortTelemetry(Path(td) / "telemetry.jsonl")
            tel.monitor.sample = lambda: None
            tel.monitor.latest = {"pids": [1001, 1002], "rss_bytes": 0}
            with patch("shared.cohort_telemetry.subprocess.run", side_effect=[gpu_ok, apps_partial]):
                tel.sample()
            self.assertIsNone(tel.peak_vram_mib)


if __name__ == "__main__":
    unittest.main()
