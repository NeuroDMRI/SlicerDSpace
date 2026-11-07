#!/usr/bin/env python3
"""Native-grid masking regressions; fake inference tests adapters, NOT PNL accuracy."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import brain_mask
from shared.brain_mask import (
    BrainMaskConfig,
    brain_mask_config_from_options,
    generate_brain_mask,
    inspect_brain_mask_backend,
    validate_brain_mask,
)


ANTS_COMMANDS = ("antsRegistrationSyNQuick.sh", "antsApplyTransforms", "ImageMath")
SHAPE = (4, 5, 3)
AFFINE = np.array([[1.5, 0, 0, -7], [0, 2, 0, 11], [0, 0, 2.5, 3], [0, 0, 0, 1]], dtype=float)
ISOLATED_ENV = {
    "PATH": str(Path(sys.executable).parent),
    "SLICERDSPACE_MRTRIX_BIN_DIR": "",
    "MRTRIX_BIN_DIR": "",
    "SLICERDSPACE_PNL_MASK_PYTHON": "",
    "SLICERDSPACE_PNL_MASK_SCRIPT": "",
    "SLICERDSPACE_PNL_MASK_MODEL_DIR": "",
    "ANTSPATH": "",
    "FREESURFER_HOME": "",
}


def mask_data() -> np.ndarray:
    data = np.zeros(SHAPE, dtype=np.uint8)
    data[1:3, 1:4, 1:3] = 1
    return data


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray = AFFINE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def make_subject(root: Path, *, supplied_mask: bool = False) -> Path:
    subject = root / "sub-01"
    inputs = subject / "input"
    spatial = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) + 1
    data = np.stack((spatial + 10, spatial + 30, spatial + 900, spatial + 1500), axis=-1)
    save_nifti(inputs / "data.nii.gz", data)
    (inputs / "bvals").write_text("0 50 51 1000\n", encoding="ascii")
    (inputs / "bvecs").write_text("0 0 1 0\n0 0 0 1\n0 0 0 0\n", encoding="ascii")
    (inputs / "data.json").write_text(
        json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.065}), encoding="ascii"
    )
    if supplied_mask:
        save_nifti(inputs / "nodif_brain_mask.nii.gz", mask_data())
    return subject


def command_record(tool: Path) -> Path:
    return tool.with_name(tool.name + ".calls.jsonl")


# No trained model or anatomical inference is present in this executable fixture.
FAKE_COMMAND = textwrap.dedent(
    """\
    import argparse
    import json
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    import nibabel as nib
    import numpy as np

    args = sys.argv[1:]
    record = {"argv": args, "python": sys.executable}
    if MODE == "pnl_cnn":
        # Stands in for PythonSlicer: argv[1] is shared/pnl_mask_native.py (T2 A1), not run here.
        script = Path(args[0])
        assert script.name == "pnl_mask_native.py" and script.is_file()
        parser = argparse.ArgumentParser()
        for flag in ("--input", "--output", "--model-dir", "--template", "--device", "--record"):
            parser.add_argument(flag, required=True)
        parsed = parser.parse_args(args[1:])
        source = Path(parsed.input)
        assert source.name == "mean_b0.nii.gz"
        record["script"] = str(script)
        record["model_dir"] = parsed.model_dir
        record["template"] = parsed.template
        record["model_files"] = sorted(p.name for p in Path(parsed.model_dir).iterdir())
        candidate = Path(parsed.output)
        Path(parsed.record).write_text(json.dumps({
            "requested": parsed.device, "resolved": "cpu", "framework": "pytorch", "route": "native-sitk",
            "status": "completed", "threads": 1, "phase_seconds": {"registration": 0.0}}), encoding="utf-8")
    elif MODE == "mrtrix":
        assert len(args) == 6 and args[2] == "-fslgrad" and args[5] == "-force"
        source, candidate = Path(args[0]), Path(args[1])
        assert Path(args[3]).name == "bvecs" and Path(args[4]).name == "bvals"
    else:
        assert len(args) == 4 and args[0] == "-i" and args[2] == "-m"
        source, candidate = Path(args[1]), Path(args[3])
    image = nib.load(str(source))
    record.update({"source": str(source), "shape": list(image.shape),
                   "affine": image.affine.tolist(), "data": np.asarray(image.dataobj).tolist(),
                   "candidate": str(candidate),
                   "env": {key: os.environ.get(key) for key in (
                       "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                       "OPENBLAS_NUM_THREADS", "PYTHONUNBUFFERED", "ANTSPATH", "SLICERDSPACE_STAGE_CPUS",
                       "FREESURFER_HOME", "PYTHONHOME", "PYTHONPATH", "PATH")}})
    with Path(RECORD).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\\n")
    if BEHAVIOR == "missing_output":
        print("fixture deliberately omitted its declared output")
        sys.exit(0)
    if BEHAVIOR == "partial_failure":
        candidate.write_bytes(b"incomplete candidate, not a NIfTI image")
        print("fixture failed after writing a partial candidate", flush=True)
        sys.exit(7)
    mask = np.zeros(image.shape[:3], dtype=np.float32)
    mask[1:3, 1:4, 1:3] = 1
    affine = image.affine.copy()
    if BEHAVIOR == "empty":
        mask[:] = 0
    elif BEHAVIOR == "nonbinary":
        mask[1, 1, 1] = 0.5
    elif BEHAVIOR == "shifted":
        affine[0, 3] += 1
    elif BEHAVIOR == "alternate":
        mask[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(mask, affine), str(candidate))
    print("adapter contract fixture completed; no anatomical accuracy claim")
    """
)


def write_fake_command(tool: Path, mode: str, *, behavior: str = "valid") -> Path:
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(
        f"#!{sys.executable}\nMODE = {mode!r}\nBEHAVIOR = {behavior!r}\n"
        f"RECORD = {str(command_record(tool))!r}\n" + FAKE_COMMAND,
        encoding="ascii",
    )
    tool.chmod(0o755)
    return tool


def write_fake_mrtrix(bin_dir: Path, *, behavior: str = "valid") -> Path:
    return write_fake_command(bin_dir / "dwi2mask", "mrtrix", behavior=behavior)


def fake_pythonslicer(root: Path) -> Path:
    return root / "Slicer with spaces" / "bin" / "PythonSlicer"


def fake_pnl_models(root: Path) -> Path:
    models = root / "PNL models with spaces"
    models.mkdir(parents=True)
    (models / "weights.json").write_text("{}\n", encoding="ascii")
    save_nifti(models / "IITmean_b0_256.nii.gz", np.ones(SHAPE, dtype=np.float32))
    for view in ("sagittal", "coronal", "axial"):
        (models / f"{view}.pt").write_bytes(b"adapter-only fake weights")
    return models


def fake_pnl_config(root: Path) -> BrainMaskConfig:
    # SC-1a: pinned to the native route so these tests do not depend on the platform of the machine.
    models = fake_pnl_models(root)
    write_fake_command(fake_pythonslicer(root), "pnl_cnn")
    return BrainMaskConfig(brain_mask_device="cpu", pnl_mask_model_dir=models, pnl_mask_route="native")


def fake_managed_pnl_config(root: Path, models: Path, **changes) -> BrainMaskConfig:
    """SC-1a: the conda runner files of the managed route (Python, ``pnl-brain-mask`` script, three ANTs commands)."""
    prefix = root / "pnl-cnn-torch prefix" / "bin"
    ants = root / "ANTs tools with spaces"
    for folder in (prefix, ants):
        folder.mkdir(parents=True)
    for tool in (prefix / "python", prefix / "pnl-brain-mask", *(ants / name for name in ANTS_COMMANDS)):
        tool.write_text("#!/bin/sh\n", encoding="ascii")
        tool.chmod(0o755)
    return BrainMaskConfig(brain_mask_device="cpu", pnl_mask_model_dir=models, pnl_mask_python=prefix / "python",
                           pnl_mask_script=prefix / "pnl-brain-mask", pnl_mask_ants_bin_dir=ants, **changes)


class BrainMaskBackendsTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="mask-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, ISOLATED_ENV)
        environment.start()
        self.addCleanup(environment.stop)
        self.subject = make_subject(self.root / "subject with spaces")
        self.inputs = self.subject / "input"
        self.dwi = self.inputs / "data.nii.gz"
        self.bvals = self.inputs / "bvals"
        self.bvecs = self.inputs / "bvecs"
        self.output = self.inputs / "nodif_brain_mask.nii.gz"
        self.manifest = self.inputs / "nodif_brain_mask.manifest.json"
        self.mrtrix = write_fake_mrtrix(self.root / "MRtrix tools with spaces")
        self.mrtrix_config = BrainMaskConfig(backend="mrtrix", mrtrix_bin_dir=self.mrtrix.parent)
        # The fake PythonSlicer exists only after fake_pnl_config(); before that no Slicer Python is found.
        self.pythonslicer = fake_pythonslicer(self.root)
        resolver = patch.object(brain_mask, "_pnl_native_python",
                                side_effect=lambda: self.pythonslicer if self.pythonslicer.exists() else None)
        resolver.start()
        self.addCleanup(resolver.stop)
        # SC-1a: `auto` depends on the platform; the tests run as on macOS unless a test sets another system.
        system = patch.object(brain_mask.platform, "system", return_value="Darwin")
        system.start()
        self.addCleanup(system.stop)

    def generate(self, config: BrainMaskConfig | None = None, **kwargs):
        return generate_brain_mask(
            self.dwi, self.bvals, self.bvecs, self.output,
            config=config if config is not None else self.mrtrix_config, **kwargs,
        )

    def calls(self, tool: Path) -> list[dict]:
        record = command_record(tool)
        return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()] if record.exists() else []

    def snapshot(self) -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in (self.dwi, self.bvals, self.bvecs, self.output, self.manifest) if p.exists()}

    def assert_preserved(self, snapshot: dict[Path, bytes]) -> None:
        for path, content in snapshot.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertEqual(list(self.inputs.glob(".brain-mask-*")), [])
        self.assertEqual(list(self.inputs.glob(".mask-record-*")), [])
        self.assertEqual(list(self.inputs.glob(".mask-backup-*")), [])

    def test_default_and_options_select_pnl_not_mrtrix(self) -> None:
        self.assertEqual(BrainMaskConfig().backend, "pnl_cnn")
        self.assertEqual(brain_mask_config_from_options(SimpleNamespace()).backend, "pnl_cnn")
        options = SimpleNamespace(
            brain_mask_backend="mrtrix", preprocessing_mrtrix_bin_dir=self.mrtrix.parent, b0_threshold=25,
            pnl_mask_python=self.root / "venv/bin/python", postprocess_freesurfer_home=self.root / "fs",
        )
        config = brain_mask_config_from_options(options)
        self.assertEqual(config.backend, "mrtrix")
        self.assertEqual(config.mrtrix_bin_dir, self.mrtrix.parent)
        self.assertEqual(config.pnl_mask_python, options.pnl_mask_python)
        self.assertEqual(config.freesurfer_home, options.postprocess_freesurfer_home)
        self.assertEqual(config.b0_threshold, 25)

    def test_windows_does_not_block_the_native_pnl_route(self):
        # T2 A1: no POSIX-only ANTs runtime is left in the pnl_cnn route.
        config = fake_pnl_config(self.root)
        result = inspect_brain_mask_backend(config, system="Windows")
        self.assertTrue(result["available"], result)

    def test_missing_default_pnl_never_falls_through_to_available_mrtrix(self) -> None:
        with patch.dict(os.environ, {"SLICERDSPACE_MRTRIX_BIN_DIR": str(self.mrtrix.parent)}):
            self.assertTrue(inspect_brain_mask_backend(self.mrtrix_config)["available"])
            inspected = inspect_brain_mask_backend()
            self.assertEqual(inspected["backend"], "pnl_cnn")
            self.assertFalse(inspected["available"])
            before = self.snapshot()
            with self.assertRaisesRegex(FileNotFoundError, "pnl_cnn unavailable.*PythonSlicer"):
                generate_brain_mask(self.dwi, self.bvals, self.bvecs, self.output)
        self.assertEqual(self.calls(self.mrtrix), [])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())
        self.assert_preserved(before)

    def test_inspection_discovers_files_without_importing_tensorflow_or_running_inference(self) -> None:
        config = fake_pnl_config(self.root)
        original_import = __import__

        def forbid_tensorflow(name, *args, **kwargs):
            if name.split(".")[0] in ("tensorflow", "keras", "torch"):
                raise AssertionError("Inspection must not import an inference runtime")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=forbid_tensorflow), patch.object(brain_mask, "_run") as run:
            inspected = inspect_brain_mask_backend(config)
        run.assert_not_called()
        self.assertTrue(inspected["available"], inspected)
        self.assertEqual(inspected["missing_required"], [])
        # Native module + UPSTREAM.json + the 7 vendored files + weights.json, template and 3 views.
        self.assertEqual(len(inspected["resources"]), 14)
        self.assertEqual(inspected["tool_paths"]["pythonslicer"], str(self.pythonslicer))
        self.assertEqual(self.calls(self.pythonslicer), [])

    def test_a7_signature_hashes_native_module_and_every_vendored_file(self) -> None:
        # T2 A7 / O5: the cache signature covers pnl_mask_native.py and every vendored file, with the JSON.
        config = fake_pnl_config(self.root)
        shared_dir = Path(brain_mask.__file__).parent
        vendor = shared_dir / "pnl_cnn"
        upstream = json.loads((vendor / "UPSTREAM.json").read_text(encoding="utf-8"))
        expected = {shared_dir / "pnl_mask_native.py", vendor / "UPSTREAM.json",
                    *(vendor / relative for relative in upstream["files"])}
        self.assertIn(vendor / "data/CompNetBasicModel.json", expected)
        self.assertIn(vendor / "model.py", expected)
        self.assertNotIn(shared_dir / "pnl_mask_runner.py", {Path(p) for p in inspect_brain_mask_backend(config)["resources"]})
        result = self.generate(config)
        hashed = {Path(item["path"]): item["sha256"] for item in result["signature"]["resources"]}
        for path in expected:
            with self.subTest(file=path.name):
                self.assertEqual(hashed[path.resolve()], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["signature"]["settings"]["pnl_route"], "native-sitk")
        self.assertEqual(result["device_execution"]["route"], "native-sitk")
        # A change in any vendored file (here the architecture JSON, in a copy) changes the signature.
        copy = self.root / "vendored copy" / "pnl_cnn"
        shutil.copytree(vendor, copy, ignore=shutil.ignore_patterns("__pycache__"))
        with patch.object(brain_mask, "PNL_VENDOR_DIR", copy):
            same = self.generate(config)
            self.assertEqual(same["status"], "completed")  # other paths, so a new signature
            again = self.generate(config)
            self.assertEqual(again["status"], "reused")
            with (copy / "data/CompNetBasicModel.json").open("ab") as stream:
                stream.write(b" ")
            changed = self.generate(config)
        self.assertEqual(changed["status"], "completed")
        self.assertNotEqual(again["signature"]["resources"], changed["signature"]["resources"])

    def test_pnl_requires_converted_weights_and_rejects_legacy_hdf5(self) -> None:
        config = fake_pnl_config(self.root)
        for view in ("sagittal", "coronal", "axial"):
            file = config.pnl_mask_model_dir / f"{view}.pt"
            content = file.read_bytes()
            with self.subTest(view=view, problem="missing"):
                file.unlink()
                inspected = inspect_brain_mask_backend(config)
                self.assertFalse(inspected["available"])
                self.assertTrue(any(item.startswith(view + ".pt") for item in inspected["missing_required"]))
                file.write_bytes(content)
            with self.subTest(view=view, problem="legacy_hdf5"):
                file.unlink()
                legacy = file.with_name(f"weights-{view}-improvement-01.h5")
                legacy.write_bytes(content)
                inspected = inspect_brain_mask_backend(config)
                self.assertFalse(inspected["available"])
                self.assertTrue(any("legacy HDF5" in item for item in inspected["missing_required"]))
                with patch.object(brain_mask, "_run") as run:
                    with self.assertRaisesRegex(FileNotFoundError, "legacy HDF5"):
                        self.generate(config)
                    run.assert_not_called()
                legacy.unlink()
                file.write_bytes(content)
        self.assertTrue(inspect_brain_mask_backend(config)["available"])

    def test_pnl_checks_model_files_interpreter_and_vendored_files(self) -> None:
        config = fake_pnl_config(self.root)
        cases = [
            (config.pnl_mask_model_dir / "weights.json", "weights.json"),
            (config.pnl_mask_model_dir / "IITmean_b0_256.nii.gz", "IITmean_b0_256.nii.gz"),
            (self.pythonslicer, brain_mask.PNL_NATIVE_PYTHON_MISSING),
        ]
        for file, missing in cases:
            with self.subTest(resource=missing):
                saved = file.with_name(file.name + ".saved")
                file.rename(saved)
                try:
                    inspected = inspect_brain_mask_backend(config)
                    self.assertFalse(inspected["available"])
                    self.assertIn(missing, inspected["missing_required"])
                finally:
                    saved.rename(file)
        with self.subTest(resource="pnl_mask_model_dir"):
            inspected = inspect_brain_mask_backend(replace(config, pnl_mask_model_dir=self.root / "missing"))
            self.assertFalse(inspected["available"])
            self.assertIn("pnl_mask_model_dir", inspected["missing_required"])
        copy = self.root / "vendored copy" / "pnl_cnn"
        shutil.copytree(Path(brain_mask.__file__).with_name("pnl_cnn"), copy,
                        ignore=shutil.ignore_patterns("__pycache__"))
        with patch.object(brain_mask, "PNL_VENDOR_DIR", copy):
            self.assertTrue(inspect_brain_mask_backend(config)["available"])
            for relative in ("data/CompNetBasicModel.json", "UPSTREAM.json"):
                with self.subTest(resource=relative):
                    (copy / relative).unlink()
                    inspected = inspect_brain_mask_backend(config)
                    self.assertFalse(inspected["available"])
                    self.assertIn(f"PNL native file {copy / relative}", inspected["missing_required"])

    def test_pnl_interpreter_must_be_executable(self) -> None:
        config = fake_pnl_config(self.root)
        self.pythonslicer.chmod(0o644)
        inspected = inspect_brain_mask_backend(config)
        self.assertFalse(inspected["available"])
        self.assertIn(brain_mask.PNL_NATIVE_PYTHON_MISSING, inspected["missing_required"])

    def test_saved_conda_runner_settings_are_not_used_by_the_native_route(self) -> None:
        # T2 A7: brain_mask runs A1 instead of the conda runner; the old paths stay in the config until A8.
        config = replace(fake_pnl_config(self.root), pnl_mask_python=self.root / "missing python",
                         pnl_mask_script=self.root / "missing script", pnl_mask_ants_bin_dir=self.root / "missing ants")
        with patch.dict(os.environ, {"SLICERDSPACE_PNL_MASK_PYTHON": str(self.root / "missing python"),
                                     "ANTSPATH": str(self.root / "missing ants")}):
            inspected = inspect_brain_mask_backend(config)
            self.assertTrue(inspected["available"], inspected)
            result = self.generate(config)
        self.assertEqual(result["command"][:2], [str(self.pythonslicer), str(brain_mask.PNL_NATIVE_SCRIPT)])
        self.assertNotIn("pnl_mask_runner.py", " ".join(result["command"]))
        self.assertEqual(set(inspected["tool_paths"]), {"pythonslicer", "model_dir"})

    def test_invalid_backend_threshold_and_threads_are_rejected(self) -> None:
        for config in (
            BrainMaskConfig(backend="auto"), BrainMaskConfig(b0_threshold=-1),
            BrainMaskConfig(b0_threshold=float("nan")), BrainMaskConfig(b0_threshold=float("inf")),
            BrainMaskConfig(threads=0),
            BrainMaskConfig(brain_mask_device="invalid"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                inspect_brain_mask_backend(config)
        with self.assertRaisesRegex(ValueError, "Unknown brain mask backend"):
            self.generate(BrainMaskConfig(backend="auto"))

    def test_pnl_native_adapter_interface_output_and_environment(self) -> None:
        config = replace(fake_pnl_config(self.root), threads=3)
        before = self.snapshot()
        logs = []
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}):  # A-O4: the child count, not config.threads
            result = self.generate(config, log=logs.append)
        call, = self.calls(self.pythonslicer)
        command = result["command"]
        record = command[command.index("--record") + 1]
        # T2 A1: [PythonSlicer, shared/pnl_mask_native.py, ...]; no --threads (the child reads SLICERDSPACE_STAGE_CPUS).
        self.assertEqual(command[:2], [str(self.pythonslicer), str(Path(brain_mask.__file__).with_name("pnl_mask_native.py"))])
        self.assertEqual(call["argv"], command[1:])
        self.assertEqual(command[2:], [
            "--input", call["source"], "--output", call["candidate"],
            "--model-dir", str(config.pnl_mask_model_dir),
            "--template", str(config.pnl_mask_model_dir / "IITmean_b0_256.nii.gz"),
            "--device", "cpu", "--record", record])
        self.assertNotIn("--threads", command)
        self.assertEqual(result["device_execution"]["resolved"], "cpu")
        self.assertEqual(result["device_execution"]["route"], "native-sitk")
        self.assertEqual(Path(call["candidate"]).name, "mask.nii.gz")
        self.assertEqual(result["device_execution"]["framework"], "pytorch")
        self.assertEqual(len(call["model_files"]), 5)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(call["env"][key], "3")
        self.assertEqual(call["env"]["PYTHONUNBUFFERED"], "1")
        self.assertFalse(call["env"]["ANTSPATH"])
        self.assertEqual(result["backend"], "pnl_cnn")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["b0_volumes"], 2)
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["qc"]["human_review"], "pending")
        self.assertEqual(set(result["phase_seconds"]), {"preflight_and_hashing", "b0_preparation", "external_process", "native_mask_qc"})
        self.assertTrue(all(np.isfinite(value) and value >= 0 for value in result["phase_seconds"].values()))
        self.assertLessEqual(sum(result["phase_seconds"].values()), result["runtime_seconds"])
        self.assertIn("not a complete sum", result["phase_timing_scope"])
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(result["thread_environment"][key], "3")
        self.assertTrue(any("no anatomical accuracy claim" in line for line in logs))
        np.testing.assert_array_equal(nib.load(str(self.output)).dataobj, mask_data())
        np.testing.assert_allclose(nib.load(str(self.output)).affine, AFFINE)
        self.assertEqual(json.loads(self.manifest.read_text(encoding="utf-8")), result)
        self.assertEqual(result["output"]["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assert_preserved(before)

    # ---- SC-1a: route of the pnl_cnn backend per platform
    def run_fake_managed(self, command, env, log):
        """Stands in for the conda runner: writes the mask and the device record the runner would write."""
        self.managed_calls.append({"command": list(command), "env": dict(env)})
        source = nib.load(command[command.index("--input") + 1])
        mask = np.zeros(source.shape[:3], dtype=np.float32)
        mask[1:3, 1:4, 1:3] = 1
        nib.save(nib.Nifti1Image(mask, source.affine), command[command.index("--output") + 1])
        Path(command[command.index("--record") + 1]).write_text(
            json.dumps({"requested": "cpu", "resolved": "cpu", "framework": "pytorch"}), encoding="utf-8")
        return "managed runner fixture"

    def managed_setup(self):
        models = fake_pnl_models(self.root)
        write_fake_command(fake_pythonslicer(self.root), "pnl_cnn")
        self.managed_calls = []
        return models

    def test_sc1a_auto_route_is_managed_on_linux_and_native_on_macos(self) -> None:
        for system, expected in (("Linux", "managed"), ("Darwin", "native"), ("Windows", "native")):
            with self.subTest(system=system):
                self.assertEqual(brain_mask.resolve_pnl_mask_route("auto", system), expected)
                with patch.object(brain_mask.platform, "system", return_value=system):
                    self.assertEqual(brain_mask.resolve_pnl_mask_route("auto"), expected)
                    self.assertEqual(brain_mask.resolve_pnl_mask_route(None), expected)
        for system in ("Linux", "Darwin"):
            self.assertEqual(brain_mask.resolve_pnl_mask_route("native", system), "native")
            self.assertEqual(brain_mask.resolve_pnl_mask_route("managed", system), "managed")
        with self.assertRaisesRegex(ValueError, "pnl_mask_route must be one of: auto, managed, native"):
            brain_mask.resolve_pnl_mask_route("conda", "Linux")
        with self.assertRaisesRegex(ValueError, "pnl_mask_route"):
            inspect_brain_mask_backend(BrainMaskConfig(backend="mrtrix", pnl_mask_route="conda"))
        self.assertEqual(BrainMaskConfig().pnl_mask_route, "auto")
        self.assertEqual(brain_mask_config_from_options(SimpleNamespace()).pnl_mask_route, "auto")
        self.assertEqual(brain_mask_config_from_options(SimpleNamespace(pnl_mask_route="managed")).pnl_mask_route, "managed")
        self.assertEqual(brain_mask_config_from_options(SimpleNamespace(pnl_mask_route=None)).pnl_mask_route, "auto")

    def test_sc1a_auto_on_linux_runs_the_managed_conda_runner(self) -> None:
        models = self.managed_setup()
        config = fake_managed_pnl_config(self.root, models)
        self.assertEqual(config.pnl_mask_route, "auto")
        inspected = inspect_brain_mask_backend(config, system="Linux")
        self.assertTrue(inspected["available"], inspected["missing_required"])
        self.assertEqual(inspected["pnl_route_resolved"], "managed")
        self.assertEqual(inspected["settings"]["pnl_route"], "managed-ants")
        self.assertEqual(set(inspected["tool_paths"]) & {"pythonslicer"}, set())
        self.assertIn(str(Path(brain_mask.__file__).with_name("pnl_mask_runner.py")), inspected["resources"])
        with patch.object(brain_mask.platform, "system", return_value="Linux"), \
                patch.object(brain_mask, "_run", side_effect=self.run_fake_managed):
            result = self.generate(replace(config, threads=3))
        call, = self.managed_calls
        command = call["command"]
        self.assertEqual(command[:2], [str(config.pnl_mask_python), str(Path(brain_mask.__file__).with_name("pnl_mask_runner.py"))])
        self.assertEqual(command[command.index("--script") + 1], str(config.pnl_mask_script))
        self.assertEqual(command[command.index("--") + 1], "mask")
        self.assertEqual(command[command.index("--percentile") + 1], "99")
        self.assertEqual(command[command.index("--threads") + 1], "3")
        self.assertEqual(command[command.index("--ants-bin") + 1], str(config.pnl_mask_ants_bin_dir))
        self.assertNotIn("pnl_mask_native.py", " ".join(command))
        self.assertEqual(call["env"]["ANTSPATH"], str(config.pnl_mask_ants_bin_dir))
        self.assertEqual(call["env"]["OMP_NUM_THREADS"], "3")
        self.assertEqual(result["route"], "managed-ants")
        self.assertNotIn("route_status", result)
        self.assertNotIn("child_threads", result)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["signature"]["settings"]["pnl_route"], "managed-ants")

    def test_sc1a_managed_route_lists_missing_runner_files_and_blocks_windows(self) -> None:
        models = self.managed_setup()
        config = BrainMaskConfig(pnl_mask_model_dir=models, pnl_mask_route="managed")
        inspected = inspect_brain_mask_backend(config, system="Linux")
        self.assertFalse(inspected["available"])
        for name in ("pnl_mask_python", "pnl_mask_script", *ANTS_COMMANDS):
            self.assertIn(name, inspected["missing_required"])
        full = fake_managed_pnl_config(self.root, models)
        self.assertTrue(inspect_brain_mask_backend(full, system="Linux")["available"])
        windows = inspect_brain_mask_backend(replace(full, pnl_mask_route="managed"), system="Windows")
        self.assertTrue(any("POSIX external runtime" in item for item in windows["missing_required"]))

    def test_sc1a_auto_on_macos_runs_the_native_route(self) -> None:
        config = replace(fake_pnl_config(self.root), pnl_mask_route="auto")
        result = self.generate(config)
        call, = self.calls(self.pythonslicer)
        self.assertEqual(result["command"][:2], [str(self.pythonslicer), str(Path(brain_mask.__file__).with_name("pnl_mask_native.py"))])
        self.assertEqual(call["argv"], result["command"][1:])
        self.assertEqual(result["route"], "native-sitk")
        self.assertEqual(result["signature"]["settings"]["pnl_route"], "native-sitk")

    def test_sc1a_explicit_native_on_linux_runs_the_native_route(self) -> None:
        models = self.managed_setup()
        config = replace(fake_managed_pnl_config(self.root, models), pnl_mask_route="native")
        with patch.object(brain_mask.platform, "system", return_value="Linux"):
            inspected = inspect_brain_mask_backend(config)
            self.assertEqual(inspected["pnl_route_resolved"], "native")
            self.assertEqual(inspected["tool_paths"]["pythonslicer"], str(self.pythonslicer))
            self.assertNotIn("pnl_mask_python", inspected["tool_paths"])
            result = self.generate(config)
        call, = self.calls(self.pythonslicer)
        self.assertEqual(result["command"][:2], [str(self.pythonslicer), str(Path(brain_mask.__file__).with_name("pnl_mask_native.py"))])
        self.assertEqual(result["route"], "native-sitk")

    # ---- SC-1b: the receipt marks the native route as experimental
    def test_sc1b_receipt_marks_native_experimental_and_managed_not(self) -> None:
        native = self.generate(fake_pnl_config(self.root))
        self.assertEqual(native["route"], "native-sitk")
        self.assertEqual(native["route_status"], "experimental")
        self.assertEqual(native["route_reason"], "did not meet the M-A parity rule on 2 of 24 cases (T2 design \u00a73)")
        self.assertEqual(native["status"], "completed")  # the receipt status keeps its meaning
        self.assertEqual(brain_mask.pnl_route_record("native"), {
            "route": "native-sitk", "status": "experimental", "reason": brain_mask.PNL_NATIVE_EXPERIMENTAL_REASON})
        self.assertEqual(brain_mask.pnl_route_record("managed"), {"route": "managed-ants"})
        self.assertEqual(brain_mask.PNL_ROUTE_LABELS["native"], "native (experimental)")
        self.assertEqual(self.generate(self.mrtrix_config).get("route"), None)

    # ---- SC-1f: the route is part of the cache signature
    def test_sc1f_route_is_part_of_the_cache_signature(self) -> None:
        models = self.managed_setup()
        managed = fake_managed_pnl_config(self.root, models, pnl_mask_route="managed")
        native = replace(managed, pnl_mask_route="native")
        signatures = {}
        for name, config in (("managed", managed), ("native", native)):
            inspected = inspect_brain_mask_backend(config, system="Linux")
            self.assertTrue(inspected["available"], inspected["missing_required"])
            signatures[name] = inspected["settings"]
        self.assertNotEqual(signatures["managed"]["pnl_route"], signatures["native"]["pnl_route"])
        with patch.object(brain_mask, "_run", side_effect=self.run_fake_managed):
            first = self.generate(managed)
            self.assertEqual(self.generate(managed)["status"], "reused")
        # A managed mask is never reused for a native request, nor the reverse.
        self.assertEqual(first["signature"]["settings"]["pnl_route"], "managed-ants")
        second = self.generate(native)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["signature"]["settings"]["pnl_route"], "native-sitk")
        self.assertNotEqual(first["signature"], second["signature"])
        self.assertEqual(self.generate(native)["status"], "reused")
        with patch.object(brain_mask, "_run", side_effect=self.run_fake_managed):
            third = self.generate(managed)
        self.assertEqual(third["status"], "completed")
        self.assertEqual(len(self.managed_calls), 2)
        # Same inputs, same route: the same signature.
        self.assertEqual(first["signature"]["settings"], third["signature"]["settings"])

    def test_sc1e_native_signature_records_one_registration_thread(self) -> None:
        settings = inspect_brain_mask_backend(fake_pnl_config(self.root))["settings"]
        self.assertEqual(settings["pnl_registration_threads"], 1)
        from shared import pnl_mask_native
        self.assertEqual(brain_mask.PNL_REGISTRATION_THREADS, pnl_mask_native.REGISTRATION_THREADS)

    def test_signature_and_receipt_record_the_effective_child_thread_count(self) -> None:
        # A-O4: the native child ignores config.threads and reads SLICERDSPACE_STAGE_CPUS (A6).
        config = replace(fake_pnl_config(self.root), threads=3)
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "2"}):
            self.assertEqual(inspect_brain_mask_backend(config)["settings"]["threads"], 2)
            result = self.generate(config)
        self.assertEqual(result["signature"]["settings"]["threads"], 2)
        self.assertEqual(result["child_threads"], 2)
        self.assertEqual(result["thread_environment"]["OMP_NUM_THREADS"], "2")
        # The signature follows the reservation: another count is another signature (no silent reuse).
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "5"}):
            self.assertEqual(self.generate(config)["status"], "completed")
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "5"}):
            self.assertEqual(self.generate(config)["status"], "reused")

    def test_scheduler_cpu_reservation_reaches_the_native_child_from_slicer_startup_environment(self) -> None:
        # A6: the child reads SLICERDSPACE_STAGE_CPUS; inside Slicer the child environment starts from
        # slicer.util.startupEnvironment(), which does not hold the scheduler's value.
        config = fake_pnl_config(self.root)
        startup = {key: value for key, value in os.environ.items() if key != "SLICERDSPACE_STAGE_CPUS"}
        slicer = SimpleNamespace(util=SimpleNamespace(startupEnvironment=lambda: dict(startup)))
        with patch.dict(sys.modules, {"slicer": slicer}), patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "2"}):
            self.generate(config)
        call, = self.calls(self.pythonslicer)
        self.assertEqual(call["env"]["SLICERDSPACE_STAGE_CPUS"], "2")

    def test_pnl_averages_only_b_values_at_or_below_50_in_native_grid(self) -> None:
        config = fake_pnl_config(self.root)
        data = nib.load(str(self.dwi)).get_fdata()
        # Nonfinite/high-signal diffusion volumes must never contaminate mean b0.
        data[..., 2] = np.nan
        data[..., 3] = np.inf
        save_nifti(self.dwi, data.astype(np.float32))
        self.generate(config)
        call, = self.calls(self.pythonslicer)
        np.testing.assert_allclose(call["data"], np.mean(data[..., :2], axis=-1), rtol=0, atol=0)
        self.assertEqual(call["shape"], list(SHAPE))
        np.testing.assert_allclose(call["affine"], AFFINE)

    @unittest.skipUnless(os.name == "posix", "PNL's official shell interface requires POSIX")
    def test_pnl_workspace_ignores_user_tmpdir_with_spaces(self) -> None:
        config = fake_pnl_config(self.root)
        unsafe = self.root / "user TMPDIR with spaces"
        unsafe.mkdir()
        # tempfile caches its directory: set both knobs to actually exercise it.
        with patch.dict(os.environ, {"TMPDIR": str(unsafe)}), patch.object(tempfile, "tempdir", str(unsafe)):
            result = self.generate(config)
        call, = self.calls(self.pythonslicer)
        work = Path(call["source"]).parent
        self.assertEqual(work.parent, Path("/tmp"))
        for value in (result["command"][3], result["command"][5], call["source"], call["candidate"]):
            self.assertNotIn(" ", value)
        self.assertFalse(work.exists(), "Backend scratch files must be cleaned after inference")

    def test_invalid_record_json_raises_instead_of_blindly_reusing_mask(self) -> None:
        self.generate()
        self.manifest.write_text('{"version": 2, broken json\n', encoding="ascii")
        before = self.snapshot()
        with patch.object(brain_mask, "_run") as run:
            with self.assertRaises(json.JSONDecodeError):
                self.generate()
        run.assert_not_called()
        self.assert_preserved(before)

    def test_no_baseline_is_rejected_without_launching_any_backend(self) -> None:
        self.bvals.write_text("51 1000 1000 2000\n", encoding="ascii")
        before = self.snapshot()
        with patch.object(brain_mask, "_run") as run:
            with self.assertRaisesRegex(ValueError, "No baseline measurements with b <= 50"):
                self.generate()
            run.assert_not_called()
        self.assert_preserved(before)
        self.assertFalse(self.output.exists())

    def test_invalid_gradients_are_rejected_before_launch(self) -> None:
        original = {p: p.read_bytes() for p in (self.bvals, self.bvecs)}
        cases = (
            (self.bvals, "0 1000\n"), (self.bvals, "0 -1 1000 2000\n"),
            (self.bvals, "0 nan 1000 2000\n"), (self.bvals, "0 inf 1000 2000\n"),
            (self.bvecs, "0 1\n0 0\n"), (self.bvecs, "nan 0 1 0\n0 0 0 1\n0 0 0 0\n"),
        )
        for path, content in cases:
            with self.subTest(path=path.name, content=content), patch.object(brain_mask, "_run") as run:
                path.write_text(content, encoding="ascii")
                with self.assertRaisesRegex(ValueError, "matching finite"):
                    self.generate()
                run.assert_not_called()
                path.write_bytes(original[path])
        self.assertFalse(self.output.exists())

    def test_transposed_b_vectors_are_accepted(self) -> None:
        np.savetxt(self.bvecs, np.loadtxt(self.bvecs).T)
        result = self.generate()
        self.assertEqual(result["b0_volumes"], 2)

    def test_invalid_baseline_signal_or_non4d_dwi_is_rejected_before_launch(self) -> None:
        original = nib.load(str(self.dwi)).get_fdata().astype(np.float32)
        cases = []
        for value in (np.nan, np.inf, 0, -1):
            data = original.copy()
            data[..., :2] = value
            cases.append(data)
        cases.append(original[..., 0])
        for index, data in enumerate(cases):
            with self.subTest(index=index), patch.object(brain_mask, "_run") as run:
                save_nifti(self.dwi, data)
                with self.assertRaises(ValueError):
                    self.generate()
                run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_provided_and_untracked_masks_are_preserved_without_dependencies(self) -> None:
        save_nifti(self.output, mask_data())
        before = self.snapshot()
        for backend in brain_mask.BRAIN_MASK_BACKENDS:
            with self.subTest(backend=backend), patch.object(brain_mask, "_run") as run:
                result = self.generate(BrainMaskConfig(backend=backend))
                run.assert_not_called()
                self.assertEqual(result["status"], "reused")
                self.assertEqual(result["backend"], "provided")
                self.assertEqual(result["requested_backend"], backend)
                self.assertTrue(result["qc"]["geometry_pass"])
                self.assertEqual(result["qc"]["brain_voxels"], 12)
                self.assert_preserved(before)
        self.assertFalse(self.manifest.exists())

    def test_legacy_manifest_does_not_turn_user_mask_into_generated_cache(self) -> None:
        save_nifti(self.output, mask_data())
        self.manifest.write_text('{"version": 1, "backend": "mrtrix"}\n', encoding="ascii")
        before = self.snapshot()
        with patch.object(brain_mask, "_run") as run:
            result = self.generate(BrainMaskConfig())
        run.assert_not_called()
        self.assertEqual(result["backend"], "provided")
        self.assertEqual(result["previous_backend"], "mrtrix")
        self.assert_preserved(before)

    def test_provided_without_mask_fails_even_with_mrtrix_available(self) -> None:
        with patch.object(brain_mask, "_run") as run:
            with self.assertRaisesRegex(FileNotFoundError, "Provided-mask route requires"):
                self.generate(BrainMaskConfig(backend="provided"))
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_identical_generated_cache_reuses_without_command_or_writes(self) -> None:
        first = self.generate()
        before = self.snapshot()
        with patch.object(brain_mask, "_run") as run:
            second = self.generate()
        run.assert_not_called()
        self.assertEqual(second["status"], "reused")
        self.assertEqual(first["signature"], second["signature"])
        self.assertEqual(len(self.calls(self.mrtrix)), 1)
        self.assert_preserved(before)

    def test_changed_input_content_invalidates_cache_even_with_same_size_and_mtime(self) -> None:
        first = self.generate()
        before = self.bvals.stat()
        self.bvals.write_text("0 50 51 2000\n", encoding="ascii")
        os.utime(self.bvals, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertEqual(self.bvals.stat().st_size, before.st_size)
        second = self.generate()
        self.assertEqual(second["status"], "completed")
        self.assertNotEqual(first["signature"]["inputs"], second["signature"]["inputs"])
        self.assertEqual(len(self.calls(self.mrtrix)), 2)

    def test_changed_dwi_and_b_vectors_each_invalidate_cache(self) -> None:
        first = self.generate()
        data = nib.load(str(self.dwi)).get_fdata().astype(np.float32)
        save_nifti(self.dwi, data + 1)
        second = self.generate()
        self.bvecs.write_text("0 0 0 1\n0 0 1 0\n0 0 0 0\n", encoding="ascii")
        third = self.generate()
        for previous, current in ((first, second), (second, third)):
            self.assertEqual(current["status"], "completed")
            self.assertNotEqual(previous["signature"]["inputs"], current["signature"]["inputs"])
        self.assertEqual(len(self.calls(self.mrtrix)), 3)

    def test_changed_pnl_model_weights_and_template_each_invalidate_cache(self) -> None:
        # Vendored code changes: test_a7_signature_hashes_native_module_and_every_vendored_file.
        config = fake_pnl_config(self.root)
        previous = self.generate(config)
        files = [
            config.pnl_mask_model_dir / "weights.json",
            config.pnl_mask_model_dir / "sagittal.pt",
            config.pnl_mask_model_dir / "coronal.pt",
            config.pnl_mask_model_dir / "axial.pt",
            config.pnl_mask_model_dir / "IITmean_b0_256.nii.gz",
        ]
        for file in files:
            with self.subTest(resource=file.name):
                if file.name.endswith(".nii.gz"):
                    save_nifti(file, np.full(SHAPE, 2, dtype=np.float32))
                elif file.suffix == ".json":
                    file.write_text('{"changed": true}\n', encoding="ascii")
                else:
                    with file.open("ab") as stream:
                        stream.write(b"\n# changed fixture resource\n")
                current = self.generate(config)
                self.assertEqual(current["status"], "completed")
                self.assertNotEqual(previous["signature"]["resources"], current["signature"]["resources"])
                previous = current
        self.assertEqual(len(self.calls(self.pythonslicer)), 6)

    def test_changed_backend_invalidates_generated_cache(self) -> None:
        first = self.generate()
        config = fake_pnl_config(self.root)
        second = self.generate(config)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["backend"], "pnl_cnn")
        self.assertNotEqual(first["signature"], second["signature"])
        third = self.generate()
        self.assertEqual(third["status"], "completed")
        self.assertEqual(third["backend"], "mrtrix")
        self.assertEqual(len(self.calls(self.mrtrix)), 2)
        self.assertEqual(len(self.calls(self.pythonslicer)), 1)

    def test_threshold_and_thread_changes_invalidate_cache(self) -> None:
        first = self.generate()
        second = self.generate(replace(self.mrtrix_config, b0_threshold=0))
        third = self.generate(replace(self.mrtrix_config, b0_threshold=0, threads=2))
        self.assertEqual(second["b0_volumes"], 1)
        for previous, current in ((first, second), (second, third)):
            self.assertEqual(current["status"], "completed")
            self.assertNotEqual(previous["signature"]["settings"], current["signature"]["settings"])
        self.assertEqual(len(self.calls(self.mrtrix)), 3)

    def test_edited_generated_mask_raises_unless_explicitly_provided(self) -> None:
        self.generate()
        edited = mask_data()
        edited[0, 0, 0] = 1
        save_nifti(self.output, edited)
        before = self.snapshot()
        with patch.object(brain_mask, "_run") as run:
            with self.assertRaisesRegex(ValueError, "Generated brain mask was edited"):
                self.generate()
            result = self.generate(BrainMaskConfig(backend="provided"))
        run.assert_not_called()
        self.assertEqual(result["backend"], "provided")
        self.assertEqual(result["qc"]["brain_voxels"], 13)
        self.assert_preserved(before)

    def test_explicit_force_regenerates_edited_generated_mask(self) -> None:
        self.generate()
        edited = mask_data()
        edited[0, 0, 0] = 1
        save_nifti(self.output, edited)
        result = self.generate(force=True)
        self.assertEqual(result["status"], "completed")
        np.testing.assert_array_equal(nib.load(str(self.output)).dataobj, mask_data())
        self.assertEqual(len(self.calls(self.mrtrix)), 2)

    def test_geometry_rejects_wrong_shape_affine_or_four_dimensional_masks(self) -> None:
        shifted = AFFINE.copy()
        shifted[1, 3] += 0.01
        for data, affine in (
            (np.ones((3, 5, 3), dtype=np.uint8), AFFINE), (mask_data(), shifted),
            (mask_data()[..., None], AFFINE),
        ):
            with self.subTest(shape=data.shape, affine=affine.tolist()):
                save_nifti(self.output, data, affine)
                with self.assertRaisesRegex(ValueError, "native 3D voxel grid and world affine"):
                    validate_brain_mask(self.dwi, self.output)

    def test_content_rejects_empty_nonbinary_nan_and_infinite_masks(self) -> None:
        for value in (0, 0.5, 2, -1, np.nan, np.inf):
            with self.subTest(value=value):
                data = np.zeros(SHAPE, dtype=np.float32)
                data[1, 1, 1] = value
                save_nifti(self.output, data)
                with self.assertRaisesRegex(ValueError, "finite, binary and nonempty"):
                    validate_brain_mask(self.dwi, self.output)

    def test_valid_mask_qc_is_integrity_only_with_human_review_pending(self) -> None:
        save_nifti(self.output, mask_data())
        qc = validate_brain_mask(self.dwi, self.output)
        self.assertTrue(qc["geometry_pass"] and qc["finite"] and qc["binary"])
        self.assertEqual(qc["brain_voxels"], 12)
        self.assertAlmostEqual(qc["mask_fov_fraction"], 0.2)
        self.assertEqual(qc["human_review"], "pending")
        self.assertTrue(qc["warnings"])
        save_nifti(self.output, np.ones(SHAPE, dtype=np.uint8))
        warnings = validate_brain_mask(self.dwi, self.output)["warnings"]
        self.assertTrue(any("Unusual mask/FOV" in warning for warning in warnings))

    def test_invalid_provided_masks_are_not_accepted_or_replaced(self) -> None:
        for backend in ("provided", "pnl_cnn", "mrtrix"):
            with self.subTest(backend=backend), patch.object(brain_mask, "_run") as run:
                save_nifti(self.output, np.zeros(SHAPE, dtype=np.uint8))
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    self.generate(BrainMaskConfig(backend=backend))
                run.assert_not_called()
                self.assert_preserved(before)

    def test_nonzero_command_with_partial_output_preserves_previous_mask_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()
        write_fake_mrtrix(self.mrtrix.parent, behavior="partial_failure")
        with self.assertRaisesRegex(RuntimeError, r"Brain-mask command failed \(7\).*partial candidate"):
            self.generate(force=True)
        self.assert_preserved(before)

    def test_success_without_declared_output_preserves_previous_mask_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()
        write_fake_mrtrix(self.mrtrix.parent, behavior="missing_output")
        with self.assertRaisesRegex(RuntimeError, "did not create its declared mask"):
            self.generate(force=True)
        self.assert_preserved(before)

    def test_invalid_backend_candidates_preserve_previous_mask_manifest_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()
        for behavior in ("empty", "nonbinary", "shifted"):
            with self.subTest(behavior=behavior):
                write_fake_mrtrix(self.mrtrix.parent, behavior=behavior)
                with self.assertRaises(ValueError):
                    self.generate(force=True)
                self.assert_preserved(before)

    def test_failed_staging_copy_preserves_previous_mask_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()

        def partial_copy(source, destination):
            Path(destination).write_bytes(b"partial copy")
            raise OSError("injected staging copy failure")

        with patch.object(brain_mask.shutil, "copyfile", side_effect=partial_copy):
            with self.assertRaisesRegex(OSError, "injected staging copy failure"):
                self.generate(force=True)
        self.assert_preserved(before)

    def test_failed_atomic_mask_replace_preserves_previous_mask_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()
        with patch.object(brain_mask.os, "replace", side_effect=OSError("injected mask replace failure")):
            with self.assertRaisesRegex(OSError, "injected mask replace failure"):
                self.generate(force=True)
        self.assert_preserved(before)

    def test_failed_provenance_replace_preserves_previous_mask_manifest_and_inputs(self) -> None:
        self.generate()
        before = self.snapshot()
        write_fake_mrtrix(self.mrtrix.parent, behavior="alternate")
        real_replace = os.replace

        def fail_record_only(source, destination):
            if Path(destination) == self.manifest:
                raise OSError("injected provenance replace failure")
            return real_replace(source, destination)

        with patch.object(brain_mask.os, "replace", side_effect=fail_record_only):
            with self.assertRaisesRegex(OSError, "injected provenance replace failure"):
                self.generate(force=True)
        self.assert_preserved(before)

    def test_first_generation_provenance_failure_leaves_no_mask_or_manifest(self) -> None:
        before = self.snapshot()
        real_replace = os.replace

        def fail_record_only(source, destination):
            if Path(destination) == self.manifest:
                raise OSError("injected first provenance failure")
            return real_replace(source, destination)

        with patch.object(brain_mask.os, "replace", side_effect=fail_record_only):
            with self.assertRaisesRegex(OSError, "injected first provenance failure"):
                self.generate()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())
        self.assert_preserved(before)

    def test_failed_provenance_serialization_preserves_previous_files_and_cleans_staging(self) -> None:
        self.generate()
        before = self.snapshot()

        def partial_record(payload, stream, **kwargs):
            stream.write('{"incomplete": ')
            raise OSError("injected provenance serialization failure")

        with patch.object(brain_mask.json, "dump", side_effect=partial_record):
            with self.assertRaisesRegex(OSError, "injected provenance serialization failure"):
                self.generate(force=True)
        self.assert_preserved(before)

    def test_output_must_not_overwrite_source_inputs(self) -> None:
        before = self.snapshot()
        for source in (self.dwi, self.bvals, self.bvecs):
            with self.subTest(source=source.name), self.assertRaisesRegex(ValueError, "source input or symlink"):
                generate_brain_mask(self.dwi, self.bvals, self.bvecs, source, config=self.mrtrix_config)
        self.assert_preserved(before)

    def test_output_and_manifest_symlinks_are_rejected_without_modifying_targets(self) -> None:
        self.output.symlink_to(self.dwi)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "source input or symlink"):
            self.generate(force=True)
        self.assert_preserved(before)
        self.output.unlink()
        target = self.root / "user-record.json"
        target.write_text("protected user data\n", encoding="ascii")
        self.manifest.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "provenance must not be a symlink"):
            self.generate()
        self.assertEqual(target.read_text(encoding="ascii"), "protected user data\n")
        self.assertFalse(self.output.exists())

    def test_explicit_mrtrix_command_uses_native_dwi_and_fsl_gradients(self) -> None:
        result = self.generate()
        call, = self.calls(self.mrtrix)
        self.assertEqual(result["command"][0], str(self.mrtrix))
        self.assertEqual(call["source"], str(self.dwi))
        self.assertEqual(call["shape"], [*SHAPE, 4])
        self.assertEqual(result["command"][3:], ["-fslgrad", str(self.bvecs), str(self.bvals), "-force"])
        self.assertEqual(result["backend"], "mrtrix")

    def test_compatibility_wrapper_explicitly_selects_mrtrix(self) -> None:
        result = brain_mask.generate_brain_mask_with_dwi2mask(
            self.dwi, self.bvals, self.bvecs, self.output, mrtrix_bin_dir=self.mrtrix.parent,
        )
        self.assertEqual(result["backend"], "mrtrix")
        self.assertEqual(len(self.calls(self.mrtrix)), 1)

    def test_synthstrip_adapter_receives_mean_b0_and_declared_mask_argument(self) -> None:
        home = self.root / "FreeSurfer with spaces"
        tool = write_fake_command(home / "bin/mri_synthstrip", "synthstrip")
        model = home / "models/synthstrip.1.pt"
        model.parent.mkdir()
        model.write_bytes(b"adapter-only fake SynthStrip model")
        config = BrainMaskConfig(backend="synthstrip", freesurfer_home=home, brain_mask_device="cpu")
        self.assertTrue(inspect_brain_mask_backend(config)["available"])
        result = self.generate(config)
        call, = self.calls(tool)
        self.assertEqual(result["command"][1::2], ["-i", "-m"])
        self.assertEqual(call["shape"], list(SHAPE))
        expected = nib.load(str(self.dwi)).get_fdata()[..., :2].mean(axis=-1)
        np.testing.assert_allclose(call["data"], expected)
        self.assertEqual(call["env"]["FREESURFER_HOME"], str(home))
        self.assertEqual(result["backend"], "synthstrip")

    def test_synthstrip_missing_model_or_command_never_falls_back_to_mrtrix(self) -> None:
        home = self.root / "freesurfer"
        config = BrainMaskConfig(backend="synthstrip", freesurfer_home=home, brain_mask_device="cpu")
        for missing in ("FreeSurfer mri_synthstrip", "FreeSurfer models/synthstrip.1.pt"):
            with self.subTest(missing=missing), patch.object(brain_mask, "_run") as run:
                inspected = inspect_brain_mask_backend(config)
                self.assertFalse(inspected["available"])
                self.assertIn(missing, inspected["missing_required"])
                with self.assertRaisesRegex(FileNotFoundError, "synthstrip unavailable"):
                    self.generate(config)
                run.assert_not_called()
            write_fake_command(home / "bin/mri_synthstrip", "synthstrip")
        self.assertEqual(self.calls(self.mrtrix), [])

    def test_run_streams_logs_and_limits_failure_diagnostics_to_last_50_lines(self) -> None:
        logs = []
        script = "import sys\nfor i in range(60): print('line-%02d' % i)\nsys.exit(9)"
        with self.assertRaises(RuntimeError) as raised:
            brain_mask._run([sys.executable, "-c", script], os.environ.copy(), logs.append)
        self.assertEqual(len(logs), 60)
        self.assertEqual(logs[0], "[Brain mask] line-00")
        self.assertNotIn("line-09", str(raised.exception))
        self.assertIn("line-10", str(raised.exception))
        self.assertIn("line-59", str(raised.exception))
        self.assertIn("failed (9)", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
