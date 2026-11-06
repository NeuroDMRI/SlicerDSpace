"""M-A mask parity harness (T2 design section 3 M-A, W-A4): mocked reference arm, tiny synthetic volumes."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import benchmark_mask_parity as parity
from shared import pnl_mask_native as native


HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None
HAS_SITK = importlib.util.find_spec("SimpleITK") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
SHAPE = (10, 9, 7)
# LAS, 2 mm, integer origin: ITK index <-> physical point is exact, so identity resampling is exact.
AFFINE = np.array([[-2.0, 0, 0, 20.0], [0, 2.0, 0, -10.0], [0, 0, 2.0, -6.0], [0, 0, 0, 1.0]])
IDENTITY = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0]


def write_matlab_transform(path: Path, parameters, fixed=(0.0, 0.0, 0.0),
                           name="MatrixOffsetTransformBase_double_3_3") -> None:
    """ITK MatlabTransformIO file as ANTs writes ``*0GenericAffine.mat`` (MAT v4, little-endian double)."""
    payload = b""
    for label, values in ((name, parameters), ("fixed", fixed)):
        values = np.asarray(values, "<f8")
        encoded = label.encode() + b"\0"
        payload += struct.pack("<5i", 0, len(values), 1, 0, len(encoded)) + encoded + values.tobytes()
    Path(path).write_bytes(payload)


def write_volume(path: Path, data, affine=AFFINE, codes=(1, 1)):
    import nibabel as nib

    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_qform(affine, code=codes[0])
    image.header.set_sform(affine, code=codes[1])
    nib.save(image, str(path))
    return path


def write_case(root: Path, name: str, *, blob=(2, 7), bvals=("0", "0", "1000", "1000"), volumes=4) -> dict:
    data = np.full(SHAPE + (volumes,), 10.0, dtype=np.float32)
    data[blob[0]:blob[1], 2:7, 1:5, :] = 100.0
    data[..., 2:] *= 0.3
    dwi = write_volume(root / f"{name}.nii.gz", data)
    (root / f"{name}.bval").write_text(" ".join(bvals) + "\n")
    vectors = np.zeros((3, volumes))
    vectors[0, 2:] = 1.0
    np.savetxt(root / f"{name}.bvec", vectors)
    return {"id": name, "dwi": str(dwi), "bval": str(root / f"{name}.bval"), "bvec": str(root / f"{name}.bvec")}


def as_paths(case: dict) -> dict:
    return dict(case, **{key: Path(case[key]) for key in ("dwi", "bval", "bvec")})


FAKE_PACKAGE = {
    "__init__.py": '__version__ = "fake-0"\n',
    "weights.py": textwrap.dedent('''\
        from pathlib import Path
        SOURCE_COMMIT = "fake-commit"
        TEMPLATE_SHA256 = "fake-template"
        def check(path, expected):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        '''),
    "inference.py": textwrap.dedent('''\
        import numpy as np
        def select_device(requested="auto"):
            return {"requested": requested, "resolved": "cpu"}
        def normalize(volume, percentile=99):
            volume = np.asarray(volume, dtype=np.float32)
            return volume / volume.max()
        def predict(volume, model_dir, *, device="cpu", batch_size=1, all_outputs=False):
            return {}, volume, (volume > 0.5).astype(np.uint8)
        '''),
    # ants_workspace as the package; finish_native writes the package's file names (identity transform).
    "pipeline.py": textwrap.dedent('''\
        import os
        from pathlib import Path
        import shutil
        import nibabel as nib
        import numpy as np
        def ants_workspace(work, template, ants_bin=None):
            if any(char.isspace() for char in str(work)):
                raise ValueError("ANTs work_dir must not contain whitespace")
            binary = work / "ants-bin"
            binary.mkdir()
            search = os.pathsep.join(filter(None, (str(ants_bin) if ants_bin else "", os.environ.get("PATH", ""))))
            for command in ("antsRegistrationSyNQuick.sh", "antsRegistration", "antsApplyTransforms", "ImageMath",
                            "PrintHeader"):
                (binary / command).symlink_to(Path(shutil.which(command, path=search)).absolute())
            reference = work / "template.nii.gz"
            reference.symlink_to(Path(template).resolve())
            return binary, reference
        def mean_b0(image, bvals=None, threshold=50):
            if image.ndim != 3:
                raise ValueError("3-D only")
            return np.maximum(image.get_fdata(), 0)
        def finish_native(template_mask, registered, baseline, matrix, output, *, ants_bin=None, threads=1):
            folder = Path(output).parent
            nib.save(nib.Nifti1Image(template_mask.astype(np.int16), registered.affine, registered.header.copy()),
                     str(folder / "template_mask.nii.gz"))
            original = nib.load(str(baseline))
            inverse = template_mask.astype(np.float32)
            nib.save(nib.Nifti1Image(inverse, original.affine), str(folder / "inverse.nii.gz"))
            nib.save(nib.Nifti1Image(inverse, original.affine), str(folder / "filled.nii.gz"))
            header = original.header.copy()
            header.set_data_dtype(np.int16)
            nib.save(nib.Nifti1Image(inverse.astype(np.uint8), original.affine, header), str(output))
        '''),
}

FAKE_SYNQUICK = '''#!{python}
import json, os, struct, sys
from pathlib import Path
import nibabel as nib
import numpy as np
arguments = sys.argv[1:]
options = dict(zip(arguments[::2], arguments[1::2]))
control = Path(__file__).resolve().parent
threads = options.get("-n", "1")
print(" Number of threads:        " + ("99" if (control / "wrong_threads").exists() else threads))
seed = os.environ.get("ANTS_RANDOM_SEED")
moving, fixed = nib.load(options["-m"]), nib.load(options["-f"])
data = moving.get_fdata()
if seed == "2" and (control / "seed2_shift").exists():
    data = np.roll(data, 1, axis=0)
prefix = options["-o"]
nib.save(nib.Nifti1Image(data.astype(np.float32), fixed.affine), prefix + "Warped.nii.gz")
payload = b""
for label, values in (("MatrixOffsetTransformBase_double_3_3", {identity}), ("fixed", [0.0, 0.0, 0.0])):
    values = np.asarray(values, "<f8")
    encoded = label.encode() + b"\\0"
    payload += struct.pack("<5i", 0, len(values), 1, 0, len(encoded)) + encoded + values.tobytes()
Path(prefix + "0GenericAffine.mat").write_bytes(payload)
Path(prefix + "call.json").write_text(json.dumps({{"n": threads, "seed": seed, "argv": arguments,
    "itk": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"), "antspath": os.environ.get("ANTSPATH")}}))
'''


def make_reference_runtime(root: Path) -> tuple[Path, Path]:
    """Fake managed PNL Python (fake pnl_brain_mask and torch) and a fake ANTs bin folder."""
    package = root / "pkg" / "pnl_brain_mask"
    package.mkdir(parents=True)
    for name, text in FAKE_PACKAGE.items():
        (package / name).write_text(text)
    (root / "pkg" / "torch.py").write_text('__version__ = "fake-torch"\ndef set_num_threads(count):\n    pass\n')
    python = root / "pnl" / "bin" / "python"
    python.parent.mkdir(parents=True)
    # The harness strips Slicer's Python variables from the reference environment; the fake restores the
    # test interpreter's own, so the test also runs inside Slicer's Python.
    exports = {"PYTHONPATH": os.pathsep.join(filter(None, (str(root / "pkg"), os.environ.get("PYTHONPATH"))))}
    exports.update({key: os.environ[key] for key in ("PYTHONHOME", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")
                    if os.environ.get(key)})
    lines = "".join(f"export {key}='{value}'\n" for key, value in exports.items())
    python.write_text(f'#!/bin/sh\n{lines}exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    ants = root / "ants" / "bin"
    ants.mkdir(parents=True)
    synquick = ants / "antsRegistrationSyNQuick.sh"
    synquick.write_text(FAKE_SYNQUICK.format(python=sys.executable, identity=IDENTITY))
    synquick.chmod(0o755)
    for command in ("antsRegistration", "antsApplyTransforms", "ImageMath", "PrintHeader"):
        (ants / command).write_text("#!/bin/sh\nexit 0\n")
        (ants / command).chmod(0o755)
    return python, ants


class HelperTest(unittest.TestCase):
    def test_printed_thread_line_is_the_last_synquick_value(self):
        text = "start\n Number of threads:        1\nmore\n Number of threads:        8\n"
        self.assertEqual(parity.printed_threads(text), 8)
        self.assertIsNone(parity.printed_threads("Number of threads: unknown\n"))

    def test_compare_dice_and_relative_count_difference_in_voxel_space(self):
        reference = np.zeros((4, 4, 4), np.uint8)
        reference[:2] = 1                                    # 32 voxels
        mask = np.zeros((4, 4, 4), np.int16)
        mask[:3] = 1                                         # 48 voxels, 32 shared
        result = parity.compare(mask, reference)
        self.assertAlmostEqual(result["dice"], 64 / 80)
        self.assertAlmostEqual(result["relative_voxel_difference"], 16 / 32)
        self.assertEqual(result["differing_voxels"], 16)
        self.assertFalse(result["equal"])
        self.assertTrue(parity.compare(reference, reference)["equal"])
        with self.assertRaises(ValueError):
            parity.compare(mask[:3], reference)

    def test_run_child_records_exit_code_log_and_peak_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "child.log"
            result = parity.run_child([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"],
                                      dict(os.environ), log)
            self.assertEqual(result["returncode"], 3)
            self.assertIn("hello", log.read_text())
            self.assertGreater(result["max_rss_mb"], 1.0)

    def test_baseline_threshold_is_the_production_default(self):
        from shared.brain_mask import BrainMaskConfig

        self.assertEqual(parity.B0_THRESHOLD, BrainMaskConfig.b0_threshold)

    def test_t6_takes_smallest_median_and_largest_grid(self):
        results = {name: {"check": {"shape": [size, size, size, 4]}}
                   for name, size in (("a", 30), ("b", 10), ("c", 20), ("d", 40), ("e", 25))}
        self.assertEqual(parity.t6_cases(results), ["b", "e", "d"])  # sizes 10, 20, 25, 30, 40
        self.assertEqual(parity.t6_cases({"only": results["a"]}), ["only"])

    def test_failure_clause_variants_change_one_a3_parameter_each(self):
        default = replace(native.PARAMETERS, **parity.VARIANTS["default"])
        sampling = replace(native.PARAMETERS, **parity.VARIANTS["sampling100"])
        once = replace(native.PARAMETERS, **parity.VARIANTS["estimate_once"])
        self.assertEqual(default, native.PARAMETERS)
        self.assertEqual(native.PARAMETERS.learning_rate_estimation, "EachIteration")
        self.assertEqual(replace(sampling, sampling_percentage=0.25), native.PARAMETERS)
        self.assertEqual(sampling.sampling_percentage, 1.0)
        self.assertEqual(replace(once, learning_rate_estimation="EachIteration"), native.PARAMETERS)

        class Method:
            Once, EachIteration = "once-constant", "each-constant"

            def __init__(self):
                self.calls = {}

            def __getattr__(self, name):
                return lambda *args, **kwargs: self.calls.__setitem__(name, (args, kwargs))

        method = Method()
        native.configure_level(method, once, 0)
        self.assertEqual(method.calls["SetOptimizerAsConjugateGradientLineSearch"][1]["estimateLearningRate"],
                         "once-constant")
        method = Method()
        native.configure_level(method, sampling, 1)
        self.assertEqual(method.calls["SetMetricSamplingPercentage"][0], (1.0, 2))  # A-O1: seed 1 + level 1
        self.assertEqual(method.calls["SetOptimizerAsConjugateGradientLineSearch"][1]["estimateLearningRate"],
                         "each-constant")

    @unittest.skipUnless(HAS_SITK, "SimpleITK is not installed")
    def test_variant_learning_rate_names_exist_in_simpleitk(self):
        import SimpleITK as sitk

        method = sitk.ImageRegistrationMethod()
        for change in parity.VARIANTS.values():
            name = replace(native.PARAMETERS, **change).learning_rate_estimation
            self.assertIsInstance(getattr(method, name), int)


@unittest.skipUnless(HAS_NIBABEL, "nibabel is not installed")
class InputTest(unittest.TestCase):
    def test_precheck_stops_on_volume_and_bvalue_count_mismatch_before_any_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = [write_case(root, "good"), write_case(root, "bad", bvals=("0", "0", "1000"))]
            (root / "cases.json").write_text(json.dumps(cases))
            args = parity.parse_arguments(["--cases", str(root / "cases.json"), "--out", str(root / "out"),
                                           "--model-dir", str(root), "--template", str(root / "t.nii.gz"),
                                           "--steps", "t1"])
            with self.assertRaisesRegex(SystemExit, "Pre-check failed"):
                parity.drive(args)
            checks = json.loads((root / "out" / "precheck.json").read_text())
            self.assertTrue(checks["good"]["counts_match"])
            self.assertEqual((checks["bad"]["volumes"], checks["bad"]["bvals"]), (4, 3))
            self.assertFalse(checks["bad"]["counts_match"])
            self.assertFalse((root / "out" / "cases").exists())

    def test_precheck_records_codes_and_flags_qform_sform_difference(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = write_case(root, "c")
            image = nib.load(case["dwi"])
            shifted = AFFINE.copy()
            shifted[0, 3] += 1e-3
            image.header.set_sform(shifted, code=2)
            nib.save(nib.Nifti1Image(np.asarray(image.dataobj), None, image.header), case["dwi"])
            check = parity.check_case(as_paths(case))
            self.assertEqual((check["qform_code"], check["sform_code"]), (1, 2))
            self.assertEqual(check["orientation"], "LAS")
            self.assertAlmostEqual(check["qform_sform_max_difference"], 1e-3, places=5)
            self.assertTrue(check["qform_sform_flag"])
            case = write_case(root, "same")
            self.assertFalse(parity.check_case(as_paths(case))["qform_sform_flag"])

    def test_b2_rotates_the_header_two_degrees_about_z_through_the_grid_centre(self):
        import nibabel as nib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
            source = write_volume(root / "b0.nii.gz", data, codes=(1, 2))
            rotated = parity.rotated_baseline(source, root / "rot.nii.gz")
            image = nib.load(str(root / "rot.nii.gz"))
            qform, qcode = image.header.get_qform(coded=True)
            sform, scode = image.header.get_sform(coded=True)
            self.assertEqual((int(qcode), int(scode)), (1, 2))
            np.testing.assert_allclose(qform, rotated, atol=1e-5)
            np.testing.assert_allclose(sform, rotated, atol=1e-5)
            np.testing.assert_array_equal(np.asarray(image.dataobj), data)
            centre = np.r_[(np.asarray(SHAPE) - 1) / 2, 1.0]
            np.testing.assert_allclose(rotated @ centre, AFFINE @ centre, atol=1e-9)
            change = rotated[:3, :3] @ np.linalg.inv(AFFINE[:3, :3])
            angle = math.radians(2.0)
            np.testing.assert_allclose(change, [[math.cos(angle), -math.sin(angle), 0],
                                                [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], atol=1e-12)


    def test_altered_b2_baseline_is_made_again_on_reuse(self):
        # A-O3 (A-C2): prepare_case checks the rotated baseline against its recorded SHA-256 before it reuses it.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = as_paths(write_case(root, "c"))
            folder = root / "out"
            first = parity.prepare_case(case, folder)
            rotated = folder / "baseline" / "mean_b0_rot2.nii.gz"
            baseline = folder / "baseline" / "mean_b0.nii.gz"
            baseline_before = baseline.stat().st_mtime_ns
            self.assertEqual(parity.prepare_case(case, folder), first)
            rotated.write_bytes(b"altered")
            second = parity.prepare_case(case, folder)
            self.assertEqual(second["rotated_sha256"], parity.sha256(rotated))
            self.assertEqual(second["rotated_sha256"], first["rotated_sha256"])
            self.assertEqual(len(list(rotated.parent.glob("mean_b0_rot2.nii.gz.stale-*"))), 1)
            self.assertEqual(baseline.stat().st_mtime_ns, baseline_before)


class NativeReuseTest(unittest.TestCase):
    """A-C2: a native run is reused only with an equal fingerprint of its inputs, code, parameters and threads."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "models").mkdir()
        (self.root / "models" / "weights.json").write_text("{}")
        self.args = SimpleNamespace(template=self.root / "t.nii.gz", model_dir=self.root / "models",
                                    variant="default")
        self.baseline = {"sha256": "baseline-hash"}
        self.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def run_native(self, threads=1):
        def child(command, env, log):
            command = [str(part) for part in command]
            self.calls.append(command)
            Path(command[command.index("--record") + 1]).write_text(json.dumps(
                {"status": "completed", "input_sha256": "baseline-hash", "threads": threads}))
            return {"returncode": 0, "seconds": 0.0, "max_rss_mb": 1.0, "log": str(log)}

        with patch.object(parity, "run_child", child):
            return parity.native_run(self.root, "N1", threads, self.args, self.baseline)

    def test_native_run_is_repeated_when_parameters_code_or_threads_change(self):
        self.run_native()
        self.run_native()
        self.assertEqual(len(self.calls), 1)                      # same fingerprint: reused
        with patch.dict(parity.VARIANTS, {"default": {"sampling_percentage": 1.0}}):
            self.run_native()                                     # the A3 variant changed (fails at 1a2e8d6)
        self.assertEqual(len(self.calls), 2)
        self.run_native()                                         # default parameters again: stale record is not it
        self.assertEqual(len(self.calls), 3)
        hashes = parity.implementation_hashes()
        self.assertIn("shared/pnl_mask_native.py", hashes)
        self.assertIn("shared/pnl_cnn/UPSTREAM.json", hashes)
        with patch.object(parity, "implementation_hashes", return_value=dict(hashes, extra="changed")):
            self.run_native()                                     # native code or a vendored file changed
        self.assertEqual(len(self.calls), 4)
        self.run_native(threads=2)                                # threads changed
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(list((self.root / "native").glob("N1.stale-*"))), 4)


@unittest.skipUnless(HAS_NIBABEL and HAS_SITK and HAS_SCIPY, "nibabel, SimpleITK and scipy are required")
class ReferenceArmTest(unittest.TestCase):
    """The real reference worker and driver against a fake pnl_brain_mask package and fake ANTs."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.python, self.ants = make_reference_runtime(self.root)
        self.models = self.root / "models"
        self.models.mkdir()
        self.template = write_volume(self.root / "template.nii.gz", np.ones(SHAPE))

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, cases, steps):
        (self.root / "cases.json").write_text(json.dumps(cases))
        return parity.parse_arguments([
            "--cases", str(self.root / "cases.json"), "--out", str(self.root / "out"),
            "--model-dir", str(self.models), "--template", str(self.template),
            "--pnl-python", str(self.python), "--ants-bin", str(self.ants), "--steps", steps])

    def test_reference_runs_noise_floor_and_t1_with_a_seed_dependent_reference(self):
        (self.ants / "seed2_shift").touch()
        cases = [write_case(self.root, "c1", blob=(2, 7)), write_case(self.root, "c2", blob=(2, 8))]
        out = self.root / "out"
        partial = out / "cases" / "c1" / "reference" / "A"
        partial.mkdir(parents=True)
        (partial / "partial.txt").write_text("interrupted run")
        args = self.arguments(cases, "reference,t1")
        self.assertEqual(parity.drive(args), 0)

        stale = [path for path in partial.parent.iterdir() if path.name.startswith("A.stale-")]
        self.assertEqual(len(stale), 1)
        self.assertTrue((stale[0] / "partial.txt").is_file())
        for case in cases:
            folder = out / "cases" / case["id"]
            link = folder / "input" / "dwi.nii.gz"
            self.assertTrue(link.is_symlink())
            self.assertEqual(Path(os.readlink(link)), Path(case["dwi"]))
            baseline = json.loads((folder / "baseline" / "baseline.json").read_text())
            result = json.loads((folder / "result.json").read_text())
            for name, (seed, threads, baseline_name) in parity.REFERENCE_RUNS.items():
                record = result["reference"][name]
                work = Path(record["work"])
                call = json.loads((work / "rigid-call.json").read_text())
                self.assertEqual((record["status"], record["valid"]), ("completed", True))
                self.assertEqual((record["seed"], record["threads"], record["printed_threads"]),
                                 (seed, threads, threads))
                self.assertEqual((call["seed"], call["n"], call["itk"]), (str(seed), str(threads), str(threads)))
                self.assertEqual(call["argv"][call["argv"].index("-t") + 1], "r")
                expected = baseline["rotated_sha256" if name == "B2" else "sha256"]
                self.assertEqual(record["baseline_sha256"], expected)
                for kept in ("rigid-0GenericAffine.mat", "rigid-Warped.nii.gz", "template_mask.nii.gz",
                             "inverse.nii.gz", "filled.nii.gz", "native_mask.nii.gz", "synquick.log"):
                    self.assertTrue((work / kept).is_file(), kept)
            comparisons = result["reference_vs_A"]
            self.assertEqual({name: item["equal"] for name, item in comparisons.items()},
                             {"A_repeat": True, "B1": True, "B2": True, "S2": False})
            self.assertTrue(result["t1"]["pass"])
            self.assertEqual(result["t1"]["inverse_max_abs_difference"], 0.0)
            self.assertEqual(result["t1"]["differing_indices"], [])
        summary = json.loads((out / "summary.json").read_text())["evaluation"]
        s2 = [json.loads((out / "cases" / name / "result.json").read_text())["reference_vs_A"]["S2"]
              for name in ("c1", "c2")]
        self.assertFalse(summary["noise_floor_zero"])
        self.assertEqual(summary["perturbation_bounds"]["one_minus_dice"], max(item["one_minus_dice"] for item in s2))
        self.assertEqual(summary["t1"]["status"], "pass")
        self.assertEqual(summary["t3"]["status"], "incomplete")
        self.assertEqual(summary["invalid_reference_runs"], [])
        self.assertEqual(len((out / "summary.tsv").read_text().splitlines()), 3)

        # A completed run is reused, not repeated.
        record = out / "cases" / "c1" / "reference" / "S2" / "reference.json"
        before = record.stat().st_mtime_ns
        parity.drive(self.arguments(cases, "reference"))
        self.assertEqual(record.stat().st_mtime_ns, before)
        # A completed run made from another baseline file is set aside and repeated.
        stored = json.loads(record.read_text())
        record.write_text(json.dumps(dict(stored, baseline_sha256="another-baseline")))
        parity.drive(self.arguments(cases, "reference"))
        self.assertEqual(json.loads(record.read_text())["baseline_sha256"], stored["baseline_sha256"])
        self.assertEqual(len(list(record.parent.parent.glob("S2.stale-*"))), 1)
        self.assertEqual(len(list(record.parent.parent.glob("B1.stale-*"))), 0)
        # Steps not selected in the second call keep their results.
        self.assertTrue(json.loads((out / "cases" / "c1" / "result.json").read_text())["t1"]["pass"])
        self.assertEqual(json.loads((out / "summary.json").read_text())["evaluation"]["t1"]["status"], "pass")

    def test_a_printed_thread_count_that_differs_from_n_makes_the_run_invalid(self):
        (self.ants / "wrong_threads").touch()
        cases = [write_case(self.root, "c1")]
        parity.drive(self.arguments(cases, "reference,t1"))
        result = json.loads((self.root / "out" / "cases" / "c1" / "result.json").read_text())
        self.assertEqual({record["printed_threads"] for record in result["reference"].values()}, {99})
        self.assertFalse(any(record["valid"] for record in result["reference"].values()))
        self.assertNotIn("reference_vs_A", result)
        self.assertNotIn("t1", result)
        evaluation = json.loads((self.root / "out" / "summary.json").read_text())["evaluation"]
        self.assertEqual(len(evaluation["invalid_reference_runs"]), 5)
        self.assertEqual(evaluation["t1"]["status"], "incomplete")
        self.assertEqual(evaluation["t3"]["status"], "incomplete")


    def test_reference_record_without_the_current_fingerprint_is_repeated(self):
        # A-C2: an older record (no fingerprint) or one made with other settings is set aside and repeated.
        cases = [write_case(self.root, "c1")]
        parity.drive(self.arguments(cases, "reference"))
        runs = self.root / "out" / "cases" / "c1" / "reference"
        record = runs / "B1" / "reference.json"
        stored = json.loads(record.read_text())
        self.assertEqual(len(stored["fingerprint"]), 64)
        record.write_text(json.dumps({key: value for key, value in stored.items() if key != "fingerprint"}))
        parity.drive(self.arguments(cases, "reference"))
        self.assertEqual(len(list(runs.glob("B1.stale-*"))), 1)
        self.assertEqual(json.loads(record.read_text())["fingerprint"], stored["fingerprint"])
        self.assertEqual(len(list(runs.glob("A.stale-*"))), 0)
        # The B2 fingerprint follows the rotated baseline's own hash and the rotation.
        b2 = json.loads((runs / "B2" / "reference.json").read_text())["fingerprint"]
        self.assertNotEqual(b2, json.loads((runs / "A" / "reference.json").read_text())["fingerprint"])

    def test_carried_t1_is_dropped_when_run_a_outputs_change(self):
        (self.ants / "seed2_shift").touch()
        cases = [write_case(self.root, "c1")]
        parity.drive(self.arguments(cases, "reference,t1"))
        result_path = self.root / "out" / "cases" / "c1" / "result.json"
        self.assertTrue(json.loads(result_path.read_text())["t1"]["pass"])
        parity.drive(self.arguments(cases, "reference"))
        self.assertTrue(json.loads(result_path.read_text())["t1"]["pass"])      # unchanged inputs: kept
        transform = Path(json.loads(result_path.read_text())["reference"]["A"]["transform"])
        write_matlab_transform(transform, [0.0, 1.0, 0, -1.0, 0.0, 0, 0, 0, 1.0, 0, 0, 0])
        parity.drive(self.arguments(cases, "reference"))
        self.assertNotIn("t1", json.loads(result_path.read_text()))               # run A's transform changed
        summary = json.loads((self.root / "out" / "summary.json").read_text())["evaluation"]
        self.assertEqual(summary["t1"]["status"], "incomplete")

    def test_zero_noise_writes_the_stop_report_without_bounds_and_skips_later_steps(self):
        # A-C3, M-A 1: every perturbation equals run A on every case.
        cases = [write_case(self.root, "c1"), write_case(self.root, "c2", blob=(2, 8))]
        arguments = self.arguments(cases, "reference,native,t1,t2,t4,t6")
        arguments.torch_python = Path(sys.executable)
        trap = AssertionError("a step after the zero-noise stop ran")
        with patch.object(parity, "native_run", side_effect=trap), patch.object(parity, "t1_case", side_effect=trap), \
                patch.object(parity, "t2_case", side_effect=trap), patch.object(parity, "t4_case", side_effect=trap):
            self.assertEqual(parity.drive(arguments), 0)
        out = self.root / "out"
        evaluation = json.loads((out / "summary.json").read_text())["evaluation"]
        self.assertTrue(evaluation["noise_floor_zero"])
        self.assertNotIn("perturbation_bounds", evaluation)
        self.assertEqual(evaluation["stop"]["status"], "stop")
        self.assertEqual({key: evaluation[key]["status"] for key in ("t1", "t3", "t4")},
                         {"t1": "stop", "t3": "stop", "t4": "stop"})
        for case in ("c1", "c2"):
            result = json.loads((out / "cases" / case / "result.json").read_text())
            self.assertEqual({name for name in ("t1", "t2", "t3", "t4", "t6") if name in result}, set())
            self.assertFalse((out / "cases" / case / "native").exists())
            self.assertTrue(all(item["equal"] for item in result["reference_vs_A"].values()))


class EvaluationTest(unittest.TestCase):
    @staticmethod
    def case(perturbations=None, t3=(0.0, 0.0), t4=(0.0, 0.0)):
        runs = {name: {"equal": True, "one_minus_dice": 0.0, "relative_voxel_difference": 0.0}
                for name in parity.PERTURBATIONS}
        for name, (dice, relative) in (perturbations or {}).items():
            runs[name] = {"equal": False, "one_minus_dice": dice, "relative_voxel_difference": relative}
        return {"reference": {}, "reference_vs_A": runs, "t1": {"pass": True},
                "t3": {"one_minus_dice": t3[0], "relative_voxel_difference": t3[1]},
                "t4": {"status": "completed", "one_minus_dice": t4[0], "relative_voxel_difference": t4[1]}}

    def test_no_bound_when_every_perturbation_equals_run_a(self):
        evaluation = parity.evaluate({"c1": self.case(), "c2": self.case()})
        self.assertTrue(evaluation["noise_floor_zero"])
        self.assertEqual(evaluation["t3"]["status"], "stop")
        self.assertEqual(evaluation["t4"]["status"], "stop")

    def test_bounds_are_the_maximum_over_cases_and_perturbations(self):
        cases = {"c1": self.case({"S2": (0.02, 0.01)}, t3=(0.015, 0.02), t4=(0.02, 0.0)),
                 "c2": self.case({"B1": (0.01, 0.03)}, t3=(0.025, 0.0), t4=(0.0, 0.031))}
        evaluation = parity.evaluate(cases)
        self.assertEqual(evaluation["perturbation_bounds"], {"one_minus_dice": 0.02, "relative_voxel_difference": 0.03})
        self.assertEqual(evaluation["t3"], {"status": "fail", "failing_cases": ["c2"]})
        self.assertEqual(evaluation["t4"], {"status": "fail", "failing_cases": ["c2"]})
        cases["c2"]["t3"]["one_minus_dice"] = 0.02
        cases["c2"]["t4"]["relative_voxel_difference"] = 0.03
        evaluation = parity.evaluate(cases)
        self.assertEqual((evaluation["t3"]["status"], evaluation["t4"]["status"]), ("pass", "pass"))

    def test_a_repeat_joins_the_perturbations_only_when_it_differs(self):
        cases = {"c1": self.case({"S2": (0.01, 0.01), "A_repeat": (0.05, 0.002)}), "c2": self.case({"S2": (0.01, 0.01)})}
        evaluation = parity.evaluate(cases)
        self.assertEqual(evaluation["a_repeat_differs"], ["c1"])
        self.assertEqual(evaluation["perturbation_bounds"]["one_minus_dice"], 0.05)

    def test_a_missing_or_failed_arm_leaves_the_verdict_incomplete(self):
        cases = {"c1": self.case({"S2": (0.01, 0.01)}), "c2": self.case({"S2": (0.01, 0.01)})}
        del cases["c2"]["reference_vs_A"]["B2"]
        cases["c1"]["t4"] = {"status": "failed"}
        evaluation = parity.evaluate(cases)
        self.assertEqual(evaluation["missing_reference_runs"], ["c2:B2"])
        self.assertEqual(evaluation["t3"]["status"], "incomplete")
        self.assertIn("c1", evaluation["t4"]["missing"])


@unittest.skipUnless(HAS_NIBABEL and HAS_SITK and HAS_SCIPY, "nibabel, SimpleITK and scipy are required")
class DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.template = write_volume(self.root / "template.nii.gz", np.ones(SHAPE))
        work = self.root / "work"
        work.mkdir()
        mask = np.zeros(SHAPE)
        mask[2:7, 3:8, 1:6] = 1
        write_volume(work / "template_mask.nii.gz", mask)
        write_volume(work / "baseline.nii.gz", np.ones(SHAPE))
        self.warped = np.random.default_rng(0).random(SHAPE)
        write_volume(work / "rigid-Warped.nii.gz", self.warped)
        write_matlab_transform(work / "rigid-0GenericAffine.mat", IDENTITY)
        self.reference = {"work": str(work), "transform": str(work / "rigid-0GenericAffine.mat"),
                          "warped": str(work / "rigid-Warped.nii.gz")}

    def tearDown(self):
        self.temporary.cleanup()

    def native_record(self, transform):
        import SimpleITK as sitk

        keep = self.root / "keep"
        keep.mkdir(exist_ok=True)
        sitk.WriteTransform(transform, str(keep / "transform.tfm"))
        write_volume(keep / "Warped.nii.gz", 2 * self.warped + 1)
        return {"keep_dir": str(keep)}

    def test_t2_translation_rotation_corner_displacement_and_ncc(self):
        import SimpleITK as sitk

        shift = sitk.TranslationTransform(3, (3.0, 4.0, 0.0))
        result = parity.t2_case(self.reference, self.native_record(shift), self.template)
        self.assertAlmostEqual(result["translation_mm"], 5.0)
        self.assertAlmostEqual(result["max_corner_displacement_mm"], 5.0)
        self.assertAlmostEqual(result["rotation_degrees"], 0.0, places=6)
        self.assertAlmostEqual(result["warped_ncc"], 1.0)

        fixed = sitk.ReadImage(str(self.template))
        centre = fixed.TransformContinuousIndexToPhysicalPoint((4.0, 5.0, 3.0))  # box [2,6] x [3,7] x [1,5]
        rotation = sitk.Euler3DTransform(centre, 0.0, 0.0, math.radians(1.0))
        result = parity.t2_case(self.reference, self.native_record(rotation), self.template)
        self.assertAlmostEqual(result["rotation_degrees"], 1.0, places=6)
        self.assertAlmostEqual(result["translation_mm"], 0.0, places=9)
        # Corner offset from the centre in the x-y plane: 2 voxels x 2 mm each axis.
        self.assertAlmostEqual(result["max_corner_displacement_mm"], 2 * math.sin(math.radians(0.5)) * math.hypot(4, 4))

    def t4_setup(self):
        mask = np.zeros(SHAPE, np.uint8)
        mask[2:7, 3:8, 1:6] = 1
        other = mask.copy()
        other[6, 7, 5] = 0
        venv = self.root / "venv" / "bin" / "python"
        venv.parent.mkdir(parents=True, exist_ok=True)
        venv.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        venv.chmod(0o755)
        parity._VERSION_CACHE.clear()
        calls = []

        def child(command, env, log):
            command = [str(part) for part in command]
            python = command[0]
            calls.append(python)
            arm_mask, version = (mask, "2.14.1") if python == sys.executable else (other, "2.2.2")
            np.save(command[command.index("--output") + 1], arm_mask)
            Path(command[command.index("--record") + 1]).write_text(json.dumps(
                {"status": "completed", "torch_version": version, "numpy_version": "1.26.4"}))
            return {"returncode": 0, "seconds": 0.0, "max_rss_mb": 1.0, "log": str(log)}

        args = SimpleNamespace(template=self.template, model_dir=self.root, torch_python=venv)
        return args, calls, child

    def test_t4_compares_both_torch_masks_after_a4_on_the_native_grid(self):
        case = self.root / "case"
        args, calls, child = self.t4_setup()
        with patch.object(parity, "run_child", child), \
                patch.object(parity, "normalized_input", return_value=np.zeros(SHAPE, np.float32)):
            result = parity.t4_case(case, self.reference, args)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["template_grid_differing_voxels"], 1)
        self.assertEqual((result["voxels"], result["reference_voxels"]), (124, 125))
        self.assertAlmostEqual(result["one_minus_dice"], 1 - 248 / 249)
        self.assertAlmostEqual(result["relative_voxel_difference"], 1 / 125)
        self.assertEqual(result["torch_versions"], {"slicer": "2.14.1", "venv": "2.2.2"})
        self.assertEqual(len(calls), 2)

    def test_t4_arms_and_input_are_repeated_only_when_their_fingerprints_change(self):
        # A-C2: --torch-python changes -> that arm runs again; run A's Warped image changes -> both arms and the input.
        case = self.root / "case"
        args, calls, child = self.t4_setup()
        normalized = patch.object(parity, "normalized_input", return_value=np.zeros(SHAPE, np.float32))
        with patch.object(parity, "run_child", child), normalized:
            first = parity.t4_case(case, self.reference, args)
            parity.t4_case(case, self.reference, args)
            self.assertEqual(len(calls), 2)                                # nothing changed: both reused
            moved = self.root / "venv2" / "bin" / "python"
            moved.parent.mkdir(parents=True)
            moved.write_bytes(args.torch_python.read_bytes())
            moved.chmod(0o755)
            second = parity.t4_case(case, self.reference, SimpleNamespace(**dict(vars(args), torch_python=moved)))
            self.assertEqual(calls[2:], [str(moved)])                      # only the venv arm (fails at 1a2e8d6)
            self.assertNotEqual(second["arm_fingerprints"]["venv"], first["arm_fingerprints"]["venv"])
            self.assertEqual(second["arm_fingerprints"]["slicer"], first["arm_fingerprints"]["slicer"])
            write_volume(Path(self.reference["warped"]), 3 * self.warped + 2)  # run A's Warped image changed
            third = parity.t4_case(case, self.reference, SimpleNamespace(**dict(vars(args), torch_python=moved)))
            self.assertEqual(len(calls), 5)                                # both arms again
            self.assertNotEqual(third["fingerprint"], first["fingerprint"])
        self.assertEqual(len(list((case).glob("t4.stale-*"))), 1)
        self.assertTrue((case / "t4" / "input.json").is_file())


if __name__ == "__main__":
    unittest.main()
