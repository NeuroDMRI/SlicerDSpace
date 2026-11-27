"""Batch 2 / task G: the Mach-O minimum-macOS reader (G1), the split FSL gate (G2), the Eddy/Topup-only
disabling in the UI (G3) and the surface-runtime macOS check (G4).

Every test names its item. Runtime files are synthetic Mach-O bytes built with ``struct``.
"""

from __future__ import annotations

import ast
import os
import struct
import sys
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]

ARM64 = 0x0100000C
X86_64 = 0x01000007


def _build_version(minos, platform_id=1):
    return struct.pack("<IIIIII", 0x32, 24, platform_id, minos, 0, 0)


def _version_min(version):
    return struct.pack("<IIII", 0x24, 16, version, 0)


def pack_version(major, minor=0, patch=0):
    return (major << 16) | (minor << 8) | patch


def thin_macho(commands=(), cputype=ARM64):
    body = b"".join(commands)
    header = struct.pack("<IiiIIIII", 0xFEEDFACF, cputype, 0, 2, len(commands), len(body), 0, 0)
    return header + body


def fat_macho(slices, wide=False):
    """``slices`` is a list of ``(cputype, thin_bytes)``."""
    record = 32 if wide else 20
    offset = 8 + record * len(slices)
    table, payload = b"", b""
    for cputype, data in slices:
        offset = (offset + 0x3FFF) & ~0x3FFF
        pad = offset - (8 + record * len(slices) + len(payload))
        payload += b"\0" * pad
        if wide:
            table += struct.pack(">IIQQII", cputype, 0, offset, len(data), 14, 0)
        else:
            table += struct.pack(">IIIII", cputype, 0, offset, len(data), 14)
        payload += data
        offset += len(data)
    return struct.pack(">II", 0xCAFEBABF if wide else 0xCAFEBABE, len(slices)) + table + payload


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class G1MachoReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def read(self, data, arch="arm64", name="f"):
        from shared.macho_min_os import macho_min_macos
        return macho_min_macos(write(self.root / name, data), arch)

    def test_g1_thin_build_version_command(self):
        self.assertEqual(self.read(thin_macho([_build_version(pack_version(15, 0))])), (15, 0, 0))
        self.assertEqual(self.read(thin_macho([_build_version(pack_version(10, 13, 2))])), (10, 13, 2))

    def test_g1_thin_version_min_macosx_command(self):
        self.assertEqual(self.read(thin_macho([_version_min(pack_version(10, 13))], X86_64)), (10, 13, 0))

    def test_g1_non_macos_platform_and_missing_command_are_unknown(self):
        self.assertIsNone(self.read(thin_macho([_build_version(pack_version(15, 0), platform_id=2)])))
        self.assertIsNone(self.read(thin_macho([])))

    def test_g1_fat_picks_the_host_slice_and_falls_back_to_the_highest(self):
        arm = thin_macho([_build_version(pack_version(11, 0))], ARM64)
        intel = thin_macho([_version_min(pack_version(10, 13))], X86_64)
        for wide in (False, True):
            data = fat_macho([(X86_64, intel), (ARM64, arm)], wide=wide)
            self.assertEqual(self.read(data, "arm64"), (11, 0, 0), wide)
            self.assertEqual(self.read(data, "x86_64"), (10, 13, 0), wide)
            self.assertEqual(self.read(data, "riscv"), (11, 0, 0), wide)  # no matching slice: the highest

    def test_g1_host_arch_comes_from_platform_info_when_not_given(self):
        from shared.macho_min_os import macho_min_macos
        arm = thin_macho([_build_version(pack_version(11, 0))], ARM64)
        intel = thin_macho([_version_min(pack_version(10, 13))], X86_64)
        path = write(self.root / "fat", fat_macho([(X86_64, intel), (ARM64, arm)]))
        with mock.patch("shared.platform_info.host_machine", return_value="arm64"):
            self.assertEqual(macho_min_macos(path), (11, 0, 0))
        with mock.patch("shared.platform_info.host_machine", return_value="x86_64"):
            self.assertEqual(macho_min_macos(path), (10, 13, 0))

    def test_g1_bad_files_return_none_and_never_raise(self):
        from shared.macho_min_os import macho_min_macos
        good = thin_macho([_build_version(pack_version(15, 0))])
        self.assertIsNone(self.read(good[:20]))  # truncated header
        self.assertIsNone(self.read(good[:-4]))  # truncated load commands
        self.assertIsNone(self.read(b"#!/bin/sh\necho hi\n"))  # script
        self.assertIsNone(self.read(b""))
        self.assertIsNone(self.read(b"\xca\xfe\xba\xbe\x00\x00\x00\x34" + b"\0" * 64))  # Java class, not a fat file
        self.assertIsNone(self.read(struct.pack("<IiiIIIII", 0xFEEDFACF, ARM64, 0, 2, 1, 0xFFFFFFF0, 0, 0)))  # huge sizeofcmds
        bad_cmd = struct.pack("<II", 0x32, 0) + b"\0" * 16  # cmdsize 0 must not loop
        self.assertIsNone(self.read(thin_macho([bad_cmd])))
        self.assertIsNone(macho_min_macos(self.root / "does-not-exist"))
        self.assertIsNone(macho_min_macos(self.root))  # a directory

    def test_g1_dir_is_the_max_over_bin_and_nested_lib_and_skips_non_macho(self):
        from shared.macho_min_os import format_version, min_macos_for_dir
        bundle = self.root / "bundle"
        write(bundle / "bin" / "a", thin_macho([_build_version(pack_version(11, 0))]))
        write(bundle / "bin" / "script", b"#!/bin/sh\n")
        write(bundle / "lib" / "sub" / "deep" / "b.dylib", thin_macho([_build_version(pack_version(13, 0))]))
        write(bundle / "share" / "ignored", thin_macho([_build_version(pack_version(99, 0))]))
        self.assertEqual(min_macos_for_dir(bundle, "arm64"), (13, 0, 0))
        self.assertEqual(format_version(min_macos_for_dir(bundle, "arm64")), "13.0")
        self.assertIsNone(min_macos_for_dir(self.root / "empty-or-missing"))

    @skip_unless_symlinks
    def test_g1_dir_skips_symlinks_so_a_loop_cannot_hang(self):
        from shared.macho_min_os import min_macos_for_dir
        bundle = self.root / "loop"
        write(bundle / "bin" / "a", thin_macho([_build_version(pack_version(12, 0))]))
        os.symlink(bundle / "bin", bundle / "bin" / "again")
        os.symlink(bundle / "bin" / "a", bundle / "bin" / "alias")
        os.symlink(bundle, bundle / "lib")
        self.assertEqual(min_macos_for_dir(bundle, "arm64"), (12, 0, 0))

    def test_g1_dir_result_is_cached_by_mtime_and_refreshed_when_the_file_changes(self):
        from shared import macho_min_os
        bundle = self.root / "cache"
        binary = write(bundle / "bin" / "a", thin_macho([_build_version(pack_version(11, 0))]))
        self.assertEqual(macho_min_os.min_macos_for_dir(bundle, "arm64"), (11, 0, 0))
        with mock.patch.object(macho_min_os, "macho_min_macos", side_effect=AssertionError("re-read")):
            self.assertEqual(macho_min_os.min_macos_for_dir(bundle, "arm64"), (11, 0, 0))
        binary.write_bytes(thin_macho([_build_version(pack_version(14, 0))]))
        stamp = binary.stat().st_mtime_ns + 5_000_000_000
        os.utime(binary, ns=(stamp, stamp))
        os.utime(bundle / "bin", ns=(stamp, stamp))
        self.assertEqual(macho_min_os.min_macos_for_dir(bundle, "arm64"), (14, 0, 0))

    def test_g1_version_helpers(self):
        from shared.macho_min_os import format_version, parse_version_text
        self.assertEqual(format_version((15, 0, 0)), "15.0")
        self.assertEqual(format_version((10, 13, 2)), "10.13.2")
        self.assertEqual(format_version(None), "")
        self.assertEqual(parse_version_text("14.5"), (14, 5, 0))
        self.assertIsNone(parse_version_text("x"))
        self.assertIsNone(parse_version_text(""))


def make_native_bundle(root, name, minos, *, commands=()):
    bundle = Path(root) / name
    tool = bundle / "bin" / "tool"
    old_mtime_ns = tool.stat().st_mtime_ns if tool.exists() else None
    write(tool, thin_macho([_build_version(pack_version(*minos))]) if minos else b"dummy")
    if old_mtime_ns is not None:
        # FLAKY-G2: a same-size rewrite in one file-clock tick keeps the mtime, and the min-OS cache (path, size,
        # mtime) then returns the old minimum; give the rewrite a later mtime, as a real reinstall has.
        stamp = old_mtime_ns + 1_000_000_000
        os.utime(tool, ns=(stamp, stamp))
    (bundle / "native-runtime.json").write_text('{"commands": %s}' % list(commands).__repr__().replace("'", '"'))
    return bundle


class BundleEnv:
    """Bundled-runtimes root and user runtime root pointing into a temp dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.bundled = Path(self.tmp.name) / "bundled-runtimes"
        self.bundled.mkdir()
        self.user = Path(self.tmp.name) / "runtimes"
        self.user.mkdir()
        patcher = mock.patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_RUNTIMES_ROOT": str(self.bundled),
                                               "SLICERDSPACE_RUNTIME_ROOT": str(self.user)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def mac(self, version):
        patcher = mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), ""))
        patcher.start()
        self.addCleanup(patcher.stop)


class G2FslGateTest(BundleEnv, unittest.TestCase):
    def test_g2_darwin_with_bundled_fsl_and_mrtrix_offers_eddy_topup_on_14_2_and_15_1(self):
        from shared.selected_setup import fsl_preprocessing_offered
        make_native_bundle(self.bundled, "fsl-2602.0-darwin-arm64", (11, 0))
        make_native_bundle(self.bundled, "mrtrix-3.0.8-darwin-arm64", (10, 13))
        for version in ("14.2", "15.1"):
            with mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")):
                self.assertTrue(fsl_preprocessing_offered("Darwin", "arm64"), version)

    def test_g2_managed_fallback_without_a_bundle_keeps_the_14_5_limit(self):
        from shared.selected_setup import fsl_preprocessing_offered
        for version, expected in (("14.2", False), ("14.4", False), ("14.5", True), ("15.1", True)):
            with mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")):
                self.assertEqual(fsl_preprocessing_offered("Darwin", "arm64"), expected, version)

    def test_g2_one_missing_bundle_or_unknown_minimum_falls_back_to_the_managed_rule(self):
        from shared.selected_setup import fsl_preprocessing_offered
        make_native_bundle(self.bundled, "fsl-2602.0-darwin-arm64", (11, 0))  # no MRtrix bundle
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertFalse(fsl_preprocessing_offered("Darwin", "arm64"))
        make_native_bundle(self.bundled, "mrtrix-3.0.8-darwin-arm64", None)  # unknown minimum
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertFalse(fsl_preprocessing_offered("Darwin", "arm64"))
        with mock.patch("platform.mac_ver", return_value=("14.6", ("", "", ""), "")):
            self.assertTrue(fsl_preprocessing_offered("Darwin", "arm64"))

    def test_g2_bundle_newer_than_the_running_macos_is_not_offered_by_the_bundle(self):
        from shared.selected_setup import fsl_preprocessing_offered
        make_native_bundle(self.bundled, "fsl-2602.0-darwin-arm64", (15, 0))
        make_native_bundle(self.bundled, "mrtrix-3.0.8-darwin-arm64", (10, 13))
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertFalse(fsl_preprocessing_offered("Darwin", "arm64"))
        with mock.patch("platform.mac_ver", return_value=("15.0", ("", "", ""), "")):
            self.assertTrue(fsl_preprocessing_offered("Darwin", "arm64"))

    def test_g2_other_platforms_keep_their_rule_and_preprocessing_offered_is_unchanged(self):
        from shared.selected_setup import fsl_preprocessing_offered, preprocessing_offered
        with mock.patch("platform.mac_ver", return_value=("15.1", ("", "", ""), "")):
            self.assertTrue(fsl_preprocessing_offered("Windows", "AMD64"))
            self.assertTrue(fsl_preprocessing_offered("Linux", "x86_64"))
            self.assertFalse(fsl_preprocessing_offered("Linux", "aarch64"))
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertFalse(preprocessing_offered("Darwin", "arm64"))  # managed-route meaning, bundles ignored

    def test_g2_unavailable_text_names_the_versions_on_a_mac(self):
        from shared.selected_setup import fsl_unavailable_text
        make_native_bundle(self.bundled, "fsl-2602.0-darwin-arm64", (13, 0))
        make_native_bundle(self.bundled, "mrtrix-3.0.8-darwin-arm64", (10, 13))
        with mock.patch("platform.mac_ver", return_value=("12.3", ("", "", ""), "")):
            text = fsl_unavailable_text("Darwin", "arm64")
            self.assertIn("Eddy and Topup need FSL", text)
            self.assertIn("Brain mask and DWI denoising still run", text)
            self.assertIn("12.3", text)
            self.assertIn("13.0", text)  # the bundle's minimum is lower than the managed 14.5
            self.assertIn("需要 FSL", fsl_unavailable_text("Darwin", "arm64", zh=True))
        make_native_bundle(self.bundled, "fsl-2602.0-darwin-arm64", (15, 0))
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertIn("macOS 14.5 or later", fsl_unavailable_text("Darwin", "arm64"))
        with mock.patch("shared.native_runtimes.installed_bundle_bin_dirs", return_value=()), \
                mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            self.assertIn("macOS 14.5 or later", fsl_unavailable_text("Darwin", "arm64"))
        plain = fsl_unavailable_text("Linux", "aarch64")
        self.assertIn("Eddy and Topup need FSL", plain)
        self.assertNotIn("macOS", plain)

    def test_flaky_g2_bundle_rewritten_in_the_same_clock_tick(self):
        """FLAKY-G2: macho_min_os caches a binary's minimum by path, size and mtime. On a coarse file clock (Windows) a
        same-size rewrite in the same tick keeps the mtime, so the old minimum comes back; this forces that tick."""
        real_write_bytes = Path.write_bytes

        def write_in_the_same_tick(path, data):
            before = path.stat() if path.exists() else None
            written = real_write_bytes(path, data)
            if before is not None:
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return written

        with mock.patch.object(Path, "write_bytes", write_in_the_same_tick):
            self.test_g2_unavailable_text_names_the_versions_on_a_mac()

    def test_g2_setup_plan_note_uses_the_new_text_for_the_managed_install_path(self):
        from shared.selected_setup import PREPROCESSING_UNSUPPORTED_TEXT, build_setup_plan
        self.assertEqual(
            PREPROCESSING_UNSUPPORTED_TEXT,
            "Eddy and Topup need FSL, which is not available on this system; Brain mask and DWI denoising still run.")
        item = SimpleNamespace(key="command:eddy", label="eddy", install_group="fsl", kind="command", required=True,
                               status="missing", targets=("preprocessing",), install_hint="x", details={})
        with mock.patch("platform.mac_ver", return_value=("14.2", ("", "", ""), "")):
            plan = build_setup_plan(SimpleNamespace(statuses=[item], targets=("preprocessing",)),
                                    system="Linux", machine="aarch64", closure_targets=("preprocessing",))
        self.assertIn(PREPROCESSING_UNSUPPORTED_TEXT, plan["unsupported_notes"].values())


class FakeCheckbox:
    def __init__(self, checked=True):
        self.checked, self.enabled, self.tip, self.calls = checked, True, "", []

    def setEnabled(self, value):
        self.calls.append(("enabled", value))
        self.enabled = value

    def setToolTip(self, text):
        self.calls.append(("tip", text))
        self.tip = text

    def setChecked(self, value):
        self.calls.append(("checked", value))
        self.checked = value

    def blockSignals(self, value):
        self.calls.append(("signals", value))


def node(stages="denoise,eddy,topup", target=True):
    return SimpleNamespace(targetPreprocessing=target, orchestrationPreprocessingStages=stages)


class G3SelectionTest(unittest.TestCase):
    def test_g3_without_fsl_stages_keeps_canonical_order_and_never_returns_the_run_everything_value(self):
        from shared.algorithm_selection import without_fsl_stages
        self.assertEqual(without_fsl_stages("denoise,eddy,topup"), "denoise")
        self.assertEqual(without_fsl_stages("topup,brain_mask,eddy,denoise"), "brain_mask,denoise")
        self.assertEqual(without_fsl_stages("eddy,topup"), "")
        self.assertEqual(without_fsl_stages(""), "")
        self.assertEqual(without_fsl_stages("all"), "brain_mask,denoise")

    def test_g3_apply_keeps_the_checkbox_enabled_and_strips_eddy_topup_where_fsl_is_not_offered(self):
        from shared.ui_parameter_sync import apply_preprocessing_availability
        box, n = FakeCheckbox(), node()
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            self.assertFalse(apply_preprocessing_availability(box, "Linux", "aarch64", node=n))
        self.assertNotIn(("enabled", False), box.calls)
        self.assertNotIn(("checked", False), box.calls)
        self.assertTrue(box.enabled and box.checked)
        self.assertEqual(n.orchestrationPreprocessingStages, "denoise")
        self.assertTrue(n.targetPreprocessing)
        self.assertIn("Eddy and Topup need FSL", box.tip)
        self.assertIn("Brain mask and DWI denoising still run", box.tip)

    def test_g3_apply_with_only_fsl_stages_selected_writes_a_non_empty_default_and_clears_the_target(self):
        from shared.ui_parameter_sync import apply_preprocessing_availability
        for stages in ("eddy,topup", "eddy", ""):
            n = node(stages)
            with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
                apply_preprocessing_availability(FakeCheckbox(), node=n)
            self.assertEqual(n.orchestrationPreprocessingStages, "denoise", stages)
            self.assertEqual(n.targetPreprocessing, stages == "", stages)

    def test_g3_apply_changes_nothing_where_fsl_is_offered(self):
        from shared.ui_parameter_sync import apply_preprocessing_availability
        box, n = FakeCheckbox(), node()
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True):
            self.assertTrue(apply_preprocessing_availability(box, node=n))
        self.assertEqual(box.calls, [])  # a tooltip that never showed the FSL reason is left alone (O4)
        self.assertEqual(n.orchestrationPreprocessingStages, "denoise,eddy,topup")

    def test_g3_apply_writes_the_node_only_when_the_value_changes(self):
        from shared.ui_parameter_sync import apply_preprocessing_availability

        class Counting(SimpleNamespace):
            writes = 0

            def __setattr__(self, name, value):
                if name == "orchestrationPreprocessingStages":
                    type(self).writes += 1
                super().__setattr__(name, value)

        n = Counting(targetPreprocessing=True, orchestrationPreprocessingStages="denoise,eddy,topup")
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            apply_preprocessing_availability(FakeCheckbox(), node=n)
            before = Counting.writes
            apply_preprocessing_availability(FakeCheckbox(), node=n)
        self.assertEqual(Counting.writes, before)

    def test_g3_the_stripped_stage_list_runs_no_eddy_or_topup_and_needs_no_fsl(self):
        """G3 check (no code change needed): orchestrator and dependency plan treat 'denoise' as no FSL."""
        from shared import ProductOrchestratorConfig
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs
        from shared.orchestrator import _resolved_preprocessing_stages
        config = ProductOrchestratorConfig(subject_id="s", subject_root=Path("/x"), output_root=Path("/y"),
                                           targets=("preprocessing",), preprocessing_stages="brain_mask,denoise")
        stages = _resolved_preprocessing_stages(config)
        self.assertNotIn("eddy", stages)
        self.assertNotIn("topup", stages)
        specs = build_dependency_specs(EnvironmentDoctorConfig(targets=("preprocessing",), repo_root=REPO_ROOT,
                                                               preprocessing_stages="brain_mask,denoise"))
        self.assertFalse([spec.key for spec in specs if spec.install_group in {"fsl", "mrtrix", "native-fsl", "native-mrtrix"}])

    def test_g3_picker_selection_helpers_follow_the_gate(self):
        from shared import ui_algorithm_picker as picker
        n = node("denoise,eddy,topup")
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            self.assertEqual(picker.selected_algorithms(n, "preprocessing"), ("denoise",))
            self.assertEqual(picker.selected_algorithms(node(""), "preprocessing"), ("brain_mask",))  # default has no FSL stage
            written = picker._apply_family_selection_to_node(n, "preprocessing", ["brain_mask", "denoise", "eddy", "topup"])
            self.assertEqual(written, ("brain_mask", "denoise"))
            self.assertEqual(n.orchestrationPreprocessingStages, "brain_mask,denoise")
            with self.assertRaises(ValueError):
                picker._apply_family_selection_to_node(n, "preprocessing", ["eddy", "topup"])
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True):
            self.assertEqual(picker.selected_algorithms(node(""), "preprocessing"), ("brain_mask",))

    def _run_popup(self, offered, stages="denoise,eddy,topup", lang="en", ui=None, node_obj=None):
        from Testing.Python.test_ui_run_history import fake_qt
        from shared import ui_algorithm_picker as picker
        qt = fake_qt()
        if ui is not None:
            qt.QComboBox = type("FakeComboBox", (), {})
            qt.QDoubleSpinBox = type("FakeDoubleSpinBox", (), {})
        owner = SimpleNamespace(
            ui=ui if ui is not None else SimpleNamespace(),
            _uiWidget=None, _pipelineActive=False, _selectedSetupActive=False,
            _onboardingLanguage=lang, _parameterNode=node_obj if node_obj is not None else node(stages),
            _lastEnvironmentResult=None, _lastEnvironmentSignature=None)
        slicer = mock.MagicMock(name="slicer")
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}), \
                mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=offered):
            picker.choose_algorithms(owner, "preprocessing")
        self._last_qt = qt
        return {w.objectName: w for w in qt.created if w.kind == "QCheckBox"}

    def test_g3_popup_disables_and_unchecks_eddy_topup_where_fsl_is_not_offered(self):
        boxes = self._run_popup(False)
        for name in ("algorithm_eddy", "algorithm_topup"):
            self.assertFalse(boxes[name].checked, name)
            boxes[name].setEnabled.assert_called_with(False)
            tip = boxes[name].setToolTip.call_args[0][0]
            self.assertIn("Eddy and Topup need FSL", tip)
        self.assertTrue(boxes["algorithm_denoise"].checked)
        boxes["algorithm_denoise"].setEnabled.assert_not_called()
        boxes["algorithm_brain_mask"].setEnabled.assert_not_called()
        zh_boxes = self._run_popup(False, lang="zh")
        self.assertIn("需要 FSL", zh_boxes["algorithm_eddy"].setToolTip.call_args[0][0])

    def test_g3_popup_shows_visible_inline_reason_and_disables_topup_gear_when_fsl_missing(self):
        from Testing.Python.test_ui_run_history import object_name

        topup_ui = SimpleNamespace(
            preprocessingPhaseEncodingDirectionComboBox=SimpleNamespace(toolTip="pe tip"),
            preprocessingPhaseEncodingDirectionLabel=SimpleNamespace(text="Phase direction"),
            preprocessingReadoutTimeSpinBox=SimpleNamespace(toolTip="ro tip"),
            preprocessingReadoutTimeLabel=SimpleNamespace(text="Readout time"),
            preprocessingReversePhaseB0Path=SimpleNamespace(toolTip="rpe tip"),
            preprocessingReversePhaseB0Label=SimpleNamespace(text="Reverse B0"),
        )
        topup_node = node("denoise,eddy,topup")
        topup_node.preprocessingPhaseEncodingDirection = "j-"
        topup_node.preprocessingReadoutTime = "0.05"
        topup_node.preprocessingReversePhaseB0 = ""

        with mac_host("13.6.0", "arm64"), \
                mock.patch("shared.selected_setup._bundled_fsl_requirement", return_value=(False, None)):
            self._run_popup(False, lang="en", ui=topup_ui, node_obj=topup_node)
            en_labels = [
                w for w in self._last_qt.created
                if w.kind == "QLabel" and object_name(w) == "algorithmPickerAvailabilityNote"
            ]
            self.assertEqual(len(en_labels), 1)
            en_text = en_labels[0].ctor_args[0]
            self.assertIn("Eddy and Topup need FSL", en_text)
            self.assertIn("macOS 13.6", en_text)
            self.assertIn("14.5", en_text)
            en_labels[0].hide.assert_not_called()

            topup_gear = [
                w for w in self._last_qt.created
                if w.kind == "QToolButton" and object_name(w) == "algorithmSettings_topup"
            ]
            self.assertEqual(len(topup_gear), 1)
            topup_gear[0].setEnabled.assert_called_with(False)
            self.assertIn("macOS 13.6", topup_gear[0].setToolTip.call_args[0][0])

            dialogs = [w for w in self._last_qt.created if w.kind == "QDialog"]
            self.assertEqual(len(dialogs), 1)
            dialogs[0].setAttribute.assert_any_call(self._last_qt.Qt.WA_AlwaysShowToolTips, True)

            self._run_popup(False, lang="zh", ui=topup_ui, node_obj=topup_node)
            zh_labels = [
                w for w in self._last_qt.created
                if w.kind == "QLabel" and object_name(w) == "algorithmPickerAvailabilityNote"
            ]
            self.assertEqual(len(zh_labels), 1)
            self.assertIn("本机 macOS 13.6", zh_labels[0].ctor_args[0])
            self.assertIn("14.5", zh_labels[0].ctor_args[0])

        self._run_popup(True, lang="en", ui=topup_ui, node_obj=topup_node)
        ok_labels = [
            w for w in self._last_qt.created
            if w.kind == "QLabel" and object_name(w) == "algorithmPickerAvailabilityNote"
        ]
        self.assertEqual(ok_labels, [])

    def test_g3_popup_forces_eddy_topup_unchecked_even_if_the_node_still_lists_them(self):
        with mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False):
            boxes = self._run_popup(False, stages="denoise,eddy,topup")
        self.assertFalse(boxes["algorithm_eddy"].checked)
        self.assertFalse(boxes["algorithm_topup"].checked)

    def test_g3_popup_keeps_all_boxes_enabled_where_fsl_is_offered(self):
        boxes = self._run_popup(True)
        for name in ("algorithm_eddy", "algorithm_denoise"):
            boxes[name].setEnabled.assert_not_called()
        self.assertTrue(boxes["algorithm_eddy"].checked)
        # Topup stays disabled until a valid reverse-PE b=0 file is provided.
        boxes["algorithm_topup"].setEnabled.assert_called_with(False)
        self.assertFalse(boxes["algorithm_topup"].checked)
        with tempfile.TemporaryDirectory() as tmp:
            rev = Path(tmp) / "rev_b0.nii.gz"
            rev.write_bytes(b"\x00")
            n = node("denoise,eddy,topup")
            n.preprocessingReversePhaseB0 = str(rev)
            boxes_with_rpe = self._run_popup(True, node_obj=n)
            for name in ("algorithm_eddy", "algorithm_topup", "algorithm_denoise"):
                boxes_with_rpe[name].setEnabled.assert_not_called()
            self.assertTrue(boxes_with_rpe["algorithm_topup"].checked)

    def test_g3_accept_with_brain_mask_and_denoise_turns_the_main_box_on_without_fsl(self):
        """Accept keeps the main checkbox enabled, so the dark-blue check shows (reads isEnabled of the main box)."""
        from Testing.Python.test_ui_run_history import fake_qt
        from shared import ui_algorithm_picker as picker
        qt = fake_qt()
        main = FakeCheckbox(checked=False)
        main.isEnabled = lambda: main.enabled
        n = node("denoise", target=False)
        owner = SimpleNamespace(
            ui=SimpleNamespace(targetPreprocessingCheckBox=main), _uiWidget=None, _pipelineActive=False,
            _selectedSetupActive=False, _onboardingLanguage="en", _parameterNode=n,
            _lastEnvironmentResult=None, _lastEnvironmentSignature=None, _sync_checkbox=lambda box, value: box.setChecked(value))
        slicer = mock.MagicMock(name="slicer")
        dialog_factory = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dialog = dialog_factory(*args, **kwargs)

            def fake_exec():
                for widget in qt.created:
                    if widget.kind == "QCheckBox":
                        widget.checked = widget.objectName in {"algorithm_brain_mask", "algorithm_denoise"}
                return qt.QDialog.Accepted

            dialog.exec_.side_effect = fake_exec
            return dialog

        qt.QDialog.side_effect = make_dialog
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}), \
                mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=False), \
                mock.patch("shared.ui_compact.refresh_compact_algorithms"), \
                mock.patch("shared.ui_on_demand_resources.maybe_prompt_on_demand_install"):
            self.assertTrue(picker.choose_algorithms(owner, "preprocessing"))
        self.assertTrue(n.targetPreprocessing)
        self.assertEqual(n.orchestrationPreprocessingStages, "brain_mask,denoise")
        self.assertTrue(main.checked)


import contextlib


@contextlib.contextmanager
def mac_host(version, machine="arm64"):
    """C4: a Mac running ``version`` on ``machine``, whatever OS and CPU the test runs on.

    ``current_macos_version`` checks ``platform.system()`` first, and the bundle lookup and the Mach-O slice choice read the
    host architecture, so all of them are patched together with ``platform.mac_ver``.
    """
    with mock.patch("platform.system", return_value="Darwin"), \
            mock.patch("platform.machine", return_value=machine), \
            mock.patch("shared.platform_info.host_machine", return_value=machine), \
            mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")):
        yield


def make_surface_bundle(root, minos=(15, 0)):
    bundle = Path(root) / "surface-runtime-darwin-arm64"
    write(bundle / "bin" / "mris_register", thin_macho([_build_version(pack_version(*minos))]))
    (bundle / "runtime.json").write_text('{"activation_key_included": false}')
    return bundle


class G4SurfaceRuntimeTest(BundleEnv, unittest.TestCase):
    def test_g4_bundled_surface_runtime_is_rejected_on_14_2_and_accepted_on_15_1(self):
        """C4: the Mac (system, version, arch) is mocked, so this passes on Linux and Windows too."""
        from shared.surface_runtime import bundled_surface_runtime
        bundle = make_surface_bundle(self.bundled)
        with mac_host("14.2"):
            self.assertIsNone(bundled_surface_runtime("Darwin", "arm64"))
        with mac_host("15.1"):
            self.assertEqual(bundled_surface_runtime("Darwin", "arm64"), bundle.resolve())

    def test_g4_unknown_minimum_or_other_platform_is_accepted(self):
        """C4: the Mac is mocked on every OS."""
        from shared.surface_runtime import bundled_surface_runtime
        bundle = self.bundled / "surface-runtime-darwin-arm64"
        write(bundle / "bin" / "tool", b"dummy")
        (bundle / "runtime.json").write_text('{"activation_key_included": false}')
        with mac_host("14.2"):
            self.assertEqual(bundled_surface_runtime("Darwin", "arm64"), bundle.resolve())
        with mac_host(""):  # not macOS: no version to compare
            self.assertEqual(bundled_surface_runtime("Darwin", "arm64"), bundle.resolve())

    def test_g4_block_message_is_clear(self):
        """C4: the Mac is mocked on every OS."""
        from shared.surface_runtime import surface_runtime_macos_block
        make_surface_bundle(self.bundled)
        with mac_host("14.2"):
            self.assertEqual(surface_runtime_macos_block("Darwin", "arm64"),
                             "The bundled surface runtime needs macOS 15.0 or later (this Mac: 14.2)")
        with mac_host("15.1"):
            self.assertIsNone(surface_runtime_macos_block("Darwin", "arm64"))

    def test_g4_inspect_reports_the_macos_reason_instead_of_a_dyld_failure(self):
        """C4: the Mac and its arm64 host arch are mocked, so the bundle lookup is the same on every OS."""
        from shared.surface_runtime import inspect_surface_runtime
        bundle = make_surface_bundle(self.bundled)
        expected = "The bundled surface runtime needs macOS 15.0 or later (this Mac: 14.2)"
        with mac_host("14.2"), \
                mock.patch("shared.environment.resolve_freesurfer_home", return_value=None), \
                mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            for runtime in (None, bundle):  # a saved path to the bundle (module_widget persists it) is checked too
                record = inspect_surface_runtime(runtime)
                self.assertFalse(record["components_available"])
                self.assertFalse(record["available"])
                self.assertTrue(record["macos_blocked"])
                self.assertEqual(record["notes"], expected)
                self.assertEqual(record["error"], expected)

    def test_g4_environment_check_shows_the_message_and_a_real_fix_and_the_plan_keeps_it(self):
        """b2-os, decisions 2-4: the fix is the action text, not the message again; no manual item forces a dialog."""
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor
        from shared.selected_setup import build_setup_plan
        message = "The bundled surface runtime needs macOS 15.0 or later (this Mac: 14.2)"
        blocked = dict(backend="native", components_available=False, license_available=False, available=False,
                       build_required=False, macos_blocked=True, notes=message, error=message)
        with mock.patch("shared.environment.build_dependency_specs", return_value=()), \
                mock.patch("shared.surface_runtime.inspect_surface_runtime", return_value=blocked):
            result = run_environment_doctor(EnvironmentDoctorConfig(targets=("surface_postprocess",), repo_root=REPO_ROOT))
        status = {item.key: item for item in result.statuses}["surface_postprocess_runtime"]
        self.assertEqual(status.status, "missing")
        self.assertEqual(status.details["notes"], message)
        self.assertIn("upgrade macOS to 15.0 or later", status.install_hint)
        self.assertNotEqual(status.install_hint, message)
        plan = build_setup_plan(result, system="Darwin", machine="arm64")
        self.assertNotIn("surface_postprocess_runtime", [item["key"] for item in plan["manual_items"]])
        reasons = {item["key"]: item["reason"] for item in plan["os_blocked_items"]}
        self.assertEqual(reasons.get("surface_postprocess_runtime"), message)


class PythonSyntaxTest(unittest.TestCase):
    def test_changed_slicer_modules_stay_python_3_9_clean(self):
        for name in ("shared/macho_min_os.py", "shared/selected_setup.py", "shared/ui_parameter_sync.py",
                     "shared/ui_algorithm_picker.py", "shared/algorithm_selection.py", "shared/surface_runtime.py",
                     "shared/environment.py"):
            source = (REPO_ROOT / name).read_text(encoding="utf-8")
            ast.parse(source, filename=name, feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
