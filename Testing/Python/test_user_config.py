"""Tests for the shared GUI/CLI user configuration."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from shared.user_config import (
    apply_user_config_to_args,
    coupled_explicit_cli_destinations,
    default_user_config_path,
    empty_user_config,
    explicit_cli_destinations,
    load_user_config,
    normalize_user_config,
    update_user_config,
    user_config_from_cli_args,
    user_config_cli_values,
    write_user_config,
)


class UserConfigTest(unittest.TestCase):
    def test_default_language_is_english(self) -> None:
        self.assertEqual(empty_user_config()["ui"]["language"], "en")
        with tempfile.TemporaryDirectory() as tmp:
            missing_path = Path(tmp) / "missing-user-config.json"
            self.assertEqual(load_user_config(missing_path)["ui"]["language"], "en")

    def test_json_language_accepts_chinese_and_english(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            write_user_config({"ui": {"language": "zh"}}, path)
            self.assertEqual(load_user_config(path)["ui"]["language"], "zh")
            update_user_config({"ui": {"language": "en"}}, path)
            self.assertEqual(load_user_config(path)["ui"]["language"], "en")

    def test_default_paths_are_platform_specific(self) -> None:
        home = Path("/home/tester")
        self.assertEqual(
            default_user_config_path(system="Linux", home=home, environ={}),
            home / ".config" / "SlicerDSpace" / "user_config.json",
        )
        self.assertEqual(
            default_user_config_path(system="Darwin", home=home, environ={}),
            home / "Library" / "Application Support" / "SlicerDSpace" / "user_config.json",
        )
        self.assertEqual(
            default_user_config_path(system="Windows", home=home, environ={"APPDATA": "C:/Users/test/AppData/Roaming"}),
            Path("C:/Users/test/AppData/Roaming") / "SlicerDSpace" / "user_config.json",
        )

    def test_roundtrip_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            write_user_config(
                {
                    "ui": {"language": "zh"},
                    "recent": {"subject_root": "/data/sub-01"},
                    "external_dependencies": {"mrtrix_bin_dir": "/opt/mrtrix3/bin"},
                },
                path,
            )
            update_user_config({"run_defaults": {"tractography_backend": "dti_det"}}, path)
            config = load_user_config(path)

        self.assertEqual(config["ui"]["language"], "zh")
        self.assertEqual(config["recent"]["subject_root"], "/data/sub-01")
        self.assertEqual(config["run_defaults"]["tractography_backend"], "dti_det")
        self.assertEqual(config["external_dependencies"]["mrtrix_bin_dir"], "/opt/mrtrix3/bin")

    def test_nested_value_can_be_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            write_user_config(
                {"external_dependencies": {"mrtrix_bin_dir": "/opt/mrtrix3/bin"}},
                path,
            )
            update_user_config(
                {"external_dependencies": {"mrtrix_bin_dir": None}},
                path,
            )
            config = load_user_config(path)

        self.assertNotIn("mrtrix_bin_dir", config["external_dependencies"])

    def test_cli_payload_preserves_gui_only_values_and_uses_resolved_selection(self) -> None:
        args = argparse.Namespace(
            subject_id="sub-01",
            subject_root=Path("/data/sub-01"),
            output_root=None,
            run_name="RunA",
            targets=["all"],
            tractography_backend="dti_det",
            bundle_backend="wmql",
            microstructure_stages=["all"],
            connectome_backend="python",
            select_streamlines=5000,
        )
        config = user_config_from_cli_args(
            args,
            language="zh",
            base_config={
                "ui": {"first_run_wizard_shown": True},
                "run_defaults": {"auto_install_python_dependencies": False},
                "external_dependencies": {"model_assets_dir": "/models"},
            },
            resolved_targets=("validation", "microstructure_freewater", "saf", "wma"),
            resolved_stages=("dti",),
        )

        self.assertTrue(config["ui"]["first_run_wizard_shown"])
        self.assertEqual(config["ui"]["language"], "zh")
        self.assertFalse(config["run_defaults"]["auto_install_python_dependencies"])
        self.assertEqual(
            config["run_defaults"]["targets"],
            ["validation", "microstructure_freewater", "saf", "wma"],
        )
        self.assertEqual(config["run_defaults"]["microstructure_stages"], ["dti", "freewater"])
        self.assertEqual(config["run_defaults"]["tractography_backend"], "saf")
        self.assertEqual(config["run_defaults"]["bundle_backend"], "wma")
        self.assertEqual(config["external_dependencies"]["model_assets_dir"], "/models")

    def test_w2a_saved_removed_preprocessing_setting_loads_without_it_with_one_note(self) -> None:
        """W2a (D1): a saved preprocessing route or denoise choice loads without it, with one note, and is not saved again."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            saved_values = ({"preprocessing_backend": "dipy"},
                            {"preprocessing_backend": "mrtrix_fsl", "preprocessing_dipy_denoise_method": "localpca"})
            for run_defaults in saved_values:
                with self.subTest(saved=run_defaults):
                    path.write_text(json.dumps({"run_defaults": {"run_name": "R1", **run_defaults}}), encoding="utf-8")
                    with self.assertLogs("shared.user_config", level="WARNING") as logged:
                        loaded = load_user_config(path)
                    for key in run_defaults:
                        self.assertNotIn(key, loaded["run_defaults"])
                    self.assertEqual(loaded["run_defaults"]["run_name"], "R1")
                    self.assertEqual(len(loaded["notes"]), 1)
                    self.assertEqual(len(logged.records), 1)
                    for key in run_defaults:
                        self.assertIn(key, loaded["notes"][0])
                    self.assertIn("one route", loaded["notes"][0])
                    self.assertNotIn("preprocessing_backend", user_config_cli_values(loaded))
                    write_user_config(loaded, path)
                    again = load_user_config(path)
                    self.assertNotIn("notes", again)
                    self.assertNotIn("preprocessing_backend", again["run_defaults"])
        self.assertNotIn("preprocessing_backend", empty_user_config()["run_defaults"])

    def test_non_object_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_user_config(path)

    def test_cli_values_include_recent_paths_and_defaults(self) -> None:
        values = user_config_cli_values(
            {
                "recent": {"subject_id": "sub-01", "subject_root": "/data/sub-01", "output_root": "/out/sub-01"},
                "run_defaults": {"targets": ["validation", "microstructure"], "tractography_backend": "dti_det"},
            }
        )

        self.assertEqual(values["subject_root"], Path("/data/sub-01"))
        self.assertEqual(values["targets"], ["validation,microstructure"])
        self.assertEqual(values["tractography_backend"], "dti_det")

    def test_explicit_cli_values_win(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--subject-root", type=Path)
        parser.add_argument("--tractography-backend", default="csd_act")
        argv = ["--tractography-backend", "dti_det"]
        args = parser.parse_args(argv)
        explicit = explicit_cli_destinations(parser, argv)

        apply_user_config_to_args(
            args,
            {
                "recent": {"subject_root": "/data/saved"},
                "run_defaults": {"tractography_backend": "csd_act"},
            },
            explicit_dests=explicit,
        )

        self.assertEqual(args.tractography_backend, "dti_det")
        self.assertEqual(args.subject_root, Path("/data/saved"))

    def test_explicit_subject_root_invalidates_saved_subject_context(self) -> None:
        explicit = coupled_explicit_cli_destinations({"subject_root"})
        args = argparse.Namespace(
            subject_root=Path("/data/new-subject"),
            subject_id=None,
            output_root=None,
        )

        apply_user_config_to_args(
            args,
            {
                "recent": {
                    "subject_root": "/data/old-subject",
                    "subject_id": "old-subject",
                    "output_root": "/out/old-subject",
                },
            },
            explicit_dests=explicit,
        )

        self.assertEqual(args.subject_root, Path("/data/new-subject"))
        self.assertIsNone(args.subject_id)
        self.assertIsNone(args.output_root)


class RememberedInputsConfigTest(unittest.TestCase):
    """Package C1: the input tiles and the distortion-correction values live under ``recent``."""

    TILE_KEYS = ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask")

    def test_c1_empty_config_has_empty_defaults_for_the_remembered_inputs(self) -> None:
        recent = empty_user_config()["recent"]
        self.assertEqual(recent["input_sources"], {})
        self.assertEqual(recent["reverse_pe_b0"], "")
        self.assertEqual(recent["pe_direction"], "")
        self.assertEqual(recent["readout_time"], 0.0)

    def test_c1_an_old_config_file_without_the_new_keys_loads_with_empty_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            path.write_text(json.dumps({"recent": {"subject_id": "s1", "subject_root": "/d", "output_root": "/o"}}),
                            encoding="utf-8")
            recent = load_user_config(path)["recent"]
        self.assertEqual(recent["subject_id"], "s1")
        self.assertEqual(recent["input_sources"], {})
        self.assertEqual((recent["reverse_pe_b0"], recent["pe_direction"], recent["readout_time"]), ("", "", 0.0))

    def test_c1_normalize_keeps_the_new_keys_through_a_write_and_load(self) -> None:
        sources = {"kind": "dicom", "dicom_source": "/data/s1/dicom", "bval": "", "mask": "/data/m.nii.gz"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "user_config.json"
            write_user_config({"recent": {"input_sources": sources, "reverse_pe_b0": "/data/rev.nii.gz",
                                          "pe_direction": "AP", "readout_time": 0.05}}, path)
            recent = load_user_config(path)["recent"]
        self.assertEqual(recent["input_sources"], sources)
        self.assertEqual(recent["reverse_pe_b0"], "/data/rev.nii.gz")
        self.assertEqual(recent["pe_direction"], "AP")
        self.assertEqual(recent["readout_time"], 0.05)

    def test_c1_malformed_remembered_values_normalize_to_empty_defaults(self) -> None:
        for bad in ("text", ["a"], 7, None):
            with self.subTest(input_sources=bad):
                self.assertEqual(normalize_user_config({"recent": {"input_sources": bad}})["recent"]["input_sources"], {})
        recent = normalize_user_config({"recent": {"readout_time": "fast", "reverse_pe_b0": None, "pe_direction": None,
                                                   "input_sources": {"bval": None, "kind": "nifti", 3: "x"}}})["recent"]
        self.assertEqual(recent["readout_time"], 0.0)
        self.assertEqual((recent["reverse_pe_b0"], recent["pe_direction"]), ("", ""))
        self.assertEqual(recent["input_sources"], {"kind": "nifti", "3": "x"})

    def test_c1_the_cli_payload_keeps_remembered_inputs_it_did_not_write(self) -> None:
        args = argparse.Namespace(subject_id="s", subject_root=Path("/d/s"), output_root=None)
        config = user_config_from_cli_args(args, base_config={"recent": {"input_sources": {"kind": "nrrd"},
                                                                         "readout_time": 0.03}})
        self.assertEqual(config["recent"]["input_sources"], {"kind": "nrrd"})
        self.assertEqual(config["recent"]["readout_time"], 0.03)

    def test_c1_the_existing_public_names_are_all_still_importable(self) -> None:
        import shared.user_config as module
        for name in ("empty_user_config", "normalize_user_config", "load_user_config", "write_user_config",
                     "update_user_config", "default_user_config_path", "user_config_cli_values",
                     "user_config_from_cli_args", "prune_stale_bundled_dependencies", "extension_root_for_module",
                     "selected_microstructure_stages", "canonical_microstructure_stages", "RUN_DEFAULTS"):
            self.assertTrue(hasattr(module, name), name)


if __name__ == "__main__":
    unittest.main()
