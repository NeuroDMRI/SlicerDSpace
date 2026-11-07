#!/usr/bin/env python3
"""SAF command boundaries preserve path values and explicit legacy quoting."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SAF_BACKEND_ROOT = REPO_ROOT / "tractography" / "saf_backend"
if str(SAF_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(SAF_BACKEND_ROOT))

# A path containing both a space and a shell command-separator metacharacter.
UNSAFE_NAME = "unsafe dir; touch pwned"


class ShellQuoteHelperTest(unittest.TestCase):
    def test_shell_quote_wraps_metacharacters_safely(self) -> None:
        from Support.shell_quote import shell_quote

        quoted = shell_quote(f"/tmp/{UNSAFE_NAME}/file.tck")
        # A POSIX-safe quoted form is a single shell word: shlex.split()
        # must parse it back to exactly the original string.
        import shlex

        self.assertEqual(shlex.split(quoted), [f"/tmp/{UNSAFE_NAME}/file.tck"])
        self.assertNotEqual(quoted, f"/tmp/{UNSAFE_NAME}/file.tck")


class ApplyTckSampleTest(unittest.TestCase):
    def test_paths_are_single_argv_entries(self) -> None:
        from Data_modules.apply_tck_sample import apply_tck_sample

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / UNSAFE_NAME / "out"
            out_dir.mkdir(parents=True)
            tracks = Path(tmp) / UNSAFE_NAME / "tracks.tck"
            scalar = Path(tmp) / UNSAFE_NAME / "fa.nii.gz"
            PS = types.SimpleNamespace(mrtrix_prefix="", force=True)

            with mock.patch("Data_modules.apply_tck_sample.sys_exec") as mock_exec:
                apply_tck_sample(str(tracks), str(scalar), str(out_dir), PS)

            mock_exec.assert_called_once()
            command, ps_arg = mock_exec.call_args.args
            self.assertIs(ps_arg, PS)
            self.assertEqual(command, ["tcksample", "-stat_tck", "mean", "-precise", "-force",
                                       str(tracks), str(scalar), str(out_dir / "fa.nii.tsf")])


class TxClassTest(unittest.TestCase):
    def test_mrconvert_paths_are_single_argv_entries(self) -> None:
        from Class_files.tx_class import tx_class

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / UNSAFE_NAME / "work"
            work_dir.mkdir(parents=True)
            fs_dir = Path(tmp) / UNSAFE_NAME / "fs"
            (fs_dir / "mri").mkdir(parents=True)
            PS = types.SimpleNamespace(work_dir=str(work_dir), fs_dir=str(fs_dir), mrtrix_prefix="")

            with mock.patch("Class_files.tx_class.sys_exec") as mock_exec:
                tx_class(PS)

            mock_exec.assert_called_once()
            command, ps_arg = mock_exec.call_args.args
            self.assertIs(ps_arg, PS)
            self.assertEqual(command, ["mrconvert", str(fs_dir / "mri" / "brain.mgz"),
                                       str(work_dir / "brain_T1.nii.gz")])


class FiltersLengthTest(unittest.TestCase):
    def test_final_init_path_is_one_argv_entry(self) -> None:
        from Filters import filters_length as filters_length_module

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / UNSAFE_NAME / "work"
            work_dir.mkdir(parents=True)
            final_init = Path(tmp) / UNSAFE_NAME / "prefiltered.tck"
            PS = types.SimpleNamespace(
                maxlen=50.0,
                work_dir=str(work_dir),
                mrtrix_prefix="",
                final_init=str(final_init),
                logt=[],
            )

            with mock.patch.object(filters_length_module, "sys_exec") as mock_exec, mock.patch.object(
                filters_length_module, "read_tsf", return_value=__import__("numpy").array([10.0, 20.0])
            ):
                filters_length_module.filters_length(PS)

            mock_exec.assert_called_once()
            command, ps_arg = mock_exec.call_args.args
            self.assertIs(ps_arg, PS)
            self.assertEqual(command, ["tckstats", "-dump", str(work_dir / "lengths.txt"),
                                       str(final_init), "-force"])


def _stub_nibabel() -> types.ModuleType:
    """Minimal nibabel stand-in so Support.mrtrix_tracks can be imported.

    tractography_mrtrix imports Support.pts_to_tck -> Support.mrtrix_tracks,
    which does `import nibabel as nib` at module scope. The real nibabel
    package is not required here because read_prefilter_tracks (the only
    caller that would actually touch nibabel's streamlines API) is mocked
    out below.
    """

    streamlines_module = types.ModuleType("nibabel.streamlines")
    streamlines_module.load = mock.Mock()
    streamlines_module.Tractogram = mock.Mock()
    streamlines_module.save = mock.Mock()
    nibabel_module = types.ModuleType("nibabel")
    nibabel_module.streamlines = streamlines_module
    return nibabel_module


class TractographyMrtrixTest(unittest.TestCase):
    def test_fod_seed_and_custom_option_paths_are_single_argv_entries(self) -> None:
        # Support.mrtrix_tracks does `import nibabel as nib` at module scope;
        # stub it for the import so this test can run without the real
        # (heavy, optional) nibabel dependency installed. Once imported, the
        # module keeps its own reference to the stub regardless of later
        # sys.modules changes, and read_prefilter_tracks (the only caller
        # that would touch nibabel's streamlines API) is mocked out below.
        nibabel_module = _stub_nibabel()
        with mock.patch.dict(
            sys.modules,
            {"nibabel": nibabel_module, "nibabel.streamlines": nibabel_module.streamlines},
        ):
            from Pipe_modules import tractography_mrtrix as tractography_mrtrix_module

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / UNSAFE_NAME / "work"
            work_dir.mkdir(parents=True)
            fod = Path(tmp) / UNSAFE_NAME / "wmfod.mif"
            sl_init = Path(tmp) / UNSAFE_NAME / "init.tck"
            seed_in = Path(tmp) / UNSAFE_NAME / "seeds_in.txt"
            # tractography_mrtrix() overwrites SD.DWI.seed_coords_file_out
            # with os.path.join(PS.work_dir, 'DWI_seeds_out.txt') itself, so
            # compute the expected value the same way rather than passing in
            # an arbitrary path for it below.
            seed_out = work_dir / "DWI_seeds_out.txt"

            PS = types.SimpleNamespace(
                work_dir=str(work_dir),
                noseed=None,
                force=True,
                surfseed=True,
                mrtrix_prefix="",
                tckgen='-select 100 -mask "custom dir/brain mask.mif"',
                fod=str(fod),
                dgn=False,
                logt=[],
            )
            SD = types.SimpleNamespace(
                DWI=types.SimpleNamespace(
                    seed_coords_file_out=str(seed_out),
                    seed_coords_file_in=str(seed_in),
                    sl_init=str(sl_init),
                )
            )

            with mock.patch.object(tractography_mrtrix_module, "sys_exec") as mock_exec, mock.patch.object(
                tractography_mrtrix_module, "read_prefilter_tracks", return_value=(PS, SD)
            ):
                tractography_mrtrix_module.tractography_mrtrix(PS, SD)

            mock_exec.assert_called_once()
            command, ps_arg = mock_exec.call_args.args
            self.assertIs(ps_arg, PS)
            self.assertEqual(command, ["tckgen", "-seed_coordinates_global", str(seed_in),
                                       "-output_seeds", str(seed_out), "-select", "100", "-mask",
                                       "custom dir/brain mask.mif", str(fod), str(sl_init), "-force"])


class InitClassSortFSTest(unittest.TestCase):
    def test_recon_all_paths_are_quoted(self) -> None:
        from Class_files.init_class import init_class

        with tempfile.TemporaryDirectory() as tmp:
            fod_dir = Path(tmp) / UNSAFE_NAME / "fod_dir"
            fod_dir.mkdir(parents=True)
            fod = fod_dir / "wmfod.nii.gz"
            fod.write_text("stub", encoding="utf-8")
            # _sortFS() only reaches the recon-all branch when fs_dir exists
            # but is not a directory, and no default "FS_reconall" directory
            # is found next to the fod file.
            fs_dir = Path(tmp) / UNSAFE_NAME / "fs_dir_as_a_file"
            fs_dir.parent.mkdir(parents=True, exist_ok=True)
            fs_dir.write_text("not a directory", encoding="utf-8")

            # Bypass __init__ (which parses a MATLAB-style positional arg
            # list) and set only the state _sortFS() needs.
            instance = object.__new__(init_class)
            instance.fod = str(fod)
            instance.fs_dir = str(fs_dir)
            instance.logt = []

            with mock.patch("Class_files.init_class.sys_exec") as mock_exec:
                instance._sortFS()

            mock_exec.assert_called_once()
            command, ps_arg = mock_exec.call_args.args
            self.assertIs(ps_arg, instance)
            import shlex

            self.assertIn(shlex.quote(str(fs_dir)), command)


if __name__ == "__main__":
    unittest.main()
