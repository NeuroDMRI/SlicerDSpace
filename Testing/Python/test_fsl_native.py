"""Contract tests for native FSL correction with stubbed external tools."""

import json
from pathlib import Path
import tempfile
import unittest

from preprocessing.fsl_native import run_fsl_correction


class FSLNativeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="FSL subject ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scratch = self.root / "scratch dir"
        self.source = self.root / "input dwi.nii.gz"
        self.bvec = self.root / "input.bvec"
        self.bval = self.root / "input.bval"
        self.output = self.root / "final.nii.gz"
        self.output_bvec = self.root / "final.bvec"
        self.output_bval = self.root / "final.bval"
        self.pair = self.root / "se_epi_pair.mif"
        self.topup_config = self.root / "FSL share" / "b02b0.cnf"
        for path in (self.source, self.bvec, self.bval, self.pair, self.topup_config):
            path.parent.mkdir(exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        self.commands = {name: str(self.root / "tool dir" / name) for name in (
            "mrinfo", "mrconvert", "dwi2mask", "maskfilter", "mrcat", "mrtransform", "mrcalc",
            "eddy", "topup", "applytopup",
        )}
        self.calls = []
        self.spatial_shape = [5, 6, 7]
        self.first_b0 = 2
        self.emit_rotated_bvecs = True
        self.regrid_pair = False

    def run_correction(self, *, pair=False):
        run_fsl_correction(
            input_dwi=self.source,
            input_bvec=self.bvec,
            input_bval=self.bval,
            output_dwi=self.output,
            output_bvec=self.output_bvec,
            output_bval=self.output_bval,
            scratch_dir=self.scratch,
            phase_encoding="LR",
            readout_time=None,
            se_epi_pair=self.pair if pair else None,
            topup_config=self.topup_config if pair else None,
            eddy_options=("--cnr_maps", "--residuals", "--nthr=2"),
            commands=self.commands,
            run_stage=self.simulate,
        )

    def simulate(self, command):
        program, *args = command
        self.assertTrue(Path(program).is_absolute())
        tool = Path(program).name
        self.calls.append((tool, args))

        def write(path, content="fixture"):
            Path(path).write_text(content, encoding="utf-8")

        if tool == "mrinfo" and "-json_all" in args:
            pair = args[0] == str(self.pair)
            size = self.spatial_shape.copy()
            if pair and self.regrid_pair:
                size[0] += 2
            size.append(2 if pair else 3)
            grad = [[1, 0, 0, 1000] for _ in range(3)]
            grad[self.first_b0] = [0, 0, 0, 0]
            write(args[args.index("-json_all") + 1], json.dumps({
                "size": size, "strides": [1, -2, 3, 4], "spacing": [2, 2, 2, 1],
                "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                "keyval": {"dw_scheme": grad},
            }))
        elif tool == "mrconvert":
            write(args[1])
            if "-export_grad_fsl" in args:
                pos = args.index("-export_grad_fsl")
                write(args[pos + 1], "1 0 1\n0 0 0\n0 0 0\n")
                write(args[pos + 2], "1000 0 1000\n")
            if "-export_pe_eddy" in args:
                pos = args.index("-export_pe_eddy")
                write(args[pos + 1], "-1 0 0 0.1\n")
                write(args[pos + 2], "1 1 1\n")
            if "-export_pe_topup" in args:
                write(args[args.index("-export_pe_topup") + 1], "-1 0 0 0.1\n1 0 0 0.1\n")
        elif tool == "topup":
            field = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--out="))
            write(field + "_fieldcoef.nii.gz")
            write(field + "_movpar.txt")
            write(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--fout=")))
        elif tool == "applytopup":
            prefix = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--out="))
            write(prefix + ".nii.gz")
        elif tool == "eddy":
            prefix = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--out="))
            write(prefix + ".nii.gz")
            if self.emit_rotated_bvecs:
                write(prefix + ".eddy_rotated_bvecs", "1 0 1\n0 0 0\n0 0 0\n")
        elif tool == "mrcat":
            write(args[2])
        elif tool == "mrtransform":
            write(args[1])
        elif tool == "mrcalc":
            write(args[-1])
        elif tool in {"dwi2mask", "maskfilter"}:
            write(args[-1])
        else:
            self.fail(f"Unexpected command: {command}")
        return "fixture subprocess output"

    def command_for_output(self, name):
        return next(args for tool, args in self.calls if tool == "mrconvert" and Path(args[1]).name == name)

    def test_eddy_uses_fsl_strides_then_restores_order_and_rotated_gradients(self):
        self.run_correction()
        prep = self.command_for_output("eddy_in.nii")
        self.assertEqual(prep[prep.index("-strides") + 1], "-1,+2,+3,+4")
        self.assertEqual(prep[prep.index("-coord") + 1:prep.index("-coord") + 3], ["3", "2,0,1"])
        self.assertEqual((self.scratch / "dwi_manual_pe_scheme.txt").read_text().splitlines(),
                         ["1 0 0 0.1"] * 3)
        restore = self.command_for_output("result.mif")
        self.assertEqual(restore[restore.index("-coord") + 1:restore.index("-coord") + 3], ["3", "1,2,0"])
        self.assertEqual(restore[restore.index("-strides") + 1], "1,-2,3,4")
        self.assertIn(str(self.scratch / "dwi_post_eddy.eddy_rotated_bvecs"), restore)
        self.assertNotIn("topup", [tool for tool, _ in self.calls])
        self.assertTrue(self.output_bvec.is_file())

    def test_pair_pads_odd_axes_and_uses_applytopup_only_for_mask(self):
        self.regrid_pair = True
        self.run_correction(pair=True)
        self.assertEqual((self.scratch / "se_epi_manual_pe_scheme.txt").read_text().splitlines(),
                         ["1 0 0 0.1", "-1 0 0 0.1"])
        topup_prep = self.command_for_output("topup_in.nii")
        self.assertTrue(topup_prep[0].endswith("se_epi_pad2.mif"))
        self.assertEqual(topup_prep[topup_prep.index("-strides") + 1], "-1,+2,+3,+4")
        self.assertIn("mrcalc", [tool for tool, _ in self.calls])
        mask_input = next(args[0] for tool, args in self.calls if tool == "dwi2mask")
        self.assertTrue(mask_input.endswith("applytopup_corrected.mif"))
        eddy_input = self.command_for_output("eddy_in.nii")
        self.assertTrue(eddy_input[0].endswith("dwi_pad2.mif"))
        eddy = next(args for tool, args in self.calls if tool == "eddy")
        self.assertIn(f"--topup={(self.scratch / 'field').as_posix()}", eddy)
        restore = self.command_for_output("result.mif")
        coords = [restore[index + 1:index + 3] for index, arg in enumerate(restore) if arg == "-coord"]
        self.assertEqual(coords, [["3", "1,2,0"], ["0", "0:4"], ["2", "0:6"]])

    def test_missing_rotated_gradients_stop_before_final_export(self):
        self.emit_rotated_bvecs = False
        with self.assertRaisesRegex(RuntimeError, "eddy_rotated_bvecs"):
            self.run_correction()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
