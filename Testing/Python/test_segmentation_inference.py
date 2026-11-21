from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anatomy.segmentation.inference import combine_views, normalize_class_scores
from anatomy.segmentation.config import DDSegConfig
from anatomy.segmentation.qc import DDSEG_PROBABILITY_SEMANTICS, write_ddseg_qc_record


class SegmentationInferenceTest(unittest.TestCase):
    def test_final_logit_normalization_preserves_labels(self) -> None:
        axial = np.linspace(-4.0, 8.0, num=32, dtype=np.float32).reshape(2, 2, 2, 4)
        sagittal = axial.copy()
        coronal = axial.copy()
        combined = combine_views(axial, sagittal, coronal)
        probabilities = normalize_class_scores(combined["prob_maps"])

        np.testing.assert_array_equal(np.argmax(probabilities, axis=3), combined["labels"])
        np.testing.assert_allclose(probabilities.sum(axis=3), 1.0, atol=1e-6)
        self.assertGreaterEqual(float(probabilities.min()), 0.0)
        self.assertLessEqual(float(probabilities.max()), 1.0)

    def test_normalization_rejects_non_volume_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "4D NHWC"):
            normalize_class_scores(np.zeros((2, 4), dtype=np.float32))

    def test_completed_ddseg_writes_probability_qc_contract(self) -> None:
        labels = np.asarray([[[1, 2, 3]]], dtype=np.int16)
        probabilities = np.zeros((1, 1, 3, 4), dtype=np.float32)
        probabilities[..., 0] = 0.1
        probabilities[0, 0, 0, 1] = 0.9
        probabilities[0, 0, 1, 2] = 0.9
        probabilities[0, 0, 2, 3] = 0.9

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            outputs = {
                name: output_dir / f"{name}.nii.gz"
                for name in ("segmentation", "gm_probability", "wm_probability", "csf_probability")
            }
            for path in outputs.values():
                path.write_bytes(b"output")
            write_ddseg_qc_record(
                DDSegConfig(
                    input_mask_nii=output_dir / "mask.nii.gz",
                    parameter_type="DTI",
                    output_folder=output_dir,
                    weights_dir=output_dir / "weights",
                ),
                labels=labels,
                probability_maps=probabilities,
                outputs=outputs,
            )
            qc = json.loads((output_dir / "ddseg_qc.json").read_text(encoding="utf-8"))

        self.assertEqual(qc["status"], "completed")
        self.assertEqual(qc["probability_semantics"], DDSEG_PROBABILITY_SEMANTICS)
        self.assertEqual(qc["probability_qc"]["tissue_mass_valid_fraction"], 1.0)
        self.assertEqual(qc["probability_qc"]["label_agreement"], 1.0)
        self.assertEqual(set(qc['output_artifacts']), set(outputs))

    def test_resume_rejects_changed_outputs_even_with_unchanged_size_and_mtime(self):
        import nibabel as nib
        import os
        from shared.orchestrator import _segmentation_complete, _segmentation_outputs

        for changed in ('segmentation', 'gm_probability', 'wm_probability', 'csf_probability'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # Gradient frame (batch 2, G3): the record names no DTI bundle, so only an input that the fix leaves
                # unchanged (axis-aligned, det < 0) keeps the segmentation reusable.
                (root / 'input').mkdir()
                nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.float32), np.diag([-1.0, 1.0, 1.0, 1.0])),
                         root / 'input' / 'data.nii.gz')
                outputs = {key: path for key, path in _segmentation_outputs(root).items() if key != 'qc'}
                output_dir = outputs['segmentation'].parent
                output_dir.mkdir(parents=True)
                labels = np.ones((3, 3, 3), dtype=np.int16)
                probabilities = np.zeros((*labels.shape, 4), dtype=np.float32)
                probabilities[..., 1] = 1
                for key, path in outputs.items():
                    data = labels if key == 'segmentation' else (probabilities[..., 1] if key == 'wm_probability' else probabilities[..., 0])
                    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
                write_ddseg_qc_record(DDSegConfig(
                    input_mask_nii=root / 'mask.nii.gz', parameter_type='DTI',
                    output_folder=output_dir, weights_dir=root / 'weights'),
                    labels=labels, probability_maps=probabilities, outputs=outputs)
                self.assertTrue(_segmentation_complete(root))
                path = outputs[changed]
                stat = path.stat()
                original = path.read_bytes()
                path.write_bytes(b'X' * len(original))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertFalse(_segmentation_complete(root))

    def test_ddseg_qc_rejects_mismatched_probability_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with self.assertRaisesRegex(ValueError, "matching four-class"):
                write_ddseg_qc_record(
                    DDSegConfig(
                        input_mask_nii=output_dir / "mask.nii.gz",
                        parameter_type="DTI",
                        output_folder=output_dir,
                        weights_dir=output_dir / "weights",
                    ),
                    labels=np.ones((2, 2, 2), dtype=np.int16),
                    probability_maps=np.ones((2, 2, 2, 3), dtype=np.float32),
                    outputs={},
                )


if __name__ == "__main__":
    unittest.main()
