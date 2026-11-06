"""Vendored PNL CNN code: byte provenance, licence records, packaging and loader (T2 design A2)."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.audit_third_party import REQUIRED_RECORDS
from scripts.extension_runtime.runtime_resources import runtime_files


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "shared/pnl_cnn"
PREFACE = ('All or portions of this licensed product (such portions are the "Software") have been obtained '
           "under license from The Brigham and Women's Hospital, Inc. and are subject to the following terms "
           "and conditions:")
HEADER = ("# Modified version of PNLBrainMask-Torch @ 8d201a5 (Brigham and Women's Hospital licence, "
          "see shared/pnl_cnn/LICENSE.txt); modified by SlicerDSpace: ")
HAS_TORCH = importlib.util.find_spec("torch") is not None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constant(name):
    tree = ast.parse((VENDOR / "weights.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def flat(text):
    return " ".join(text.split())


class VendoredPnlCnnTest(unittest.TestCase):
    def test_architecture_json_matches_package_architecture_hash(self):
        # load_model checks only the receipt's copy of this hash (R3-n4), so the vendored file is hashed here.
        self.assertEqual(sha256(VENDOR / "data/CompNetBasicModel.json"), constant("ARCHITECTURE_SHA256"))

    def test_upstream_record_matches_vendored_bytes_and_modification_marks(self):
        record = json.loads((VENDOR / "UPSTREAM.json").read_text(encoding="utf-8"))
        self.assertEqual(record["commit"], "8d201a50447ebea8b10a4f252bfd262931603970")
        self.assertEqual(record["license_file"], "shared/pnl_cnn/LICENSE.txt")
        on_disk = {path.relative_to(VENDOR).as_posix() for path in VENDOR.rglob("*")
                   if path.is_file() and "__pycache__" not in path.parts}
        self.assertEqual(on_disk, set(record["files"]) | {"UPSTREAM.json"})
        for relative, item in record["files"].items():
            with self.subTest(file=relative):
                path = VENDOR / relative
                self.assertEqual(sha256(path), item["sha256"])
                first = path.read_text(encoding="utf-8").splitlines()[0]
                if item["patches"]:
                    self.assertTrue(relative.endswith(".py"))
                    self.assertNotEqual(item["sha256"], item["upstream_sha256"])
                    self.assertTrue(first.startswith(HEADER), first)
                    for patch in item["patches"]:
                        if not patch.startswith("clause-(d)"):
                            self.assertIn(patch, first)
                else:
                    self.assertEqual(item["sha256"], item["upstream_sha256"])
                    self.assertFalse(first.startswith("# Modified version"), first)

    def test_licence_record_and_runtime_packaging(self):
        self.assertEqual(REQUIRED_RECORDS["pnl_cnn"], "shared/pnl_cnn/LICENSE.txt")
        spec = json.loads((ROOT / "licenses/provenance.json").read_text(encoding="utf-8"))
        record = next(item for item in spec["records"] if item["id"] == "pnl_cnn")
        self.assertEqual(record["license"], "LicenseRef-PNL-CNN-BWH")
        self.assertEqual(record["path"], "shared/pnl_cnn/LICENSE.txt")
        self.assertEqual(record["sha256"], sha256(VENDOR / "LICENSE.txt"))
        self.assertEqual(record["source"], "https://github.com/ChengjinLii/PNLBrainMask-Torch/blob/"
                                           "8d201a50447ebea8b10a4f252bfd262931603970/LICENSE")
        self.assertEqual(record["covers"], ["shared/pnl_cnn"])
        for clause in ("(b)", "(c)", "(d)"):
            self.assertIn(clause, record["conditions"])
        selected = set(runtime_files(ROOT))
        for relative in ("shared/pnl_cnn/LICENSE.txt", "shared/pnl_cnn/UPSTREAM.json",
                         "shared/pnl_cnn/data/CompNetBasicModel.json", "shared/pnl_cnn/model.py",
                         "shared/pnl_cnn/inference.py", "shared/pnl_cnn/weights.py", "shared/pnl_cnn/pipeline.py"):
            self.assertIn(relative, selected)

    def test_user_documentation_carries_preface_and_full_terms(self):
        terms = (VENDOR / "LICENSE.txt").read_text(encoding="utf-8")
        self.assertIn(flat(PREFACE), flat(terms))
        page = (ROOT / "docs/licenses/pnl_brain_mask.md").read_text(encoding="utf-8")
        self.assertIn(terms, page)
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn(terms, notices)
        self.assertIn("modified integration", notices)
        self.assertIn("8d201a50447ebea8b10a4f252bfd262931603970", notices)
        guide = (ROOT / "docs/setup/brain_mask_gui_config.md").read_text(encoding="utf-8")
        self.assertIn(flat(PREFACE), flat(guide))
        self.assertIn("(../licenses/pnl_brain_mask.md)", guide)

    @unittest.skipUnless(HAS_TORCH, "torch is not installed in this interpreter")
    def test_compnet_builds_from_vendored_json_and_loader_checks_receipt(self):
        import torch
        from shared.pnl_cnn import weights
        from shared.pnl_cnn.model import CompNet

        model = CompNet()
        # Parameter and tensor counts recorded by the converted-weights receipt (weights.json).
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 18705959)
        state = model.state_dict()
        self.assertEqual(len(state), 414)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            torch.save(state, folder / "axial.pt")
            digest = sha256(folder / "axial.pt")
            for view in ("coronal", "sagittal"):
                os.symlink(folder / "axial.pt", folder / f"{view}.pt")
            receipt = {"format": 1, "architecture_sha256": weights.ARCHITECTURE_SHA256,
                       "views": {view: {"file": f"{view}.pt", "sha256": digest,
                                        "source_sha256": weights.AUTHOR_FILES[view][1]}
                                 for view in ("sagittal", "coronal", "axial")}}
            (folder / "weights.json").write_text(json.dumps(receipt))
            for view in ("sagittal", "coronal", "axial"):
                loaded = weights.load_model(folder, view)
                self.assertFalse(loaded.training)
                restored = loaded.state_dict()
                self.assertTrue(all(torch.equal(restored[key], value) for key, value in state.items()))
            receipt["architecture_sha256"] = "0" * 64
            (folder / "weights.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "Unrecognized converted model"):
                weights.load_model(folder, "axial")

    @unittest.skipUnless(HAS_TORCH and os.environ.get("SLICERDSPACE_PNL_MASK_MODEL_DIR"),
                         "set SLICERDSPACE_PNL_MASK_MODEL_DIR to the pnl_cnn_torch asset folder")
    def test_vendored_loader_loads_three_views_from_asset_folder(self):
        from shared.pnl_cnn.weights import load_model

        folder = Path(os.environ["SLICERDSPACE_PNL_MASK_MODEL_DIR"])
        for view in ("sagittal", "coronal", "axial"):
            with self.subTest(view=view):
                model = load_model(folder, view)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 18705959)
                self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
