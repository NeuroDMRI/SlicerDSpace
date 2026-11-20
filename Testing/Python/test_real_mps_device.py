"""Real-device Apple MPS parity tests (finding tests-mock-mps).

The mock based MPS tests elsewhere cannot catch kernels that are missing or numerically
wrong on a real Metal device. These tests run on a real MPS device when one is available
and skip cleanly when torch is missing or MPS is unavailable (CI, Linux, Windows, Intel).
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # torch is optional for the fast test run
    import torch
    import torch.nn.functional as F

    _MPS_OK = bool(torch.backends.mps.is_available())
except Exception:  # pragma: no cover - depends on host
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _MPS_OK = False

GIB = 1024 ** 3


def _find_tractcloud_models_file() -> Path | None:
    """Locate tractcloud/models.py from the plugin's verified source, else a bundled copy."""
    candidates: list[Path] = []
    try:
        from shared.backend_sources import available_source

        root = available_source("tractcloud")
        if root is not None:
            candidates.append(Path(root) / "src" / "tractcloud" / "models.py")
            candidates.append(Path(root) / "tractcloud" / "models.py")
    except Exception:
        pass
    env_root = os.environ.get("SLICERDSPACE_BUNDLED_SOURCES_ROOT")
    patterns = []
    if env_root:
        patterns.append(os.path.join(env_root, "tractcloud-*", "src", "tractcloud", "models.py"))
    patterns.append(str(REPO_ROOT / "bundled-sources" / "tractcloud-*" / "src" / "tractcloud" / "models.py"))
    patterns.append(
        "/Applications/Slicer.app/Contents/Extensions-*/SlicerDSpace/bundled-sources/"
        "tractcloud-*/src/tractcloud/models.py"
    )
    for pattern in patterns:
        candidates.extend(Path(p) for p in sorted(glob.glob(pattern)))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_surface_tanet():
    """Import TANet the way the backend runs: its folder on sys.path, ``model`` as a top-level package.

    The generic top-level name ``model`` is removed again afterwards so other tests are not affected.
    """
    backend = str(REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend")
    saved = {k: v for k, v in sys.modules.items() if k == "model" or k.startswith("model.")}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, backend)
    try:
        import importlib

        return importlib.import_module("model.ddsurfer").TANet
    finally:
        try:
            sys.path.remove(backend)
        except ValueError:
            pass
        for key in [k for k in sys.modules if k == "model" or k.startswith("model.")]:
            del sys.modules[key]
        sys.modules.update(saved)


@unittest.skipUnless(_MPS_OK, "torch with an available MPS device is required")
class RealMpsDeviceTests(unittest.TestCase):
    def test_tractcloud_dgcnn_forward_mps_matches_cpu(self) -> None:
        models_file = _find_tractcloud_models_file()
        if models_file is None:
            self.skipTest("TractCloud source (tractcloud/models.py) is not available")
        models = _load_module_from_file("_real_mps_tractcloud_models", models_file)

        torch.manual_seed(11)
        batch, points, k_local, k_global, classes = 64, 15, 20, 80, 40
        model = models.TractDGCNN(
            num_classes=classes, k=k_local, k_global=k_global, k_point_level=5, emb_dims=256
        ).eval()
        with torch.no_grad():
            # Widen class margins so argmax is stable against legitimate float noise.
            model.linear3.weight.mul_(25.0)
        x = torch.randn(batch, 3, points)
        info = torch.randn(batch, 3, points, k_local + k_global)

        with torch.no_grad():
            cpu_out = model(x, info)
            model_mps = models.TractDGCNN(
                num_classes=classes, k=k_local, k_global=k_global, k_point_level=5, emb_dims=256,
                device=torch.device("mps"),
            ).eval()
            model_mps.load_state_dict(model.state_dict())
            model_mps = model_mps.to("mps")
            mps_out = model_mps(x.to("mps"), info.to("mps")).cpu()

        self.assertEqual(tuple(mps_out.shape), (batch, classes))
        self.assertTrue(torch.isfinite(mps_out).all().item())
        agreement = (cpu_out.argmax(1) == mps_out.argmax(1)).float().mean().item()
        self.assertGreaterEqual(agreement, 0.99, "argmax agreement %.4f" % agreement)
        self.assertLess((cpu_out - mps_out).abs().max().item(), 5e-2)

    def test_pnl_compnet_2d_cnn_mps_matches_cpu(self) -> None:
        from shared.pnl_cnn.model import CompNet

        torch.manual_seed(5)
        model = CompNet().eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            cpu_out = model(x, segmentation_only=True)
            mps_out = model.to("mps")(x.to("mps"), segmentation_only=True).cpu()

        self.assertEqual(cpu_out.shape, mps_out.shape)
        self.assertTrue(torch.isfinite(mps_out).all().item())
        self.assertLess((cpu_out - mps_out).abs().max().item(), 1e-3)

    def test_surface_model_vf_net_stays_on_cpu_when_mps_lacks_conv3d(self) -> None:
        try:
            x = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32, device="mps")
            w = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32, device="mps")
            F.conv3d(x, w)
            conv3d_ok = True
        except Exception:
            conv3d_ok = False

        TANet = _import_surface_tanet()
        model = TANet(
            C_in=5,
            C_hid=(4, 8, 8, 8, 8),
            inshape=(16, 16, 16),
            depths=(1, 1, 1, 1, 1),
            kernels=(3, 3, 3, 3, 3),
            device="cpu",
        )
        model = model.to("mps")
        vf_types = {p.device.type for p in model.vf_net.parameters()}
        att_types = {p.device.type for p in model.att_net.parameters()}
        self.assertEqual(att_types, {"mps"})
        if conv3d_ok:
            self.assertEqual(vf_types, {"mps"})
        else:
            self.assertEqual(vf_types, {"cpu"})

    def test_query_mps_free_total_bytes_is_sane(self) -> None:
        from shared.hardware_adapt import query_mps_free_total_bytes

        result = query_mps_free_total_bytes()
        self.assertIsNotNone(result)
        free_b, total_b = result
        self.assertGreaterEqual(total_b, 2 * GIB)
        self.assertGreater(free_b, 0)
        self.assertLessEqual(free_b, total_b)


if __name__ == "__main__":
    unittest.main()
