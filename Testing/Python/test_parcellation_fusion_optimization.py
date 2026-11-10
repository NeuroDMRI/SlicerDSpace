"""Ordered maxout batching, including full-network output and gradient parity."""

import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def reference_fusion(value, backbones, slot):
    merged = torch.unsqueeze(value, 4)
    for result in backbones:
        merged = torch.cat((merged, result[slot].unsqueeze(4)), dim=4)
    return torch.max(merged, 4)[0]


@unittest.skipIf(torch is None, 'PyTorch is not installed in this interpreter')
class ParcellationFusionTests(unittest.TestCase):
    def test_order_values_ties_nonfinite_and_gradients(self):
        from anatomy.parcellation.models import networks
        torch.manual_seed(42)
        for count in (0, 1, 4):
            tensors = [torch.randn(2, 4, 8, 8) for _ in range(count + 1)]
            for array in tensors:
                array[:, :, 0, :4] = torch.tensor([0., -0., float('nan'), float('inf')])
            expected = reference_fusion(tensors[0], [[v] for v in tensors[1:]], 0)
            actual = networks._fuse_backbone_features(tensors[0], [[v] for v in tensors[1:]], 0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            self.assertTrue(torch.equal(torch.signbit(actual), torch.signbit(expected)))
            inputs = [torch.ones(1, 2, 3, 4, requires_grad=True) for _ in range(count + 1)]
            expected = reference_fusion(inputs[0], [[v] for v in inputs[1:]], 0)
            old_grad = torch.autograd.grad(expected.sum(), inputs)
            actual = networks._fuse_backbone_features(inputs[0], [[v] for v in inputs[1:]], 0)
            new_grad = torch.autograd.grad(actual.sum(), inputs)
            for left, right in zip(old_grad, new_grad):
                self.assertTrue(torch.equal(left, right))

    def test_full_network_and_logit_alias_are_unchanged(self):
        import sys
        from anatomy.parcellation.models import networks
        torch.manual_seed(2026)
        params = dict(num_channels=7, num_modality=4, num_filters=4,
                      kernel_h=3, kernel_w=3, stride_conv=1, pool=2,
                      stride_pool=2, num_classes=6, kernel_c=1, backbone_model=None)
        before_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = networks.FastSurferCNN_Fuse_Unet_v3_extended(params).eval()
            data = torch.randn(1, 28, 32, 32)
            with torch.no_grad():
                with patch.object(networks, '_fuse_backbone_features', side_effect=reference_fusion) as old:
                    expected = model(data)
                    self.assertEqual(old.call_count, 5)
                actual = model(data)
            self.assertEqual(actual[0].data_ptr(), actual[1][0].data_ptr())
            for left, right in zip((expected[0], *expected[1]), (actual[0], *actual[1])):
                self.assertTrue(torch.equal(left, right))
        finally:
            if not (sys.platform == 'darwin' and 'dipy' in sys.modules):
                torch.set_num_threads(before_threads)


if __name__ == '__main__':
    unittest.main()
