import unittest

import torch

from src.models import build_model


class ModelTests(unittest.TestCase):
    def test_all_model_output_shapes(self):
        for name in ("efficientnet_b0", "efficientnet_vgg16_fusion", "vit_b_16"):
            model = build_model(name, 3, pretrained=False, freeze=True).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.zeros(1, 3, 224, 224)).shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
