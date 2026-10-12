import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dataset import BrainTumorDataset, load_manifest
from src.data.transforms import build_transforms


class DatasetTests(unittest.TestCase):
    def test_manifest_loading_and_deterministic_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("L", (30, 20), color=128).save(root / "images" / "sample.png")
            manifest = root / "manifest.csv"
            manifest.write_text("path,label,group\nimages/sample.png,glioma,p1\n", encoding="utf-8")
            rows = load_manifest(manifest)
            dataset = BrainTumorDataset(rows, root, build_transforms(32, training=False),
                                        {"glioma": 0})
            first, label = dataset[0]
            second, _ = dataset[0]
            self.assertEqual(tuple(first.shape), (3, 32, 32))
            self.assertEqual(label, 0)
            self.assertTrue((first == second).all().item())


if __name__ == "__main__":
    unittest.main()
