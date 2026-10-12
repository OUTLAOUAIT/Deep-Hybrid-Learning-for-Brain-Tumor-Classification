import tempfile
import unittest
from pathlib import Path

from src.data.splitting import create_splits, validate_splits


class SplitTests(unittest.TestCase):
    def test_splits_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in ("a", "b"):
                for i in range(10):
                    path = root / (label + str(i) + ".bin")
                    path.write_bytes((label + str(i)).encode())
                    rows.append({"path": path.name, "label": label, "group": ""})
            first = create_splits(rows, root, root / "s1", seed=19)
            second = create_splits(rows, root, root / "s2", seed=19)
            self.assertTrue(validate_splits(first, root))
            self.assertEqual([[r["path"] for r in part] for part in first],
                             [[r["path"] for r in part] for part in second])

    def test_duplicate_bytes_never_cross_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for label in ("a", "b"):
                for i in range(5):
                    path = root / (label + str(i) + ".bin")
                    path.write_bytes(label.encode() if i == 4 else (label + str(i)).encode())
                    rows.append({"path": path.name, "label": label, "group": ""})
            parts = create_splits(rows, root, root / "splits")
            self.assertTrue(validate_splits(parts, root))


if __name__ == "__main__":
    unittest.main()
