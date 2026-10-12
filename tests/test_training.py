import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import fit


class TrainingTests(unittest.TestCase):
    def test_tiny_training_saves_best_checkpoint_and_history(self):
        model = torch.nn.Linear(2, 2)
        data = TensorDataset(torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]]),
                             torch.tensor([0, 1, 0, 1]))
        loader = DataLoader(data, batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "tiny.pt"
            fit(model, loader, loader, {"epochs": 2, "patience": 2, "lr": 0.01},
                checkpoint, torch.device("cpu"))
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_suffix(".history.json").exists())


if __name__ == "__main__":
    unittest.main()
