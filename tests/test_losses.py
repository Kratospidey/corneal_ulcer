from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.losses import ClassBalancedFocalLoss, LogitAdjustedCrossEntropyLoss


class LossTests(unittest.TestCase):
    def test_class_balanced_focal_accepts_cpu_weights_with_cuda_logits(self) -> None:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available in test environment")
        logits = torch.tensor([[0.1, 0.2, -0.3], [1.2, -0.4, 0.7]], device="cuda")
        targets = torch.tensor([1, 0], device="cuda")
        criterion = ClassBalancedFocalLoss(weight=torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32), gamma=2.0)
        loss = criterion(logits, targets)
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_logit_adjusted_ce_accepts_cpu_tensors_with_cuda_logits(self) -> None:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available in test environment")
        logits = torch.tensor([[0.3, -0.1, 0.7], [0.8, 0.4, -0.2]], device="cuda")
        targets = torch.tensor([2, 1], device="cuda")
        criterion = LogitAdjustedCrossEntropyLoss(
            logit_adjustment=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
            weight=torch.tensor([1.0, 1.5, 0.7], dtype=torch.float32),
        )
        loss = criterion(logits, targets)
        self.assertGreaterEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
