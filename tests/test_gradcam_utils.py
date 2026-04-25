from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explainability.gradcam_utils import GradCAM


class GradCAMUtilsTests(unittest.TestCase):
    def test_gradcam_accepts_dict_model_outputs(self) -> None:
        import torch  # type: ignore

        class TinyWrappedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = torch.nn.Linear(4, 3)

            def forward(self, inputs):
                feats = self.features(inputs)
                pooled = self.pool(feats).flatten(1)
                return {"logits": self.classifier(pooled), "features": feats}

        model = TinyWrappedModel()
        cam = GradCAM(model, model.features)
        try:
            cam_array = cam.generate(torch.randn(1, 3, 16, 16), target_index=1)
        finally:
            cam.close()

        self.assertEqual(cam_array.shape, (16, 16))


if __name__ == "__main__":
    unittest.main()
