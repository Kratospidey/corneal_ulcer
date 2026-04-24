from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_utils import load_model_init_checkpoint


class CheckpointInitTests(unittest.TestCase):
    def test_load_model_init_checkpoint_skips_shape_mismatches(self) -> None:
        import torch  # type: ignore

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "init.pt"
            source = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
            target = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 4))
            torch.save({"model_state_dict": source.state_dict()}, checkpoint_path)

            summary = load_model_init_checkpoint(target, checkpoint_path)

        self.assertEqual(summary["loaded_keys"], 2)
        self.assertIn("1.weight->1.weight", summary["skipped_shape_mismatch_keys"])
        self.assertIn("1.bias->1.bias", summary["skipped_shape_mismatch_keys"])
        self.assertIn("1.weight", summary["missing_keys"])
        self.assertIn("1.bias", summary["missing_keys"])

    def test_load_model_init_checkpoint_remaps_plain_convnext_keys_into_wrapped_model(self) -> None:
        import torch  # type: ignore

        class WrappedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Module()
                self.backbone.stem = torch.nn.Linear(4, 4)
                self.backbone.head = torch.nn.Module()
                self.backbone.head.norm = torch.nn.LayerNorm(4)
                self.feature_norm = torch.nn.LayerNorm(4)
                self.classifier = torch.nn.Linear(4, 3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "init.pt"
            stem_weight = torch.randn(4, 4)
            stem_bias = torch.randn(4)
            head_norm_weight = torch.randn(4)
            head_norm_bias = torch.randn(4)
            head_fc_weight = torch.randn(3, 4)
            head_fc_bias = torch.randn(3)
            torch.save(
                {
                    "model_state_dict": {
                        "stem.weight": stem_weight,
                        "stem.bias": stem_bias,
                        "head.norm.weight": head_norm_weight,
                        "head.norm.bias": head_norm_bias,
                        "head.fc.weight": head_fc_weight,
                        "head.fc.bias": head_fc_bias,
                    }
                },
                checkpoint_path,
            )

            target = WrappedModel()
            summary = load_model_init_checkpoint(target, checkpoint_path)

        self.assertGreaterEqual(summary["loaded_keys"], 6)
        self.assertFalse(summary["skipped_shape_mismatch_keys"])
        self.assertFalse(summary["unexpected_keys"])
        self.assertTrue(torch.equal(target.backbone.stem.weight.detach(), stem_weight))
        self.assertTrue(torch.equal(target.backbone.stem.bias.detach(), stem_bias))
        self.assertTrue(torch.equal(target.feature_norm.weight.detach(), head_norm_weight))
        self.assertTrue(torch.equal(target.feature_norm.bias.detach(), head_norm_bias))
        self.assertTrue(torch.equal(target.classifier.weight.detach(), head_fc_weight))
        self.assertTrue(torch.equal(target.classifier.bias.detach(), head_fc_bias))


if __name__ == "__main__":
    unittest.main()
