from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint_utils import interpolate_model_states, load_model_init_checkpoint


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

    def test_interpolate_model_states_remaps_wrapped_classifier_into_plain_head(self) -> None:
        import torch  # type: ignore

        target_state = {
            "head.fc.weight": torch.zeros(3, 4),
            "head.fc.bias": torch.zeros(3),
            "head.norm.weight": torch.zeros(4),
            "head.norm.bias": torch.zeros(4),
        }
        official_state = {
            "head.fc.weight": torch.full((3, 4), 1.0),
            "head.fc.bias": torch.full((3,), 2.0),
            "head.norm.weight": torch.full((4,), 3.0),
            "head.norm.bias": torch.full((4,), 4.0),
        }
        challenger_state = {
            "backbone.stem.weight": torch.full((4, 4), 9.0),
            "classifier.weight": torch.full((3, 4), 5.0),
            "classifier.bias": torch.full((3,), 6.0),
            "feature_norm.weight": torch.full((4,), 7.0),
            "feature_norm.bias": torch.full((4,), 8.0),
        }
        target_state["stem.weight"] = torch.zeros(4, 4)
        official_state["stem.weight"] = torch.full((4, 4), 1.0)

        interpolated, metadata = interpolate_model_states(
            official_state,
            challenger_state,
            target_state,
            alpha=0.25,
        )

        self.assertEqual(metadata["interpolated_keys"], 5)
        self.assertFalse(metadata["missing_in_b"])
        self.assertTrue(torch.allclose(interpolated["stem.weight"], torch.full((4, 4), 3.0)))
        self.assertTrue(torch.allclose(interpolated["head.fc.weight"], torch.full((3, 4), 2.0)))
        self.assertTrue(torch.allclose(interpolated["head.fc.bias"], torch.full((3,), 3.0)))
        self.assertTrue(torch.allclose(interpolated["head.norm.weight"], torch.full((4,), 4.0)))
        self.assertTrue(torch.allclose(interpolated["head.norm.bias"], torch.full((4,), 5.0)))


if __name__ == "__main__":
    unittest.main()
