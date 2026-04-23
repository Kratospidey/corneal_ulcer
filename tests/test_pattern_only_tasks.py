from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import resolve_config
from data.label_utils import get_task_definition
from model_factory import create_model


class PatternOnlyTaskTests(unittest.TestCase):
    def test_archived_tasks_are_rejected_by_default(self) -> None:
        with self.assertRaises(KeyError):
            get_task_definition("severity_5class")
        with self.assertRaises(KeyError):
            get_task_definition("binary")

    def test_pattern_config_resolves_expected_overrides(self) -> None:
        config = resolve_config("configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml")
        self.assertEqual(config["preprocessing_mode"], "cornea_crop_scale_v1")
        self.assertEqual(config["train_transform_profile"], "pattern_augplus_v2")
        self.assertEqual(config["sampler"], "weighted_sampler_tempered")
        self.assertEqual(config["sampler_temperature"], 0.65)

    def test_maxvit_tiny_model_is_supported(self) -> None:
        model = create_model(
            {"name": "maxvit_tiny_tf_224.in1k", "pretrained": False, "freeze_backbone": False},
            num_classes=3,
        )
        self.assertIsNotNone(model)

    def test_maxvit_pattern_config_resolves_expected_overrides(self) -> None:
        config = resolve_config("configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml")
        self.assertEqual(config["preprocessing_mode"], "cornea_crop_scale_v1")
        self.assertEqual(config["train_transform_profile"], "pattern_augplus_v2")
        self.assertEqual(config["sampler"], "weighted_sampler_tempered")
        self.assertEqual(config["sampler_temperature"], 0.65)
        self.assertEqual(config["model"]["name"], "maxvit_tiny_tf_224.in1k")

    def test_late_fusion_config_points_at_pattern_task(self) -> None:
        config = resolve_config("configs/inference_pattern_latefusion_v1.yaml")
        task_config = resolve_config(config["task_config"])
        self.assertEqual(task_config["task_name"], "pattern_3class")
        self.assertEqual(len(config["inference"]["models"]), 2)

    def test_maxvit_ensemble_configs_point_at_pattern_task(self) -> None:
        for path in (
            "configs/inference_pattern_convnext_maxvit_avgprob_eq.yaml",
            "configs/inference_pattern_convnext_maxvit_avgprob_valtuned.yaml",
        ):
            config = resolve_config(path)
            task_config = resolve_config(config["task_config"])
            self.assertEqual(task_config["task_name"], "pattern_3class")
            self.assertEqual(len(config["inference"]["models"]), 2)
            self.assertEqual(
                config["inference"]["models"][1]["config_path"],
                "configs/train_maxvit_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml",
            )


if __name__ == "__main__":
    unittest.main()
