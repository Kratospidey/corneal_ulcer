from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import resolve_config
from data.label_utils import get_task_definition


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

    def test_late_fusion_config_points_at_pattern_task(self) -> None:
        config = resolve_config("configs/inference_pattern_latefusion_v1.yaml")
        task_config = resolve_config(config["task_config"])
        self.assertEqual(task_config["task_name"], "pattern_3class")
        self.assertEqual(len(config["inference"]["models"]), 2)

    def test_phase1_baseline_config_disables_test_eval(self) -> None:
        config = resolve_config("configs/phase1/train_pattern3_phase1_A0_baseline.yaml")
        self.assertEqual(config["task_config"], "configs/task_3class_pattern.yaml")
        self.assertFalse(config["evaluate_test_after_train"])
        self.assertFalse(config["auto_generate_paper_figures"])
        self.assertFalse(config["show_progress"])
        self.assertEqual(config["output_root"], "outputs/phase1_pattern_a0a6")

    def test_phase1_multiscale_config_enables_optional_head(self) -> None:
        config = resolve_config("configs/phase1/train_pattern3_phase1_A1_multiscale.yaml")
        multiscale = config["model"]["multiscale_head"]
        self.assertTrue(multiscale["enabled"])
        self.assertEqual(multiscale["stage_index"], 2)
        self.assertEqual(multiscale["fusion_dim"], 256)

    def test_phase1_ordinal_config_enables_aux_head(self) -> None:
        config = resolve_config("configs/phase1/train_pattern3_phase1_A3_ordinal_aux.yaml")
        self.assertTrue(config["model"]["ordinal_aux"]["enabled"])
        self.assertEqual(config["ordinal_aux_weight"], 0.25)

    def test_phase1_loss_variants_resolve_expected_overrides(self) -> None:
        logit_adjusted = resolve_config("configs/phase1/train_pattern3_phase1_A4_logit_adjusted.yaml")
        class_balanced = resolve_config("configs/phase1/train_pattern3_phase1_A5_class_balanced_focal.yaml")
        combined = resolve_config("configs/phase1/train_pattern3_phase1_A6_multiscale_ordinal_smooth.yaml")
        self.assertEqual(logit_adjusted["loss_name"], "logit_adjusted_ce")
        self.assertFalse(logit_adjusted["use_class_weights"])
        self.assertEqual(class_balanced["loss_name"], "class_balanced_focal")
        self.assertEqual(class_balanced["class_balanced_beta"], 0.999)
        self.assertEqual(combined["label_smoothing"], 0.05)
        self.assertTrue(combined["model"]["multiscale_head"]["enabled"])
        self.assertTrue(combined["model"]["ordinal_aux"]["enabled"])


if __name__ == "__main__":
    unittest.main()
