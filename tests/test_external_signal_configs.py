from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import resolve_config
from model_factory import create_model, load_backbone_warmstart


class ExternalSignalConfigTests(unittest.TestCase):
    def test_slid_pretrain_config_resolves_expected_paths(self) -> None:
        config = resolve_config("configs/pretrain_slid_convnextv2_tiny_cornea_mask.yaml")
        self.assertEqual(config["experiment_name"], "pretrain__slid__convnextv2_tiny__cornea_mask__seed42")
        self.assertEqual(config["manifest_path"], "data/interim/slid/manifest.csv")
        self.assertEqual(config["split_file"], "data/interim/slid/split_files/slid_cornea_pretrain_holdout.csv")
        self.assertEqual(config["image_zip_path"], "data/external/slid/Original_Slit-lamp_Images.zip")
        self.assertEqual(config["model"]["name"], "convnextv2_tiny")

    def test_control_and_warmstart_configs_share_downstream_recipe(self) -> None:
        control = resolve_config(
            "configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__currentbranch_control.yaml"
        )
        warm = resolve_config(
            "configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slid_cornea_pretrain.yaml"
        )
        for key in ("preprocessing_mode", "train_transform_profile", "sampler", "sampler_temperature", "task_config"):
            self.assertEqual(control[key], warm[key])
        self.assertNotEqual(control["experiment_name"], warm["experiment_name"])
        self.assertEqual(
            warm["warmstart_checkpoint"],
            "models/exported/pretrain__slid__convnextv2_tiny__cornea_mask__seed42/best.pt",
        )

    def test_slitnet_pretrain_config_resolves_expected_paths(self) -> None:
        config = resolve_config("configs/pretrain_slitnet_convnextv2_tiny_white7_fold1.yaml")
        self.assertEqual(config["experiment_name"], "pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42")
        self.assertEqual(config["annotations_mat_path"], "Slitnet_Datasets/annotations.mat")
        self.assertEqual(config["idxs_mat_path"], "Slitnet_Datasets/idxs.mat")
        self.assertEqual(config["white_light_dir"], "Slitnet_Datasets/White_Light")
        self.assertEqual(config["model"]["name"], "convnextv2_tiny")

    def test_control_and_slitnet_warmstart_configs_share_downstream_recipe(self) -> None:
        control = resolve_config(
            "configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__currentbranch_control.yaml"
        )
        warm = resolve_config(
            "configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered__from_slitnet_white7_pretrain.yaml"
        )
        for key in ("preprocessing_mode", "train_transform_profile", "sampler", "sampler_temperature", "task_config"):
            self.assertEqual(control[key], warm[key])
        self.assertNotEqual(control["experiment_name"], warm["experiment_name"])
        self.assertEqual(
            warm["warmstart_checkpoint"],
            "models/exported/pretrain__slitnet__convnextv2_tiny__white7_fold1__seed42/best.pt",
        )

    def test_warmstart_loader_rejects_shape_mismatch(self) -> None:
        model = create_model({"name": "convnextv2_tiny", "pretrained": False, "freeze_backbone": False}, num_classes=3)
        backbone_state = {key: value.clone() for key, value in model.state_dict().items() if not key.startswith("head.")}
        first_key = next(iter(backbone_state))
        backbone_state[first_key] = backbone_state[first_key][0:1]
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "bad.pt"
            torch.save({"backbone_state_dict": backbone_state}, checkpoint_path)
            with self.assertRaises(ValueError):
                load_backbone_warmstart(model, checkpoint_path)


if __name__ == "__main__":
    unittest.main()
