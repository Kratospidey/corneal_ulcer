from __future__ import annotations

from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.transforms import build_train_transform
from utils_preprocessing import apply_variant, normalize_cornea_mask


class PatternRecipeRegressionTests(unittest.TestCase):
    def test_normalize_cornea_mask_inverts_white_background_masks(self) -> None:
        mask = Image.new("L", (10, 10), 255)
        for x in range(3, 7):
            for y in range(2, 8):
                mask.putpixel((x, y), 0)

        normalized = normalize_cornea_mask(mask)

        self.assertEqual(normalized.getpixel((0, 0)), 0)
        self.assertEqual(normalized.getpixel((4, 4)), 255)
        self.assertEqual(normalized.getbbox(), (3, 2, 7, 8))

    def test_cornea_crop_scale_v1_restores_scale_normalized_square_crop(self) -> None:
        image = Image.new("RGB", (100, 120), (10, 20, 30))
        mask = Image.new("L", (100, 120), 0)
        for x in range(10, 21):
            for y in range(15, 26):
                mask.putpixel((x, y), 255)

        roi = apply_variant(image, "cornea_crop_scale_v1", mask)

        self.assertEqual(roi.size, (72, 72))

    def test_pattern_augplus_v2_matches_frozen_winning_train_stack(self) -> None:
        transform = build_train_transform(224, "pattern_augplus_v2")
        ops = transform.transforms

        self.assertEqual(
            [type(op).__name__ for op in ops],
            [
                "Resize",
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "RandomAffine",
                "ColorJitter",
                "RandomApply",
                "RandomAdjustSharpness",
                "ToTensor",
                "Normalize",
            ],
        )
        self.assertEqual(ops[0].size, (256, 256))
        self.assertEqual(ops[1].scale, (0.78, 1.0))
        self.assertEqual(ops[1].ratio, (0.94, 1.06))
        self.assertEqual(ops[3].degrees, [-14.0, 14.0])
        self.assertEqual(ops[3].translate, (0.04, 0.04))
        self.assertEqual(ops[3].scale, (0.94, 1.06))
        self.assertEqual([type(op).__name__ for op in ops[5].transforms], ["GaussianBlur"])
        self.assertEqual(ops[6].sharpness_factor, 1.15)


if __name__ == "__main__":
    unittest.main()
