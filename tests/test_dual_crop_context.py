import unittest
import torch
from PIL import Image

from utils_preprocessing import extract_cornea_crop_wide_context_v1, extract_cornea_crop_scale_v1
from model_factory import create_model

class DualCropContextTests(unittest.TestCase):
    def test_cornea_crop_wide_context_v1_returns_valid_image(self):
        img = Image.new("RGB", (500, 500), color="white")
        mask = Image.new("L", (500, 500), color=255) # all white mask
        
        # Tight crop
        tight = extract_cornea_crop_scale_v1(img, mask)
        
        # Wide crop
        wide = extract_cornea_crop_wide_context_v1(img, mask)
        
        self.assertIsInstance(wide, Image.Image)
        # Wide crop should theoretically have a larger side before resizing, but here we just check it doesn't crash
        self.assertTrue(wide.width > 0 and wide.height > 0)

    def test_model_forward_works_for_dual_crop_batch(self):
        config = {
            "name": "convnextv2_tiny",
            "pretrained": False,
            "input_mode": "dual_crop_context_v1"
        }
        model = create_model(config, num_classes=3)
        
        # Fake dual crop batch
        batch_tight = torch.randn(2, 3, 224, 224)
        batch_wide = torch.randn(2, 3, 224, 224)
        
        outputs = model((batch_tight, batch_wide))
        
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertIn("features", outputs)

if __name__ == "__main__":
    unittest.main()
