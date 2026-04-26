import unittest
from pathlib import Path
import pandas as pd
import yaml

class CVSplitsTests(unittest.TestCase):
    def test_fold_files_exist_and_valid(self):
        cv_dir = Path("data/interim/split_files/cv_pattern_3class")
        self.assertTrue(cv_dir.exists())
        
        folds = list(cv_dir.glob("fold_*.csv"))
        self.assertEqual(len(folds), 10)
        
        for fold_csv in folds:
            df = pd.read_csv(fold_csv)
            self.assertTrue("split" in df.columns)
            self.assertTrue("image_id" in df.columns)
            splits = df["split"].unique()
            self.assertIn("train", splits)
            self.assertIn("val", splits)
            self.assertIn("test", splits)
            
    def test_configs_point_to_cv_splits_not_holdout(self):
        config_dir = Path("configs/cv_pattern_3class/w0035_style")
        for config_path in config_dir.glob("fold_*.yaml"):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.assertTrue("cv_pattern_3class" in cfg.get("split_file", ""))
            self.assertNotIn("holdout", cfg.get("split_file", ""))
            # Also check no forbidden checkpoint init
            self.assertNotIn("init_checkpoint_path", cfg)
            
if __name__ == "__main__":
    unittest.main()
