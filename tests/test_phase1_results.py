from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.phase1_results import collect_phase1_rows, render_phase1_markdown


class Phase1ResultsTests(unittest.TestCase):
    def test_collect_phase1_rows_ranks_by_val_balanced_accuracy_then_macro_f1(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            configs_dir = tmp_root / "configs" / "phase1_smoke"
            outputs_root = tmp_root / "outputs" / "phase1_pattern_a0a6" / "metrics"
            configs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(repo_root / "configs", tmp_root / "configs", dirs_exist_ok=True)

            phase_payloads = {
                "A0": {"val_balanced_accuracy": 0.80, "val_macro_f1": 0.75, "best_epoch": 4},
                "A1": {"val_balanced_accuracy": 0.83, "val_macro_f1": 0.76, "best_epoch": 5},
                "A2": {"val_balanced_accuracy": 0.83, "val_macro_f1": 0.79, "best_epoch": 6, "test_balanced_accuracy": 0.81, "test_macro_f1": 0.78},
            }
            for phase_id, payload in phase_payloads.items():
                config_name = f"train_pattern3_phase1_{phase_id}.yaml"
                config_path = configs_dir / config_name
                config_path.write_text(
                    "\n".join(
                        [
                            "base_config: ../phase1/train_pattern3_phase1_A0_baseline.yaml",
                            f"experiment_name: pattern3__phase1_{phase_id.lower()}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                metric_dir = outputs_root / f"pattern3__phase1_{phase_id.lower()}"
                metric_dir.mkdir(parents=True, exist_ok=True)
                (metric_dir / "val_metrics.json").write_text(
                    json.dumps(
                        {
                            "balanced_accuracy": payload["val_balanced_accuracy"],
                            "macro_f1": payload["val_macro_f1"],
                            "weighted_f1": 0.8,
                            "ece": 0.1,
                        }
                    ),
                    encoding="utf-8",
                )
                (metric_dir / "training_summary.json").write_text(
                    json.dumps({"best_epoch": payload["best_epoch"]}),
                    encoding="utf-8",
                )
                if "test_balanced_accuracy" in payload:
                    (metric_dir / "test_metrics.json").write_text(
                        json.dumps(
                            {
                                "balanced_accuracy": payload["test_balanced_accuracy"],
                                "macro_f1": payload["test_macro_f1"],
                                "weighted_f1": 0.79,
                                "ece": 0.09,
                            }
                        ),
                        encoding="utf-8",
                    )

            try:
                import os

                os.chdir(tmp_root)
                rows = collect_phase1_rows("configs/phase1_smoke/train_pattern3_phase1_A*.yaml", "outputs/phase1_pattern_a0a6")
            finally:
                import os

                os.chdir(original_cwd)

        self.assertEqual([row["phase_id"] for row in rows], ["A2", "A1", "A0"])
        self.assertTrue(rows[0]["tested"])
        self.assertFalse(rows[-1]["tested"])
        markdown = render_phase1_markdown(rows)
        self.assertIn("| 1 | A2 | 0.8300 | 0.7900 | 6 | yes | 0.8100 | 0.7800 |", markdown)


if __name__ == "__main__":
    unittest.main()
