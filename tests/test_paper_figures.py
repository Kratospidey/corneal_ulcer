from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.paper_figures import _load_history_rows, build_reliability_bins, generate_paper_figure_bundle, select_xai_rows


class PaperFigureTests(unittest.TestCase):
    def test_generate_paper_figure_bundle_accepts_resolved_config_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            experiment_name = "pattern3__convnextv2_tiny__raw_rgb__holdout_v1__seed42"
            experiment_output = output_root / "predictions" / experiment_name
            experiment_output.mkdir(parents=True, exist_ok=True)
            (output_root / "metrics" / experiment_name).mkdir(parents=True, exist_ok=True)

            predictions_csv = experiment_output / "test_predictions.csv"
            predictions_csv.write_text(
                "image_id,split,target_index,predicted_index,prob_point_like,prob_point_flaky_mixed,prob_flaky,confidence,true_label,pred_label,correct,raw_image_path,cornea_mask_path,ulcer_mask_path\n"
                "a,test,0,0,0.90,0.05,0.05,0.90,point_like,point_like,True,/tmp/a.png,,\n"
                "b,test,1,1,0.10,0.80,0.10,0.80,point_flaky_mixed,point_flaky_mixed,True,/tmp/b.png,,\n"
                "c,test,2,2,0.05,0.10,0.85,0.85,flaky,flaky,True,/tmp/c.png,,\n",
                encoding="utf-8",
            )
            metrics_json = output_root / "metrics" / experiment_name / "test_metrics.json"
            metrics_json.write_text(
                (
                    "{"
                    "\"accuracy\": 1.0,"
                    "\"balanced_accuracy\": 1.0,"
                    "\"macro_f1\": 1.0,"
                    "\"weighted_f1\": 1.0,"
                    "\"roc_auc_macro_ovr\": 1.0,"
                    "\"pr_auc_macro_ovr\": 1.0,"
                    "\"ece\": 0.0,"
                    "\"classification_report\": {"
                    "\"point_like\": {\"precision\": 1.0, \"recall\": 1.0, \"f1-score\": 1.0, \"support\": 1},"
                    "\"point_flaky_mixed\": {\"precision\": 1.0, \"recall\": 1.0, \"f1-score\": 1.0, \"support\": 1},"
                    "\"flaky\": {\"precision\": 1.0, \"recall\": 1.0, \"f1-score\": 1.0, \"support\": 1},"
                    "\"macro avg\": {\"precision\": 1.0, \"recall\": 1.0, \"f1-score\": 1.0, \"support\": 3},"
                    "\"weighted avg\": {\"precision\": 1.0, \"recall\": 1.0, \"f1-score\": 1.0, \"support\": 3}"
                    "}"
                    "}"
                ),
                encoding="utf-8",
            )
            history_csv = output_root / "metrics" / experiment_name / "history.csv"
            history_csv.write_text(
                "epoch,train_loss,val_loss,val_balanced_accuracy,val_macro_f1,lr\n1,0.9,0.8,0.7,0.7,0.0001\n",
                encoding="utf-8",
            )

            figure_bundle = generate_paper_figure_bundle(
                train_config={
                    "task_config": "configs/task_3class_pattern.yaml",
                    "output_root": str(output_root),
                    "experiment_name": experiment_name,
                    "preprocessing_mode": "raw_rgb",
                    "image_size": 224,
                    "model": {"name": "convnextv2_tiny"},
                },
                checkpoint_path="models/exported/fake/best.pt",
                predictions_csv=predictions_csv,
                metrics_json=metrics_json,
                output_root=tmpdir_path / "paper_figures",
                history_csv=history_csv,
                xai_count=0,
            )

        self.assertTrue(figure_bundle["manifest_path"].endswith("figure_manifest.md"))

    def test_load_history_rows_parses_training_history_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            history_path.write_text(
                "epoch,train_loss,val_loss,val_balanced_accuracy,val_macro_f1,lr\n"
                "1,0.9,0.8,0.70,0.68,0.0001\n"
                "2,0.7,0.6,0.75,0.73,0.00009\n",
                encoding="utf-8",
            )

            rows = _load_history_rows(history_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["epoch"], 1.0)
        self.assertAlmostEqual(rows[1]["val_balanced_accuracy"], 0.75)

    def test_build_reliability_bins_preserves_sample_count(self) -> None:
        probabilities = [
            [0.9, 0.1, 0.0],
            [0.2, 0.7, 0.1],
            [0.3, 0.2, 0.5],
            [0.4, 0.5, 0.1],
        ]
        y_true = [0, 1, 2, 0]

        bins = build_reliability_bins(probabilities, y_true, num_bins=5)

        self.assertEqual(len(bins), 5)
        self.assertEqual(sum(int(row["count"]) for row in bins), 4)

    def test_select_xai_rows_prioritizes_balanced_correct_and_error_examples(self) -> None:
        rows = [
            {"image_id": "a1", "true_label": "point_like", "pred_label": "point_like", "correct": "True", "confidence": "0.98"},
            {"image_id": "a2", "true_label": "point_flaky_mixed", "pred_label": "point_flaky_mixed", "correct": "True", "confidence": "0.95"},
            {"image_id": "a3", "true_label": "flaky", "pred_label": "flaky", "correct": "True", "confidence": "0.93"},
            {"image_id": "b1", "true_label": "point_like", "pred_label": "flaky", "correct": "False", "confidence": "0.91"},
            {"image_id": "b2", "true_label": "point_flaky_mixed", "pred_label": "point_like", "correct": "False", "confidence": "0.92"},
            {"image_id": "b3", "true_label": "flaky", "pred_label": "point_flaky_mixed", "correct": "False", "confidence": "0.89"},
            {"image_id": "c1", "true_label": "point_like", "pred_label": "point_like", "correct": "True", "confidence": "0.88"},
        ]

        selected = select_xai_rows(rows, ["point_like", "point_flaky_mixed", "flaky"], xai_count=6)

        self.assertEqual(len(selected), 6)
        self.assertEqual({row["image_id"] for row in selected[:3]}, {"a1", "a2", "a3"})
        self.assertEqual({row["image_id"] for row in selected[3:]}, {"b1", "b2", "b3"})


if __name__ == "__main__":
    unittest.main()
