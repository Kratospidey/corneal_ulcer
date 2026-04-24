from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.paper_figures import build_reliability_bins, select_xai_rows


class PaperFigureTests(unittest.TestCase):
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
