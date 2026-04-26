from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.prediction_contract import (
    build_prediction_provenance,
    build_prediction_row,
    logit_column_names,
    probability_column_names,
    validate_prediction_provenance,
    validate_prediction_rows,
)
from run_late_fusion import _validate_component_tables


CLASS_NAMES = ("point_like", "point_flaky_mixed", "flaky")


def _prediction_row(image_id: str, target_index: int, predicted_index: int, probabilities: list[float]) -> dict[str, object]:
    return build_prediction_row(
        base_row={
            "image_id": image_id,
            "split": "val",
            "target_index": target_index,
            "predicted_index": predicted_index,
        },
        class_names=CLASS_NAMES,
        probabilities=probabilities,
        extras={"confidence": max(probabilities)},
    )


class PredictionContractTests(unittest.TestCase):
    def test_prediction_row_contains_required_columns(self) -> None:
        row = _prediction_row("5", target_index=0, predicted_index=1, probabilities=[0.1, 0.7, 0.2])
        self.assertEqual(
            list(row.keys())[:4],
            ["image_id", "split", "target_index", "predicted_index"],
        )

    def test_probability_column_order_is_fixed(self) -> None:
        self.assertEqual(
            probability_column_names(CLASS_NAMES),
            ["prob_point_like", "prob_point_flaky_mixed", "prob_flaky"],
        )

    def test_logit_column_order_is_fixed(self) -> None:
        self.assertEqual(
            logit_column_names(CLASS_NAMES),
            ["logit_point_like", "logit_point_flaky_mixed", "logit_flaky"],
        )

    def test_prediction_provenance_matches_class_order(self) -> None:
        provenance = build_prediction_provenance(
            task_name="pattern_3class",
            class_names=CLASS_NAMES,
            split_name="val",
            source_config_path="configs/train_convnextv2_tiny_cornea_crop_scale_v1_augplus_v2_weighted_sampler_tempered.yaml",
        )
        validate_prediction_provenance(provenance, task_name="pattern_3class", class_names=CLASS_NAMES, split_name="val")

    def test_prediction_provenance_with_logits_matches_class_order(self) -> None:
        provenance = build_prediction_provenance(
            task_name="pattern_3class",
            class_names=CLASS_NAMES,
            split_name="val",
            source_config_path="configs/phase1/train_pattern3_phase1_A3_ordinal_aux.yaml",
            include_logits=True,
        )
        self.assertEqual(provenance["logit_columns"], logit_column_names(CLASS_NAMES))
        validate_prediction_provenance(provenance, task_name="pattern_3class", class_names=CLASS_NAMES, split_name="val")

    def test_validate_prediction_rows_rejects_missing_required_column(self) -> None:
        rows = [_prediction_row("5", target_index=0, predicted_index=1, probabilities=[0.1, 0.7, 0.2])]
        rows[0].pop("predicted_index")
        with self.assertRaises(ValueError):
            validate_prediction_rows(rows, class_names=CLASS_NAMES, split_name="val")

    def test_validate_prediction_rows_rejects_duplicate_image_id(self) -> None:
        rows = [
            _prediction_row("5", target_index=0, predicted_index=1, probabilities=[0.1, 0.7, 0.2]),
            _prediction_row("5", target_index=0, predicted_index=0, probabilities=[0.8, 0.1, 0.1]),
        ]
        with self.assertRaises(ValueError):
            validate_prediction_rows(rows, class_names=CLASS_NAMES, split_name="val")

    def test_validate_prediction_rows_accepts_optional_logits(self) -> None:
        rows = [
            build_prediction_row(
                base_row={
                    "image_id": "5",
                    "split": "val",
                    "target_index": 0,
                    "predicted_index": 1,
                },
                class_names=CLASS_NAMES,
                probabilities=[0.1, 0.7, 0.2],
                logits=[-1.2, 0.8, -0.3],
            )
        ]
        validate_prediction_rows(rows, class_names=CLASS_NAMES, split_name="val")

    def test_validate_prediction_rows_rejects_misordered_logit_columns(self) -> None:
        rows = [
            build_prediction_row(
                base_row={
                    "image_id": "5",
                    "split": "val",
                    "target_index": 0,
                    "predicted_index": 1,
                },
                class_names=CLASS_NAMES,
                probabilities=[0.1, 0.7, 0.2],
                logits=[-1.2, 0.8, -0.3],
            )
        ]
        rows[0]["logit_point_like_alt"] = rows[0].pop("logit_point_like")
        with self.assertRaises(ValueError):
            validate_prediction_rows(rows, class_names=CLASS_NAMES, split_name="val")

    def test_fusion_rejects_target_index_mismatch(self) -> None:
        left_rows = [_prediction_row("5", target_index=0, predicted_index=1, probabilities=[0.1, 0.7, 0.2])]
        right_rows = [_prediction_row("5", target_index=2, predicted_index=2, probabilities=[0.1, 0.2, 0.7])]
        left_payload = {
            "lookup": {row["image_id"]: row for row in left_rows},
            "rows": left_rows,
            "provenance": build_prediction_provenance("pattern_3class", CLASS_NAMES, "val", "left.yaml"),
        }
        right_payload = {
            "lookup": {row["image_id"]: row for row in right_rows},
            "rows": right_rows,
            "provenance": build_prediction_provenance("pattern_3class", CLASS_NAMES, "val", "right.yaml"),
        }
        with self.assertRaises(ValueError):
            _validate_component_tables([left_payload, right_payload], "pattern_3class", CLASS_NAMES, "val")

    def test_fusion_rejects_probability_schema_mismatch(self) -> None:
        left_rows = [_prediction_row("5", target_index=0, predicted_index=1, probabilities=[0.1, 0.7, 0.2])]
        right_rows = [_prediction_row("5", target_index=0, predicted_index=2, probabilities=[0.1, 0.2, 0.7])]
        right_rows[0]["prob_point_like_alt"] = right_rows[0].pop("prob_point_like")
        left_payload = {
            "lookup": {row["image_id"]: row for row in left_rows},
            "rows": left_rows,
            "provenance": build_prediction_provenance("pattern_3class", CLASS_NAMES, "val", "left.yaml"),
        }
        right_payload = {
            "lookup": {row["image_id"]: row for row in right_rows},
            "rows": right_rows,
            "provenance": build_prediction_provenance("pattern_3class", CLASS_NAMES, "val", "right.yaml"),
        }
        with self.assertRaises(ValueError):
            _validate_component_tables([left_payload, right_payload], "pattern_3class", CLASS_NAMES, "val")


if __name__ == "__main__":
    unittest.main()
