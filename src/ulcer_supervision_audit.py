from __future__ import annotations

from typing import Any


HISTORICAL_TASK_STATUS_PRESENT = "historical_mask_present_positive"
HISTORICAL_TASK_STATUS_ABSENT_UNKNOWN = "historical_mask_absent_unknown"
HISTORICAL_TASK_STATUS_UNCERTAIN = "historical_mask_semantics_uncertain"

NEWTASK_STATUS_PRESENT = "newtask_mask_present_positive"
NEWTASK_STATUS_REVIEWED_EMPTY = "newtask_reviewed_empty"
NEWTASK_STATUS_MISSING_POSITIVE = "newtask_missing_positive"
NEWTASK_STATUS_NOT_APPLICABLE = "newtask_not_applicable"
NEWTASK_STATUS_NEEDS_REVIEW = "newtask_needs_review"
NEWTASK_STATUS_UNCERTAIN = "newtask_semantics_uncertain"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _priority_fields(priority: str) -> tuple[int, str]:
    mapping = {
        "p1_candidate_reviewed_empty": (1, "review_candidate_negative_for_reviewed_empty"),
        "p2_candidate_missing_positive": (2, "annotate_positive_lesion_mask"),
        "p3_semantics_uncertain": (3, "resolve_semantics_with_expert_review"),
        "p4_historical_positive_qc": (4, "qc_historical_positive_mask_for_new_task"),
        "p5_complete": (5, "preserve_without_immediate_review"),
    }
    return mapping.get(priority, (99, "manual_review"))


def classify_ulcer_supervision_row(row: dict[str, Any]) -> dict[str, Any]:
    pattern_label = str(row.get("task_pattern_3class", ""))
    binary_label = str(row.get("task_binary", ""))
    has_ulcer_mask = _as_bool(row.get("has_ulcer_mask", False))
    ulcer_mask_valid = _as_bool(row.get("ulcer_mask_valid", has_ulcer_mask))

    status = "mask_semantics_uncertain"
    historical_task_status = HISTORICAL_TASK_STATUS_UNCERTAIN
    candidate_new_task_status = NEWTASK_STATUS_UNCERTAIN
    interpretation = "requires_manual_review"
    candidate_positive_missing = False
    candidate_negative = False
    uncertain_semantics = True
    needs_review = True
    review_status = "review_pending"
    review_priority = "p3_semantics_uncertain"
    repair_action = "manual_semantics_review"
    notes = "Historical ulcer-mask semantics require explicit review before lesion-guided use."

    if has_ulcer_mask and binary_label == "ulcer_present" and ulcer_mask_valid:
        status = "mask_present_positive"
        historical_task_status = HISTORICAL_TASK_STATUS_PRESENT
        candidate_new_task_status = NEWTASK_STATUS_PRESENT
        interpretation = "historical_auxiliary_positive_mask_present"
        uncertain_semantics = False
        needs_review = False
        review_status = "qc_recommended"
        review_priority = "p4_historical_positive_qc"
        repair_action = "preserve_original_mask_in_historical_namespace"
        notes = "Historical positive mask can seed the new lesion-mask task only after QC in the new namespace."
    elif not has_ulcer_mask and pattern_label == "point_like":
        status = "mask_not_applicable"
        historical_task_status = HISTORICAL_TASK_STATUS_ABSENT_UNKNOWN
        interpretation = "historical_ulcer_mask_task_not_applied_to_point_like"
        candidate_positive_missing = binary_label == "ulcer_present"
        candidate_negative = binary_label == "no_ulcer"
        uncertain_semantics = True
        needs_review = True
        if candidate_positive_missing:
            candidate_new_task_status = NEWTASK_STATUS_MISSING_POSITIVE
            review_priority = "p2_candidate_missing_positive"
            repair_action = "annotate_positive_lesion_mask"
            notes = "Point-like ulcer-present row lacks historical lesion supervision and must be newly annotated."
        else:
            candidate_new_task_status = NEWTASK_STATUS_NEEDS_REVIEW
            review_priority = "p1_candidate_reviewed_empty"
            repair_action = "review_for_explicit_empty_mask"
            notes = "Point-like no-ulcer row requires human confirmation before it can become a reviewed-empty mask."
    elif not has_ulcer_mask and binary_label == "ulcer_present":
        status = "mask_missing_positive"
        historical_task_status = HISTORICAL_TASK_STATUS_ABSENT_UNKNOWN
        candidate_new_task_status = NEWTASK_STATUS_MISSING_POSITIVE
        interpretation = "positive_case_without_historical_mask"
        candidate_positive_missing = True
        uncertain_semantics = False
        needs_review = True
        review_status = "review_pending"
        review_priority = "p2_candidate_missing_positive"
        repair_action = "annotate_positive_lesion_mask"
        notes = "Ulcer-present row is missing lesion supervision for the new label-complete task."
    elif not has_ulcer_mask and binary_label == "no_ulcer":
        status = "mask_semantics_uncertain"
        historical_task_status = HISTORICAL_TASK_STATUS_ABSENT_UNKNOWN
        candidate_new_task_status = NEWTASK_STATUS_NEEDS_REVIEW
        interpretation = "negative_case_without_reviewed_empty_mask"
        candidate_negative = True
        uncertain_semantics = False
        needs_review = True
        review_status = "review_pending"
        review_priority = "p1_candidate_reviewed_empty"
        repair_action = "review_for_explicit_empty_mask"
        notes = "Mask absence is not an empty negative; this row requires explicit reviewed-empty handling for the new task."
    elif has_ulcer_mask and binary_label == "no_ulcer":
        status = "mask_semantics_uncertain"
        historical_task_status = HISTORICAL_TASK_STATUS_UNCERTAIN
        candidate_new_task_status = NEWTASK_STATUS_UNCERTAIN
        interpretation = "mask_present_on_negative_case"
        uncertain_semantics = True
        needs_review = True
        review_status = "expert_semantics_review"
        review_priority = "p3_semantics_uncertain"
        repair_action = "manual_semantics_review"
        notes = "Historical lesion mask conflicts with the binary no-ulcer label and requires expert resolution."
    elif has_ulcer_mask and not ulcer_mask_valid:
        status = "mask_semantics_uncertain"
        historical_task_status = HISTORICAL_TASK_STATUS_UNCERTAIN
        candidate_new_task_status = NEWTASK_STATUS_UNCERTAIN
        interpretation = "mask_file_present_but_invalid"
        uncertain_semantics = True
        needs_review = True
        review_status = "expert_semantics_review"
        review_priority = "p3_semantics_uncertain"
        repair_action = "repair_or_reannotate_mask"
        notes = "Historical mask file exists but is structurally invalid for supervised use."

    review_priority_rank, annotation_action_requested = _priority_fields(review_priority)

    return {
        **row,
        "annotation_status": status,
        "historical_task_status": historical_task_status,
        "candidate_new_task_status": candidate_new_task_status,
        "supervision_interpretation": interpretation,
        "candidate_positive_missing": candidate_positive_missing,
        "candidate_negative": candidate_negative,
        "uncertain_semantics": uncertain_semantics,
        "needs_review": needs_review,
        "review_status": review_status,
        "review_priority": review_priority,
        "review_priority_rank": review_priority_rank,
        "annotation_action_requested": annotation_action_requested,
        "recommended_action": repair_action,
        "repair_action": repair_action,
        "notes": notes,
    }
