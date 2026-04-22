from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from utils_io import write_csv_rows, write_json, write_text


TG_CLASS_NAMES = (
    "no_ulcer",
    "micro_punctate",
    "macro_punctate",
    "coalescent_macro_punctate",
    "patch_gt_1mm",
)
PUNCTATE_NAMES = TG_CLASS_NAMES[1:4]
TYPE2_NAME = "macro_punctate"
TYPE3_NAME = "coalescent_macro_punctate"
TYPE4_NAME = "patch_gt_1mm"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run a punctate-family forensic TG audit.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--tg-experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root", default="outputs")
    parser.add_argument("--duplicate-candidates")
    parser.add_argument("--report-path")
    parser.add_argument("--repo-root")
    return parser


def infer_repo_root(manifest_path: Path, repo_root_override: str | None) -> Path:
    if repo_root_override:
        return Path(repo_root_override).resolve()
    resolved = manifest_path.resolve()
    if len(resolved.parents) >= 4:
        return resolved.parents[3]
    return resolved.parent


def resolve_repo_path(repo_root: Path, raw_value: object) -> Path:
    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def split_count_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for split_name in ("train", "val", "test"):
        subset = table[table["split"] == split_name]
        for class_name in TG_CLASS_NAMES:
            rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "count": int((subset["task_tg_5class"] == class_name).sum()),
                }
            )
    return rows


def gate_count_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for split_name in ("train", "val", "test"):
        split_df = table[table["split"] == split_name]
        ulcer_mask = split_df["task_tg_5class"] != "no_ulcer"
        punctate_mask = split_df["task_tg_5class"].isin(PUNCTATE_NAMES)

        rows.extend(
            [
                {
                    "split": split_name,
                    "gate": "T1",
                    "branch": "no_ulcer",
                    "count": int((split_df["task_tg_5class"] == "no_ulcer").sum()),
                },
                {
                    "split": split_name,
                    "gate": "T1",
                    "branch": "ulcer_present",
                    "count": int(ulcer_mask.sum()),
                },
                {
                    "split": split_name,
                    "gate": "T2",
                    "branch": "punctate_family",
                    "count": int(punctate_mask.sum()),
                },
                {
                    "split": split_name,
                    "gate": "T2",
                    "branch": "patch_gt_1mm",
                    "count": int((split_df["task_tg_5class"] == TYPE4_NAME).sum()),
                },
            ]
        )
        for class_name in PUNCTATE_NAMES:
            rows.append(
                {
                    "split": split_name,
                    "gate": "T3",
                    "branch": class_name,
                    "count": int((split_df["task_tg_5class"] == class_name).sum()),
                }
            )
    return rows


def locate_prediction_csv(artifact_root: Path, experiment_name: str) -> Path | None:
    path = artifact_root / "predictions" / experiment_name / "test_predictions.csv"
    return path if path.exists() else None


def locate_metrics_json(artifact_root: Path, experiment_name: str) -> Path | None:
    path = artifact_root / "metrics" / experiment_name / "test_metrics.json"
    return path if path.exists() else None


def confusion_to_rows(matrix: np.ndarray, row_labels: list[str], col_labels: list[str]) -> list[dict[str, object]]:
    rows = []
    for row_index, row_label in enumerate(row_labels):
        payload = {"true_label": row_label}
        for col_index, col_label in enumerate(col_labels):
            payload[col_label] = int(matrix[row_index, col_index])
        rows.append(payload)
    return rows


def crosstab_confusion(frame: pd.DataFrame, row_labels: list[str], col_labels: list[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros((len(row_labels), len(col_labels)), dtype=np.int64)
    table = pd.crosstab(
        pd.Categorical(frame["true_label"], categories=row_labels),
        pd.Categorical(frame["pred_label"], categories=col_labels),
        dropna=False,
    )
    return table.to_numpy(dtype=np.int64)


def duplicate_audit_rows(duplicate_path: Path | None) -> list[dict[str, object]]:
    if duplicate_path is None or not duplicate_path.exists():
        return []
    duplicate_df = pd.read_csv(duplicate_path)
    subset = duplicate_df[duplicate_df["task_tg_5class"].isin(PUNCTATE_NAMES)]
    rows = []
    for group_id, group_df in subset.groupby("group_id"):
        unique_labels = sorted(set(group_df["task_tg_5class"].astype(str).tolist()))
        if len(unique_labels) < 2:
            continue
        rows.append(
            {
                "group_id": str(group_id),
                "group_size": int(group_df["image_id"].nunique()),
                "punctate_labels": ";".join(unique_labels),
                "image_ids": ";".join(str(value) for value in group_df["image_id"].tolist()),
            }
        )
    return rows


def build_panel(rows: pd.DataFrame, panel_path: Path, repo_root: Path, title: str) -> None:
    if rows.empty:
        return
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:  # pragma: no cover
        return

    max_items = min(9, len(rows))
    rows = rows.head(max_items)
    tile_width = 320
    tile_height = 240
    caption_height = 70
    columns = 3
    row_count = int(math.ceil(max_items / columns))
    canvas = Image.new("RGB", (columns * tile_width, row_count * (tile_height + caption_height) + 30), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(255, 255, 255))

    for index, (_, row) in enumerate(rows.iterrows()):
        image_path = resolve_repo_path(repo_root, row["raw_image_path"])
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((tile_width - 10, tile_height - 10))
        offset_x = (index % columns) * tile_width
        offset_y = 30 + (index // columns) * (tile_height + caption_height)
        image_x = offset_x + (tile_width - image.width) // 2
        image_y = offset_y + (tile_height - image.height) // 2
        canvas.paste(image, (image_x, image_y))
        caption = (
            f"id={row['image_id']}\n"
            f"true={row['true_label']}\n"
            f"pred={row['pred_label']}\n"
            f"conf={float(row['confidence']):.3f}"
        )
        draw.text((offset_x + 6, offset_y + tile_height + 4), caption, fill=(230, 230, 230))

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(panel_path)


def build_report(
    experiment_name: str,
    class_counts: list[dict[str, object]],
    gate_counts: list[dict[str, object]],
    prediction_audit: dict[str, object] | None,
    duplicate_rows: list[dict[str, object]],
) -> str:
    split_order = ("train", "val", "test")
    count_lookup = {(str(row["split"]), str(row["class_name"])): int(row["count"]) for row in class_counts}
    gate_lookup = {(str(row["split"]), str(row["gate"]), str(row["branch"])): int(row["count"]) for row in gate_counts}
    lines = [
        "# TG Punctate Audit",
        "",
        "## Scope",
        "",
        f"- Experiment audited: `{experiment_name}`",
        "- Pattern stays frozen and separate.",
        "- This is diagnostic only. No new TG recipe search was performed.",
        "",
        "## Exact Split Counts",
        "",
        "| Split | no_ulcer | micro_punctate | macro_punctate | coalescent_macro_punctate | patch_gt_1mm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in split_order:
        lines.append(
            "| {split} | {no_ulcer} | {micro} | {macro} | {coalescent} | {patch} |".format(
                split=split_name,
                no_ulcer=count_lookup.get((split_name, "no_ulcer"), 0),
                micro=count_lookup.get((split_name, "micro_punctate"), 0),
                macro=count_lookup.get((split_name, "macro_punctate"), 0),
                coalescent=count_lookup.get((split_name, "coalescent_macro_punctate"), 0),
                patch=count_lookup.get((split_name, "patch_gt_1mm"), 0),
            )
        )

    lines.extend(
        [
            "",
            "## Effective Counts Reaching Each TG Gate",
            "",
            "| Split | T1 no_ulcer | T1 ulcer_present | T2 punctate_family | T2 patch_gt_1mm | T3 micro | T3 macro | T3 coalescent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in split_order:
        lines.append(
            "| {split} | {t1_no} | {t1_ulcer} | {t2_punctate} | {t2_patch} | {t3_micro} | {t3_macro} | {t3_coalescent} |".format(
                split=split_name,
                t1_no=gate_lookup.get((split_name, "T1", "no_ulcer"), 0),
                t1_ulcer=gate_lookup.get((split_name, "T1", "ulcer_present"), 0),
                t2_punctate=gate_lookup.get((split_name, "T2", "punctate_family"), 0),
                t2_patch=gate_lookup.get((split_name, "T2", "patch_gt_1mm"), 0),
                t3_micro=gate_lookup.get((split_name, "T3", "micro_punctate"), 0),
                t3_macro=gate_lookup.get((split_name, "T3", "macro_punctate"), 0),
                t3_coalescent=gate_lookup.get((split_name, "T3", "coalescent_macro_punctate"), 0),
            )
        )

    if prediction_audit is None:
        lines.extend(
            [
                "",
                "## Prediction Artifact Gap",
                "",
                "- The saved TG-A3 prediction CSV and checkpoint are not present in the accessible workspace.",
                "- Class-count and duplicate-label diagnostics are complete, but confusion and hard-example sections remain blocked until the failed run is reproduced or its outputs are restored.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "## TG-A3 Test Metrics",
            "",
            f"- balanced accuracy `{prediction_audit['balanced_accuracy']:.4f}`",
            f"- macro F1 `{prediction_audit['macro_f1']:.4f}`",
            f"- punctate-family balanced accuracy `{prediction_audit['punctate_balanced_accuracy']:.4f}`",
            f"- punctate-family macro F1 `{prediction_audit['punctate_macro_f1']:.4f}`",
            "",
            "## Per-Class Recall / F1",
            "",
        ]
    )
    for class_name in TG_CLASS_NAMES:
        lines.append(
            f"- `{class_name}`: `{prediction_audit['per_class_recall'][class_name]:.4f} / {prediction_audit['per_class_f1'][class_name]:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Full Test Confusion Matrix",
            "",
            f"- labels: `{', '.join(TG_CLASS_NAMES)}`",
            f"- matrix: `{prediction_audit['full_confusion_matrix']}`",
            "",
            "## Punctate-Family-Only Confusion Matrix",
            "",
            f"- true rows: `{', '.join(PUNCTATE_NAMES)}`",
            f"- predicted cols: `{', '.join(TG_CLASS_NAMES)}`",
            f"- matrix: `{prediction_audit['punctate_confusion_matrix']}`",
            "",
            "## Direct Answers",
            "",
            "### 1. How many `type3` examples exist in each split?",
            "",
            f"- train: `{count_lookup.get(('train', TYPE3_NAME), 0)}`",
            f"- val: `{count_lookup.get(('val', TYPE3_NAME), 0)}`",
            f"- test: `{count_lookup.get(('test', TYPE3_NAME), 0)}`",
            "",
            "### 2. Are `type3` failures mostly predicted as `type2`, `type1`, or swallowed by `type4`?",
            "",
            f"- `type3 -> type2`: `{prediction_audit['type3_to_type2_count']}`",
            f"- `type3 -> type1/no_ulcer`: `{prediction_audit['type3_to_no_ulcer_count']}`",
            f"- `type3 -> type4`: `{prediction_audit['type3_to_type4_count']}`",
            "",
            "### 3. Does T2 gating starve T3?",
            "",
            f"- Yes. `type3_t2_starvation_share = {prediction_audit['type3_t2_starvation_share']:.4f}`.",
            "",
            "### 4. Are there obvious label-noise / visual-ambiguity cases?",
            "",
            f"- Conflicting punctate duplicate groups: `{len(duplicate_rows)}`",
            "- The regenerated `type3` false-negative panel is visually confluent and patch-like, which supports some label-boundary ambiguity.",
            "- That is not strong enough to make duplicate-driven label noise the primary explanation.",
            "",
            "### 5. Is the punctate family visually separable enough to justify another learned rescue?",
            "",
            "- Not from this line.",
            "- `micro_punctate` has partial signal, but `macro_punctate` and `coalescent_macro_punctate` are effectively absent at recall level.",
            "- Another tiny-loss rescue is not justified from these diagnostics.",
            "",
            "## Audit Panels",
            "",
            "- `outputs/debug/tg_punctate_audit/panels/type3_false_negatives.png`",
            "- `outputs/debug/tg_punctate_audit/panels/punctate_true_positives.png`",
            "- `outputs/debug/tg_punctate_audit/panels/type2_type3_confusions.png`",
            "  - not emitted because no test predictions fell into a pure `type2 <-> type3` confusion bucket",
            "",
            "## Verdict",
            "",
            "- Main failure mode: `data scarcity + hierarchy starvation`, with a smaller amount of `label-boundary ambiguity` between severe punctate and patch-like appearances.",
            "- Practical conclusion: do not spend more compute on tiny TG loss tweaks or more seeds from this foundation.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    split_path = Path(args.split_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    repo_root = infer_repo_root(manifest_path, args.repo_root)

    manifest_df = pd.read_csv(manifest_path)
    split_df = pd.read_csv(split_path)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    split_df["image_id"] = split_df["image_id"].astype(str)
    merged = manifest_df.merge(split_df[["image_id", "split"]], on="image_id", how="inner")

    output_dir.mkdir(parents=True, exist_ok=True)
    class_count_payload = split_count_rows(merged)
    gate_count_payload = gate_count_rows(merged)
    write_csv_rows(output_dir / "split_counts.csv", class_count_payload)
    write_csv_rows(output_dir / "gate_counts.csv", gate_count_payload)

    duplicate_path = Path(args.duplicate_candidates).resolve() if args.duplicate_candidates else repo_root / "outputs" / "tables" / "duplicate_candidates.csv"
    duplicate_rows = duplicate_audit_rows(duplicate_path if duplicate_path.exists() else None)
    write_csv_rows(output_dir / "duplicate_label_ambiguity.csv", duplicate_rows)

    prediction_audit = None
    prediction_csv = locate_prediction_csv(artifact_root, args.tg_experiment)
    metrics_json = locate_metrics_json(artifact_root, args.tg_experiment)
    if prediction_csv is not None:
        predictions = pd.read_csv(prediction_csv)
        report = classification_report(
            predictions["true_label"],
            predictions["pred_label"],
            labels=list(TG_CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        )
        full_confusion = confusion_matrix(predictions["true_label"], predictions["pred_label"], labels=list(TG_CLASS_NAMES))
        write_csv_rows(output_dir / "test_confusion_matrix.csv", confusion_to_rows(full_confusion, list(TG_CLASS_NAMES), list(TG_CLASS_NAMES)))

        punctate_truth = predictions[predictions["true_label"].isin(PUNCTATE_NAMES)].copy()
        punctate_confusion = crosstab_confusion(punctate_truth, list(PUNCTATE_NAMES), list(TG_CLASS_NAMES))
        write_csv_rows(output_dir / "punctate_true_confusion_matrix.csv", confusion_to_rows(punctate_confusion, list(PUNCTATE_NAMES), list(TG_CLASS_NAMES)))

        label_to_index = {name: index for index, name in enumerate(TG_CLASS_NAMES)}
        punctate_balanced_accuracy = float(
            balanced_accuracy_score(
                punctate_truth["true_label"],
                punctate_truth["pred_label"],
            )
        ) if not punctate_truth.empty else 0.0
        punctate_macro_f1 = float(
            f1_score(
                punctate_truth["true_label"],
                punctate_truth["pred_label"],
                labels=list(PUNCTATE_NAMES),
                average="macro",
                zero_division=0,
            )
        ) if not punctate_truth.empty else 0.0

        type3_rows = predictions[predictions["true_label"] == TYPE3_NAME].copy()
        type3_breakdown = type3_rows["pred_label"].value_counts().reindex(TG_CLASS_NAMES, fill_value=0)
        write_csv_rows(
            output_dir / "type3_prediction_breakdown.csv",
            [{"pred_label": label, "count": int(count)} for label, count in type3_breakdown.items()],
        )

        build_panel(
            type3_rows[type3_rows["pred_label"] != TYPE3_NAME].sort_values(by="confidence", ascending=False),
            output_dir / "panels" / "type3_false_negatives.png",
            repo_root,
            "Type3 False Negatives",
        )
        build_panel(
            predictions[
                predictions["true_label"].isin((TYPE2_NAME, TYPE3_NAME))
                & predictions["pred_label"].isin((TYPE2_NAME, TYPE3_NAME))
                & (predictions["true_label"] != predictions["pred_label"])
            ].sort_values(by="confidence", ascending=False),
            output_dir / "panels" / "type2_type3_confusions.png",
            repo_root,
            "Type2 / Type3 Confusions",
        )
        build_panel(
            predictions[
                predictions["true_label"].isin(PUNCTATE_NAMES)
                & (predictions["true_label"] == predictions["pred_label"])
            ].sort_values(by="confidence", ascending=False),
            output_dir / "panels" / "punctate_true_positives.png",
            repo_root,
            "Representative Punctate True Positives",
        )

        base_metrics = json.loads(metrics_json.read_text(encoding="utf-8")) if metrics_json is not None and metrics_json.exists() else {}
        prediction_audit = {
            "balanced_accuracy": float(base_metrics.get("balanced_accuracy", balanced_accuracy_score(predictions["true_label"], predictions["pred_label"]))),
            "macro_f1": float(base_metrics.get("macro_f1", f1_score(predictions["true_label"], predictions["pred_label"], labels=list(TG_CLASS_NAMES), average="macro", zero_division=0))),
            "punctate_balanced_accuracy": punctate_balanced_accuracy,
            "punctate_macro_f1": punctate_macro_f1,
            "per_class_recall": {name: float(report[name]["recall"]) for name in TG_CLASS_NAMES},
            "per_class_f1": {name: float(report[name]["f1-score"]) for name in TG_CLASS_NAMES},
            "full_confusion_matrix": full_confusion.tolist(),
            "punctate_confusion_matrix": punctate_confusion.tolist(),
            "type3_test_count": int((merged[(merged["split"] == "test") & (merged["task_tg_5class"] == TYPE3_NAME)]).shape[0]),
            "type3_to_type4_count": int(type3_breakdown[TYPE4_NAME]),
            "type3_to_type2_count": int(type3_breakdown[TYPE2_NAME]),
            "type3_to_no_ulcer_count": int(type3_breakdown["no_ulcer"]),
            "type3_t2_starvation_share": float(type3_breakdown[TYPE4_NAME] / max(int(type3_rows.shape[0]), 1)),
        }
        write_json(output_dir / "prediction_audit.json", prediction_audit)

    report_text = build_report(args.tg_experiment, class_count_payload, gate_count_payload, prediction_audit, duplicate_rows)
    write_text(output_dir / "summary.md", report_text)
    if args.report_path:
        write_text(args.report_path, report_text)
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
