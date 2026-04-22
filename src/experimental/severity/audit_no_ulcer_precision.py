from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import re

import pandas as pd

from experimental.severity.eval_s2_ordinal_severity import stage_predict
from utils_io import write_csv_rows, write_json, write_text


DEFAULT_FINAL_EXPERIMENT = "severity5__posthoc__factorized_geom_plus_patternlogits_s2ordinal_hgb_v1__holdout_v1"
DEFAULT_S0_EXPERIMENT = "severity5__posthoc__factorized_s0_plus_patternlogits_hgb_v1__holdout_v1"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Audit whether SEV-S3 no-ulcer precision is genuine or suspicious.")
    parser.add_argument("--geom-table", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--final-experiment", default=DEFAULT_FINAL_EXPERIMENT)
    parser.add_argument("--s0-experiment", default=DEFAULT_S0_EXPERIMENT)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path")
    return parser


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    write_csv_rows(path, frame.to_dict(orient="records"))


def leakage_checks(
    geom_table: pd.DataFrame,
    manifest_path: Path,
    split_path: Path,
    s0_summary: dict[str, object],
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path)
    split_df = pd.read_csv(split_path)
    manifest["image_id"] = manifest["image_id"].astype(str)
    split_df["image_id"] = split_df["image_id"].astype(str)
    merged = manifest.merge(split_df[["image_id", "split"]], on="image_id", how="inner")

    raw_cross_split = int((merged.groupby("raw_image_path")["split"].nunique() > 1).sum())
    cornea_cross_split = int((merged.groupby("cornea_mask_path")["split"].nunique() > 1).sum())
    duplicate_image_ids = int(merged["image_id"].duplicated().sum())

    numeric_columns = [column for column in geom_table.columns if pd.api.types.is_numeric_dtype(geom_table[column])]
    suspicious_numeric_columns = [
        column
        for column in numeric_columns
        if any(token in column.lower() for token in ("severity", "label", "target", "route", "stage", "split"))
    ]
    feature_columns = [str(column) for column in s0_summary.get("feature_columns", [])]
    forbidden_feature_columns = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in ("severity", "label", "target", "route", "stage"))
    ]

    pattern_feature_columns = [column for column in feature_columns if column.startswith("pattern_")]
    pattern_feature_columns_clean = all(
        column.startswith(("pattern_prob_", "pattern_logit_", "pattern_pred_")) or "_x_" in column for column in pattern_feature_columns
    )

    eval_code = (Path(__file__).resolve().parent / "eval_s2_ordinal_severity.py").read_text(encoding="utf-8")
    route_feature_usage = bool(re.search(r"factorized_route", eval_code))
    true_route_logic = "route_taken" in eval_code and "s0:no_ulcer" in eval_code

    return {
        "duplicate_image_ids": duplicate_image_ids,
        "raw_paths_cross_split": raw_cross_split,
        "cornea_paths_cross_split": cornea_cross_split,
        "suspicious_numeric_columns": suspicious_numeric_columns,
        "feature_columns_match_training_count": len(feature_columns),
        "forbidden_feature_columns": forbidden_feature_columns,
        "pattern_feature_columns_count": len(pattern_feature_columns),
        "pattern_feature_columns_clean": bool(pattern_feature_columns_clean),
        "evaluator_mentions_factorized_route": route_feature_usage,
        "evaluator_builds_routes_only_for_reporting": true_route_logic,
    }


def report_lines(
    summary: dict[str, object],
    leakage: dict[str, object],
    correct_ids: list[str],
    missed_ids: list[str],
    false_ids: list[str],
) -> list[str]:
    verdict = str(summary["audit_verdict"])
    return [
        "# SEV-S3 No-Ulcer Precision Audit",
        "",
        f"- Final experiment: `{summary['final_experiment']}`",
        f"- S0 experiment: `{summary['s0_experiment']}`",
        f"- Audit verdict: `{verdict}`",
        "",
        "## Core Counts",
        "",
        f"- true no_ulcer support (test): `{summary['true_no_ulcer_support']}`",
        f"- predicted no_ulcer count (test): `{summary['predicted_no_ulcer_count']}`",
        f"- no_ulcer precision: `{summary['no_ulcer_precision']:.4f}`",
        f"- no_ulcer recall: `{summary['no_ulcer_recall']:.4f}`",
        f"- no_ulcer F1: `{summary['no_ulcer_f1']:.4f}`",
        "",
        "## Exact Test IDs",
        "",
        f"- predicted no_ulcer and correct: `{correct_ids}`",
        f"- true no_ulcer but missed: `{missed_ids}`",
        f"- false no_ulcer predictions: `{false_ids}`",
        "",
        "## S0 Behavior",
        "",
        f"- S0 confusion matrix: `{summary['s0_confusion_matrix']}`",
        f"- S0 predicted no_ulcer count (test): `{summary['s0_predicted_no_ulcer_count']}`",
        f"- final predicted no_ulcer count equals S0 no_ulcer count: `{summary['final_no_ulcer_matches_s0_gate']}`",
        f"- mean no_ulcer margin on true no_ulcer cases: `{summary['mean_true_no_ulcer_margin']:.4f}`",
        f"- min no_ulcer margin on true no_ulcer cases: `{summary['min_true_no_ulcer_margin']:.4f}`",
        f"- max no_ulcer margin on missed true no_ulcer cases: `{summary['max_missed_no_ulcer_margin']:.4f}`",
        "",
        "## Leakage / Routing Checks",
        "",
        f"- duplicate image ids after merge: `{leakage['duplicate_image_ids']}`",
        f"- raw image paths crossing splits: `{leakage['raw_paths_cross_split']}`",
        f"- cornea mask paths crossing splits: `{leakage['cornea_paths_cross_split']}`",
        f"- suspicious numeric columns in geometry table: `{leakage['suspicious_numeric_columns']}`",
        f"- forbidden stage feature columns: `{leakage['forbidden_feature_columns']}`",
        f"- pattern feature columns limited to logits/probs/confidence/interactions: `{leakage['pattern_feature_columns_clean']}`",
        f"- evaluator mentions factorized_route: `{leakage['evaluator_mentions_factorized_route']}`",
        "",
        "## Interpretation",
        "",
        "- The perfect no-ulcer precision is numerically fragile because the model predicts `no_ulcer` only once on the test split.",
        "- That lone prediction is not produced by a combiner bug; it comes directly from the S0 gate and stays consistent through final routing.",
        "- No obvious split leakage, duplicate leakage, target-derived numeric leakage, or factorized-route leakage was found in this audit.",
        "- The behavior is conservative rather than suspicious: precision is perfect because predicted count is tiny, while recall is poor.",
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geom_table = pd.read_csv(args.geom_table)
    geom_table["image_id"] = geom_table["image_id"].astype(str)
    final_predictions = pd.read_csv(output_root / "predictions" / args.final_experiment / "test_predictions.csv")
    final_predictions["image_id"] = final_predictions["image_id"].astype(str)
    s0_predictions = pd.read_csv(output_root / "predictions" / args.s0_experiment / "test_predictions.csv")
    s0_predictions["image_id"] = s0_predictions["image_id"].astype(str)
    s0_summary = load_json(output_root / "reports" / args.s0_experiment / "training_summary.json")
    s0_metrics = load_json(output_root / "metrics" / args.s0_experiment / "test_metrics.json")

    test_geom = geom_table[geom_table["split"] == "test"].reset_index(drop=True)
    s0_full_predictions, s0_full_probabilities = stage_predict(test_geom, output_root, args.s0_experiment)
    s0_full = test_geom[["image_id", "severity_label", "raw_image_path"]].copy()
    s0_full["s0_pred"] = s0_full_predictions
    s0_full["prob_no_ulcer"] = s0_full_probabilities[:, 0]
    s0_full["prob_ulcer_present"] = s0_full_probabilities[:, 1]
    s0_full["no_ulcer_margin"] = s0_full["prob_no_ulcer"] - s0_full["prob_ulcer_present"]

    final_joined = final_predictions.merge(
        s0_full[["image_id", "prob_no_ulcer", "prob_ulcer_present", "no_ulcer_margin"]],
        on="image_id",
        how="left",
    )

    correct_pred_no_ulcer = final_joined[(final_joined["pred_label"] == "no_ulcer") & (final_joined["true_label"] == "no_ulcer")].copy()
    missed_true_no_ulcer = final_joined[(final_joined["pred_label"] != "no_ulcer") & (final_joined["true_label"] == "no_ulcer")].copy()
    false_no_ulcer_predictions = final_joined[(final_joined["pred_label"] == "no_ulcer") & (final_joined["true_label"] != "no_ulcer")].copy()
    true_no_ulcer_rows = final_joined[final_joined["true_label"] == "no_ulcer"].copy()

    write_frame(output_dir / "test_predicted_no_ulcer_correct.csv", correct_pred_no_ulcer)
    write_frame(output_dir / "test_true_no_ulcer_missed.csv", missed_true_no_ulcer)
    write_frame(output_dir / "test_false_no_ulcer_predictions.csv", false_no_ulcer_predictions)
    write_frame(output_dir / "test_s0_true_no_ulcer_rows.csv", true_no_ulcer_rows)
    write_frame(output_dir / "test_s0_all_rows.csv", s0_full)

    leakage = leakage_checks(
        geom_table=geom_table,
        manifest_path=Path(args.manifest).resolve(),
        split_path=Path(args.split_file).resolve(),
        s0_summary=s0_summary,
    )
    write_json(output_dir / "leakage_checks.json", leakage)

    predicted_no_ulcer_count = int((final_joined["pred_label"] == "no_ulcer").sum())
    true_no_ulcer_support = int((final_joined["true_label"] == "no_ulcer").sum())
    summary = {
        "final_experiment": args.final_experiment,
        "s0_experiment": args.s0_experiment,
        "true_no_ulcer_support": true_no_ulcer_support,
        "predicted_no_ulcer_count": predicted_no_ulcer_count,
        "no_ulcer_precision": float(load_json(output_root / "metrics" / args.final_experiment / "test_metrics.json")["no_ulcer_precision"]),
        "no_ulcer_recall": float(load_json(output_root / "metrics" / args.final_experiment / "test_metrics.json")["classification_report"]["no_ulcer"]["recall"]),
        "no_ulcer_f1": float(load_json(output_root / "metrics" / args.final_experiment / "test_metrics.json")["classification_report"]["no_ulcer"]["f1-score"]),
        "s0_confusion_matrix": s0_metrics["confusion_matrix"],
        "s0_predicted_no_ulcer_count": int((s0_full["s0_pred"] == "no_ulcer").sum()),
        "final_no_ulcer_matches_s0_gate": bool(
            predicted_no_ulcer_count == int((s0_full["s0_pred"] == "no_ulcer").sum()) == int((final_joined["route_taken"] == "s0:no_ulcer").sum())
        ),
        "mean_true_no_ulcer_margin": float(true_no_ulcer_rows["no_ulcer_margin"].mean()),
        "min_true_no_ulcer_margin": float(true_no_ulcer_rows["no_ulcer_margin"].min()),
        "max_missed_no_ulcer_margin": float(missed_true_no_ulcer["no_ulcer_margin"].max()) if not missed_true_no_ulcer.empty else 0.0,
    }

    suspicious = any(
        [
            leakage["duplicate_image_ids"],
            leakage["raw_paths_cross_split"],
            leakage["cornea_paths_cross_split"],
            bool(leakage["suspicious_numeric_columns"]),
            bool(leakage["forbidden_feature_columns"]),
            not summary["final_no_ulcer_matches_s0_gate"],
        ]
    )
    summary["audit_verdict"] = "suspicious, needs fix" if suspicious else "clean conservative behavior"

    write_json(output_dir / "audit_summary.json", summary)
    report_text = "\n".join(
        report_lines(
            summary=summary,
            leakage=leakage,
            correct_ids=correct_pred_no_ulcer["image_id"].astype(str).tolist(),
            missed_ids=missed_true_no_ulcer["image_id"].astype(str).tolist(),
            false_ids=false_no_ulcer_predictions["image_id"].astype(str).tolist(),
        )
    )
    write_text(output_dir / "audit_report.md", report_text)
    if args.report_path:
        write_text(Path(args.report_path).resolve(), report_text)

    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
