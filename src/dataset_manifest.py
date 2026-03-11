from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile
import logging
import os

from utils_io import try_write_parquet, write_csv_rows


PATTERN_LABELS = {
    "0": "point_like",
    "1": "point_flaky_mixed",
    "2": "flaky",
}

SEVERITY_LABELS = {
    "0": "no_ulcer",
    "1": "ulcer_leq_25pct",
    "2": "ulcer_leq_50pct",
    "3": "ulcer_geq_75pct",
    "4": "central_ulcer",
}

TG_LABELS = {
    "0": "no_ulcer",
    "1": "micro_punctate",
    "2": "macro_punctate",
    "3": "coalescent_macro_punctate",
    "4": "patch_gt_1mm",
}


def parse_category_workbook(workbook_path: Path) -> list[dict[str, str]]:
    namespace = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(workbook_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared_strings.append("".join(node.text or "" for node in item.iterfind(".//a:t", namespace)))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
        sheets = workbook.find("a:sheets", namespace)
        if sheets is None or len(sheets) == 0:
            return []

        sheet = sheets[0]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        sheet_xml = ET.fromstring(archive.read(f"xl/{rel_map[rel_id]}"))
        rows = sheet_xml.findall(".//a:sheetData/a:row", namespace)
        parsed: list[dict[str, str]] = []
        for row in rows[1:]:
            record: dict[str, str] = {}
            for cell in row.findall("a:c", namespace):
                ref = cell.attrib["r"][0]
                value_node = cell.find("a:v", namespace)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                record[ref] = value
            if record:
                parsed.append(
                    {
                        "image_filename": record.get("A", ""),
                        "category_code": record.get("B", ""),
                        "type_code": record.get("C", ""),
                        "grade_code": record.get("D", ""),
                    }
                )
    return parsed


def _path_or_empty(path: Path) -> str:
    return str(path) if path.exists() else ""


def build_manifest(data_root: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    workbook_path = data_root / "Category information.xlsx"
    rows = parse_category_workbook(workbook_path)
    manifest: list[dict[str, Any]] = []
    for row in rows:
        filename = row["image_filename"]
        image_id = Path(filename).stem
        raw_image_path = data_root / "rawImages" / filename
        cornea_mask_path = data_root / "corneaLabels" / f"{image_id}.png"
        cornea_overlay_path = data_root / "corneaOverlay" / f"{image_id}.jpg"
        ulcer_mask_path = data_root / "ulcerLabels" / f"{image_id}.png"
        ulcer_overlay_path = data_root / "ulcerOverlay" / f"{image_id}.jpg"

        manifest.append(
            {
                "image_id": image_id,
                "image_filename": filename,
                "raw_image_path": _path_or_empty(raw_image_path),
                "cornea_mask_path": _path_or_empty(cornea_mask_path),
                "cornea_overlay_path": _path_or_empty(cornea_overlay_path),
                "ulcer_mask_path": _path_or_empty(ulcer_mask_path),
                "ulcer_overlay_path": _path_or_empty(ulcer_overlay_path),
                "has_raw_image": raw_image_path.exists(),
                "has_cornea_mask": cornea_mask_path.exists(),
                "has_cornea_overlay": cornea_overlay_path.exists(),
                "has_ulcer_mask": ulcer_mask_path.exists(),
                "has_ulcer_overlay": ulcer_overlay_path.exists(),
                "category_code": row["category_code"],
                "type_code": row["type_code"],
                "grade_code": row["grade_code"],
                "task_pattern_3class": PATTERN_LABELS.get(row["category_code"], "unknown"),
                "task_severity_5class": SEVERITY_LABELS.get(row["type_code"], "unknown"),
                "task_tg_5class": TG_LABELS.get(row["grade_code"], "unknown"),
                "filesize_bytes": raw_image_path.stat().st_size if raw_image_path.exists() else 0,
            }
        )

    logger.info("Built manifest with %d rows from %s", len(manifest), data_root)
    return manifest


def export_manifest(rows: list[dict[str, Any]], csv_path: Path, parquet_path: Path, logger: logging.Logger) -> None:
    write_csv_rows(csv_path, rows)
    try_write_parquet(parquet_path, rows, logger)


def label_distribution_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label_key in ("task_pattern_3class", "task_severity_5class", "task_tg_5class"):
        counter = Counter(str(row[label_key]) for row in manifest)
        for label, count in sorted(counter.items()):
            output.append({"task": label_key, "label": label, "count": count})
    return output


def audit_rows(manifest: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    counts = {
        "raw_images": len(list((data_root / "rawImages").glob("*"))),
        "cornea_masks": len(list((data_root / "corneaLabels").glob("*"))),
        "cornea_overlays": len(list((data_root / "corneaOverlay").glob("*"))),
        "ulcer_masks": len(list((data_root / "ulcerLabels").glob("*"))),
        "ulcer_overlays": len(list((data_root / "ulcerOverlay").glob("*"))),
        "manifest_rows": len(manifest),
        "missing_raw_image": sum(1 for row in manifest if not row["has_raw_image"]),
        "missing_cornea_mask": sum(1 for row in manifest if not row["has_cornea_mask"]),
        "missing_cornea_overlay": sum(1 for row in manifest if not row["has_cornea_overlay"]),
        "missing_ulcer_mask": sum(1 for row in manifest if not row["has_ulcer_mask"]),
        "missing_ulcer_overlay": sum(1 for row in manifest if not row["has_ulcer_overlay"]),
        "root_video_files": len(list(data_root.glob("*.mp4"))) + len(list(data_root.glob("*.wmv"))),
    }
    return [{"metric": key, "value": value} for key, value in counts.items()]


def manifest_support_summary(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_pattern_3class_supported": bool(Counter(row["task_pattern_3class"] for row in manifest)),
        "task_severity_5class_supported": bool(Counter(row["task_severity_5class"] for row in manifest)),
        "task_tg_5class_supported": bool(Counter(row["task_tg_5class"] for row in manifest)),
        "binary_supported": True,
        "segmentation_assisted_supported": any(bool(row["has_ulcer_mask"]) for row in manifest),
    }


def binary_label(row: dict[str, Any]) -> str:
    return "no_ulcer" if row["task_severity_5class"] == "no_ulcer" else "ulcer_present"


def add_binary_task(manifest: list[dict[str, Any]]) -> None:
    for row in manifest:
        row["task_binary"] = binary_label(row)


def manifest_overview_markdown(manifest: list[dict[str, Any]], audit: list[dict[str, Any]]) -> str:
    audit_map = {row["metric"]: row["value"] for row in audit}
    return "\n".join(
        [
            "# Dataset Audit Report",
            "",
            "## Inventory",
            "",
            f"- Manifest rows: {audit_map.get('manifest_rows', 0)}",
            f"- Raw images: {audit_map.get('raw_images', 0)}",
            f"- Cornea masks: {audit_map.get('cornea_masks', 0)}",
            f"- Cornea overlays: {audit_map.get('cornea_overlays', 0)}",
            f"- Ulcer masks: {audit_map.get('ulcer_masks', 0)}",
            f"- Ulcer overlays: {audit_map.get('ulcer_overlays', 0)}",
            f"- Root-level process videos: {audit_map.get('root_video_files', 0)}",
            "",
            "## Missingness",
            "",
            f"- Missing raw images: {audit_map.get('missing_raw_image', 0)}",
            f"- Missing cornea masks: {audit_map.get('missing_cornea_mask', 0)}",
            f"- Missing cornea overlays: {audit_map.get('missing_cornea_overlay', 0)}",
            f"- Missing ulcer masks: {audit_map.get('missing_ulcer_mask', 0)}",
            f"- Missing ulcer overlays: {audit_map.get('missing_ulcer_overlay', 0)}",
            "",
            "## Task Support",
            "",
            f"- Pattern 3-class task rows: {sum(1 for _ in manifest)}",
            f"- Binary task rows: {sum(1 for _ in manifest)}",
            f"- Segmentation-assisted rows: {sum(1 for row in manifest if row['has_ulcer_mask'])}",
        ]
    )


def folder_tree_rows(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in sorted(data_root.iterdir()):
        if child.is_dir():
            file_count = sum(1 for entry in child.iterdir() if entry.is_file())
            rows.append({"name": child.name, "kind": "directory", "file_count": file_count})
        else:
            rows.append(
                {
                    "name": child.name,
                    "kind": "file",
                    "file_count": 1,
                    "size_bytes": child.stat().st_size,
                    "extension": child.suffix.lower(),
                }
            )
    return rows
