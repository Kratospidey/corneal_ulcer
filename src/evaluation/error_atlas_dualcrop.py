import csv
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import os

from utils_io import safe_open_image
from utils_preprocessing import apply_variant
from config_utils import write_text

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create a dual-crop error atlas comparing official and w0035 predictions.")
    parser.add_argument("--official-csv", required=True)
    parser.add_argument("--w0035-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--thumbnail-dir", required=True)
    parser.add_argument("--xai-manifest")
    return parser

def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def _make_thumbnails(
    row: dict[str, str],
    thumbnail_dir: Path,
    image_id: str,
) -> tuple[str, str]:
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    
    current_path = thumbnail_dir / f"{image_id}_current.png"
    wide_path = thumbnail_dir / f"{image_id}_wide.png"
    
    raw_path = Path(row["raw_image_path"])
    if not raw_path.exists():
        return "", ""
        
    image = safe_open_image(raw_path)
    cornea_mask_path = row.get("cornea_mask_path", "")
    cornea_mask = safe_open_image(Path(cornea_mask_path)) if cornea_mask_path and cornea_mask_path != "nan" else None
    
    # Current crop
    if not current_path.exists():
        current_crop = apply_variant(image, "cornea_crop_scale_v1", cornea_mask=cornea_mask)
        current_thumb = current_crop.copy()
        current_thumb.thumbnail((224, 224))
        current_thumb.save(current_path)
        
    # Wide crop
    if not wide_path.exists():
        wide_crop = apply_variant(image, "cornea_crop_wide_context_v1", cornea_mask=cornea_mask)
        wide_thumb = wide_crop.copy()
        wide_thumb.thumbnail((224, 224))
        wide_thumb.save(wide_path)
        
    return str(current_path), str(wide_path)

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    official_rows = {r["image_id"]: r for r in _read_rows(args.official_csv)}
    w0035_rows = {r["image_id"]: r for r in _read_rows(args.w0035_csv)}
    
    xai_map = {}
    if args.xai_manifest and Path(args.xai_manifest).exists():
        with open(args.xai_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            for item in manifest:
                xai_map[item["image_id"]] = item["gradcam_path"]
                
    common_ids = sorted(set(official_rows.keys()) & set(w0035_rows.keys()), key=lambda x: int(x))
    
    out_rows = []
    
    for image_id in common_ids:
        o_row = official_rows[image_id]
        w_row = w0035_rows[image_id]
        
        true_label = o_row["true_label"]
        o_pred = o_row["pred_label"]
        w_pred = w_row["pred_label"]
        o_correct = (o_pred == true_label)
        w_correct = (w_pred == true_label)
        
        groups = []
        if o_correct and not w_correct:
            groups.append("official_correct_w0035_wrong")
        if w_correct and not o_correct:
            groups.append("w0035_correct_official_wrong")
        if not o_correct and not w_correct:
            groups.append("both_wrong")
            
        if not w_correct:
            if true_label == "flaky":
                groups.append("true_flaky_missed_by_w0035")
            elif true_label == "point_flaky_mixed":
                groups.append("true_point_flaky_mixed_missed_by_w0035")
            elif true_label == "point_like":
                groups.append("true_point_like_missed_by_w0035")
                
            w_conf = float(w_row["confidence"])
            if w_conf < 0.5:
                groups.append("w0035_wrong_low_confidence")
            if o_correct:
                groups.append("w0035_wrong_official_correct")
        
        if not groups:
            continue
            
        group_str = ";".join(groups)
        
        # Prob vectors
        o_probs = f"[{o_row.get('prob_point_like', '0')}, {o_row.get('prob_point_flaky_mixed', '0')}, {o_row.get('prob_flaky', '0')}]"
        w_probs = f"[{w_row.get('prob_point_like', '0')}, {w_row.get('prob_point_flaky_mixed', '0')}, {w_row.get('prob_flaky', '0')}]"
        
        curr_thumb, wide_thumb = _make_thumbnails(o_row, Path(args.thumbnail_dir), image_id)
        
        out_rows.append({
            "image_id": image_id,
            "groups": group_str,
            "true_label": true_label,
            "official_pred": o_pred,
            "official_confidence": o_row["confidence"],
            "w0035_pred": w_pred,
            "w0035_confidence": w_row["confidence"],
            "official_probs": o_probs,
            "w0035_probs": w_probs,
            "raw_image_path": o_row["raw_image_path"],
            "cornea_mask_path": o_row.get("cornea_mask_path", ""),
            "current_crop_thumbnail": curr_thumb,
            "wide_crop_thumbnail": wide_thumb,
            "gradcam_path": xai_map.get(image_id, ""),
            "notes": ""
        })
        
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader()
            writer.writerows(out_rows)
            
    md_lines = ["# Dual Crop Error Atlas: Official vs w0035", ""]
    
    categories = [
        "official_correct_w0035_wrong",
        "w0035_correct_official_wrong",
        "both_wrong",
        "true_flaky_missed_by_w0035",
        "true_point_flaky_mixed_missed_by_w0035",
        "true_point_like_missed_by_w0035",
        "w0035_wrong_low_confidence",
        "w0035_wrong_official_correct"
    ]
    
    for cat in categories:
        cat_rows = [r for r in out_rows if cat in r["groups"].split(";")]
        md_lines.append(f"## {cat} ({len(cat_rows)} cases)")
        if cat_rows:
            md_lines.append("| Image ID | True Label | Off. Pred | Off. Conf | w0035 Pred | w0035 Conf | Current Crop | Wide Crop | XAI |")
            md_lines.append("|---|---|---|---|---|---|---|---|---|")
            for r in cat_rows:
                thumb1 = f"<img src='../../../../{r['current_crop_thumbnail']}' width='120'/>" if r['current_crop_thumbnail'] else ""
                thumb2 = f"<img src='../../../../{r['wide_crop_thumbnail']}' width='120'/>" if r['wide_crop_thumbnail'] else ""
                xai = f"<img src='../../../../{r['gradcam_path']}' width='120'/>" if r['gradcam_path'] else ""
                md_lines.append(f"| {r['image_id']} | {r['true_label']} | {r['official_pred']} | {float(r['official_confidence']):.3f} | {r['w0035_pred']} | {float(r['w0035_confidence']):.3f} | {thumb1} | {thumb2} | {xai} |")
        md_lines.append("")
        
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    
    print(f"Generated dual crop error atlas with {len(out_rows)} interesting cases.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
