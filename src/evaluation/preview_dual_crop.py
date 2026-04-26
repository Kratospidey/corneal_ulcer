import csv
from argparse import ArgumentParser
from pathlib import Path
from utils_io import safe_open_image
from utils_preprocessing import apply_variant

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Preview dual crop variants.")
    parser.add_argument("--error-atlas-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser

def _make_preview(row, output_dir: Path):
    raw_path = Path(row["raw_image_path"])
    if not raw_path.exists():
        return "", "", ""
        
    image_id = row["image_id"]
    image = safe_open_image(raw_path)
    cornea_mask_path = row.get("cornea_mask_path", "")
    cornea_mask = safe_open_image(Path(cornea_mask_path)) if cornea_mask_path and cornea_mask_path != "nan" else None
    
    # Save original thumbnail
    raw_thumb = image.copy()
    raw_thumb.thumbnail((224, 224))
    raw_thumb_path = output_dir / f"{image_id}_raw.png"
    raw_thumb.save(raw_thumb_path)
    
    # Save tight crop
    tight_crop = apply_variant(image, "cornea_crop_scale_v1", cornea_mask=cornea_mask)
    tight_thumb = tight_crop.copy()
    tight_thumb.thumbnail((224, 224))
    tight_thumb_path = output_dir / f"{image_id}_tight.png"
    tight_thumb.save(tight_thumb_path)
    
    # Save wide crop
    wide_crop = apply_variant(image, "cornea_crop_wide_context_v1", cornea_mask=cornea_mask)
    wide_thumb = wide_crop.copy()
    wide_thumb.thumbnail((224, 224))
    wide_thumb_path = output_dir / f"{image_id}_wide.png"
    wide_thumb.save(wide_thumb_path)
    
    return str(raw_thumb_path), str(tight_thumb_path), str(wide_thumb_path)

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    with open(args.error_atlas_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    # Select cases: all w0035 missed flaky, all w0035 missed point_flaky_mixed
    selected = [r for r in rows if "true_flaky_missed_by_w0035" in r["groups"] or "true_point_flaky_mixed_missed_by_w0035" in r["groups"]]
    # A few correct examples per class: we didn't add correct cases to error_atlas... let's just use what's available
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    md_lines = ["# Dual Crop Preview", "", "| Image ID | True Label | Pred Label | Raw | Tight (Scale V1) | Wide (Context V1) |", "|---|---|---|---|---|---|"]
    for r in selected:
        raw_p, tight_p, wide_p = _make_preview(r, output_dir)
        raw_img = f"<img src='../../../../{raw_p}' width='120'/>" if raw_p else ""
        tight_img = f"<img src='../../../../{tight_p}' width='120'/>" if tight_p else ""
        wide_img = f"<img src='../../../../{wide_p}' width='120'/>" if wide_p else ""
        md_lines.append(f"| {r['image_id']} | {r['true_label']} | {r['w0035_pred']} | {raw_img} | {tight_img} | {wide_img} |")
        
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated preview with {len(selected)} cases.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
