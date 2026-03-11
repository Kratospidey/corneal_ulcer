# Dataset Manifest Summary

## Root inventory

Dataset root: `data/raw/sustech_sysu`

Files present at root:
- `Category information.xlsx`
- `automatic extraction.mp4`
- `label inspection.mp4`
- `manual correction.wmv`

Directories present:
- `rawImages`
- `corneaLabels`
- `corneaOverlay`
- `ulcerLabels`
- `ulcerOverlay`

## File inventory

| Asset | Count | Extension | Notes |
|---|---:|---|---|
| Raw images | 712 | `.jpg` | Primary image records |
| Cornea masks | 712 | `.png` | One per raw image |
| Cornea overlays | 712 | `.jpg` | One per raw image |
| Ulcer masks | 354 | `.png` | Present for a subset only |
| Ulcer overlays | 354 | `.jpg` | Same subset as ulcer masks |
| Spreadsheet metadata | 1 | `.xlsx` | Only structured label file found |
| Videos | 3 | `.mp4`, `.wmv` | Process/demo assets, not tabular labels |

## Naming and linkage observations
- Raw image IDs are contiguous from `1.jpg` through `712.jpg`.
- `corneaLabels` and `corneaOverlay` cover all 712 raw image IDs with no missing pairings.
- `ulcerLabels` and `ulcerOverlay` cover the same 354-image subset.
- No official split files were found under `data/raw/sustech_sysu` or elsewhere in `data/`.
- No README, CSV manifest, JSON metadata, YAML split definition, or extra label text file was found alongside the raw dataset.

## Spreadsheet-backed label availability

The only populated sheet in `Category information.xlsx` is `Sheet1`, with 712 image rows plus one header row. It provides one filename column and three label columns that can be aligned to the Diagnostics 2024 scenarios by count.

Observed label counts:
- `Category`: `358 / 263 / 91`
- `Type`: `36 / 98 / 203 / 273 / 102`
- `Grade`: `36 / 78 / 40 / 10 / 548`

These counts match the paper-reported class distributions for:
- 3-class ulcer type.
- 5-class severity.
- 5-class TG/type-grade.

## Verified task support

- 3-class ulcer pattern classification: supported.
  - Reconstructable from spreadsheet column `Category`.
- 5-class TG/type-grade analysis: supported.
  - Reconstructable from spreadsheet column `Grade`.
- 5-class severity analysis: supported.
  - Reconstructable from spreadsheet column `Type`.
- Segmentation / mask-based study: partially supported.
  - Cornea masks exist for all images, but ulcer masks exist for only 354 of 712 images.
- Binary ulcer-vs-non-ulcer study: derivable.
  - Can be inferred from `Category` or from ulcer-mask presence, but is not shipped as a dedicated standalone split or label file.

## Missing or inconsistent items
- No official train/validation/test split definition is provided.
- No patient IDs or encounter IDs are present.
- The spreadsheet header names are semantically confusing relative to the Diagnostics paper:
  - `Type` aligns with the paper’s severity counts.
  - `Grade` aligns with the paper’s TG/type-grade counts.
- Folder names alone do not reveal the paper’s three scenario tasks; they must be reconstructed from the spreadsheet.
- Ulcer-mask availability is incomplete and label-correlated, so segmentation-derived analyses cannot be assumed to generalize to the full 712-image cohort.
