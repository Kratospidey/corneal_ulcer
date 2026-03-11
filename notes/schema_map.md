# Schema Map

## Primary record structure

The dataset is keyed by numeric image ID.

Primary record:
- `rawImages/{id}.jpg`

Derived assets linked by the same `{id}`:
- `corneaLabels/{id}.png`
- `corneaOverlay/{id}.jpg`
- `ulcerLabels/{id}.png` for a subset only
- `ulcerOverlay/{id}.jpg` for the same subset only

## Spreadsheet schema

Source file:
- `data/raw/sustech_sysu/Category information.xlsx`

Observed workbook structure:
- `Sheet1`: populated
- `Sheet2`: empty
- `Sheet3`: empty

Observed `Sheet1` columns by actual cell position:

| Excel column | Header text | Meaning inferred from counts | Class counts |
|---|---|---|---|
| `A` | unlabeled in row 1 | Image filename | 712 unique filenames |
| `B` | `Category` | 3-class ulcer type / pattern | `358 / 263 / 91` |
| `C` | `Type` | 5-class severity | `36 / 98 / 203 / 273 / 102` |
| `D` | `Grade` | 5-class TG/type-grade | `36 / 78 / 40 / 10 / 548` |

Important note:
- The header text is not self-explanatory.
- Semantic meaning should be taken from paper-count alignment, not from column names alone.

## Reconstructed task map

### Task 1: 3-class ulcer type
- Source: spreadsheet column `Category`
- Values:
  - `0`: point-like corneal ulcer
  - `1`: point-flaky mixed corneal ulcer
  - `2`: flaky corneal ulcer

### Task 2: 5-class severity
- Source: spreadsheet column `Type`
- Values inferred from Diagnostics 2024:
  - `0`: no ulcer
  - `1`: ulcer area no more than 25% of cornea
  - `2`: ulcer area up to 50% of cornea
  - `3`: ulcer area at least 75% of cornea
  - `4`: ulcer surrounding the center of the cornea

### Task 3: 5-class TG / type-grade
- Source: spreadsheet column `Grade`
- Values inferred from Diagnostics 2024:
  - `0`: no ulcer
  - `1`: micro-punctate ulcers
  - `2`: macro-punctate ulcers
  - `3`: coalescent macro-punctate ulcers
  - `4`: ulcers with patch value greater than 1 mm

## Asset-availability map

Coverage by image ID:
- Raw image: 712 / 712
- Cornea mask: 712 / 712
- Cornea overlay: 712 / 712
- Ulcer mask: 354 / 712
- Ulcer overlay: 354 / 712

Observed relationship between `Category` and ulcer-mask presence:
- `Category = 0`: 358 images, all without ulcer masks.
- `Category = 1`: 263 images, all with ulcer masks.
- `Category = 2`: 91 images, all with ulcer masks.

This is a file-level observation from the shipped dataset. It should be treated as an empirical dataset property, not as a clinical rule.

## Practical mapping guidance

- Use `rawImages/{id}.jpg` as the canonical join key.
- Join spreadsheet labels by stripping `.jpg` from column `A`.
- Join masks and overlays by the same numeric stem.
- Keep cornea-wide assets separate from ulcer-subset assets in downstream manifests.
- Do not infer scenario availability from folder names alone; use the spreadsheet as the authoritative label source.
