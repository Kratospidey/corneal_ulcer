# External Signal Dataset Audit

Date: `2026-04-23`

## Scope

Audited the real accessibility and practical usability of the two requested public slit-lamp datasets:

- `SLID`
- `SLIT-Net`

This audit distinguishes:

- what is publicly reachable now
- what is already present locally
- what current-branch code can reuse cleanly
- what is not trustworthy enough to use as the base external-signal run

## Summary Verdict

### `SLID`

Status:

- practically usable
- best base dataset for the first external-signal run

Why:

- public GitHub repository is reachable
- official `Annotations.csv` is reachable
- official image zip is already present locally
- local prepared cornea-mask artifacts are internally consistent
- the simplest honest upstream task is binary cornea segmentation from the verified cornea annotations

Important caveat:

- the current local prepared manifest is not directly consumable as-is because its `raw_image_path` values point to an extracted image directory that is not currently present on disk
- this is repairable because every manifest image path maps cleanly to an image inside the local zip

### `SLIT-Net`

Status:

- code is publicly reachable
- dataset is not currently verified as downloadable in practice
- do not use for the base run

Why:

- public GitHub repository is reachable
- repo README points to Duke Box dataset and model download links
- on `2026-04-23`, the advertised Box links returned HTTP `404` when probed directly
- repo contents do not include the actual dataset; they only include code plus README stubs pointing back to Box

Consequence:

- `SLIT-Net` remains conditional only
- it should not delay or bloat the base `SLID` warm-start experiment

## Dataset 1: `SLID`

### Verified Public Access

Public sources observed:

- GitHub repo: `https://github.com/xumingyu-hub/SLID`
- reachable raw annotation CSV:
  - `https://raw.githubusercontent.com/xumingyu-hub/SLID/main/Annotations.csv`

Repo observations:

- repo is public
- repo exposes:
  - `Annotations.csv`
  - `Original_Slit-lamp_Images.zip`
- no explicit `LICENSE` file was observed on the repo landing page during audit

Practical access conclusion:

- public access is real
- usage / redistribution terms are not fully explicit from the repo listing alone
- safest interpretation is research-use public data with citation required, but not a clearly licensed redistribution package

### What Is Actually Present Locally

Local source artifact:

- `data/external/slid/Original_Slit-lamp_Images.zip`

Integrity:

- file size: `720,857,744` bytes
- SHA-256:
  - `cf74278dbd0c1782e58db834e8826a87f4038c007a5a4133b26131cf1b2a9607`

Zip contents:

- total entries: `5236`
- real PNG images inside `Original_Slit-lamp_Images/`: `2617`
- no CSV, README, or license file inside the zip itself

### Official Annotation Structure

Verified from the public `Annotations.csv`:

- annotation rows: `10204`
- unique image files: `2617`

Region annotations observed:

- empty region field: `2712`
- `Conjunctiva`: `2616`
- `Cornea`: `2573`
- `Pupil`: `2303`

Non-empty lesion labels observed:

- `Corneal / Conjunctival tumor`: `537`
- `Pigmented nevus`: `500`
- `Pinguecula`: `422`
- `Conjunctival injection`: `307`
- `Corneal dystrophy`: `307`
- `Subconjunctival hemorrhage`: `301`
- `Keratitis`: `270`
- `Cataract`: `221`
- `Pterygium`: `164`
- `Conjunctival cyst`: `134`
- `Intraocular lens`: `119`
- `Corneal scarring`: `77`
- `Lens dislocation`: `36`
- `Lens dislocation/Cataract`: `4`

Important alignment fact:

- files with a `Cornea` annotation: `2573`
- this exactly matches the locally prepared cornea-mask count

### Recommended Role In This Repo

Use `SLID` for:

- upstream slit-lamp anatomy supervision
- specifically binary cornea segmentation pretraining

Do not use `SLID` in the base run for:

- direct remapping to `pattern_3class`
- lesion multi-task expansion
- derived weak-label experiments as the primary claim

Reason:

- the cornea annotations are the cleanest, best-supported, and already locally prepared signal
- lesion labels are heterogeneous and farther from the repo's frozen pattern target

## Reused Local `SLID` Artifacts

### 1. Source image zip

Path:

- `data/external/slid/Original_Slit-lamp_Images.zip`

Expected contents:

- official slit-lamp PNG images from the public GitHub repo

Integrity result:

- zip readable
- `2617` PNG images found
- local manifest image IDs are all recoverable from the zip

Reuse decision:

- reuse
- but current-branch code must either extract the zip or use a zip-aware image-loading repair step before training

### 2. Prepared manifest

Path:

- `data/interim/slid/manifest.csv`

Expected schema:

- `image_id`
- `image_filename`
- `raw_image_path`
- `cornea_mask_path`
- `cornea_overlay_path`
- `ulcer_mask_path`
- `ulcer_overlay_path`
- `has_raw_image`
- `has_cornea_mask`
- `has_cornea_overlay`
- `has_ulcer_mask`
- `has_ulcer_overlay`
- `task_slid_cornea_pretrain`
- `task_slid_keratitis_binary_pretrain`
- `task_binary`

Observed counts:

- rows: `2573`
- unique image IDs: `2573`
- cornea-pretrain rows: `2573`
- derived keratitis positives: `222`
- derived keratitis negatives: `2351`

Integrity result:

- all `2573` `cornea_mask_path` files exist
- `raw_image_path` existence check currently fails for all rows because the extracted image directory is missing
- all `2573` manifest image paths map to real files inside the local zip
- `has_raw_image=True` in the manifest is therefore stale relative to the current filesystem state

Reuse decision:

- reuse with repair
- not directly consumable by current-branch code until raw-image extraction or equivalent path repair is done

### 3. Prepared cornea masks

Path:

- `data/interim/slid/cornea_masks/`

Expected contents:

- one binary cornea mask PNG per retained image

Observed counts:

- mask PNGs: `2573`

Integrity result:

- all sampled masks opened successfully
- sampled masks were grayscale `L`
- sampled masks contained only `0` and `255`
- mask filename stems match manifest image filename stems exactly

Reuse decision:

- reuse
- this is the strongest existing local artifact for the base upstream task

### 4. Duplicate candidates

Path:

- `data/interim/slid/duplicate_candidates.csv`

Observed contents:

- zero rows

Integrity result:

- file exists but contains no grouped duplicate candidates

Implication:

- no patient-level grouping is currently available from the prepared artifacts
- current split grouping effectively falls back to image-level grouping

### 5. Prepared split files

Paths:

- `data/interim/slid/split_files/slid_cornea_pretrain_holdout.csv`
- `data/interim/slid/split_files/slid_cornea_pretrain_repeated_cv.csv`
- `data/interim/slid/split_files/slid_keratitis_binary_pretrain_holdout.csv`
- `data/interim/slid/split_files/slid_keratitis_binary_pretrain_repeated_cv.csv`

Observed holdout split counts:

- `slid_cornea_pretrain_holdout.csv`
  - train: `1801`
  - val: `386`
  - test: `386`
- `slid_keratitis_binary_pretrain_holdout.csv`
  - train: `1801`
  - val: `386`
  - test: `386`

Integrity result:

- split files are readable
- schema is consistent:
  - `image_id`
  - `group_id`
  - `task_name`
  - `label`
  - `split`
- current `group_id` appears to equal `image_id`, so no stronger grouping signal is present

Reuse decision:

- reusable if we continue with image-level grouping assumptions
- not sufficient for any patient-level generalization claim

## `SLID` Gaps And Mismatch Notes

### Missing extracted raw image directory

Manifest points to:

- `data/external/slid/Original_Slit-lamp_Images/<id>.png`

Current state:

- directory missing on disk
- images exist only inside the zip

Impact:

- current branch cannot consume the prepared manifest directly

Repair options:

- extract the image zip into the expected directory
- or rewrite the manifest and loader contract to use a new extracted path

Preferred action:

- extract or repair paths explicitly and record the repair in the plan/results docs

### Zip contains more images than the prepared manifest

Observed:

- zip images: `2617`
- manifest rows: `2573`
- extra images in zip not present in manifest: `44`

Interpretation:

- local prep intentionally retained only the subset with verified `Cornea` annotations
- this is aligned with the recommended cornea-segmentation upstream task

## Dataset 2: `SLIT-Net`

### Verified Public Access

Public sources observed:

- GitHub repo: `https://github.com/jessicaloohw/SLIT-Net`
- README:
  - `https://raw.githubusercontent.com/jessicaloohw/SLIT-Net/master/README.md`

Repo observations:

- repo is public
- code is present
- repo contains `Datasets/README.md` and `Trained_Models/README.md`
- those README files only point to Duke Box downloads
- no actual dataset files are present in the GitHub repo contents

### Practical Download Check

Advertised links from the public README:

- dataset:
  - `https://duke.box.com/s/iess8rryyn6oo5607aj83tl1hg3o49kk`
- trained models:
  - `https://duke.box.com/s/lm1im3y5oy2k574ng0yjxm89ubis8z40`

Probe result on `2026-04-23`:

- dataset link returned HTTP `404`
- trained-model link returned HTTP `404`

Practical access conclusion:

- code is accessible
- data is not currently verified as downloadable in practice from the advertised public links

### Label Type And Domain Relevance

From the public README and paper metadata:

- `SLIT-Net` is a microbial keratitis slit-lamp segmentation project
- supervision is closer to:
  - ocular structure segmentation
  - microbial keratitis biomarker segmentation
  - white-light and blue-light slit-lamp image analysis

Domain relevance:

- high
- probably useful if accessible

Current repo role:

- conditional only
- do not block the base run on it

## What Is Usable Right Now

Usable now:

- local `SLID` image zip
- local `SLID` prepared cornea masks
- local `SLID` split files
- local `SLID` manifest after raw-image path repair or extraction
- public `SLID` annotation CSV as a verification source

Not usable right now:

- local `SLID` manifest without raw-image path repair
- `SLIT-Net` dataset as a current executable base dataset

## Recommended Dataset Roles

### Recommended Base Role: `SLID`

Role:

- external anatomy-aware warm-start through binary cornea segmentation

Why:

- verified public source
- strong local artifact reuse
- single-task objective is simple and current-branch-feasible

### Conditional Future Role: `SLIT-Net`

Role:

- optional follow-up only if data access is restored and the integration tax stays low

Why:

- domain fit is good
- current practical accessibility is not good enough for the base experiment

## Audit Decision For Phase 2

Proceed with:

- `SLID` cornea-mask pretraining as the default upstream task

Do not make the base run depend on:

- `SLIT-Net`
- weak-label `SLID` keratitis binary pretraining
- multi-objective lesion expansion

This keeps the first executed external-signal experiment aligned with the approved spec:

- one upstream change only
- current-branch rerun only
- smallest honest external-signal path with the highest chance of clean execution
