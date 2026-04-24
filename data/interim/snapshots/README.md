# Data Snapshots

This directory stores point-in-time copies of benchmark-critical CSV files so
future runs can reference a tracked manifest/split pair instead of relying on
mutable local files outside Git history.

Active snapshot for the official pattern benchmark forensic pass:

- `manifest__2026-04-25_sha256_a7efafece192.csv`
- `pattern_3class_holdout__2026-04-25_sha256_9e37a668b20a.csv`

These were copied from:

- `data/interim/manifests/manifest.csv`
- `data/interim/split_files/pattern_3class_holdout.csv`
