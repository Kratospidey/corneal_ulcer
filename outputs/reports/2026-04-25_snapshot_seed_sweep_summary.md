# Snapshot Seed Sweep Summary

Date: 2026-04-25

## Scope

This sweep used the restored official image-only recipe:

- task: `pattern_3class`
- backbone: `convnextv2_tiny`
- preprocessing: `cornea_crop_scale_v1`
- train augmentation: `pattern_augplus_v2`
- sampler: `weighted_sampler_tempered`

To remove one known reproducibility gap, the sweep was run against snapshot copies of:

- `data/interim/snapshots/manifest__2026-04-25_sha256_a7efafece192.csv`
- `data/interim/snapshots/pattern_3class_holdout__2026-04-25_sha256_9e37a668b20a.csv`

The seed42 entry below reuses the already-completed restored-recipe run from the same machine state and same current manifest/split contents.

## Seed Results

| Seed | Val BA | Test BA | Test Macro F1 | Point-like Recall | Point-flaky-mixed Recall | Flaky Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | `0.7128` | `0.8021` | `0.7516` | `0.9259` | `0.6341` | `0.8462` |
| 42 | `0.6912` | `0.7416` | `0.6590` | `0.7407` | `0.5610` | `0.9231` |
| 73 | `0.6876` | `0.7491` | `0.6884` | `0.8889` | `0.5122` | `0.8462` |
| 101 | `0.7569` | `0.7158` | `0.7068` | `0.9259` | `0.6829` | `0.5385` |
| 211 | `0.6836` | `0.7294` | `0.7183` | `0.9630` | `0.6098` | `0.6154` |

## Family Statistics

- fresh-run val BA mean: `0.7064`
- fresh-run val BA std: `0.0304`
- fresh-run test BA mean: `0.7476`
- fresh-run test BA std: `0.0330`

Official frozen checkpoint:

- val BA (current re-eval): `0.7253`
- test BA: `0.8482`

Relative to the fresh five-seed family:

- official val BA is only about `0.62` standard deviations above the fresh-run mean
- official test BA is about `3.05` standard deviations above the fresh-run mean

## Interpretation

This is the strongest evidence gathered in the current forensic pass.

The official checkpoint's current validation score is not uniquely exceptional relative to fresh retrains. Fresh runs can match or beat its validation balanced accuracy:

- seed101 fresh retrain val BA: `0.7569`
- official current val BA: `0.7253`

But those same fresh runs do not reproduce the official test score:

- best fresh test BA in this sweep: `0.8021` (seed17)
- official test BA: `0.8482`

That pattern argues against a simple present-day training-code bug and instead supports one or both of:

1. a provenance gap around the original winning run
2. unusually favorable test-set realization for the frozen official checkpoint on this small benchmark

## Practical Takeaway

The restored recipe is real and functional, but it does not produce a stable family centered near `0.8482` test BA.

Current best evidence:

- the recipe is not broken
- the official winner is not reproduced as a normal outcome of fresh training
- the benchmark remains highly seed-sensitive
- validation balanced accuracy alone is not enough to explain the official winner
