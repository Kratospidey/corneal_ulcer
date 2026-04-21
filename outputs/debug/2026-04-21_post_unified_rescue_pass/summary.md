# Post-Unified Rescue Pass Summary

## Scope
This pass did not run another large unified 3-task end-to-end training job. It split the work into:
- TG rescue as the main learned branch
- severity salvage on frozen geometry features
- pattern protection as a guardrail

## Why the unified pass failed
The prior unified branch was not stable enough to extend directly:
- pattern regressed badly under shared multitask training
- validation winners did not transfer reliably to held-out test
- learned severity geometry heads collapsed and even lost to the weak rule baseline
- TG improved over the flat baseline only marginally, while `type3` recall remained effectively dead
- the shared setup mixed too many objectives before either TG or severity had proved themselves independently

See also:
- `failure_audit.md`
- `failure_audit.csv`

## What changed
### TG rescue branch
Implemented a dedicated TG rescue path with:
- official pattern checkpoint warm-start
- shared `convnextv2_tiny` backbone
- structured TG hierarchy
- optional TG-only multiscale fusion
- punctate-aware sampling support
- optional `T3`-focused smoothing
- pattern-teacher distillation as a guardrail

### Severity salvage branch
Implemented a frozen-feature severity salvage pipeline using the existing cornea-normalized image-space fluorescein geometry features and compared:
- geometry-rule baseline
- logistic regression
- small MLP
- sklearn histogram gradient boosting fallback

## Verification
The following code paths were verified locally:
- `PYTHONPATH=src micromamba run -n corneal-train python -m unittest tests.test_tg_rescue -v`
- `PYTHONPATH=src micromamba run -n corneal-train python -m unittest tests.test_severity_salvage -v`
- `micromamba run -n corneal-train python src/main_train_tg_rescue.py --help`
- `micromamba run -n corneal-train python src/run_severity_salvage.py --help`

## TG rescue results
### Main ladder
Held-out unrestricted TG results for the serious ladder were:
- `TG-A1_serious_distill`: BA `0.5102`, macro F1 `0.4920`, pattern BA `0.7914`, pattern delta `-0.0567`
- `TG-A2_multiscale_distill_seed42`: BA `0.4523`, macro F1 `0.4601`, pattern BA `0.8580`, pattern delta `0.0098`
- `TG-A3_punctate_seed42`: BA `0.4523`, macro F1 `0.4595`, pattern BA `0.8336`, pattern delta `-0.0145`
- `TG-A4_t3smooth_seed42`: BA `0.4523`, macro F1 `0.4595`, pattern BA `0.8593`, pattern delta `0.0111`

Key interpretation:
- A1 had the best held-out TG score in the serious ladder, but remained pattern-regressing relative to the 0.03 guardrail.
- A2 was the only main-path run that produced a guardrail-valid validation checkpoint. Its held-out unrestricted pattern was strong, but its held-out TG dropped below A1.
- A3 and A4 raised unrestricted validation TG, but that did not survive on held-out TG.
- `type3` recall remained `0.0` in every unrestricted held-out TG result from A1 to A4.

### TG-safe checkpoint view
The only main-path guardrail-valid run was A2. Its held-out guardrail checkpoint produced:
- TG BA `0.4970`
- TG macro F1 `0.4500`
- pattern BA `0.8388`
- pattern delta vs official `-0.0094`

That makes A2 the best TG-safe branch from this pass, even though it is not the best unrestricted TG branch.

## TG stability
A small 3-seed stability check was completed on A2, the only credible TG-safe branch:
- seed 42: guardrail-valid checkpoint existed; best unrestricted val TG BA `0.7833`; test unrestricted TG BA `0.4523`
- seed 43: guardrail-valid checkpoint existed only at epoch 1; best unrestricted val TG BA `0.8236`; test unrestricted TG BA `0.5999`
- seed 44: no guardrail-valid checkpoint; best unrestricted val TG BA `0.6722`; test unrestricted TG BA `0.5226`

Mean unrestricted metrics across the 3 A2 seeds:
- mean best validation TG BA: `0.7597`
- mean held-out unrestricted TG BA: `0.5249`
- guardrail-valid seeds: `2/3`

Interpretation:
- A2 is not stable enough to claim as a clean winner yet
- validation ranking is still seed-sensitive
- only two of the three seeds produced any guardrail-valid checkpoint at all

## Severity salvage results
Severity salvage completed on frozen geometry features. Best held-out severity result:
- `hgb_fallback`
- BA `0.3280`
- macro F1 `0.3327`

Selection mismatch remained present:
- validation selected `logreg`
- held-out test favored `hgb_fallback`

No learned severity salvage model beat the prior severity reference.

## Current decision state
### Pattern
- the official pattern model does not change

### TG
- structured TG is still the right modeling direction
- A1 is the best unrestricted TG result from the serious rescue ladder
- A2 is the best TG-safe branch from this pass
- A3 and A4 did not rescue `type3` and did not produce a better deployable outcome than A2
- `type3` remains effectively unlearned on held-out test

### Severity
- severity should remain a post-hoc frozen-feature module for now
- `hgb_fallback` is the best severity salvage model from this pass
- severity is still not strong enough to promote over the prior severity reference

## Bottom line
This rescue pass is no longer blocked by the GPU driver. The full A1 to A4 TG ladder ran, and the required small stability check ran on A2. The honest conclusion is still conservative:
- no new model replaces the official pattern model
- A1 is the best unrestricted TG rescue result
- A2 is the best TG-safe branch, but it is not stable enough across seeds to promote confidently
- A3 and A4 did not fix `type3`
- severity remains a post-hoc module, with `hgb_fallback` as the best salvage baseline from this pass
