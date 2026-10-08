---
name: ml-core
description: Acceptance criteria and definition of done for the captioning library core — config schema, preprocessing, tokenizer, models, data pipeline, training, and notebook parity. Use whenever changing src/captioning/{config,preprocessing,models,data,training,utils}, configs/*.yaml, scripts/train.py, or anything that could affect notebook parity.
---

# ML core — acceptance criteria

## Scope
`src/captioning/{config,preprocessing,models,data,training,utils}/**`, `configs/**`,
`scripts/train.py`, `scripts/notebook_module_audit.py`, `scripts/bootstrap_dev_artifacts.py`,
the frozen notebook (read-only), and tests under `tests/unit/`.
Decoding (`inference/`) → see `inference-api`. Metrics → see `evaluation`.

## Expected behaviour
- GIVEN default config (`configs/base.yaml`) WHEN any module runs THEN behaviour matches the IEEE notebook (parity audit 4/4).
- GIVEN a YAML or env override with an unknown or misspelled key WHEN loading `AppConfig` THEN load fails with a `ValidationError` naming the field.
- GIVEN a YAML value and a `CAPTIONING__*` env var for the same field WHEN `load_config` runs THEN the env value wins and every other YAML value is kept; a list is replaced whole (ADR-029).
- GIVEN a new training or decoding improvement WHEN added THEN it's behind a config flag whose default preserves notebook behaviour (pattern: `TrainConfig` stability flags, `configs/train/stabilized.yaml`).
- GIVEN any stochastic code path WHEN run twice with the same `train.seed` THEN results are identical (`set_global_seed`).
- GIVEN a tokenizer save → load round-trip THEN vocab and encodings are identical (pickle + JSON sidecar).

## Edge cases (each needs a test)
- Config validators: out-of-range split, label smoothing, warmup, beam width, repetition penalty.
- Image preprocessing: grayscale/RGBA input → 3 channels, 299×299, InceptionV3 normalisation.
- Caption preprocessing: punctuation, extra whitespace, casing, `[start]`/`[end]` wrapping.
- Train/val split is image-level (no image in both splits).

## Required tests
| Level | File | Must cover |
|---|---|---|
| unit | `tests/unit/test_config.py` | new fields, validators, `extra="forbid"` |
| unit | `tests/unit/test_{caption,image}_preprocessing.py`, `test_tokenizer.py` | preprocessing changes |
| unit | `tests/unit/test_training_stability.py` | losses/schedules/flags |
| gate | `scripts/notebook_module_audit.py` | stays 4/4; extend a stage if you touch its seam |

## Definition of done — ALL must pass
- [ ] New or changed behaviour has a test that fails without the change.
- [ ] `pytest tests backend/app/tests -q` green, no tests skipped or deleted.
- [ ] `ruff check` + `ruff format --check` + mypy clean (commands in CLAUDE.md).
- [ ] `python -m scripts.notebook_module_audit` → `[OK] parity audit: 4/4 checks passed`.
- [ ] Notebook SHA-256 matches `.paper-notebook.sha256`.
- [ ] New hyperparameters live in `schema.py` + YAML, never hardcoded in scripts.
- [ ] Default values unchanged unless the task explicitly changes the published baseline (then: ADR in `docs/DECISIONS.md`).
- [ ] Any architecture or tokenizer change ⇒ new model version dir + HF Hub tag, never overwrite `models/v1.0.0/`.

## Verification commands
```bash
.venv/Scripts/pytest.exe tests -q
.venv/Scripts/python.exe -m scripts.notebook_module_audit
.venv/Scripts/ruff.exe check src/captioning backend scripts tests && .venv/Scripts/ruff.exe format --check src/captioning backend scripts tests
MYPYPATH="src;backend" .venv/Scripts/mypy.exe --explicit-package-bases --namespace-packages src/captioning backend/app scripts
```

## Non-negotiables
- Never edit or re-lock the frozen notebook. Never "fix" a notebook quirk by changing a default (e.g. `honour_training_flag_in_test_step` stays opt-in).
- Keep TF out of import paths that don't need it (`models/factory.py` lazy builders exist for this).
- Retraining is never part of a code task. It's a user-run Kaggle job (`docs/STABILIZED_TRAINING_RUNBOOK.md`).
