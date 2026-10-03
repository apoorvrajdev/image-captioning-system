---
name: evaluation
description: Acceptance criteria and definition of done for evaluation and benchmarking — BLEU/CIDEr/METEOR/ROUGE-L, the benchmark artefact contract, results/<run_id>/ sets, eval-methodology audits, and cross-model comparisons (Phase 3 harness). Use whenever changing src/captioning/evaluation/**, scripts/{evaluate,inspect_predictions,rescore_nltk_bleu,categorize_predictions}.py, or results/**, or publishing any metric.
---

# Evaluation — acceptance criteria

## Scope
`src/captioning/evaluation/**`, `scripts/evaluate.py`, `scripts/inspect_predictions.py`,
`scripts/rescore_nltk_bleu.py`, `scripts/categorize_predictions.py`, `results/**`,
`docs/EVAL_METHODOLOGY.md`, metric tables in `README.md`.

## Expected behaviour
- Every eval run writes one directory `results/<run_id>/` via `write_run_artifacts`: `run_meta.json`, `metrics.json`, `predictions.jsonl`, `diagnostics.jsonl`, `report.md`.
- A run is one (model, decode strategy, dataset slice). `run_meta.json` records enough to reproduce it (model id, decode params, n_samples, timestamp).
- Corpus metrics: BLEU-1..4 (sacrebleu, deterministic tokenisation in `evaluation/tokenization.py`), CIDEr, METEOR, ROUGE-L.
- Two runs are compared only when the slice, reference count, tokenisation, and smoothing are identical, and that sameness is stated.

## Edge cases (each needs a test)
- Empty prediction / empty reference list, single-token captions, duplicate references.
- Metric values on hand-computed tiny corpora (`tests/unit/test_evaluation_metrics.py`).
- Reference count matters: the committed slice averages ~1.46 refs/image, while COCO standard is 5.

## Required tests
| Level | File | Must cover |
|---|---|---|
| unit | `tests/unit/test_evaluation_metrics.py`, `test_evaluation.py` | any metric/tokenisation change |
| artefact | new `results/<run_id>/` | produced by the script, not hand-edited |

## Definition of done — ALL must pass
- [ ] Metric code changes have hand-checkable unit tests. `pytest tests -q` green.
- [ ] Existing `results/*` directories untouched (append-only). New numbers → new run dir.
- [ ] Methodology change (tokeniser, refs, smoothing, slice) documented in `docs/EVAL_METHODOLOGY.md` and noted in `docs/DECISIONS.md`.
- [ ] Hypothesis-testing audits are **pre-registered**: thresholds and decision rule committed before results. Qualitative judging blinded to metrics.
- [ ] README numbers cite the exact `results/<run_id>/` they come from, with no claims of superiority from mismatched eval setups.
- [ ] ruff + mypy clean on touched scripts/modules.

## Verification commands
```bash
.venv/Scripts/pytest.exe tests/unit/test_evaluation_metrics.py tests/unit/test_evaluation.py -q
.venv/Scripts/ruff.exe check src/captioning scripts && .venv/Scripts/ruff.exe format --check src/captioning scripts
```
(Full COCO evals need the dataset and GPU/Kaggle. They're user-run; agents prepare commands and never fabricate numbers.)

## Non-negotiables
- Never report a metric that wasn't produced by a committed script run.
- Research experiments stay isolated and attributable: one named change per run (ablatable flags).
- Phase 3 baselines (BLIP / ViT-GPT2 / GIT) must reuse this artefact contract and the same slice + tokenisation.
