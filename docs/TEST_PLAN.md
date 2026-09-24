# Test plan — what "working" means

Every change is verified against the rows for the layers it touches. The same checks run in CI
([`CI.md`](CI.md)), so local green should mean CI green. Commands assume the repo venv
(`.venv/Scripts/*.exe` on Windows; plain names elsewhere).

## Gates by layer

| Layer touched | Required checks | Pass condition |
|---|---|---|
| any Python | `ruff check src/captioning backend scripts tests` · `ruff format --check …` · `mypy` (see `docs/CI.md`) | exit 0 |
| `src/captioning/**`, `configs/**` | `pytest tests backend/app/tests -q` · `python -m scripts.notebook_module_audit` · notebook SHA-256 | all pass · `[OK] 4/4` · hash equals `.paper-notebook.sha256` |
| `backend/app/**` | `pytest backend/app/tests -q` then the full suite | pass; backend slice imports no TensorFlow |
| `src/captioning/evaluation/**`, eval scripts | `pytest tests/unit/test_evaluation*.py -q` | pass; existing `results/*` unchanged |
| `frontend/**` | `npm run lint` · `npm run build` · manual/browser flow (below) | exit 0; flow verified or reported "not browser-verified" |
| workflows, Dockerfile, deps | YAML parses · full suite · `docker build .` if Docker is available | no gate removed or weakened |

## What each area must demonstrate

**Notebook parity (reproducibility).** The frozen notebook is byte-stable, and the four-stage audit
(caption preprocessing = string-equal, tokenizer vocab = set-equal, image preprocessing
`tf.allclose` atol 1e-5, decoder forward at fixed weights `tf.allclose` atol 1e-4) passes.

**Configuration.** Unknown keys and out-of-range values fail at load (`tests/unit/test_config.py`).
Env overrides work at any depth.

**Preprocessing and tokenizer.** Deterministic text cleanup, 299×299×3 InceptionV3-normalised tensors,
and a lossless tokenizer save/load round-trip (`test_caption_preprocessing.py`,
`test_image_preprocessing.py`, `test_tokenizer.py`). Splits are image-level (`test_splits.py`).

**Training recipe.** Label smoothing, warmup + cosine schedule, and flag defaults that keep parity
(`test_training_stability.py`). Full training is owner-run on Kaggle
([`STABILIZED_TRAINING_RUNBOOK.md`](STABILIZED_TRAINING_RUNBOOK.md)) and isn't part of a code change's DoD.

**Inference.** Beam-search components (length/repetition penalties, n-gram blocking, EOS termination,
detokenisation) in `test_beam_decoder.py`. Greedy stays the default.

**API contract.** `/healthz` always 200 with readiness in the body. `/v1/captions` returns
200/400/413/415/422/503 exactly as documented, and `x-request-id` is echoed or generated
(`backend/app/tests/test_captions.py`, `test_health.py`). Weights resolve from Hub or local paths offline
(`test_weights_loader.py`).

**Evaluation.** Metric implementations match hand-checkable tiny corpora, and run artefacts follow the
`write_run_artifacts` contract (`test_evaluation_metrics.py`, `test_evaluation.py`).

**Frontend (manual until a runner exists).** With backend + `npm run dev` running: upload a valid image,
then Generate shows the caption card with version/strategy/latency/request ID. A disallowed type or a >10 MB file
is rejected client-side with no request. With the backend stopped, the badge goes offline and Generate shows
"Cannot reach backend". Browser console has zero errors.

**Deployment (post-deploy smoke, owner-run).** Follow `PHASE_2C_DEPLOYMENT_RUNBOOK.md` § 8: Space
`/healthz` reports `model_loaded: true`, and one caption round-trip works from the Vercel origin (CORS).

## Regression protection

- Every bug fix adds a test that fails before the fix.
- Tests are CPU-only, offline, seeded (`tests/conftest.py` autouse seed 42), and never download models.
- Never delete, skip, or loosen a test to get green. Tolerances in the parity audit are part of the contract.

## Known gaps (not yet covered)

No frontend unit/e2e runner, no coverage measured in CI, no load tests, no test for beam width 1 ≡ greedy,
and no end-to-end lifespan test with real weights (manual smoke only).
