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
| `frontend/**` | `npm run lint` · `npm run build` · `npm run test:e2e` (Playwright, Chromium; first run `npx playwright install chromium`) | exit 0; all specs pass with zero console errors |
| workflows, Dockerfile, deps, hooks | YAML parses · `SKIP=mypy pre-commit run --all-files` · full suite · `docker build .` if Docker is available | no gate removed or weakened |
| any commit | pre-commit hooks (locally on commit; all files in CI) | pass with no rewrites |

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
`write_run_artifacts` contract (`test_evaluation_metrics.py`, `test_evaluation.py`). The SPA's Phase 3 dashboard data
(`frontend/src/generated/phase3-dashboard.json`) equals a fresh export from the committed results
(`test_dashboard_export.py`; `python -m scripts.export_dashboard_data --check`).

**Frontend browser E2E (TASK-007, ADR-023).** `cd frontend && npm run test:e2e` runs Playwright on Chromium against
`vite preview` of the production build. It also runs in the CI `frontend` job. `e2e/support.js` mocks `/healthz` and
`/v1/captions`, so no backend or TensorFlow is needed. Every test fails on a console error or an uncaught page
error, and on any off-origin request outside those two endpoints.

`e2e/caption-flow.spec.js` covers the caption flow:
- A PNG uploaded against a healthy API, then Generate, shows the caption card with version, strategy, latency
  and request ID.
- A disallowed type or a >10 MB file is rejected inline with no request.
- With the API unreachable, the badge goes offline and Generate shows "Cannot reach backend". In this case only,
  Chromium's own `net::ERR_CONNECTION_REFUSED` line for the two API URLs is allowed.

**Phase 3 dashboard (`e2e/phase3-dashboard.spec.js`, TASK-018).** On "Phase 3 comparison":

- Rendering:
  - The quality table has a row for every model's quality run. Each cell's exact value and its 2-decimal display equal the committed `phase3-dashboard.json`, which `test_dashboard_export.py` ties to `results/`.
  - The CPU and GPU latency tables have a row for every model and batch size. Each value equals the JSON, shown in seconds to 0.0001 s, with load time to 0.1 s.
  - Every row names its run id.
  - The caveats are visible: not live, not held-out, not a ranking, CPU/GPU from different hosts, and sequential CNN batches.
- Exact values: "Show exact values" shows the JSON's unrounded numbers.
- Missing values show "n/a": the null revisions and decode settings in the data are checked. The real data has no missing metric or latency.
- Requests: switching views makes no request, and the dashboard still renders with the API down.
- Caption flow: switching back keeps the chosen file and the result.
- Layout, at 390 and 1280 px:
  - The page never scrolls sideways.
  - Each table region sits inside the viewport and is a focusable `overflow-x: auto` container, so a wide table
    scrolls instead of losing columns. At 390 px every table does scroll.
  - Whether every column fits at 1280 px without scrolling depends on the platform's fonts, so it isn't asserted.
    With Windows fonts the slack is 5.6–10.9 %.
- Console: zero errors.

**Deployment (post-deploy smoke, owner-run).** Follow `PHASE_2C_DEPLOYMENT_RUNBOOK.md` § 8: Space
`/healthz` reports `model_loaded: true`, and one caption round-trip works from the Vercel origin (CORS).

## Regression protection

- Every bug fix adds a test that fails before the fix.
- Tests are CPU-only, offline, seeded (`tests/conftest.py` autouse seed 42), and never download models.
- Never delete, skip, or loosen a test to get green. Tolerances in the parity audit are part of the contract.

## Known gaps (not yet covered)

No frontend unit tests (the SPA is covered by Playwright E2E only, Chromium only), no coverage measured in CI, no load tests, no test for beam width 1 ≡ greedy,
and no end-to-end lifespan test with real weights (manual smoke only).
