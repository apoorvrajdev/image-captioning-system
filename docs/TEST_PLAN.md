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
| workflows, Dockerfile, deps, hooks | YAML parses · `SKIP=mypy pre-commit run --all-files` · full suite · the security scans (`docs/CI.md` § Local equivalents) · `docker build .` if Docker is available | no gate removed or weakened; the pip-audit gate and `npm audit --omit=dev` pass |
| any commit | pre-commit hooks (locally on commit; all files in CI) | pass with no rewrites |

## What each area must demonstrate

**Notebook parity (reproducibility).** The frozen notebook is byte-stable, and the four-stage audit
(caption preprocessing = string-equal, tokenizer vocab = set-equal, image preprocessing
`tf.allclose` atol 1e-5, decoder forward at fixed weights `tf.allclose` atol 1e-4) passes.

**Configuration.** Unknown keys and out-of-range values fail at load (`tests/unit/test_config.py`).
Env overrides work at any depth.

**Preprocessing and tokenizer.** Deterministic text cleanup, 299×299×3 InceptionV3-normalised tensors,
and a lossless tokenizer save/load round-trip (`test_caption_preprocessing.py`,
`test_image_preprocessing.py`, `test_tokenizer.py`). Loading reads `vocab.json` before any pickle, builds the same
ids either way, rejects malformed or inconsistent vocabularies, and refuses a pickle that names any global but
NumPy's two (`test_tokenizer_vocabulary.py`, TensorFlow-free). Splits are image-level (`test_splits.py`).

**Training recipe.** Label smoothing, warmup + cosine schedule, and flag defaults that keep parity
(`test_training_stability.py`). Full training is owner-run on Kaggle
([`STABILIZED_TRAINING_RUNBOOK.md`](STABILIZED_TRAINING_RUNBOOK.md)) and isn't part of a code change's DoD.

**Inference.** Beam-search components (length/repetition penalties, n-gram blocking, EOS termination,
detokenisation) in `test_beam_decoder.py`. Greedy stays the default. On a scripted bigram decoder, beam width 1 gives
exactly the greedy caption at length penalties 0, 0.7 and 1. That holds when `[end]` is only the runner-up and when
neither decoder finishes within the budget. Width 2 finds a better caption than greedy on the same fixture, so the
fixture does tell the decoders apart (TASK-031).

**API contract.** `/healthz` always 200 with readiness in the body. `/v1/captions` returns
200/400/413/415/422/503 exactly as documented, and `x-request-id` is echoed or generated
(`backend/app/tests/test_captions.py`, `test_health.py`). A body over the upload limit plus multipart framing gets
the same 413 before it's buffered, with or without `Content-Length` (`test_body_size_limit.py`). An image whose header
is over the pixel limits gets a 413 before TensorFlow decodes it, and other bytes, GIF included, get a 422
(`test_image_guard.py`, using header-only bytes). An image within the limits produces the same tensor as before the
check (`tests/unit/test_upload_image_decode.py`). At most two requests decode or predict at once: the next gets
an immediate 503 with `Retry-After`, slots come back after success and after errors, a cancelled request holds its
slot until its worker thread ends, and `/healthz` answers while inference holds every worker thread
(`test_inference_admission.py`, with gated worker threads rather than sleeps). Weights resolve from local paths,
or from the Hub by commit id only: a pinned tag resolves to its commit, branches and unpinned tags are refused
before any download, only the weights file and `vocab.json` are requested, and a commit missing either fails by name
(`test_weights_loader.py`, offline). Both files must match the SHA-256 pinned for their commit (TASK-029):
- a one-byte change in either file is refused, naming only that file, and two changed files are both named;
- a file replaced by another file's pinned bytes is refused;
- a commit with no digests, a missing digest and a malformed digest are refused before any download;
- `v1.0.0` and `v2.0.0` are each checked against their own commit's digests;
- `v1.0.0`'s vocabulary digest equals the repo's committed `models/v1.0.0/vocab.json`;
- hashing reads in chunks.

Through the real lifespan, with the downloader and predictor stubbed, a tampered weights file or vocabulary stops
startup before the predictor reads anything. That holds even for a vocabulary that would pass validation
(`test_startup_integrity.py`).

**Supply chain (TASK-021, ADR-026).** The pip-audit gate passes baselined findings, and fails on a new finding, a
version change, an alias standing in for an id, a stale entry, an unaudited dependency or malformed input. The committed baseline covers only the
reviewed packages (`test_check_pip_audit.py`). CI runs the scans themselves (`docs/SECURITY.md` § CI scanning policy).

**Deploy scope (TASK-022, ADR-027).** The deploy rule, run against throwaway git repositories:
- docs-only commits skip;
- `README.md`, dependency, config and build changes deploy;
- an image change anywhere in the range deploys, including one whose deploy failed or was skipped before a docs-only
  commit, and a file moved out of an image directory;
- the Space being off the recorded commit, no record, or an unreadable one all deploy.

Every Dockerfile `COPY` source must be in the build context. The deploy workflow keeps its guard → decision → snapshot
→ push → health → record order, with no step after the decision running on failure, and pushes only the validated
snapshot commit, with the token passed through `GIT_ASKPASS` (`test_deploy_scope.py`).

**Space snapshot (TASK-025, ADR-030).** Built by the real git plumbing from throwaway repositories: the snapshot holds
exactly the build context, byte for byte with modes, and none of the demo media, docs, results, notebooks, frontend,
tests or untracked local files. It's one commit with no parents, names the exact tested commit, gets the Space header,
and leaves the checkout untouched. A missing build input, or a binary, LFS, secret- or cache-like file refuses the
build, and no secret is printed. One test snapshots this repository's own HEAD (`test_space_snapshot.py`).

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
- The "Try an example" gallery (TASK-033). `support.js` records each caption request's multipart body:
  - an example sends its exact file bytes, file name and `image/jpeg`;
  - its "Dataset reference" card appears only after the caption, with the COCO credit, and never holds the
    generated caption;
  - a second example replaces the first;
  - an upload after an example clears the selection and sends the upload;
  - a sample fetch held open can't overwrite a later upload or a later example. Both tests fail without the
    stale-fetch guard;
  - Enter and Space select;
  - a sample that fails to load (404 on the fetch only) shows an error and sends nothing. That test alone allows
    Chromium's failed-load line for that one URL;
  - the gallery is hidden on the Phase 3 view;
  - at 390 px it shows two columns without horizontal scrolling.

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

**Deployment (post-deploy smoke, TASK-023, ADR-028).** Every deploy captions a generated PNG on the live Space after
the health gate (`scripts/smoke_caption.py`). It checks:
- HTTP 200 and the `CaptionResponse` shape;
- a non-empty caption, with no exact text;
- the `/healthz` model version;
- the echoed `x-request-id`;
- the Vercel origin allowed.

`test_smoke_caption.py` covers those checks, the retry rules, the image decoding through the serving decoder, and the
whole check against `create_app()` with a stand-in predictor. The frontend half of
`PHASE_2C_DEPLOYMENT_RUNBOOK.md` § 8 stays owner-run.

## Regression protection

- Every bug fix adds a test that fails before the fix.
- Tests are CPU-only, offline, seeded (`tests/conftest.py` autouse seed 42), and never download models.
- Never delete, skip, or loosen a test to get green. Tolerances in the parity audit are part of the contract.

## Known gaps (not yet covered)

No frontend unit tests (the SPA is covered by Playwright E2E only, Chromium only), no coverage measured in CI, no load tests,
and no end-to-end lifespan test with real weights (manual smoke only).
