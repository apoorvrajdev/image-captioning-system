# Architecture decisions

Permanent technical decisions, one short entry each. **Append-only**: to change a decision,
add a new entry that supersedes the old one; don't edit history. Current state lives in
[`MEMORY.md`](MEMORY.md). Longer rationale for early decisions: [`PHASE_0_NOTES.md`](PHASE_0_NOTES.md),
[`restructure-plan.md`](restructure-plan.md) § 5, README § Engineering Decisions.

Format: **Decision · Why · Evidence**.

---

### ADR-001 — The IEEE notebook is frozen and canonical
- **Decision:** `notebooks/01_ieee_inceptionv3_transformer.ipynb` is never edited. All improvements go into `src/captioning/`. A SHA-256 lock (`.paper-notebook.sha256`) is enforced in pre-commit and CI.
- **Why:** it's the only artefact that reproduces the published result. Editing it destroys reproducibility.
- **Evidence:** `notebooks/README.md`, `Makefile` `freeze-paper-notebook`, `ci.yml` `notebook-freeze`.

### ADR-002 — Structure-only refactor gated by a parity audit
- **Decision:** the modular package must match the notebook at four seams (caption preprocessing, tokenizer vocabulary, image preprocessing, decoder forward pass). Behaviour changes are opt-in flags whose defaults preserve parity.
- **Why:** when metrics move, every change has to be attributable to one named intervention.
- **Evidence:** `scripts/notebook_module_audit.py`, `TrainConfig` stability flags, `configs/train/stabilized.yaml`.

### ADR-003 — Pin `tensorflow-cpu==2.15.0` and `numpy<2`
- **Decision:** hard pin. Upgrading is a deliberate future task.
- **Why:** TF 2.16 defaults to Keras 3, which breaks `TextVectorization` save/load. NumPy 2 breaks TF 2.15 binaries. The CPU wheel suits CPU-only Spaces.
- **Evidence:** `pyproject.toml` dependency comments, `PHASE_0_NOTES.md` § 3.

### ADR-004 — Strict typed configuration, split by audience
- **Decision:** Pydantic v2 with `extra="forbid"`. Research config `AppConfig` (YAML + `CAPTIONING__*` env) is separate from serving config `BackendSettings` (`BACKEND_*` env).
- **Why:** a hyperparameter typo must fail at load time, and research and serving knobs change on different cadences.
- **Evidence:** `src/captioning/config/schema.py`, `backend/app/core/config.py`.

### ADR-005 — Lifespan-managed single predictor, single worker
- **Decision:** one `CaptionPredictor` is built and warmed in the FastAPI lifespan, shared by all requests, with inference offloaded via `anyio.to_thread.run_sync`. Uvicorn runs `--workers 1`.
- **Why:** avoids per-request graph rebuilds and event-loop blocking. Multiple workers would duplicate TF + InceptionV3 in memory.
- **Evidence:** `backend/app/main.py`, `backend/app/services/predictor_service.py`, `Dockerfile` CMD, `restructure-plan.md` § 5.

### ADR-006 — Shared train/serve preprocessing
- **Decision:** serving decodes uploads with `tf.io.decode_image` and then the training `preprocess_image_tensor`. No separate serve-side normalisation.
- **Why:** train/serve skew is ruled out by construction.
- **Evidence:** `backend/app/utils/image.py`, `src/captioning/preprocessing/image.py`.

### ADR-007 — Versioned, immutable model artefacts on HF Hub
- **Decision:** weights and vocab are published as tagged HF Hub revisions and pulled at startup via `snapshot_download`. `models/vX.Y.Z/` and published tags are never modified in place; a new checkpoint gets a new version.
- **Why:** keeps the Space image small and lets weights be rotated without a rebuild, and any served caption stays traceable to a checkpoint.
- **Evidence:** `backend/app/services/weights_loader.py`, `PHASE_2C_DEPLOYMENT_RUNBOOK.md` § 3.

### ADR-008 — Split deployment topology on free tiers
- **Decision:** backend on HF Spaces (Docker SDK), deployed by `deploy-backend.yml` only after CI is green. Frontend on Vercel via its Git integration. Production CORS comes from the Space variable, not code.
- **Why:** free-tier constraint. Frontend and backend deploy independently, and the only coupling is the typed HTTP contract.
- **Evidence:** `.github/workflows/deploy-backend.yml`, `PHASE_2C_DEPLOYMENT_RUNBOOK.md`, `docs/CI.md`.

### ADR-009 — Multipart uploads for images
- **Decision:** `POST /v1/captions` accepts `multipart/form-data`, not base64 JSON.
- **Why:** base64 adds ~33 % overhead and can't stream.
- **Evidence:** `restructure-plan.md` § 5, `backend/app/api/routes.py`.

### ADR-010 — Backend tests never load TensorFlow
- **Decision:** route tests use a duck-typed `FakePredictorService` on a freshly built app, and Hub tests inject a stub downloader. All tests are CPU-only and offline.
- **Why:** sub-second, deterministic contract tests, and CI needs no network or weights.
- **Evidence:** `backend/app/tests/conftest.py`, `backend/app/tests/test_weights_loader.py`.

### ADR-011 — Evaluation artefact contract and methodology discipline
- **Decision:** every eval run writes `results/<run_id>/` (`run_meta.json`, `metrics.json`, `predictions.jsonl`, `diagnostics.jsonl`, `report.md`). Runs are only compared under identical slice, reference count, tokenisation and smoothing. Hypothesis-testing audits are pre-registered and blinded.
- **Why:** the Stage 0 audit showed reference count alone moved beam BLEU-4 from 10.39 to 25.91. Metric deltas across different eval setups aren't model-quality evidence.
- **Evidence:** `src/captioning/evaluation/benchmark.py`, `docs/EVAL_METHODOLOGY.md`, `results/stabilized-beam-w4-lp07-rp12/verdict.md`.

### ADR-012 — Reframe, don't retrain (Stage 0 outcome)
- **Decision:** the original-recipe retrain ("Option B / Stage 1") isn't needed to close a BLEU gap. It stays as an optional future ablation. Caption specificity is left to Phase 3 architectures.
- **Why:** the 5-reference rescore reached the IEEE range (25.91 BLEU-4), and the blinded review found generic, not wrong, captions.
- **Evidence:** `results/stabilized-beam-w4-lp07-rp12/verdict.md`, `docs/EVAL_METHODOLOGY.md`.

### ADR-013 — Phase 3 baselines isolated in an optional extra
- **Decision:** foundation-model baselines (BLIP, ViT-GPT2, GIT) install via the `[hf]` extra (`transformers`, `torch`) and don't change the core pins or the default Docker image.
- **Why:** keeps the serving image slim and the research pipeline reproducible.
- **Evidence:** `pyproject.toml` `[project.optional-dependencies].hf`, README § Engineering Decisions.

### ADR-014 — Frozen notebook checks out with LF line endings
- **Decision:** `.gitattributes` sets `eol=lf` on the frozen notebook.
- **Why:** with `core.autocrlf=true`, the Windows working copy was CRLF, so the SHA-256 freeze check failed locally even though the committed blob matched.
- **Evidence:** `.gitattributes`, `.paper-notebook.sha256`.

### ADR-015 — Agent workflow config is tracked; invariants are enforced, not just documented
- **Decision:** the agent workflow config under `.claude/` (`settings.json`, `skills/`, `agents/`, `context/repo-map.md`, `context/build_index.py`) is versioned and reviewed like code. Personal settings and the regenerated code index stay gitignored. `settings.json` denies edits to the frozen notebook, its hash, `models/**`, and `results/**`, denies reading `.env` files, prompts before `git commit`/`push`/`tag`, and rebuilds the index at session start.
- **Why:** unversioned workflow config can't be reviewed or reproduced on another machine. Invariants enforced by permission rules hold even when instructions are skimmed.
- **Evidence:** `.gitignore`, `.claude/settings.json`, `CLAUDE.md` § Traps.

### ADR-016 — pre-commit hooks also run in CI
- **Decision:** CI runs `pre-commit run --all-files` with the pinned hook config (`SKIP=mypy`, because `python-quality` runs mypy with the full dependency set).
- **Why:** hooks only run on machines where they're installed. CI makes hygiene, nbstripout, and prettier binding for every commit, which matches what `.pre-commit-config.yaml` already promised. The gitleaks hook scans staged changes only, so it's a no-op in CI and doesn't count as CI secret scanning.
- **Evidence:** `.github/workflows/ci.yml` `pre-commit` job, `.pre-commit-config.yaml`.

### ADR-017 — The HF Space is a deployment target fed by force-pushed deploy commits (refines ADR-008)
- **Decision:** GitHub `main` is the only source of truth, and only CI-verified commits deploy. `deploy-backend.yml` checks out the exact commit CI tested and skips it if `main` has moved past it. Manual `workflow_dispatch` runs must first prove, via the GitHub API, that the exact SHA has a completed, successful CI run on `main`; otherwise they fail before checkout. It adds a deploy commit that prepends the Space's README config header (the block removed from GitHub in `befac80`), then **force-pushes** that commit to the Space. A deploy passes only once the HF API shows a rebuild of the new commit reaching `RUNNING` and `/healthz` reports `model_loaded: true`.
- **Why:** mirroring GitHub history with plain pushes broke twice.
  - On 2026-06-16, `302e907` was deployed, then GitHub `main` was rewritten into atomic commits. Every later push was rejected as non-fast-forward.
  - Since `befac80` (2026-06-02), the Space has had no YAML config, so it sits in `CONFIG_ERROR` while pushes still showed "success".
- **Why force-push is safe:**
  - The Space's history is derived from GitHub. Its only divergent commit, `302e907`, has a tree identical to `64f80e8` on GitHub.
  - Space variables and secrets live in Space settings, not git.
  - The guard against superseded commits stops an older run from rolling the Space back.
- **Consequences:**
  - Don't push to the Space by hand; a raw GitHub commit has no config header.
  - Rollback is `git revert` on `main`.
  - Space-only edits made in the HF UI are overwritten on the next deploy.
- **Evidence:** `.github/workflows/deploy-backend.yml`, `docs/CI.md`, `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md` §§ 4, 7, 10, HF runtime API `errorMessage: "Missing configuration in README"`.

### ADR-018 — `BACKEND_MODEL_VERSION` is set alongside `BACKEND_WEIGHTS_HUB_REVISION`, never derived
- **Decision:** the reported `model_version` stays an operator-set Space variable (`BACKEND_MODEL_VERSION`). Every promotion or rollback sets it to the same Hub tag as `BACKEND_WEIGHTS_HUB_REVISION`, in the same change, then checks `/healthz`. The backend code is unchanged; its `"v1.0.0"` default applies only when the variable is unset.
- **Why:** the two are independent fields in `BackendSettings`, and nothing links them. TASK-004 found production serving tag `v2.0.0` while reporting the default `v1.0.0`, because the documented promotion step bumped only the revision. Fixing this with a procedure and a Space variable needed no code change and no redeploy. Deriving the label from the revision in code is a possible future change, not a requirement.
- **Evidence:** `backend/app/core/config.py` (`model_version`, `weights_hub_revision`), `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md` §§ 3, 4, 10, live `/healthz` 2026-10-03 (`model_version: "v2.0.0"`, `model_loaded: true`).

### ADR-019 — Phase 3 baselines are an evaluation workflow with lazily imported Hugging Face code
- **Decision:**
  - The Phase 3 comparison is an offline evaluation workflow, not backend serving. No baseline is served, and the serving image and the Space don't change (ADR-013).
  - Code location:
    - Model adapters (the shared captioner interface, the CNN + Transformer wrapper and the Hugging Face adapter) live in `src/captioning/baselines/`.
    - Slice loading and cross-run comparison live in `src/captioning/evaluation/`.
    - CLI entrypoints live in `scripts/`.
    - Baseline settings go in a new section of `src/captioning/config/schema.py` plus YAML under `configs/`.
    - Tests live in `tests/unit/`, using fakes.
  - `transformers` and `torch` are imported only inside `src/captioning/baselines/`, and only lazily (inside functions, never at module import). Importing `captioning` or `captioning.baselines` without `[hf]` works; using a Hugging Face adapter without it raises an error naming `pip install -e ".[hf]"`.
  - `[hf]` stays an optional extra:
    - CI keeps installing only `requirements-dev.txt`, `requirements-eval.txt` and `pip install -e .`.
    - Tests never need `transformers`, `torch` or a model download.
    - `backend/` never imports `captioning.baselines`.
  - `tensorflow-cpu==2.15.0` and `numpy<2` stay pinned and unchanged. GPU runs of the CNN + Transformer install `tensorflow==2.15.0` in that run environment only.
- **Why:** the comparison needs `transformers` and `torch` only where runs actually happen (owner-run, on Kaggle). Keeping them out of CI, the backend and the image keeps CI fast and offline, keeps the image slim, and leaves the TF 2.15 pin (Keras 2 `TextVectorization` save/load) untouched. Both are already installed with `[hf]` in the local dev venv, so nothing new needs installing.
- **Evidence:** `docs/EVAL_METHODOLOGY.md` § 8, `docs/TASKS.md` (TASK-009 – TASK-018), `pyproject.toml` `[project.optional-dependencies].hf` and the mypy overrides, `.github/workflows/ci.yml` install steps, ADR-013.

### ADR-020 — Phase 3 latency is a separate artefact, timed through the shared captioner interface
- **Decision:**
  - The artefact contract (ADR-011) gains a latency run: a new `results/<prefix><model_id>-<decoding>-<device>/` holding only `latency.json`. It never sits in or next to a quality run, and holds no metrics or predictions.
  - Latency is timed around `Captioner.caption()` (TASK-011), the call that produced the quality runs' captions. One sample is one call on one batch, read from `time.perf_counter`. Load (construction plus `load()`) is timed once and kept separate.
  - Each batch size gets at least one untimed warmup pass, then the measured passes. The statistics are count, mean, median, min and max per batch size, with the raw samples stored. Nothing is filtered, and any failed call ends the run with nothing written.
  - One model runs per invocation. For the CNN + Transformer, which TensorFlow places itself, `--device` is checked against the GPUs TensorFlow can see rather than forced.
  - The protocol and its defaults are `EVAL_METHODOLOGY.md` § 9. The defaults are the first 32 slice images, batch sizes 1 and 8, 1 warmup pass and 5 measured passes.
- **Why:**
  - Quality runs are append-only and their files are already defined, so latency can't be added to them without rewriting committed results.
  - Timing the same call that captioned the quality slice measures those exact model setups, with no second inference path to drift.
  - Raw samples let any other statistic be recomputed later without re-running. Fixing the statistics before any run means none can be picked to suit the results.
  - Checking the CNN's device keeps the device label true without changing the adapter.
- **Evidence:** `docs/EVAL_METHODOLOGY.md` § 9, `src/captioning/evaluation/latency.py`, `scripts/benchmark_latency.py`, `tests/unit/test_latency_benchmark.py`.

### ADR-021 — The Phase 3 dashboard reads static data exported from committed results
- **Decision:**
  - The SPA's comparison dashboard (TASK-018) reads one static JSON file, `frontend/src/generated/phase3-dashboard.json`, imported at build time. There is no backend endpoint and no live comparison. The Space never computes or serves Phase 3 results.
  - `python -m scripts.export_dashboard_data` (logic in `captioning.evaluation.dashboard`) is the file's only writer. It reads `results/phase3-comparison/comparison.json` and every `results/phase3-latency-*/latency.json`.
    - Values are copied verbatim: nothing is rounded, recomputed, ranked or converted.
    - It refuses sources that disagree on the slice, protocol, inputs, settings, timing or a model's Hub id and revision, a run directory whose name doesn't match its file, and a summary that doesn't recompute from its raw samples.
    - The only hand-written content is each model's display name and the caveat notes, which restate `EVAL_METHODOLOGY.md` §§ 8 and 9 with their section numbers.
  - Contents, per model: display name, Hub id and revision, quality rows (metrics, decoding, run id), latency per device and batch size (the § 9.4 summary, load time, environment, batch mode, run id), and its source run ids. Shared: the slice description, the § 8.5 overlap caveat copied from `comparison.json`, and the quality and latency notes. Raw latency samples stay in the run directories.
  - The file is generated, never edited by hand. `tests/unit/test_dashboard_export.py` regenerates it from `results/` and fails on any difference, so a new comparison summary or latency run is followed by a re-export in the same change. `--check` runs the same comparison from the command line.
  - Prettier's pre-commit hook excludes `frontend/src/generated/`, because the drift test owns that file's exact bytes.
- **Why:**
  - It keeps ADR-013 and ADR-019 intact. The Space image has neither `torch` nor `results/`: the `Dockerfile` copies only `src/`, `backend/`, `configs/` and `models/`, and `requirements.txt` has no `torch`. A live endpoint would need one of them in the image.
  - Results are append-only and change only by commit, so a file built with the SPA is as current as the repository. The dashboard needs no network request, and the drift test stops it from disagreeing with `results/`.
  - Copying verbatim and refusing inconsistent sources keeps every number traceable to one committed run, with the caveats that apply to it.
- **Evidence:** `src/captioning/evaluation/dashboard.py`, `scripts/export_dashboard_data.py`, `tests/unit/test_dashboard_export.py`, `frontend/src/generated/phase3-dashboard.json`, `.pre-commit-config.yaml` (prettier `exclude`), `Dockerfile` `COPY` lines, ADR-013, ADR-019, ADR-020.

### ADR-022 — The SPA switches views in `App` state, and the dashboard shows the exported values without reinterpreting them
- **Decision:**
  - The SPA has two views, the caption flow and the Phase 3 dashboard. A button pair in `App.jsx` switches them through React state. There is no router and no URL change (TASK-018 rules out a router without separate approval).
  - The caption flow stays mounted behind the `hidden` attribute while the dashboard shows, so its file, result and any in-flight request survive a switch. The dashboard mounts only when chosen.
  - `components/Phase3Dashboard.jsx` reads only `frontend/src/generated/phase3-dashboard.json` (ADR-021), imported at build time. It makes no request.
  - Values are shown as exported. The dashboard computes no metric, ranking, winner or combined score, and colours nothing by value. Models keep the file's order (by model id), and each row names its run id.
  - Display rounding follows the committed reports:
    - metrics to two decimals, as in `comparison.md`;
    - latency to 0.0001 s, the 0.1 ms of `EVAL_METHODOLOGY.md` § 9.9, kept in the file's unit, seconds;
    - load time to 0.1 s.
  - Every number keeps its exact value in a `<data value>` element, and a "Show exact values" toggle displays it. Missing values show "n/a".
  - The caveats are shown above the tables, ahead of any number: not live, not held-out, not a ranking, CPU/GPU from different hosts, sequential CNN batches. The file's own notes are also shown in full.
- **Why:**
  - Two views don't justify a routing dependency.
  - Keeping the caption flow mounted is the smallest way to leave it unchanged.
  - The dashboard's numbers have to be traceable to one committed run. A recomputed or ranked value would be a new result without a run behind it.
  - Reusing the reports' rounding keeps the page consistent with what is already published, and the exact values stay one click away.
- **Evidence:** `frontend/src/App.jsx`, `frontend/src/components/Phase3Dashboard.jsx`, `docs/TEST_PLAN.md` (Phase 3 dashboard), `results/phase3-comparison/comparison.md`, `docs/EVAL_METHODOLOGY.md` §§ 9.9–9.10, ADR-021.

### ADR-023 — Browser E2E runs Playwright on Chromium against the production bundle, with the API mocked
- **Decision:**
  - `@playwright/test` is the SPA's only test framework, a devDependency of `frontend/`. Only Chromium is installed: the full build locally, and only the headless shell in CI.
  - `playwright.config.js` builds the app and serves it with `npm run preview`, so the specs exercise the production bundle the deploy ships. There are no component or unit tests.
  - The specs live in `frontend/e2e/`.
    - `e2e/support.js` answers every request that leaves the app's origin.
    - `/healthz` and `/v1/captions` get bodies shaped like `backend/app/schemas/caption.py`, or a refused connection when a test marks the API down. Any other off-origin request fails the test.
    - No test needs a backend, TensorFlow or the network.
  - Every test fails on a console error or an uncaught page error, with one exception. While a test has marked the API down, Chromium's `Failed to load resource: net::ERR_CONNECTION_REFUSED` line is allowed for the two API URLs. The browser's network stack logs that line even though the app handles the failure; any other console error still fails.
  - Expected dashboard values come from the committed `phase3-dashboard.json`, the same file the bundle imports. The specs don't keep a second copy.
  - Retries are off. `npm run test:e2e` runs in the CI `frontend` job after lint and build, and the traces are uploaded only when it fails.
- **Why:**
  - Mocking at the browser's network layer keeps the caption flow testable without the 0.5 GB model, and keeps the specs deterministic.
  - Testing the preview build catches bundling problems that a dev server would hide.
  - One browser keeps the CI download and runtime small. The SPA has no browser-specific code.
  - Without retries, a flaky test fails visibly instead of passing on a second try.
  - The narrow console allowance keeps "zero console errors" meaningful. Without it, the "API unreachable" case could never pass.
- **Evidence:** `frontend/playwright.config.js`, `frontend/e2e/`, `frontend/package.json`, `.github/workflows/ci.yml` (`frontend` job), `docs/TEST_PLAN.md`, ADR-021, ADR-022.

### ADR-024 — CI runs on a pinned platform: `ubuntu-24.04`, Node 24 action majors and Node 24 LTS; Python 3.10 stays past its end of life
- **Decision:**
  - Every job in `ci.yml`, `deploy-backend.yml` and `no-ai-attribution.yml` runs on `ubuntu-24.04`. No job uses `ubuntu-latest`. Moving to a newer image, such as Ubuntu 26, is a separate, reviewed change.
  - Actions use their current major tags, all of which declare the `node24` runtime: `actions/checkout@v7`, `actions/setup-python@v7`, `actions/setup-node@v7`, `actions/cache@v6` and `actions/upload-artifact@v7`. They stay pinned to major tags (`@vN`), as before. Pinning to commit SHAs isn't part of this decision.
  - The `frontend` job runs on Node 24 (`node-version: "24"`). Nothing else pins Node: `frontend/package.json` gets no `engines` field and there's no `.nvmrc`, because `engines` would also change the Node version Vercel builds with. Vercel's Node version stays in its project settings.
  - Python 3.10 stays in the pytest matrix (3.10 and 3.11) and in `requires-python`, although it reached end of life on 2026-10-01. Production isn't affected: the image runs Python 3.11 (`python:3.11-slim-bookworm`), which the matrix and the other Python jobs test.
- **Why:**
  - `ubuntu-latest` moves to Ubuntu 26 from 2026-10-19. Every job already ran on the `ubuntu-24.04` image (24.04.5) through that label, so pinning it keeps the exact platform CI and the deploy gate were verified on.
  - Every job warned that `checkout@v4`, `setup-python@v5`, `setup-node@v4` and `cache@v4` target the deprecated Node 20 runtime and were being forced onto Node 24. The latest majors are the ones that get fixes, and their breaking changes don't touch these workflows' inputs:
    - checkout v6 keeps persisted credentials in a separate file. The deploy pushes to the Space through its own token URL.
    - checkout v7 refuses to check out fork pull-request code under `workflow_run`. Deploys come from `push` CI runs on `main`, which the check skips. A fork PR from a branch named `main`, already skipped by the superseded-commit guard, is now refused at checkout instead.
    - setup-python v7 removed the `pip-install` input, which isn't used. setup-node v5+ auto-caches only when `package.json` has a `packageManager` field, and the job sets `cache: npm` itself.
    - upload-artifact v7 still zips uploads by default.
  - Node 20 reached end of life on 2026-04-30. Node 24 is the current LTS line: active until 2026-10-20, then maintenance to 2028-04-30. Node 26 becomes LTS only on 2026-10-28.
    - Node 24 satisfies all 135 `engines.node` ranges in `frontend/package-lock.json`. Node 22 would need 22.13 or later, for ESLint 10.
    - Local development already uses Node 24. Lint, build and the Playwright suite (14/14) passed on Node 24.12.0 before CI moved.
  - Python 3.10:
    - It's the declared development interpreter: `.python-version`, the local venv (3.10.11), ruff `target-version = "py310"` and mypy `python_version = "3.10"`. The committed Kaggle GPU latency runs used Python 3.10 environments (TASK-016).
    - `tensorflow-cpu==2.15.0` publishes wheels for Python 3.9–3.11 only. Dropping 3.10 would leave CI testing a single interpreter and force a development-environment migration, and the matrix can't add 3.12 without the TensorFlow migration.
    - The 3.10 leg still resolves on the pinned runner, to CPython 3.10.21 from the image's tool cache.
    - Revisit at the TensorFlow / Keras migration, or as soon as the pinned runner or `setup-python` stops providing 3.10, whichever comes first. Then move `.python-version`, the venv, `requires-python`, the ruff and mypy targets, the matrix and `CLAUDE.md` together.
- **Evidence:** `.github/workflows/{ci,deploy-backend,no-ai-attribution}.yml`, `docs/CI.md`; the annotations of CI run `37657374067` and deploy run `37653146361` (Node 20 deprecation warning, `ubuntu-latest` migration notice, image `ubuntu-24.04` 20260927.320.1); each action's release notes and `action.yml` (`runs.using: node24`); the Node.js release schedule; PEP 619; `frontend/package-lock.json`; `pyproject.toml`; `.python-version`; `Dockerfile`.

### ADR-025 — Serving dependencies move to FastAPI 0.133.0, Starlette 1.3.1 (pinned), `python-multipart` 0.0.31, Pillow 12.3.0 and `anyio` 4.14.2, and a request-body cap closes the upload-buffering gap; TensorFlow 2.15 stays
- **Decision:**
  - FastAPI 0.111.0 → 0.133.0. It's the first release whose Starlette range admits 1.x ("Add support for Starlette 1.0.0+"). No later release notes a Starlette or upload fix the app needs. From 0.135.2, FastAPI requires Pydantic ≥ 2.9, which would force an unrelated Pydantic upgrade.
  - Starlette 0.37.2 → 1.3.1, pinned in `requirements.txt` and as `starlette>=1.3.1,<2.0` in `pyproject.toml`. FastAPI 0.133.0 declares only `starlette>=0.40.0`, which still admits vulnerable releases. 1.3.1 is the lowest release that fixes all seven Starlette advisories.
  - `python-multipart` 0.0.9 → 0.0.31, the lowest release that fixes all eight of its advisories. 0.0.32 is a performance change only.
  - Pillow 10.3.0 → 12.3.0, and its `pyproject.toml` range moves from `>=10.0,<11.0` to `>=12.3,<13.0`. Every fix is in 12.x, and the `<13.0` bound keeps the next major a reviewed change. Pillow stays in the serving requirements, where `pyproject.toml`'s core dependencies put it.
  - `anyio` 4.4.0 → 4.14.2, and its range moves to `>=4.14.2,<5.0`. It's outside the four packages TASK-020 named, and was added with the owner's approval (see Why).
  - `routes.py` uses Starlette's RFC 9110 status names, `HTTP_413_CONTENT_TOO_LARGE` and `HTTP_422_UNPROCESSABLE_CONTENT`. The codes are the same. Starlette 1.x warns, as a `UserWarning`, on every use of the old names.
  - `BodySizeLimitMiddleware` (`backend/app/core/body_limit.py`) caps a request body at `serve.max_upload_bytes` plus 64 KiB of multipart framing:
    - a declared `Content-Length` over the cap gets a 413 before any byte is read;
    - without one, reading stops as soon as the received bytes cross the cap;
    - under the cap the middleware is transparent, and the route's exact limit still decides (10 MiB accepted, 10 MiB + 1 byte → 413).
  - Unchanged: `tensorflow-cpu==2.15.0`, `numpy==1.26.4`, Pydantic 2.7.4, uvicorn 0.30.1 and every other pin.
- **Why:**
  - Every `POST /v1/captions` runs Starlette's multipart parser on `python-multipart`. On the old pins, `pip-audit -r requirements.txt` found 89 vulnerabilities in 7 packages, including multipart DoS advisories in both parser packages. After the upgrade it finds 26 in 3 packages, none in FastAPI, Starlette, `python-multipart`, Pillow or `anyio`. `SECURITY.md` § Dependency audit lists every advisory fixed and every finding left.
  - Pillow isn't on the serving path: uploads decode through `tf.io.decode_image` and `preprocess_image_tensor`. It ships in the image, and the `[hf]` baseline adapter uses it. The adapter only calls `Image.open`, `convert`, `new` and `save`, which the Pillow 11 and 12 removals don't touch.
  - The new releases need Python ≥ 3.10, which the 3.10 / 3.11 matrix (ADR-024) and the 3.11 image already satisfy.
  - The upgrade alone doesn't change buffering. FastAPI parses the form before any route or dependency runs, and Starlette spools a file part to a temporary file past 1 MiB, with no size limit. Measured with the body fed in 64 KiB chunks, as uvicorn does:
    - old and new pins alike read all 50 MB of a 50 MB upload, and spool it to disk, before the 413;
    - with the cap, a 50 MB upload with `Content-Length` is refused after 0 bytes, and a chunked one after 10.6 MB.
    - Against the real app, curl's 12 MB upload got the 413 in under 3 ms without sending its body, because `Expect: 100-continue` was answered with the final status.
  - The cap needs a middleware because the body is read before any dependency runs. It raises the route's `HTTPException(413)` from inside `receive()`, which FastAPI 0.133.0 re-raises when it occurs while reading the body ("If a middleware raises an HTTPException, it should be raised again"). The 413 then has the same `{"detail": ...}` body, CORS headers and `x-request-id` as any other error, and the middleware never builds a response itself.
  - `anyio`: Starlette writes a spooled upload chunk through `anyio.to_thread`. Starlette doesn't close a file part when the cap interrupts the parse, and in anyio 4.4.0 an idle worker thread kept its last work item, the file's bound `write`, referenced. A refused upload's temporary file (up to about 10 MiB) therefore stayed open until that thread got new work, which `gc.collect()` didn't change. 4.14.2 releases each work item, so the file closes as soon as the request ends, measured over real uvicorn with no garbage collection. It also fixes anyio's two advisories (CVE-2026-64847, CVE-2026-63374), though neither is reachable from serving.
  - TensorFlow 2.15 stays. No TensorFlow advisory requires action, and TF 2.16+ brings Keras 3, which breaks the `TextVectorization` save/load the model depends on (ADR-003). The `keras` 2.15.0 and `protobuf` 4.25.9 findings are fixed only in Keras 3 and protobuf ≥ 5.29.6, which TF 2.15 excludes. They need an untrusted model file or untrusted protobuf JSON, and serving loads only its own weights from the pinned Hub tag.
- **Consequences:**
  - Status codes and response shapes are unchanged. One precedence change: a body over the cap now gets 413 even when it would otherwise have got 415 (wrong type) or 503 (predictor loading), because the cap applies before the route and its dependencies run.
  - OpenAPI describes the upload as `contentMediaType: application/octet-stream` instead of `format: binary`, FastAPI's OpenAPI 3.1 form. Nothing else in the generated document changed. Nothing in the repository consumes it, and `/docs` still renders the upload field.
  - FastAPI ≥ 0.112 no longer installs `fastapi-cli`, `jinja2`, `email-validator`, `orjson`, `ujson` or `httpx` by default. Nothing in `src/`, `backend/` or `scripts/` imports them, so the image drops them. `httpx` stays a dev dependency for `TestClient`.
  - Not solved:
    - bodies up to the cap are still received, and without `Content-Length` up to about 10 MiB can be spooled to disk before the 413 (the file is released when the request ends);
    - a client that keeps sending after the 413 still uses bandwidth, though uvicorn discards the bytes;
    - the HF Spaces proxy can't be configured, and there's still no rate limiting.
  - The `click` 8.1.7 finding stays, with the `keras` and `protobuf` ones. It's outside TASK-020's scope and not reachable from serving. TASK-021 decides whether its scan blocks on these.
  - `test_refused_upload_leaves_no_temp_file_open` fails on anyio 4.4.0, so a downgrade is caught.
  - Under Starlette 1.x, `TestClient` warns that `httpx` is deprecated in favour of `httpx2`. It's test-only, and dev dependencies are out of scope here.
- **Evidence:** `requirements.txt`, `pyproject.toml`, `backend/app/core/body_limit.py`, `backend/app/main.py`, `backend/app/api/routes.py`, `backend/app/tests/test_body_size_limit.py`, `docs/SECURITY.md` § Dependency audit; OSV and `pip-audit` 2.10.1 output for both requirement sets; the FastAPI 0.133.0 and 0.135.2 release notes and its per-release `requires_dist` on PyPI; the Starlette 1.0–1.3.1 and `python-multipart` 0.0.31–0.0.32 release notes; the Pillow 11.0 and 12.0 release notes; `fastapi/routing.py` (0.133.0), `starlette/formparsers.py` (1.3.1) and `WorkerThread.run` in `anyio/_backends/_asyncio.py` (4.4.0 vs 4.14.2); TASK-020 in `TASKS.md` for the measurements.

### ADR-027 — The Space rebuilds only when an image input changed since the last successful deploy (revises ADR-017)
- **Decision:**
  - `deploy-backend.yml` still runs after every green CI run on `main`. It still deploys only the exact tested commit, skips a commit `main` has moved past, and verifies manual runs' CI first (ADR-017). After those guards, `python3 -m scripts.deploy_scope decide` decides. If no image input changed, the run ends green with a "Space deploy skipped" notice and job summary giving the reason, and nothing is pushed to the Space.
  - Image inputs (`IMAGE_INPUTS` in `scripts/deploy_scope.py`):
    - the Dockerfile's `COPY` sources: `requirements.txt`, `pyproject.toml`, `README.md`, `src/`, `backend/`, `configs/`, `models/`;
    - `Dockerfile` and `.dockerignore`, the build recipe and its context filter;
    - `.gitattributes`, because the Space builds from a git checkout of the deploy commit and this file decides how that checkout materialises files (LFS filters, line endings);
    - `.github/workflows/deploy-backend.yml`, which writes the Space's README config header, and `scripts/deploy_scope.py`. A change to the deploy procedure is proven by the run that introduces it.
    - A unit test fails if a Dockerfile `COPY` source isn't covered, so a new one can't fall outside the rule.
  - The comparison is a tree diff, `git diff --name-only --no-renames <baseline> <commit>`. A change reverted before it was ever pushed doesn't count, and a file moved out of an image directory does.
  - The baseline is the last successful deploy, recorded as a GitHub deployment in the `huggingface-space` environment.
    - The workflow creates it as its last step, only after the Space reports `RUNNING` and `/healthz` reports `model_loaded: true`. It names the GitHub commit and, in its payload, the commit pushed to the Space (`space_commit`), and gets a `success` status.
    - The next run reads the newest record created by `github-actions[bot]`. It skips only if that record has a `success` status, the Space's repository is still on its `space_commit` (public HF API, no token), and no image input differs.
  - Anything unknown deploys: no record, a record without `success`, an API error, a Space head other than `space_commit`, a baseline missing from the checkout, or a failed diff.
  - A manual `workflow_dispatch` run always deploys, after its CI verification.
  - Permissions: the workflow adds `deployments: write`. Its checkout sets `persist-credentials: false`, so the token reaches only the two steps given it.
  - `README.md` is an image input, so a README-only commit redeploys.
- **Why:**
  - Under ADR-017 every green commit force-pushed and rebuilt the Space, and restarted the model, docs-only commits included. `workflow_run` has no path filter, so the check has to run inside the job.
  - The baseline is the last deploy that actually reached the Space and passed its gate, not the parent commit. With A deployed, B (an image change) skipped, superseded, cancelled or failed, and then C (docs only), comparing C with its parent B would skip, and the Space would never get B. A run that doesn't pass the gate writes no record, so the baseline never advances past what the Space really serves.
  - The `space_commit` check covers a deploy that pushed and then failed its gate. That deploy leaves the Space on its commit with no record. If the next commit reverts it, its tree equals the baseline's, but the Space isn't on the baseline. Comparing the Space's head with the recorded commit catches this, and any push made outside the workflow.
  - GitHub deployments are the repository's own record of what is deployed where. They hold an explicit SHA, show under the repository's Environments, and need no new service or storage. `deployments: write` can't change code or secrets. The alternatives each fell short:
    - a git tag or ref needs `contents: write`;
    - the run history has no explicit deployed SHA, and ties the rule to step names;
    - artifacts expire;
    - Space variables restart the Space when they're written.
  - `README.md` is copied into the image on purpose. `pyproject.toml` declares `readme = "README.md"` for the in-image `pip install -e .`, so it's package metadata, and the deploy commit prepends the Space's config header, so the deployed copy is the Space's card. Skipping README-only commits would leave both out of step with `main`. The cost is that README edits still rebuild.
  - Race safety: the `deploy-backend` concurrency group (no cancellation) runs deploys one at a time, so a lookup and a record can't interleave with another deploy. The superseded-commit guard runs before the decision, and only `DEPLOY_SHA` is pushed and recorded.
- **Consequences:**
  - Docs, tests, frontend, results, notebooks and CI-only commits no longer restart the Space. Changes under `backend/`, including `backend/app/tests/`, still redeploy, because `backend/` is copied whole.
  - A skip doesn't check the live Space's health. A Space that breaks without a push (a runtime error, HF infrastructure) is no longer rebuilt by the next unrelated commit; run the workflow manually.
  - Skipped runs don't pick up a new `python:3.11-slim-bookworm` base image or newer unpinned transitive dependencies. Those arrive with the next image change or a manual run.
  - Space variables and secrets (weights revision, CORS origins) live outside git and outside this rule. Changing one restarts the Space by itself (ADR-018).
  - The first run after this change finds no record, so it deploys and writes the first baseline.
  - If writing the record fails after a healthy deploy, that step fails the run, and the next run, finding no trusted baseline, redeploys.
- **Evidence:** `.github/workflows/deploy-backend.yml`, `scripts/deploy_scope.py`, `tests/unit/test_deploy_scope.py`, `Dockerfile`, `.dockerignore`, `pyproject.toml` (`readme`), `docs/CI.md` § `deploy-backend.yml`, `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md` § 7; the rule run over real ranges and the deploy runs recorded in TASK-022 in `TASKS.md`; ADR-017, ADR-018.

### ADR-028 — A deploy passes only after one real caption from the live Space (extends ADR-017 and ADR-027)
- **Decision:**
  - After the health gate, `deploy-backend.yml` runs `python3 -m scripts.smoke_caption` against the Space domain that gate verified. The domain is passed as the `health` step's `space_url` output, checked against a hostname pattern first.
  - It sends one `POST /v1/captions`, the same multipart `image` upload the SPA sends, with:
    - a 64×64 RGB gradient PNG (about 8 KB) built in code;
    - `Origin: https://image-captioning-system.vercel.app`;
    - its own `x-request-id`.
  - It passes only on HTTP 200 with a `CaptionResponse`-shaped JSON body:
    - a non-empty `caption`;
    - the `model_version` that `/healthz` reports;
    - a non-empty `decode_strategy` and a positive `latency_ms`;
    - the sent request id, in the body and in the `x-request-id` header;
    - `Access-Control-Allow-Origin` equal to the Vercel origin.
  - Retries and logging:
    - These count as a Space still waking and are retried every 10 s, for up to 5 minutes:
      - connection, TLS and timeout errors;
      - a body cut off mid-read;
      - HTTP 502/503/504;
      - a `/healthz` reporting the model not loaded yet.
    - No attempt starts after that deadline or runs past it, and the step has a 10-minute timeout.
    - Any other status fails at once, logged with the status and the error's `detail`.
    - A repeated `Access-Control-Allow-Origin` or `x-request-id` header fails, because browsers reject two origin values.
    - Log lines are escaped as GitHub's workflow commands require, so a server-supplied message can't inject a command.
    - Nothing of the image or the caption is logged, and no token is used.
  - A failure fails the deploy, so the deploy record isn't written (ADR-027) and the next green commit deploys again.
  - The smoke script is an image input, like the rest of the deploy procedure, so a change to it is proven by a deploy.
- **Why:**
  - `/healthz` reports `model_loaded: true` as soon as the predictor service exists. It doesn't decode an image or run InceptionV3 and the transformer. A broken TensorFlow graph, a preprocessing error, a multipart or middleware regression, or a response-contract change could pass that gate while every real upload fails.
  - The caption text isn't asserted. It depends on the weights and isn't a stable contract. A non-empty caption with the reported model version proves the real model ran, and asserting the text would turn a deliberate weights promotion (ADR-018) into a failed deploy.
  - A generated image keeps binaries out of the repository and can't drift with a frontend asset. Two tests tie it to the real stack:
    - one decodes it through the serving decoder, `bytes_to_tensor`;
    - another runs the whole check against `create_app()` with a stand-in predictor, so the request and the checks match the real middleware.
  - Request id: the HF proxy injects its own `x-request-id` when a client sends none, so the check sends its own and requires it echoed.
  - CORS: only the allowed case is checked.
    - The HF Spaces proxy answers CORS itself and reflects any `Origin`, preflights included. A disallowed origin is therefore allowed at the edge whatever the app does, and TASK-023's negative criterion can't be asserted against production.
    - Separately, `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` doesn't reach the app: `load_config` passes the YAML as constructor arguments, which outrank environment variables. Today the SPA works only through the proxy.
    - The positive check catches the SPA losing access, for example if the proxy stopped reflecting. Both findings are in `SECURITY.md` § Known gaps.
- **Consequences:**
  - Each deploy sends one public caption request, about 1–2 s of inference on cpu-basic. Skipped runs (ADR-027) send none.
  - A Space that is healthy but can't caption now fails its deploy instead of passing it.
  - Not covered: caption quality, latency budgets, uptime monitoring between deploys, and enforcing the CORS allow-list.
- **Evidence:** `.github/workflows/deploy-backend.yml`, `scripts/smoke_caption.py`, `tests/unit/test_smoke_caption.py`, `backend/app/api/routes.py`, `backend/app/core/logging.py`, `backend/app/main.py`, `src/captioning/config/loader.py`; the live probes and the pre-push smoke run recorded in TASK-023 in `TASKS.md`; ADR-017, ADR-018, ADR-027.
