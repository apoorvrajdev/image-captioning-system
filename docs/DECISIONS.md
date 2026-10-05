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
