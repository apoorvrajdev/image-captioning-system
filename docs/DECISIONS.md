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
