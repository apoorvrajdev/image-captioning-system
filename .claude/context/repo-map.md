# Repo map

Read this first. Then grep `symbols.tsv` for the symbol, check `deps.json` for
blast radius (`imported_by`), and open only those files.
Rebuild the generated files after adding/moving modules:
`.venv/Scripts/python.exe .claude/context/build_index.py`

## Areas

| Path | Owns | Depends on | Touch when | Skill |
|---|---|---|---|---|
| `src/captioning/config/` | `AppConfig` (Pydantic v2, `extra="forbid"`, env prefix `CAPTIONING__`) + YAML loader | — | adding any hyperparameter / serve knob | ml-core |
| `src/captioning/preprocessing/` | caption text cleanup, `preprocess_image_tensor` (shared train+serve), `CaptionTokenizer` (TextVectorization, pickle + JSON vocab) | config | tokenisation or image-normalisation change (parity-sensitive) | ml-core |
| `src/captioning/models/` | InceptionV3 encoder, 1-layer TF encoder, 8-head decoder, `ImageCaptioningModel` (custom train/test step), `factory.py` (lazy TF class builders) | config | architecture change (needs retrain + new model version) | ml-core |
| `src/captioning/data/` | COCO loading, image-level train/val split, `tf.data` pipeline | preprocessing, config | dataset / sampling change | ml-core |
| `src/captioning/training/` | losses (label smoothing), warmup+cosine schedule, callbacks, trainer | models, data | training-recipe change (behind opt-in flags) | ml-core |
| `src/captioning/inference/` | `CaptionPredictor` (`from_artifacts`, `warmup`, `predict_tensor/path`), greedy + beam decoders, disk image loader | models, preprocessing | decode behaviour | inference-api / ml-core |
| `src/captioning/evaluation/` | BLEU (sacrebleu), CIDEr, METEOR, ROUGE-L, corpus runner, `benchmark.py` (`RunMeta`, `write_run_artifacts`), per-sample inspection, `slice.py` (`load_eval_slice`, `slice_fingerprint`), `comparison.py` (`load_run`, `build_summary`: Phase 3 cross-run summary), `latency.py` (`LatencySettings`, `time_load`, `measure_latency`: Phase 3 latency timing, CLI `scripts/benchmark_latency.py`), `dashboard.py` (`build_dashboard_data`: the SPA's Phase 3 dashboard JSON from committed results, CLI `scripts/export_dashboard_data.py`, ADR-021) | — | metrics / artefact contract | evaluation |
| `src/captioning/baselines/` | Phase 3 `Captioner` interface (normalises via `preprocess_caption` → `strip_sentinels`), `CNNCaptioner` (wraps `CaptionPredictor`), `HFCaptioner` (pinned Hub revision; `torch`/`transformers` imported lazily, `[hf]` extra) | config, evaluation, preprocessing; inference (lazy) | adding or changing a compared model (ADR-019) | evaluation |
| `src/captioning/utils/` | structlog setup, `set_global_seed`, SHA-256 hashing | — | rarely | — |
| `backend/app/main.py` | `create_app()` factory + lifespan (load config → resolve weights → predictor → warmup → `app.state.predictor_service`) | captioning.inference, services | startup / middleware / CORS | inference-api |
| `backend/app/api/routes.py` | `/healthz`, `POST /v1/captions` (thin: validate → service → schema) | schemas, services, utils | endpoint contract | inference-api |
| `backend/app/core/` | `BackendSettings` (env prefix `BACKEND_`), structlog + `RequestContextMiddleware` (`x-request-id`), `BodySizeLimitMiddleware` (request-body cap) | — | serving knobs, logging | inference-api |
| `backend/app/schemas/caption.py` | `CaptionResponse`, `HealthResponse`, `ErrorResponse` — the wire contract | — | any response shape change (frontend must follow) | inference-api |
| `backend/app/services/` | `PredictorService` (anyio thread offload, latency), `weights_loader.resolve_weights` (HF Hub `snapshot_download`, injectable downloader) | captioning.inference | serving logic | inference-api |
| `backend/app/utils/image.py` | content-type allow-list, `bytes_to_tensor` → `preprocess_image_tensor`, `ImageDecodeError` | preprocessing | upload decoding | inference-api |
| `backend/app/tests/` | route tests with `FakePredictorService` (no TF), weights-loader tests (offline stub) | app | any backend change | inference-api |
| `frontend/src/services/api.js` | the only backend boundary: `checkHealth`, `captionImage`, `ApiError{kind}`, `VITE_API_BASE`, timeouts | — | any API call | frontend |
| `frontend/src/generated/phase3-dashboard.json` | Phase 3 dashboard data, **generated** by `scripts/export_dashboard_data.py` from `results/` (ADR-021). Never hand-edited; `test_dashboard_export.py` fails if it drifts. Imported at build time by `components/Phase3Dashboard.jsx` | `results/phase3-comparison/`, `results/phase3-latency-*/` | after any new comparison summary or latency run: re-export | evaluation |
| `frontend/src/App.jsx` + `components/` | request lifecycle state (`file/result/error/loading`), the view switch (caption flow / Phase 3 dashboard, no router, ADR-022), UploadZone (client validation), StatusBadge (10 s health poll), CaptionResult, ErrorBanner, `Phase3Dashboard` (quality + per-device latency tables, caveats, provenance; static data, no request) | services/api.js; `generated/phase3-dashboard.json` | UI change | frontend |
| `frontend/e2e/` + `frontend/playwright.config.js` | Playwright browser E2E on Chromium against `vite preview` of the production build (ADR-023): `support.js` (API mock fixture + console-error fixture), `caption-flow.spec.js` (TASK-007), `phase3-dashboard.spec.js` (TASK-018) | `src/` bundle, `src/generated/phase3-dashboard.json` | any UI change: add or extend a spec | frontend |
| `scripts/` | CLIs: `train`, `evaluate`, `predict`, `inspect_predictions`, `notebook_module_audit` (parity gate), `bootstrap_dev_artifacts`, `rescore_nltk_bleu` + `categorize_predictions` (Stage 0 eval audit), `check_pip_audit` (CI pip-audit baseline gate, stdlib only), `deploy_scope` (deploy-backend's image-change decision + deploy record, stdlib only, ADR-027), `smoke_caption` (deploy-backend's post-deploy caption check, stdlib only, ADR-028) | captioning | new CLI entrypoint | evaluation / ml-core |
| `configs/` | `base.yaml` (mirrors notebook cell 6), `train/debug.yaml` (smoke), `train/stabilized.yaml` (4 ablatable flags) | schema | experiment configs | ml-core |
| `tests/unit/` | CPU-only offline unit tests (config, tokenizer, preprocessing, splits, beam, metrics, training stability, hashing) | captioning | any library change | all |
| `notebooks/01_ieee_*.ipynb` | **FROZEN** IEEE research artefact (SHA-256 in `.paper-notebook.sha256`) | — | never | ml-core |
| `models/v1.0.0/` | local artefacts (`vocab.json` tracked; `model.h5`, `vocab.pkl` untracked). Prod weights come from HF Hub | — | never in place; bump version dir | deployment |
| `results/<run_id>/` | committed evaluation artefact sets (`run_meta.json`, `metrics.json`, `predictions.jsonl`, `diagnostics.jsonl`, `report.md`; beam run also has Stage 0 audit files `metrics_5ref.json`, `categories.jsonl`, `verdict.md`; Phase 3 `phase3-*-greedy/` runs add `comparison_meta.json`; `phase3-comparison/` holds `comparison.json` + `comparison.md`; Phase 3 latency runs `phase3-latency-*/` hold only `latency.json`) | — | new eval run → new dir; never rewrite old runs | evaluation |
| `docs/` | phase notes, runbooks, `EVAL_METHODOLOGY.md`, `CI.md`, plus `MEMORY.md` / `TASKS.md` / `DECISIONS.md` / `TEST_PLAN.md` / `SECURITY.md` | — | end of every task (MEMORY/TASKS) | — |
| `.github/workflows/` | `ci.yml` (6 jobs incl. `pre-commit` and `security`: pip-audit gate + full-history gitleaks; `frontend` runs lint, build, Playwright E2E and `npm audit --omit=dev`), `.github/pip-audit-baseline.txt` (reviewed pip-audit exceptions, ADR-026), `deploy-backend.yml` (push to HF Space after green CI, only when an image input changed since the last successful deploy, ADR-027), `no-ai-attribution.yml` (commit-msg policy) | — | CI change | deployment |
| `Dockerfile` (root) | HF Space image: python 3.11-slim, UID 1000, port 7860, 1 uvicorn worker, HEALTHCHECK `/healthz` | requirements.txt | runtime change | deployment |
| `.claude/` | agent workflow config: `settings.json` (edit guardrails, git confirm prompts, index-rebuild hook), `skills/*/SKILL.md` (acceptance criteria + DoD), `agents/*.md` (lanes), this map + `build_index.py` | — | workflow change; review like code | — |

## Request flow (serving)

`UploadZone` → `App.jsx` → `services/api.js captionImage` → `POST /v1/captions`
→ `routes.caption_image` (415/400/413 checks) → `PredictorService.caption_image_bytes`
(anyio thread) → `utils/image.bytes_to_tensor` → `preprocess_image_tensor`
→ `CaptionPredictor.predict_tensor` → greedy/beam decode → `CaptionResponse`.

## Not tracked / local only

`.venv/`, `frontend/node_modules/`, `models/**/model.h5`, `outputs/`, `mlruns/`,
`reference notebook/`, `AI_MODEL_SELECTION_GUIDE.txt`, `.claude/settings.local.json`, and the
generated index here (`files.txt`, `dir-weights.txt`, `symbols.tsv`, `deps.json`, `hotspots.txt`).
