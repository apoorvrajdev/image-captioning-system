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
| `src/captioning/evaluation/` | BLEU (sacrebleu), CIDEr, METEOR, ROUGE-L, corpus runner, `benchmark.py` (`RunMeta`, `write_run_artifacts`), per-sample inspection | — | metrics / artefact contract | evaluation |
| `src/captioning/utils/` | structlog setup, `set_global_seed`, SHA-256 hashing | — | rarely | — |
| `backend/app/main.py` | `create_app()` factory + lifespan (load config → resolve weights → predictor → warmup → `app.state.predictor_service`) | captioning.inference, services | startup / middleware / CORS | inference-api |
| `backend/app/api/routes.py` | `/healthz`, `POST /v1/captions` (thin: validate → service → schema) | schemas, services, utils | endpoint contract | inference-api |
| `backend/app/core/` | `BackendSettings` (env prefix `BACKEND_`), structlog + `RequestContextMiddleware` (`x-request-id`) | — | serving knobs, logging | inference-api |
| `backend/app/schemas/caption.py` | `CaptionResponse`, `HealthResponse`, `ErrorResponse` — the wire contract | — | any response shape change (frontend must follow) | inference-api |
| `backend/app/services/` | `PredictorService` (anyio thread offload, latency), `weights_loader.resolve_weights` (HF Hub `snapshot_download`, injectable downloader) | captioning.inference | serving logic | inference-api |
| `backend/app/utils/image.py` | content-type allow-list, `bytes_to_tensor` → `preprocess_image_tensor`, `ImageDecodeError` | preprocessing | upload decoding | inference-api |
| `backend/app/tests/` | route tests with `FakePredictorService` (no TF), weights-loader tests (offline stub) | app | any backend change | inference-api |
| `frontend/src/services/api.js` | the only backend boundary: `checkHealth`, `captionImage`, `ApiError{kind}`, `VITE_API_BASE`, timeouts | — | any API call | frontend |
| `frontend/src/App.jsx` + `components/` | request lifecycle state (`file/result/error/loading`), UploadZone (client validation), StatusBadge (10 s health poll), CaptionResult, ErrorBanner | services/api.js | UI change | frontend |
| `scripts/` | CLIs: `train`, `evaluate`, `predict`, `inspect_predictions`, `notebook_module_audit` (parity gate), `bootstrap_dev_artifacts`, `rescore_nltk_bleu` + `categorize_predictions` (Stage 0 eval audit) | captioning | new CLI entrypoint | evaluation / ml-core |
| `configs/` | `base.yaml` (mirrors notebook cell 6), `train/debug.yaml` (smoke), `train/stabilized.yaml` (4 ablatable flags) | schema | experiment configs | ml-core |
| `tests/unit/` | CPU-only offline unit tests (config, tokenizer, preprocessing, splits, beam, metrics, training stability, hashing) | captioning | any library change | all |
| `notebooks/01_ieee_*.ipynb` | **FROZEN** IEEE research artefact (SHA-256 in `.paper-notebook.sha256`) | — | never | ml-core |
| `models/v1.0.0/` | local artefacts (`vocab.json` tracked; `model.h5`, `vocab.pkl` untracked). Prod weights come from HF Hub | — | never in place; bump version dir | deployment |
| `results/<run_id>/` | committed evaluation artefact sets (`run_meta.json`, `metrics.json`, `predictions.jsonl`, `diagnostics.jsonl`, `report.md`; beam run also has Stage 0 audit files `metrics_5ref.json`, `categories.jsonl`, `verdict.md`) | — | new eval run → new dir; never rewrite old runs | evaluation |
| `docs/` | phase notes, runbooks, `EVAL_METHODOLOGY.md`, `CI.md`, plus `MEMORY.md` / `TASKS.md` / `DECISIONS.md` / `TEST_PLAN.md` / `SECURITY.md` | — | end of every task (MEMORY/TASKS) | — |
| `.github/workflows/` | `ci.yml` (5 jobs incl. `pre-commit`), `deploy-backend.yml` (push to HF Space after green CI), `no-ai-attribution.yml` (commit-msg policy) | — | CI change | deployment |
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
