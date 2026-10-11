---
name: inference-api
description: Acceptance criteria and definition of done for the FastAPI inference service and the decoding path behind it — routes, schemas, PredictorService, weights loader, lifespan, CaptionPredictor, greedy/beam decoders. Use whenever changing backend/app/** or src/captioning/inference/**, or the /healthz or /v1/captions contract.
---

# Inference API — acceptance criteria

## Scope
`backend/app/**` (main, api/routes, core/config + logging, schemas, services, utils/image, tests),
`src/captioning/inference/**`, `serve:` section of `configs/base.yaml`.
Wire contract consumers: `frontend/src/services/api.js`, `CaptionResult.jsx`, `StatusBadge.jsx`.

## Expected behaviour (contract)
- GIVEN the predictor is loaded WHEN `GET /healthz` THEN 200 `{status:"ok", model_loaded:true, model_version, api_version, timestamp}`.
- GIVEN the lifespan hasn't finished WHEN `GET /healthz` THEN still 200 with `status:"loading", model_loaded:false`; WHEN `POST /v1/captions` THEN 503.
- GIVEN a valid JPEG/PNG/WebP/BMP upload ≤ `serve.max_upload_bytes` THEN 200 `CaptionResponse{caption, model_version, decode_strategy, latency_ms, request_id}` and an `x-request-id` response header.
- Content type not in `ALLOWED_CONTENT_TYPES` → 415. Empty body → 400. Over the limit → 413. Undecodable bytes → 422 (`ImageDecodeError`). Errors use `{"detail": ...}`.
- A request body over `serve.max_upload_bytes` + 64 KiB of multipart framing → the same 413 from `BodySizeLimitMiddleware` (`core/body_limit.py`), before the parser buffers it: no byte read when `Content-Length` declares it, and reading stops at the cap when it doesn't. Under the cap the middleware is transparent and the route's exact limit decides (ADR-025).
- Before any decode, `check_image_dimensions` reads the header (Pillow, JPEG/PNG/BMP parsers only, no pixel decode). Over `MAX_IMAGE_SIDE` (16,384 px) or `MAX_IMAGE_PIXELS` (40 MP) → 413. Bytes without a readable JPEG/PNG/BMP header, GIF included → 422 (TASK-026).
- Uploaded bytes pass through `bytes_to_tensor` → `preprocess_image_tensor` (the training function). No other preprocessing path.
- One `CaptionPredictor` per process, built in the lifespan, `warmup()` when `BACKEND_WARMUP=true`, TF work via `anyio.to_thread.run_sync`.
- At most `MAX_CONCURRENT_INFERENCES` (2) requests decode or predict at once. The next → 503 `Retry-After: 2` at once, never queued. A slot frees only when its worker thread has finished, even if the caller was cancelled. `/healthz` and `get_predictor_service` stay async, so neither needs a worker thread (TASK-027).
- `BACKEND_WEIGHTS_HUB_REPO` set → `resolve_weights` turns `BACKEND_WEIGHTS_HUB_REVISION` into a commit id (a full 40-character id as given, or a tag in `PINNED_HUB_REVISIONS`; anything else → `ValueError` before any download) and calls `snapshot_download` at that commit for the weights file and `vocab.json` only. Either missing → `FileNotFoundError` naming it. Unset → local paths (ADR-032).

## Edge cases (each needs a test)
- Every status code above (existing: `backend/app/tests/test_captions.py`, `test_health.py`).
- Body-size cap: declared oversize body (0 bytes read), undeclared oversize body (stops at the cap), under-cap body reaches the route, a refused upload leaves no temp file open (needs `anyio` ≥ 4.14.2) (`test_body_size_limit.py`, raw ASGI so bytes read can be counted).
- Admission: a free slot admits; a full house → immediate 503 with nothing decoded; slots come back after success, a predict error and a decode error; a cancelled request holds its slot until its thread ends; `/healthz` → 200 while inference holds every worker thread (`test_inference_admission.py`, gated worker threads, no sleeps).
- Pixel limits: exactly at the limits passes; one over on either side or the product → 413; zero, truncated, malformed or non-JPEG/PNG/BMP headers → 422; the guard runs before `tf.io.decode_image` (`test_image_guard.py` with header-only bytes, `tests/unit/test_upload_image_decode.py`).
- Incoming `x-request-id` is echoed. Missing → a generated UUID.
- Weights loader: local vs hub mode, downloader args incl. the commit id and `allow_patterns`, cache dir, custom filename, a pinned tag → its commit, refused revisions (branch, unpinned tag, short or upper-case id, another repo's tag), a commit missing a required file (`test_weights_loader.py`, injected downloader, no network). Not yet covered: a network failure during download → startup error.
- Beam decoder: length penalty, repetition penalty, n-gram blocking, EOS termination, detokenise (`tests/unit/test_beam_decoder.py`). Not yet covered: beam width 1 ≡ greedy.

## Required tests
| Level | File | Must cover |
|---|---|---|
| route | `backend/app/tests/test_*.py` | contract changes via `FakePredictorService` (no TF) |
| unit | `tests/unit/test_beam_decoder.py` | decoding changes |
| manual smoke (optional, needs weights) | `uvicorn …` + `curl -F image=@x.jpg /v1/captions` | lifespan end-to-end |

## Definition of done — ALL must pass
- [ ] Contract change ⇒ Pydantic schema updated + route test per new status/field + `frontend/src/services/api.js` / UI updated in the same task (see `frontend` skill).
- [ ] `pytest backend/app/tests -q` green (<1 s, and must not import TensorFlow).
- [ ] Full `pytest tests backend/app/tests -q`, ruff, mypy green.
- [ ] Parity audit 4/4 if `src/captioning/inference` or preprocessing changed.
- [ ] No model or TF code in `api/routes.py`. No research knobs in `BackendSettings` and no serving paths in `AppConfig`.
- [ ] New env vars documented in `.env.example` and `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md` if deploy-relevant.

## Verification commands
```bash
.venv/Scripts/pytest.exe backend/app/tests -q
.venv/Scripts/pytest.exe tests backend/app/tests -q
.venv/Scripts/python.exe -m scripts.notebook_module_audit   # if inference/preprocessing touched
```

## Non-negotiables
- Default `serve.decode_strategy: greedy` (notebook parity). Beam is a config override.
- Validation stays at the boundary: allow-list, not block-list. Never log image bytes or full exception payloads containing user data.
- CORS origins come from config/env (`CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS`), never `*`. `allow_credentials=False`.
