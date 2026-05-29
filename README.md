<h1 align="center">Image Captioning System</h1>

<p align="center">
  <strong>CNN + Transformer image-to-language pipeline, lifted from an IEEE-published research notebook into a typed, tested, full-stack production codebase.</strong>
</p>

<p align="center">
  <img alt="Python 3.10+"     src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="TensorFlow 2.15"  src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">
  <img alt="FastAPI"          src="https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img alt="Pydantic v2"      src="https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white">
  <img alt="React 19"         src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black">
  <img alt="Vite 8"           src="https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white">
</p>

<p align="center">
  <img alt="Ruff"             src="https://img.shields.io/badge/lint-ruff-261230?style=flat-square&logo=ruff&logoColor=white">
  <img alt="mypy strict"      src="https://img.shields.io/badge/typed-mypy%20strict-1F5082?style=flat-square">
  <img alt="Tests"            src="https://img.shields.io/badge/tests-90%20passing-brightgreen?style=flat-square">
  <img alt="Pre-commit"       src="https://img.shields.io/badge/pre--commit-enabled-FAB040?style=flat-square&logo=pre-commit&logoColor=white">
  <img alt="IEEE Published"   src="https://img.shields.io/badge/IEEE-published-00629B?style=flat-square&logo=ieee&logoColor=white">
  <img alt="License: MIT"     src="https://img.shields.io/badge/license-MIT-blue?style=flat-square">
</p>

<p align="center">
  A deliberately scoped multimodal-AI showcase that takes a published research notebook and turns it into the kind of codebase a serving team would actually maintain — typed configuration, a structured FastAPI inference service, a polished React SPA, a parity-audit gate against the original notebook, and an honest roadmap that names what is shipped and what is not.
</p>

---

## Status

> 🚧 **Active build.** The research → modular conversion (Phase 1) is complete and the full inference stack (Phase 2A backend + 2B frontend) is operational end-to-end: a React 19 / Vite 8 SPA posts multipart uploads to `POST /v1/captions`, the FastAPI service returns a typed `CaptionResponse`, and the lifespan-managed `CaptionPredictor` is reused across every request with a warm graph and no per-call TF rebuilds. The IEEE notebook is preserved verbatim and protected by a SHA-256 freeze check. A four-stage parity audit ([`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py)) re-implements caption preprocessing, tokenizer vocabulary + encoding, image preprocessing, and the decoder forward pass inline and asserts the modular path is byte-identical (or `tf.allclose`-identical) to the notebook. Phase 1b (training stabilization) shipped beam search, the full corpus metric suite (BLEU-1..4 / CIDEr / METEOR / ROUGE-L), a benchmark runner that emits one machine-readable artefact set per evaluation, and a stabilized training config that gates label smoothing / cosine LR / warmup / dropout-free validation behind ablatable flags. Phase 2C (public deployment) is now in flight — workstream **D (backend test suite)** is complete: 12 new FastAPI route tests use a duck-typed fake predictor service to cover the full 200 / 400 / 413 / 415 / 422 / 503 contract end-to-end without loading TensorFlow, dropping the backend slice from a cold-start liability to a 0.3-second suite. The remaining workstreams (Dockerfile, HuggingFace Hub weights hosting, HF Spaces deploy, Vercel deploy, production CORS, GitHub Actions CI/CD, runbook) are sequenced in the [Roadmap](#-roadmap) below.

> ⚠️ **Caption quality disclaimer.** The weights committed under [`models/v1.0.0/`](models/v1.0.0/) are **bootstrap dev artefacts** produced by [`scripts/bootstrap_dev_artifacts.py`](scripts/bootstrap_dev_artifacts.py): the architecture is wired correctly but every weight is randomly initialised. They exist to exercise the serving stack (lifespan, predictor wiring, multipart upload, frontend integration) before a real COCO-trained checkpoint is dropped in. Live captions therefore look like noise today — that is the *intended* state of the bootstrap path, not a regression. See [Current model quality status](#-current-model-quality-status) for what is being done about it.

---

## 📌 What Is This Project?

Image Captioning System is a research-to-production conversion of the IEEE paper *"AI Narratives: Bridging Visual Content and Linguistic Expression"*. The original work — a Kaggle notebook training an InceptionV3-encoder + multi-head Transformer-decoder on MS COCO — is preserved verbatim as the canonical research artefact. Around it sits a typed Python package, a FastAPI inference service, and a React SPA that together turn the published model into something a serving team could actually run, version, and reason about.

It is **not** a hosted product (yet — Phase 2C is shipping that), and it is **not** a thin Streamlit wrapper around `model.predict`. What this project *is* is a deliberate engineering showcase aimed at hiring teams evaluating ML, multimodal-AI, and backend skills, and at anyone who has ever wondered what it actually takes to lift a research notebook into a codebase the rest of an engineering org can build on. Every architectural decision in this repository is one I can defend in an interview.

---

## 🎯 Why It Matters

Research notebooks and production ML systems are different artefacts with different audiences. A notebook proves an idea works. A production system has to **survive being maintained** — by people who did not write it, on schedules nobody planned, against inputs the original author never anticipated. The hardest part of an ML career is not getting a model to converge once; it is making the resulting pipeline *legible, typed, testable, deployable, and replaceable* without losing the behaviour the paper claimed.

This project demonstrates that conversion end-to-end at a scale one engineer can build and reason about:

- **Parity-gated refactor** — the notebook stays byte-stable and a four-stage audit script asserts the modular package reproduces the notebook's behaviour at every behavioural seam.
- **Strict typed configuration** — Pydantic v2 with `extra="forbid"` so a typo in a hyperparameter is a load-time error, not a silent training run that produces wrong numbers.
- **Lifespan-managed inference** — one warm `CaptionPredictor` shared across every HTTP request, not a graph rebuilt per call.
- **Train/serve shared preprocessing** — the same `preprocess_image_tensor` runs in `tf.data` pipelines and at inference, so the bytes that enter the model in training are byte-identical to the bytes that enter it at serve time.
- **Stabilized training experiments behind ablatable flags** — every quality intervention is opt-in, so any delta between two runs is attributable to one named change rather than a tangled rewrite.
- **Reproducible benchmarking** — every evaluation writes a machine-readable `metrics.json` + `diagnostics.jsonl` set, so two checkpoints (or one checkpoint with two decoders) can be diffed without bespoke parsers.

---

## 💡 What This Project Demonstrates

- Lifting a research notebook into an **installable, typed Python package** (`src/` layout) without breaking the published architecture.
- A production-style **FastAPI** inference service with lifespan-managed model loading, structured logging, request-ID propagation, and a typed Pydantic schema for every payload.
- A polished **React 19 + Vite 8 + Tailwind v4** SPA with drag-and-drop upload, client-side validation, `AbortController` timeouts, typed `ApiError` classification, and a polled health badge.
- **Pydantic v2 strict configuration** with YAML + env-var overrides and `extra="forbid"` to eliminate the silent-defaults failure mode.
- **Custom multi-head Transformer decoder** with masked sparse-categorical cross-entropy, masked accuracy, learned (not sinusoidal) positional embeddings, and the IEEE paper's exact dropout / head configuration.
- **Beam search decoder** with length normalisation and n-gram repetition suppression alongside greedy, selectable per inference call and per evaluation run.
- **Corpus-level metric suite** — BLEU-1..4 (sacrebleu), CIDEr, METEOR, ROUGE-L — emitted as one typed artefact per run.
- **Notebook freeze + parity audit** — SHA-256 lock on the IEEE notebook plus a four-stage inline re-implementation that fails CI if the modular path drifts.
- **Pre-commit governance** — Ruff, mypy (strict), `nbstripout`, `gitleaks`, line-ending and TOML/YAML hygiene, all enforced before commits land.
- **Clean Git workflow** with Conventional Commits and small, reviewable changesets ([`CLAUDE.md`](CLAUDE.md) codifies the contribution rules).

---

## 🏗️ Architecture

```
                       ┌───────────────────────────────────────┐
                       │     React 19 + Vite 8 SPA             │
                       │   Tailwind v4 · AbortController · ApiError │
                       └──────────────────┬────────────────────┘
                                          │ multipart/form-data
                       ┌──────────────────▼────────────────────┐
                       │      FastAPI 0.111 (Pydantic v2)      │
                       │  RequestContextMiddleware · /healthz · /v1/captions  │
                       └──────────────────┬────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────┐
                       │       PredictorService (anyio thread) │
                       │   bytes → tensor → predict → caption  │
                       └──────────────────┬────────────────────┘
                                          │ singleton, warmed in lifespan
                       ┌──────────────────▼────────────────────┐
                       │       CaptionPredictor (TensorFlow)   │
                       │   InceptionV3 → TF encoder → TF decoder → tokenizer │
                       └──────────────────┬────────────────────┘
                                          │
                       ┌──────────────────▼────────────────────┐
                       │       models/vX.Y.Z/ artefacts        │
                       │   model.h5 · vocab.json (versioned)   │
                       └───────────────────────────────────────┘

                 ┌───────────────────────────────────────────────┐
                 │  configs/*.yaml (Pydantic v2, extra="forbid") │
                 │  drives training, evaluation, AND serving     │
                 └───────────────────────────────────────────────┘
```

### Model topology

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────┐
│  Input image │──▶│  InceptionV3     │──▶│  Transformer     │──▶│  Transformer     │──▶│  Caption   │
│  299×299×3   │   │  encoder         │   │  encoder         │   │  decoder         │   │  string    │
└──────────────┘   │  (ImageNet,      │   │  (1 layer,       │   │  (2 layers,      │   └────────────┘
                   │   frozen)        │   │   1 head)        │   │   8 heads)       │
                   └──────────────────┘   └──────────────────┘   └──────────────────┘
                          ▼                       ▼                       ▼
                    [B, 64, 2048]          [B, 64, 512]            [B, T, vocab=15000]
```

### Components

- **CNN encoder** — [`models/encoder_cnn.py`](src/captioning/models/encoder_cnn.py). Pretrained InceptionV3 with the classification head removed; output reshaped to 64 spatial positions × 2048 channels. Weights frozen during training.
- **Transformer encoder** — [`models/transformer_encoder.py`](src/captioning/models/transformer_encoder.py). Single layer, one attention head. Projects InceptionV3 features into the decoder's embedding dimension.
- **Embeddings** — [`models/embeddings.py`](src/captioning/models/embeddings.py). Sum of token + *learned* positional embeddings, preserved verbatim from the published architecture.
- **Transformer decoder** — [`models/transformer_decoder.py`](src/captioning/models/transformer_decoder.py). Causal self-attention over partial captions, cross-attention over image features, feed-forward sub-block. 8 heads, `embedding_dim=512`, dropouts (0.1 / 0.3 / 0.5) preserved from the IEEE configuration.
- **Captioning model** — [`models/captioning_model.py`](src/captioning/models/captioning_model.py). Custom `train_step` / `test_step` with masked sparse-categorical cross-entropy and masked accuracy.
- **Tokenizer** — [`preprocessing/tokenizer.py`](src/captioning/preprocessing/tokenizer.py). `CaptionTokenizer` wraps `tf.keras.layers.TextVectorization`; persists vocabulary as both pickle (notebook-compatible) and JSON sidecar.
- **Inference** — [`inference/predictor.py`](src/captioning/inference/predictor.py). `CaptionPredictor.from_artifacts(weights, vocab, config)` loads everything once at boot, exposes `predict_path(...)` and `predict_tensor(...)` for stateless calls, and `warmup()` to amortise first-request latency.
- **Configuration** — [`config/schema.py`](src/captioning/config/schema.py). Pydantic v2 (`AppConfig` / `ModelConfig` / `TrainConfig` / `DataConfig` / `ServeConfig`); strict so typos in YAML or env vars become load-time errors.

**Why a monolith on a single process?** Splitting training, evaluation, and serving across services would burn the project's budget on Kubernetes manifests instead of the things a reviewer can actually click. A layered package + one FastAPI app captures the same separation-of-concerns thinking with a tenth of the operational surface area, and the seams are placed so pulling serving into its own container (Phase 2C) is a deployment change, not a refactor.

**Why TensorFlow 2.15 specifically?** TF 2.16 ships Keras 3 by default and silently breaks `TextVectorization` save/load — the project's `tensorflow-cpu==2.15.0` pin is deliberate. Documented in [`requirements.txt`](requirements.txt) and in the engineering-decisions section below.

---

## 🖼️ Sample outputs

| Image | Generated caption |
|---|---|
| ![](https://github.com/user-attachments/assets/64e8412b-1d49-404c-a5b2-1da121b224e2) | *a man is standing on a beach with a surfboard* |
| ![](https://github.com/user-attachments/assets/c802d420-a1c1-48be-8e79-599f193c72cd) | *a man riding a motorcycle on a street* |

Outputs above are from the IEEE notebook; the modular pipeline reproduces these via the parity audit ([`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py)). Live captions from the current bootstrap weights will *not* match — see [Current model quality status](#-current-model-quality-status).

---

## 📚 Research backing

The model architecture and the BLEU-4 ~24 baseline below come from the IEEE paper and its accompanying notebook:

- **Paper:** [AI Narratives: Bridging Visual Content and Linguistic Expression](https://ieeexplore.ieee.org/document/10675203) (IEEE)
- **Original notebook:** [Kaggle — image-captioning-using-dl](https://www.kaggle.com/code/apoorvujjwal/image-captionin-using-dl)
- **Frozen artefact in this repo:** [`notebooks/01_ieee_inceptionv3_transformer.ipynb`](notebooks/01_ieee_inceptionv3_transformer.ipynb) — byte-stable; pre-commit + CI enforce its SHA-256.

The notebook is preserved verbatim as the canonical research artefact. Improvements happen in the modular package; the notebook does not.

---

## 📊 Performance

| Metric | Value | Source |
|---|---|---|
| BLEU-4 (IEEE baseline) | ~24 | Reported in the IEEE paper / Kaggle notebook |
| Vocabulary size | 15,000 tokens | `TextVectorization` adapt over preprocessed COCO captions |
| Training set | ~120k captions sampled from COCO 2017 | `data.sample_size` in [`configs/base.yaml`](configs/base.yaml) |
| Image resolution | 299 × 299 (InceptionV3) | [`preprocessing/image.py`](src/captioning/preprocessing/image.py) |
| Max caption length | 40 tokens | `model.max_length` in [`configs/base.yaml`](configs/base.yaml) |
| Backend test suite | 12 tests · 0.3 s · no TF loaded | [`backend/app/tests/`](backend/app/tests/) |
| Full suite | **90 tests passing** | `pytest` (unit + backend + parity) |

> Re-training on the modular pipeline is a Phase 1b deliverable; once a fresh checkpoint exists, this table will publish corpus BLEU-1..4, CIDEr, METEOR, and ROUGE-L (the harnesses already exist under [`evaluation/`](src/captioning/evaluation/)).

---

## ⚠️ Current model quality status

The frontend, backend, and inference pipeline are operational end-to-end against the modular package, but **caption quality from the current modular pipeline is still below expectations**. The IEEE notebook reported BLEU-4 ~24; a freshly trained checkpoint produced by the modular trainer has not yet reproduced that figure on COCO. The serving stack is production-style and ready for a real checkpoint — what is missing is the checkpoint itself.

Current engineering effort is focused on:

- **Training stability** — diagnosing why early modular training runs collapse onto a small set of high-frequency captions instead of generalising.
- **Evaluation correctness** — moving from a single BLEU score to a full corpus-level metric suite with deterministic tokenisation, so two runs against the same slice are mechanically comparable.
- **Decoding improvements** — replacing greedy-only generation with beam search, repetition controls, and length normalisation.
- **Reproducible benchmarking** — emitting one consistent artefact set per evaluation run so any two runs (or any two models) can be diffed without bespoke parsing per checkpoint.

The weights currently committed under [`models/v1.0.0/`](models/v1.0.0/) are the **bootstrap dev artefacts** produced by [`scripts/bootstrap_dev_artifacts.py`](scripts/bootstrap_dev_artifacts.py). Captions returned by the live API today will look like noise; that is the *intended* state of the bootstrap path, not a regression. Poor caption quality at this stage is expected until a properly COCO-trained checkpoint replaces those files.

This gap is being addressed through the **stabilized training workflow** at [`configs/train/stabilized.yaml`](configs/train/stabilized.yaml), which gates convergence-stability primitives behind explicit, ablatable flags rather than rewriting the baseline.

### Accuracy investigation (ongoing)

- **Greedy decoding limited caption quality and diversity.** Argmax-per-step routinely picked the locally-most-probable token regardless of how that affected the overall sequence likelihood, biasing outputs toward a small "safe captions" basin. Beam-search infrastructure now lives at [`src/captioning/inference/beam.py`](src/captioning/inference/beam.py) and dispatches through `CaptionPredictor` alongside the existing greedy path; decode strategy is selectable per inference call and per evaluation run.
- **BLEU-only evaluation hid behaviour the score did not reflect.** CIDEr, METEOR, and ROUGE-L are implemented under [`src/captioning/evaluation/`](src/captioning/evaluation/) and run through the same corpus-level runner that already produces BLEU-1..4. Every evaluation now emits the full metric set in a single `metrics.json`.
- **Validation-time dropout parity quirks** inherited from the notebook (`compute_loss_and_acc` ignoring its `training` argument, so dropout stayed active during validation) were identified during the parity audit. They are now gated behind an explicit config flag (`train.honour_training_flag_in_test_step`) so notebook parity is preserved by default and the conventional dropout-free validation path is opt-in via [`configs/train/stabilized.yaml`](configs/train/stabilized.yaml).
- **Training stabilization experiments** are introduced as opt-in flags so they can be ablated cleanly rather than entangled with the baseline:
  - label smoothing (`train.label_smoothing`),
  - cosine LR schedule (`train.lr_schedule: cosine`),
  - warmup steps (`train.warmup_steps`),
  - dropout-free validation path (`train.honour_training_flag_in_test_step`).

A complete experimental training config — not a thin override — lives at [`configs/train/stabilized.yaml`](configs/train/stabilized.yaml). It is byte-for-byte identical to [`configs/base.yaml`](configs/base.yaml) except for those four flags, so any quality delta between the two runs is attributable to those flags alone.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Core ML** | Python 3.10–3.12, TensorFlow-CPU 2.15.0 (pinned), NumPy, Pillow |
| **Model** | InceptionV3 encoder (frozen) + custom multi-head Transformer decoder |
| **Backend** | FastAPI 0.111, Pydantic v2, `pydantic-settings` 2.x, structlog 24, anyio 4 |
| **Frontend** | React 19, Vite 8, Tailwind v4, ESLint flat config |
| **Evaluation** | sacrebleu, custom CIDEr / METEOR / ROUGE-L implementations |
| **Tooling** | Ruff (lint + format), mypy (strict), pytest 8, pre-commit, nbstripout, gitleaks |
| **Infra (planned, Phase 2C)** | HuggingFace Hub (weights), HuggingFace Spaces (backend), Vercel (frontend), GitHub Actions (CI/CD) |

---

## 📁 Repository Structure

```
image-captioning-system/
├── notebooks/
│   ├── 01_ieee_inceptionv3_transformer.ipynb   # FROZEN — IEEE research artefact
│   └── README.md                                # Frozen-notebook policy
│
├── src/captioning/                              # Installable package
│   ├── config/         schema.py · loader.py
│   ├── preprocessing/  caption.py · image.py · tokenizer.py · augmentation.py
│   ├── data/           coco.py · splits.py · pipeline.py
│   ├── models/         encoder_cnn.py · transformer_encoder.py · embeddings.py
│   │                   transformer_decoder.py · captioning_model.py · factory.py
│   ├── training/       losses.py · callbacks.py · trainer.py
│   ├── inference/      image_loader.py · greedy.py · beam.py · predictor.py
│   ├── evaluation/     bleu.py · cider.py · meteor.py · rouge.py
│   │                   runner.py · benchmark.py · inspection.py · tokenization.py
│   └── utils/          logging.py · seed.py · hashing.py
│
├── backend/                                     # Phase 2A — FastAPI inference service
│   └── app/
│       ├── main.py                              # App factory + lifespan-managed predictor singleton
│       ├── api/routes.py                        # Thin HTTP — /healthz, /v1/captions
│       ├── core/                                # BackendSettings, structlog setup, RequestContextMiddleware
│       ├── schemas/                             # Pydantic request/response models
│       ├── services/predictor_service.py        # bytes → caption + latency (anyio thread offload)
│       ├── utils/image.py                       # Content-type allow-list + ImageDecodeError
│       └── tests/                               # Phase 2C WS-D — 12 route tests, no TF loaded
│
├── frontend/                                    # Phase 2B — React 19 + Vite 8 + Tailwind v4 SPA
│   ├── vite.config.js · eslint.config.js · package.json · .env.example
│   └── src/
│       ├── main.jsx · App.jsx · index.css
│       ├── services/api.js                      # checkHealth / captionImage — AbortController + typed ApiError
│       └── components/
│           ├── Header.jsx · StatusBadge.jsx     # Sticky brand bar + 10s health poller
│           ├── UploadZone.jsx · ImagePreview.jsx
│           ├── CaptionResult.jsx · ErrorBanner.jsx · Spinner.jsx
│
├── configs/
│   ├── base.yaml                                # IEEE hyperparameters (notebook cell 6 mirror)
│   └── train/
│       ├── debug.yaml                           # CI smoke override (1 epoch, 64 captions)
│       └── stabilized.yaml                      # Phase 1b stability experiment (4 ablatable flags)
│
├── scripts/
│   ├── train.py · evaluate.py · predict.py
│   ├── inspect_predictions.py                   # Per-sample diagnostics + diagnostics.jsonl
│   ├── bootstrap_dev_artifacts.py               # Smoke-test artefacts so the API can boot pre-training
│   └── notebook_module_audit.py                 # 4-stage parity gate vs. notebook
│
├── tests/unit/                                  # 78 unit tests (parity, tokenizer, eval, splits, …)
├── docs/                                        # restructure-plan · PHASE_0_NOTES · PHASE_1_NOTES · STABILIZED_TRAINING_RUNBOOK
├── pyproject.toml · requirements*.txt · Makefile
├── .pre-commit-config.yaml · .python-version · .env.example
├── .paper-notebook.sha256                       # Locked notebook hash for the freeze check
├── CLAUDE.md                                    # Contribution + commit governance
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10 – 3.12** (TensorFlow 2.15 has no 3.13 wheels)
- Node **20+**
- Git

### Backend

```powershell
# PowerShell (Windows)
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements-eval.txt
pip install -e ".[hf,mlflow]"
pre-commit install
```

```bash
# bash (Linux / macOS)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-eval.txt
pip install -e ".[hf,mlflow]"
pre-commit install
```

Boot the API:

```bash
uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port 8000
```

Interactive Swagger UI is live at **http://localhost:8000/docs**; raw OpenAPI 3.1 at **http://localhost:8000/openapi.json**.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The SPA is live at **http://localhost:5173** (Vite picks the next free port if 5173 is busy). `VITE_API_BASE` (see [`frontend/.env.example`](frontend/.env.example)) points it at any backend origin; absent the env var, it falls back to `http://127.0.0.1:8000`.

### Tests

```bash
pytest -q                          # All 90 tests (unit + backend + parity)
pytest backend/app/tests/ -v       # Backend route tests only (0.3 s, no TF loaded)
make freeze-paper-notebook         # Asserts the IEEE notebook SHA-256 has not changed
```

### One-shot caption (CLI)

```bash
python -m scripts.predict \
    --config configs/base.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --image samples/photo.jpg
```

### One-shot caption (HTTP)

```bash
curl -X POST http://localhost:8000/v1/captions -F "image=@samples/photo.jpg"
```

### Reproduce training

```bash
python -m scripts.train --config configs/base.yaml
# Or with the stabilization experiment flags enabled:
python -m scripts.train --config configs/base.yaml --override configs/train/stabilized.yaml
# Or a 64-caption CI smoke run:
python -m scripts.train --config configs/base.yaml --override configs/train/debug.yaml
```

Outputs (`weights.h5`, `vocab.pkl` + `vocab.json` sidecar, `history.json`, `training_log.csv`) land under `outputs/runs/latest/` by default.

`make help` lists every available command (lint, format, type-check, test, train, serve, evaluate, predict, Docker, freeze-paper-notebook, …).

---

## 🌐 FastAPI backend (Phase 2A)

Phase 2A delivers a production-style inference service rather than a thin demo wrapper:

- **App factory + lifespan** — [`backend/app/main.py`](backend/app/main.py). `create_app()` builds the FastAPI instance; the lifespan loads the YAML `AppConfig`, instantiates a `CaptionPredictor`, calls `warmup()`, and stashes a `PredictorService` singleton on `app.state` so every request reuses one warm model.
- **Routes** — [`backend/app/api/routes.py`](backend/app/api/routes.py). Intentionally thin: validate inputs, delegate, shape the response. No TF imports leak into the HTTP layer.
- **Service layer** — [`backend/app/services/predictor_service.py`](backend/app/services/predictor_service.py). Wraps the predictor, decodes uploaded bytes off the event loop via `anyio.to_thread.run_sync`, measures per-request latency, returns `(caption, latency_ms)`.
- **Schemas** — [`backend/app/schemas/caption.py`](backend/app/schemas/caption.py). Pydantic v2 (`CaptionResponse`, `HealthResponse`, `ErrorResponse`); every payload that crosses the wire is typed and OpenAPI-documented.
- **Backend settings** — [`backend/app/core/config.py`](backend/app/core/config.py). Separate `BackendSettings` (env-overridable: weights path, tokenizer dir, model version, warmup toggle) layered on top of the research-side `AppConfig`. Research hyperparameters and serving knobs change on different cadences and live in different settings objects.
- **Structured logging + request IDs** — [`backend/app/core/logging.py`](backend/app/core/logging.py). `RequestContextMiddleware` stamps each request with a UUID; `structlog` carries it through every log line so a single failed caption can be traced end-to-end.
- **Image safety** — [`backend/app/utils/image.py`](backend/app/utils/image.py). Content-type allow-list (JPEG / PNG / WebP / BMP), explicit `ImageDecodeError` so malformed bytes produce a clean 422 rather than a 500.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz`      | Liveness + readiness — reports `model_loaded`, `model_version`, `api_version`. Always 200; readiness is conveyed in the body. |
| `POST` | `/v1/captions`  | Multipart image upload → generated caption + decode strategy + latency + request ID. |
| `GET`  | `/docs`         | Interactive Swagger UI, auto-generated from the Pydantic schemas. |
| `GET`  | `/openapi.json` | Raw OpenAPI 3.1 spec for client codegen. |

`POST /v1/captions` enforces input validation at the boundary: **415** on disallowed content types, **413** on oversized uploads (`serve.max_upload_bytes`), **422** on undecodable image bytes, **400** on empty uploads, **503** while the predictor is still loading during a rolling restart. All six status codes are covered by the [`backend/app/tests/`](backend/app/tests/) suite added in Phase 2C WS-D.

---

## 🎨 Frontend UI (Phase 2B)

Phase 2B ships a single-page inference UI under [`frontend/`](frontend/) — not a styled demo. The split mirrors the backend's separation between transport, service, and presentation:

- **Application shell** — [`frontend/src/App.jsx`](frontend/src/App.jsx). Owns the request lifecycle (selected file → preview → generate → result). The preview `URL.createObjectURL` is `useMemo`-derived and revoked through an effect cleanup so previews never leak across uploads. Four `useState` slots (`file`, `result`, `error`, `loading`) cover every UI state — no Redux, no React Query, no context.
- **API service layer** — [`frontend/src/services/api.js`](frontend/src/services/api.js). Single boundary for every backend call. Reads `import.meta.env.VITE_API_BASE` once at module load (falls back to `http://127.0.0.1:8000`), wraps `fetch` with `AbortController`-driven timeouts (3 s for `/healthz`, 60 s for `/v1/captions`), and classifies failures into `timeout` / `network` / `http` / `unknown` kinds on a typed `ApiError`.
- **Upload zone** — [`frontend/src/components/UploadZone.jsx`](frontend/src/components/UploadZone.jsx). Drag/drop + click-to-browse + keyboard activation. Validates content-type (JPEG / PNG / WebP) and size (10 MB) before the file ever touches the network — invalid uploads are rejected client-side with the same wording the backend would have returned.
- **Status badge** — [`frontend/src/components/StatusBadge.jsx`](frontend/src/components/StatusBadge.jsx). Polls `/healthz` every 10 seconds and on window focus, runs a three-state machine (`checking` / `online` / `offline`), recovers automatically when the backend comes back.
- **Error banner** — [`frontend/src/components/ErrorBanner.jsx`](frontend/src/components/ErrorBanner.jsx). Single surface for every failure class. Reads `ApiError.message` so the user sees "Cannot reach backend" or "Request timed out" instead of a raw browser error.
- **Caption result** — [`frontend/src/components/CaptionResult.jsx`](frontend/src/components/CaptionResult.jsx). Consumes the backend's typed `CaptionResponse` directly: caption text plus model version, decode strategy, latency, and the request ID echoed from the `x-request-id` header.

```
┌──────────────┐  drag/drop   ┌─────────────┐  validate   ┌──────────────┐
│ UploadZone   │ ───────────▶ │  App state  │ ──────────▶ │ ImagePreview │
└──────────────┘              └─────────────┘             └──────────────┘
                                     │ click "Generate"
                                     ▼
                            ┌─────────────────┐  multipart   POST /v1/captions
                            │ services/api.js │ ───────────▶ FastAPI backend
                            └─────────────────┘
                                     │   typed CaptionResponse / ApiError
                                     ▼
                         ┌──────────────────────┐
                         │ CaptionResult /      │
                         │ ErrorBanner          │
                         └──────────────────────┘
```

Frontend and backend are deployed independently. The SPA only knows the backend's origin via `VITE_API_BASE`; the backend only trusts SPAs whose origin appears in `serve.cors_allowed_origins`. Dev origins are pre-allowed in [`configs/base.yaml`](configs/base.yaml); production origins join the same list at deploy time (Phase 2C WS-F). No shared build, no shared runtime — only the typed Pydantic schemas in [`backend/app/schemas/caption.py`](backend/app/schemas/caption.py) cross the wire.

---

## ⚙️ Configuration system

Hyperparameters are not globals. They live in YAML validated by Pydantic v2:

```yaml
# configs/base.yaml — mirrors the IEEE notebook cell 6 verbatim
model:
  embedding_dim: 512
  units: 512
  max_length: 40
  vocabulary_size: 15000
  decoder_num_heads: 8
  decoder_dropout_inner: 0.3
  decoder_dropout_outer: 0.5
  decoder_attention_dropout: 0.1
train:
  epochs: 10
  batch_size: 64
  early_stopping_patience: 3
  seed: 42
data:
  sample_size: 120000
  train_val_split: 0.8
```

Three load-time guarantees:

1. **Type validation.** `batch_size: "64"` (string instead of int) raises a `ValidationError` pointing at the field, not a downstream tensor-shape error.
2. **No silent typos.** `extra="forbid"` rejects unknown keys — typos in ML hyperparameters silently using defaults is the worst failure mode, and `extra="forbid"` eliminates it.
3. **Env overrides.** `CAPTIONING__TRAIN__BATCH_SIZE=32` overrides at any nesting depth — useful for CI smoke tests, ablations, and serve-time tuning without rebuilding images.

Schema in [`src/captioning/config/schema.py`](src/captioning/config/schema.py); loader in [`src/captioning/config/loader.py`](src/captioning/config/loader.py).

---

## 🧪 Testing & code quality

```bash
make test            # pytest — 90/90 (unit + backend route tests + parity)
make lint            # Ruff lint + format check
make typecheck       # mypy strict on src/captioning + scripts
make pre-commit      # All hooks across all files
make freeze-paper-notebook   # Asserts notebook SHA-256 unchanged
```

| Layer | Tool | Status |
|---|---|---|
| Lint + format | [Ruff](https://docs.astral.sh/ruff/) (replaces black + isort + flake8) | ✅ clean |
| Type-check | [mypy](https://mypy.readthedocs.io/) with `pandas-stubs`, `types-PyYAML`, `types-requests` | ✅ 0 errors |
| Tests | pytest + pytest-cov + pytest-asyncio | ✅ 90 passing |
| Notebook hygiene | [`nbstripout`](https://github.com/kynan/nbstripout) (pre-commit) | ✅ outputs stripped on commit |
| Secret scanning | [`gitleaks`](https://github.com/gitleaks/gitleaks) (pre-commit) | ✅ enabled |
| Notebook integrity | SHA-256 freeze via [`make freeze-paper-notebook`](Makefile) | ✅ locked |
| Parity audit | [`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py) — 4 stages | ✅ all passing |

The parity audit re-implements four notebook stages inline (caption preprocessing, tokenizer vocabulary + encoding, image preprocessing, decoder forward pass) and asserts the modular path produces byte-identical (or `tf.allclose`-identical) output. It is the contract that gates any behavioural improvement.

The backend test suite ([`backend/app/tests/`](backend/app/tests/)) introduced in Phase 2C WS-D uses a duck-typed `FakePredictorService` to exercise every status code in the `/v1/captions` contract — 200 / 400 / 413 / 415 / 422 / 503 — plus the `/healthz` readiness flip and `x-request-id` propagation, all without loading TensorFlow. The full backend slice runs in **0.3 seconds**.

---

## 🗺️ Roadmap

### Phase 0 — Bootstrap ✅

- [x] **0A** — Repo scaffolding, `pyproject.toml`, Makefile, Conventional Commits
- [x] **0B** — Pre-commit hooks (Ruff, mypy, nbstripout, gitleaks, line-ending + TOML/YAML hygiene)
- [x] **0C** — Notebook freeze policy + `.paper-notebook.sha256` SHA-256 lock
- [x] **0D** — Pinned dependency surface (`requirements*.txt` + `pyproject.toml` extras: `hf`, `eval`, `mlflow`, `dev`)

### Phase 1 — Modularisation ✅

- [x] **1A** — Notebook → installable `captioning` package (`src/` layout)
- [x] **1B** — Pydantic v2 strict config (`AppConfig` / `ModelConfig` / `TrainConfig` / `DataConfig` / `ServeConfig`) with YAML loader + env-var overrides
- [x] **1C** — Preprocessing modules (`caption.py`, `image.py`, `tokenizer.py`, `augmentation.py`) — shared train/serve preprocessing
- [x] **1D** — Data pipeline (`coco.py`, `splits.py`, `pipeline.py`) with seeded sampling
- [x] **1E** — Model factory (`encoder_cnn.py`, `transformer_encoder.py`, `embeddings.py`, `transformer_decoder.py`, `captioning_model.py`, `factory.py`)
- [x] **1F** — Training loop (`losses.py`, `callbacks.py`, `trainer.py`) with structured logging + history serialisation
- [x] **1G** — Greedy inference (`predictor.py`, `image_loader.py`, `greedy.py`) with lifespan-friendly `from_artifacts(...)` + `warmup()`
- [x] **1H** — Notebook parity audit ([`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py)) — 4 stages, byte/tensor-identical
- [x] **1I** — Unit test suite (parity, tokenizer, evaluation, splits, hashing, image preprocessing, caption preprocessing)

### Phase 1b — Training stabilization ✅ (training validation in progress)

- [x] **1b-A** — Beam-search decoder ([`inference/beam.py`](src/captioning/inference/beam.py)) with length normalisation + n-gram repetition suppression, selectable per call/run
- [x] **1b-B** — CIDEr implementation ([`evaluation/cider.py`](src/captioning/evaluation/cider.py))
- [x] **1b-C** — METEOR implementation ([`evaluation/meteor.py`](src/captioning/evaluation/meteor.py))
- [x] **1b-D** — ROUGE-L implementation ([`evaluation/rouge.py`](src/captioning/evaluation/rouge.py))
- [x] **1b-E** — Benchmark runner ([`evaluation/benchmark.py`](src/captioning/evaluation/benchmark.py)) emitting one `metrics.json` + `diagnostics.jsonl` per run
- [x] **1b-F** — Per-sample inspection tool ([`scripts/inspect_predictions.py`](scripts/inspect_predictions.py)) — sentence-level BLEU/ROUGE, length, longest repeated-token run, failure flags
- [x] **1b-G** — Stabilization config ([`configs/train/stabilized.yaml`](configs/train/stabilized.yaml)) — label smoothing, cosine LR, warmup, dropout-free validation, all ablatable
- [x] **1b-H** — Stabilized training runbook ([`docs/STABILIZED_TRAINING_RUNBOOK.md`](docs/STABILIZED_TRAINING_RUNBOOK.md))
- [ ] **1b-I** — Fresh stabilized COCO-trained checkpoint committed to [`models/`](models/) (under a bumped `vX.Y.Z/`)
- [ ] **1b-J** — Headline numbers (BLEU-1..4, CIDEr, METEOR, ROUGE-L) published in [Performance](#-performance)

### Phase 2A — FastAPI inference service ✅

- [x] **2A-1** — App factory + lifespan-managed `CaptionPredictor` singleton with `warmup()` on boot
- [x] **2A-2** — Thin `/healthz` and `POST /v1/captions` routes with full status-code contract (200 / 400 / 413 / 415 / 422 / 503)
- [x] **2A-3** — Pydantic v2 schemas (`CaptionResponse`, `HealthResponse`, `ErrorResponse`) with auto-generated Swagger + OpenAPI 3.1
- [x] **2A-4** — `PredictorService` with `anyio.to_thread.run_sync` offload so TF inference never blocks the event loop
- [x] **2A-5** — Structured logging (`structlog`) + `RequestContextMiddleware` propagating `x-request-id` across log lines
- [x] **2A-6** — `BackendSettings` separated from research `AppConfig` (different change cadences, different env prefixes)
- [x] **2A-7** — Bootstrap dev artefacts script so the API boots before training has produced real weights

### Phase 2B — Frontend SPA ✅

- [x] **2B-1** — React 19 + Vite 8 + Tailwind v4 scaffolding, flat ESLint config with `eslint-plugin-react-hooks` + `eslint-plugin-react-refresh`
- [x] **2B-2** — Drag/drop + click-to-browse upload zone with keyboard activation and client-side content-type + size validation
- [x] **2B-3** — `services/api.js` boundary: `VITE_API_BASE` env, `AbortController` timeouts (3 s health / 60 s caption), typed `ApiError` classification
- [x] **2B-4** — Polled `/healthz` status badge with three-state machine, window-focus refetch, and automatic recovery
- [x] **2B-5** — Typed `CaptionResponse` rendering — caption, model version, decode strategy, latency, request ID — with copy-to-clipboard
- [x] **2B-6** — Single `ErrorBanner` surface mapping every `ApiError.kind` to actionable copy
- [x] **2B-7** — CORS allow-list wired through backend YAML (`serve.cors_allowed_origins`), dev origins pre-allowed

### Phase 2C — Public deployment 🚧 (in progress)

- [x] **WS-A** — Backend containerisation: `Dockerfile` (python:3.11-slim, non-root UID 1000, EXPOSE 7860, HEALTHCHECK on `/healthz`) + `.dockerignore` + corrected `.env.example` schema
- [x] **WS-A4** — Lifespan integration with HuggingFace Hub: extended `BackendSettings` with `weights_hub_repo` / `weights_hub_revision` / `weights_hub_filename` / `weights_cache_dir`; new `app.services.weights_loader.resolve_weights` calls `huggingface_hub.snapshot_download` when configured, falls back to local paths otherwise (4 new unit tests, downloader injected for offline testing)
- [ ] **WS-B** — Upload trained weights + tokenizer to a HuggingFace Hub model repo
- [ ] **WS-C** — First manual deploy to a HuggingFace Space (Docker SDK, cpu-basic, port 7860, single worker)
- [x] **WS-D** — **Backend test suite** ([`backend/app/tests/`](backend/app/tests/)): 12 route tests covering the full `/healthz` + `/v1/captions` contract (200 / 400 / 413 / 415 / 422 / 503) with a duck-typed `FakePredictorService` — no TF loaded, full slice runs in 0.3 s
- [ ] **WS-E** — Frontend deploy to Vercel (static SPA, `VITE_API_BASE` baked at build time, SPA rewrites)
- [ ] **WS-F** — Production CORS: add the deployed Vercel origin to `serve.cors_allowed_origins`
- [ ] **WS-G** — GitHub Actions CI/CD:
  - [x] `ci.yml` — Python quality (ruff lint + format check, mypy), pytest matrix on 3.10/3.11/3.12, notebook SHA-256 freeze check, frontend lint + build, concurrency cancel-in-progress, pip + npm caching
  - [ ] `deploy-backend.yml` — gated on `needs: ci`, pushes to the HF Space
  - [ ] `deploy-frontend.yml` *(optional)* — Vercel-native GitHub integration is the recommended path
- [ ] **WS-H** — README "Live Demo" section (badges swapped to live HF Space + Vercel URLs) + `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md` + `docs/CI.md`

### Phase 3 — Multimodal baselines ⏳ (planned)

- [ ] **3A** — Side-by-side comparison harness: original CNN + Transformer vs. BLIP-base vs. ViT-GPT2 vs. GIT-base-coco
- [ ] **3B** — Per-model BLEU / CIDEr / METEOR / ROUGE-L on a shared COCO slice with deterministic tokenisation
- [ ] **3C** — Per-model latency benchmarking (single-image, batch, CPU vs. GPU)
- [ ] **3D** — Comparison-result dashboard exposed through the existing SPA

### Phase 4 — Observability ⏳ (planned)

- [ ] **4A** — Sentry error tracking on backend + frontend
- [ ] **4B** — Prometheus metrics (per-route latency histograms, predictor cache hits, lifespan boot duration)
- [ ] **4C** — DagsHub-hosted MLflow tracking link surfaced in the README
- [ ] **4D** — Architecture Decision Records (`docs/adr/`) — every non-trivial choice (TF version pin, anyio offload, env-var prefix separation, etc.) gets a one-page ADR

Detailed phase notes live under [`docs/`](docs/): [restructure plan](docs/restructure-plan.md) · [Phase 0 notes](docs/PHASE_0_NOTES.md) · [Phase 1 notes](docs/PHASE_1_NOTES.md) · [Stabilized training runbook](docs/STABILIZED_TRAINING_RUNBOOK.md).

---

## 🎯 Engineering Decisions

> **Why preserve the notebook verbatim instead of refactoring it in place?**
> The notebook is the published research artefact and the only thing that can credibly produce the BLEU-4 ~24 baseline the IEEE paper claims. Editing it would silently destroy that reproducibility. The freeze + parity-audit pattern keeps the published result anchored while the modular package evolves; if the audit ever fails, the modular path has drifted from the paper and the diff is exactly where to start debugging.

> **Why pin `tensorflow-cpu==2.15.0`?**
> TF 2.16 ships Keras 3 as the default backend, and Keras 3 silently breaks `TextVectorization` save/load — the tokenizer round-trip the entire serving stack depends on. The pin is documented in [`requirements.txt`](requirements.txt) and protected by the env setup commands above. Phase 3's foundation-model baselines will live in optional dependency groups so they can install on a newer TF without unpinning the research pipeline.

> **Why two separate settings objects (`AppConfig` + `BackendSettings`)?**
> Research hyperparameters (`model.*`, `train.*`, `data.*`) and serving knobs (weights path, model version, warmup toggle, request-id header) change on different cadences and have different audiences. Folding them into one object would mean every backend env var lived in a research YAML, and every research-side schema change risked breaking a deploy. Two objects with two prefixes (`CAPTIONING__*` vs `BACKEND_*`) gives each surface its own change schedule.

> **Why `anyio.to_thread.run_sync` for inference instead of `async def predict`?**
> TensorFlow's `predict` call is synchronous and CPU-bound. Calling it directly from an async route handler would block the event loop and starve every other request. Offloading via `anyio.to_thread.run_sync` lets the event loop keep serving health checks and concurrent uploads while the model runs.

> **Why is the bootstrap-weights script committed?**
> The serving stack (lifespan, predictor wiring, multipart upload, frontend integration) has to be verifiable before a real COCO-trained checkpoint exists. The bootstrap script makes the entire path runnable from a fresh clone, which is what lets reviewers actually evaluate the architectural work independently of the model-quality work. The captions are gibberish — by design — and the README states that prominently to keep expectations honest.

> **Why `extra="forbid"` on every config schema?**
> ML projects fail catastrophically when a typo in a hyperparameter silently uses a default. `vocabularsy_size: 30000` should be a load-time error, not a quiet retraining run on the wrong vocabulary size. Strict configs are the cheapest possible insurance against the most expensive class of bug in this domain.

> **Why ship the metric suite and beam search *before* publishing new numbers?**
> Without deterministic tokenisation + a corpus-level runner + a non-greedy decoder, any "improved" number is unfalsifiable — it could be a real gain, a decoding artefact, or a tokenisation difference. The harness is the prerequisite to making the next training run mean something. Publishing the bar before the harness exists is how research projects accumulate numbers nobody can reproduce.

---

## 🔬 Experimental evaluation pipeline

The repository is evolving from a "research notebook reproduction" into a reproducible experimentation platform. Evaluation is no longer a single BLEU number printed at the end of training — it is a structured set of artefacts any future run, including the Phase 3 multimodal baselines, can be diffed against.

- **[`scripts/evaluate.py`](scripts/evaluate.py)** — single entrypoint for full corpus evaluation. Loads a checkpoint + tokenizer, runs decoding (greedy or beam) over the COCO validation slice, computes BLEU-1..4 / CIDEr / METEOR / ROUGE-L, and writes a versioned artefact set under `results/<run_id>/`.
- **[`scripts/inspect_predictions.py`](scripts/inspect_predictions.py)** — per-sample diagnostic view. Prints N random predictions vs. references with sentence-level BLEU-4 / ROUGE-L, prediction length, longest repeated-token run, and failure flags (`empty` / `very_short` / `repetitive` / `under_length`). Used when the aggregate metric moves but the qualitative behaviour does not.
- **[`evaluation/benchmark.py`](src/captioning/evaluation/benchmark.py)** — `RunMeta` and `write_run_artifacts(...)`, the contract every evaluation run honours. Phase 3 cross-model comparison code joins multiple `results/<run_id>/` directories without bespoke parsers per model.
- **Greedy vs. beam evaluation support** — the same evaluator accepts `--decode-strategy greedy|beam` plus beam-search controls (`--beam-width`, `--length-penalty`, `--no-repeat-ngram-size`), so a single command-line difference produces directly comparable artefact sets for the same checkpoint.

---

## ⚖️ Limitations

- The model produces generic captions on cluttered or rare-object scenes — a known limitation of the IEEE-era architecture, addressed in Phase 3 by adding modern foundation-model baselines for side-by-side comparison.
- The modular pipeline has not yet reproduced the IEEE notebook's BLEU-4 ~24 on a freshly trained checkpoint; see [Current model quality status](#-current-model-quality-status). The bootstrap weights shipped under [`models/v1.0.0/`](models/v1.0.0/) are intentionally random and exist only for architectural smoke testing.
- Beam search is implemented and selectable, but a head-to-head benchmark against greedy on a real checkpoint is part of in-progress Phase 1b validation, not a published result yet.
- CIDEr / METEOR / ROUGE-L are implemented and emitted into `metrics.json` per run; finalised numbers from the modular pipeline are pending a stabilized COCO-trained checkpoint.
- Validation pipeline includes a leftover `shuffle()` from the notebook (functionally harmless, removed in Phase 1b).

These are explicitly tracked rather than hidden; full list in [`docs/PHASE_1_NOTES.md` § Technical debt](docs/PHASE_1_NOTES.md#technical-debt-remaining).

---

## 🧭 What I'd Build Next

Clear extension paths beyond the current scope, ordered by how much I'd learn building them:

- **Foundation-model fine-tuning** — fine-tune BLIP-2 or LLaVA on COCO and benchmark per-token cost vs. caption quality against the InceptionV3 + Transformer baseline.
- **Streaming generation** — server-sent events from `/v1/captions` so the SPA renders tokens as the decoder produces them, instead of waiting for the full sequence.
- **Batch inference endpoint** — a second route that accepts an array of images, runs them through one TF graph call, and amortises the per-request Python overhead — useful for any downstream pipeline that needs to caption a folder.
- **Visual Question Answering** — extend the same encoder + decoder pattern to `POST /v1/vqa` taking image + question, sharing the warmed CNN encoder.
- **VLM-backed comparison endpoint** — an opt-in route that runs the same image through Anthropic Claude vision or OpenAI Vision behind a feature flag, returns both captions, and surfaces a side-by-side card in the SPA. The framing is *"here's what a 2024 VLM does for the same input"*, not a replacement for the local model.
- **Online evaluation** — a background job that periodically scores the latest checkpoint against a held-out COCO slice and pushes BLEU / CIDEr / latency to a Grafana dashboard, so model regressions surface without a human running `scripts/evaluate.py`.
- **Active-learning loop** — surface low-confidence captions in the SPA, capture user corrections, and route them into a labelled corpus for the next training run.

---

## 📚 Lessons Being Learned

> The hardest engineering skill on a research → production conversion is not the code — it is the discipline of *not improving the model* while you fix the codebase around it. Every quality intervention you fold in mid-refactor makes the parity audit ambiguous: when the numbers change, you cannot tell whether the new metric harness, the new tokenisation, the new decoder, or the new training schedule was responsible. The four ablatable flags in [`configs/train/stabilized.yaml`](configs/train/stabilized.yaml) exist specifically so each change can be diffed in isolation.

> Pydantic with `extra="forbid"` has caught more real bugs in this codebase than every other tool combined. A typo in a YAML key that silently uses a default is the single most expensive class of bug in ML, and the fix is one config option.

> The split between research config (`AppConfig`) and serving config (`BackendSettings`) felt over-engineered the day it was introduced and has paid for itself every week since. The two surfaces change on different cadences, ship on different schedules, and need different env-var prefixes for the deploy story to make sense. Conflating them would have meant every backend-only env var lived in a research YAML.

> Notebook freezing is the smallest possible piece of engineering that earns the largest amount of trust. A SHA-256 file plus a pre-commit hook plus one CI step is enough to guarantee the published research is exactly what reviewers think it is, three years from now.

---

## 📝 License & Contact

This project is released under the [MIT License](LICENSE).

**Built by [apoorvrajdev](https://github.com/apoorvrajdev)** — reach me at [apoorvrajmgr@gmail.com](mailto:apoorvrajmgr@gmail.com).

Contribution + commit governance for this repo is codified in [`CLAUDE.md`](CLAUDE.md).

---

<p align="center">
  <em>Built as a flagship portfolio project for ML and multimodal-AI engineering roles.</em>
</p>
