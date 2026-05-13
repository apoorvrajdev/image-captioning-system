# Image Captioning System

> CNN + Transformer architecture for visual-to-language generation, restructured from an IEEE-published research notebook into a production-style multimodal AI codebase.

<p align="left">
  <img alt="Python 3.10+"   src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow 2.15" src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Pydantic v2"    src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white">
  <img alt="FastAPI ready"  src="https://img.shields.io/badge/FastAPI-ready-009688?logo=fastapi&logoColor=white">
</p>

<p align="left">
  <img alt="Ruff"         src="https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white">
  <img alt="mypy"         src="https://img.shields.io/badge/typed-mypy-1F5082">
  <img alt="Tests"        src="https://img.shields.io/badge/tests-37%20passing-brightgreen">
  <img alt="Pre-commit"   src="https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit&logoColor=white">
</p>

<p align="left">
  <img alt="IEEE Published" src="https://img.shields.io/badge/IEEE-published-00629B?logo=ieee&logoColor=white">
  <img alt="License: MIT"   src="https://img.shields.io/badge/license-MIT-lightgrey">
  <img alt="Phase 1"        src="https://img.shields.io/badge/Phase%201-complete-brightgreen">
  <img alt="Phase 2A"       src="https://img.shields.io/badge/Phase%202A-complete-brightgreen">
  <img alt="Phase 2B"       src="https://img.shields.io/badge/Phase%202B-planned-blue">
</p>

---

## Overview

This repository implements an **end-to-end image-captioning pipeline** built around an InceptionV3 visual encoder and a custom multi-head Transformer decoder. The architecture is the basis of the IEEE-published paper *“AI Narratives: Bridging Visual Content and Linguistic Expression”*; this codebase lifts the original Kaggle research notebook into a typed, tested, configuration-driven Python package that can be reused from CLI, scripts, or a future serving layer.

The repository is structured in deliberate phases:

| Phase | Focus | Status |
|---|---|---|
| 0 — Bootstrap | Tooling, packaging, freeze policy | ✅ complete |
| 1 — Modularisation | Notebook → typed Python package, parity audit, unit tests | ✅ complete |
| 2A — Backend Infrastructure | FastAPI inference API, structured logging, schemas, health checks, Swagger/OpenAPI, predictor lifecycle | ✅ complete |
| 2B — Frontend UI | React/Vite frontend + upload UX + API integration | ⏳ planned |
| 3 — Multimodal baselines | BLIP / ViT-GPT2 / GIT side-by-side comparison | ⏳ planned |
| 4 — Observability | Sentry, Prometheus metrics, ADRs | ⏳ planned |

Phase notes live under [`docs/`](docs/): [restructure plan](docs/restructure-plan.md) · [Phase 0 notes](docs/PHASE_0_NOTES.md) · [Phase 1 notes](docs/PHASE_1_NOTES.md).

---

## Research backing

The model architecture and the BLEU-4 ~24 baseline below come from the IEEE paper and its accompanying notebook:

- **Paper:** [AI Narratives: Bridging Visual Content and Linguistic Expression](https://ieeexplore.ieee.org/document/10675203) (IEEE)
- **Original notebook:** [Kaggle — image-captioning-using-dl](https://www.kaggle.com/code/apoorvujjwal/image-captionin-using-dl)
- **Frozen artefact in this repo:** [`notebooks/01_ieee_inceptionv3_transformer.ipynb`](notebooks/01_ieee_inceptionv3_transformer.ipynb) — byte-stable; CI enforces its SHA-256.

The notebook is preserved verbatim as the canonical research artefact. Improvements happen in the modular package; the notebook does not.

---

## Architecture

```
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────┐
│  Input image │──▶│  InceptionV3    │──▶│  Transformer     │──▶│  Transformer     │──▶│  Caption   │
│  299x299x3   │   │  encoder        │   │  encoder         │   │  decoder         │   │  string    │
└──────────────┘   │  (ImageNet,     │   │  (1 layer,       │   │  (2 layers,      │   └────────────┘
                   │   frozen)       │   │   1 head)        │   │   8 heads)       │
                   └─────────────────┘   └──────────────────┘   └──────────────────┘
                          ▼                       ▼                       ▼
                    [B, 64, 2048]          [B, 64, 512]            [B, T, vocab]
                    patch features         projected features      softmax over 15k tokens
```

### Components

- **CNN encoder** — [`models/encoder_cnn.py`](src/captioning/models/encoder_cnn.py). Pretrained InceptionV3 with the classification head removed; output reshaped to a sequence of 64 spatial positions × 2048 channels. Weights are frozen during training.
- **Transformer encoder** — [`models/transformer_encoder.py`](src/captioning/models/transformer_encoder.py). Single layer with one attention head. Projects InceptionV3 features into the decoder’s embedding dimension and lets the decoder attend across spatial positions.
- **Embeddings** — [`models/embeddings.py`](src/captioning/models/embeddings.py). Sum of token and *learned* positional embeddings (not sinusoidal — preserved from the published architecture).
- **Transformer decoder** — [`models/transformer_decoder.py`](src/captioning/models/transformer_decoder.py). Causal self-attention over partial captions, cross-attention over image features, and a feed-forward sub-block. 8 heads, ``embedding_dim=512``, dropouts (0.1 / 0.3 / 0.5) preserved from the IEEE configuration.
- **Captioning model** — [`models/captioning_model.py`](src/captioning/models/captioning_model.py). Custom `train_step` / `test_step` with masked sparse-categorical cross-entropy and masked accuracy.
- **Tokenizer** — [`preprocessing/tokenizer.py`](src/captioning/preprocessing/tokenizer.py). `CaptionTokenizer` wraps `tf.keras.layers.TextVectorization`; persists the vocabulary as both pickle (notebook-compatible) and JSON sidecar.
- **Inference** — [`inference/predictor.py`](src/captioning/inference/predictor.py). `CaptionPredictor.from_artifacts(weights, vocab, config)` loads everything once at boot, exposes `predict_path(...)` and `predict_tensor(...)` for stateless calls, and `warmup()` for first-request latency.
- **Configuration** — [`config/schema.py`](src/captioning/config/schema.py). Pydantic v2 schemas (`AppConfig` / `ModelConfig` / `TrainConfig` / `DataConfig` / `ServeConfig`); strict (`extra="forbid"`) so typos in YAML or env vars become load-time errors instead of silent drift.

---

## Sample outputs

| Image | Generated caption |
|---|---|
| ![](https://github.com/user-attachments/assets/64e8412b-1d49-404c-a5b2-1da121b224e2) | *a man is standing on a beach with a surfboard* |
| ![](https://github.com/user-attachments/assets/c802d420-a1c1-48be-8e79-599f193c72cd) | *a man riding a motorcycle on a street* |

Outputs above are from the IEEE notebook; the modular pipeline reproduces these via the parity audit ([`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py)).

---

## Performance

| Metric | Value | Source |
|---|---|---|
| BLEU-4 | ~24 | Reported in the IEEE paper / Kaggle notebook |
| Vocabulary size | 15,000 tokens | TextVectorization adapt over preprocessed COCO captions |
| Training set | ~120k captions sampled from COCO 2017 | `data.sample_size` in [`configs/base.yaml`](configs/base.yaml) |
| Image resolution | 299 × 299 (InceptionV3) | [`preprocessing/image.py`](src/captioning/preprocessing/image.py) |
| Max caption length | 40 tokens | `model.max_length` in [`configs/base.yaml`](configs/base.yaml) |

> Re-training on the modular pipeline is a Phase 2 deliverable; once a fresh checkpoint exists, this table will be expanded with corpus BLEU-1..4, CIDEr, METEOR, and ROUGE-L (already implemented in [`evaluation/`](src/captioning/evaluation/)).

---

## Project structure

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
│   ├── inference/      image_loader.py · greedy.py · predictor.py
│   ├── evaluation/     bleu.py
│   └── utils/          logging.py · seed.py · hashing.py
│
├── backend/                                     # Phase 2A — FastAPI inference service
│   └── app/
│       ├── main.py                              # App factory + lifespan-managed predictor singleton
│       ├── api/                                 # Thin HTTP routes — /healthz, /v1/captions
│       ├── core/                                # BackendSettings, structured logging, request IDs
│       ├── schemas/                             # Pydantic request/response schemas
│       ├── services/                            # PredictorService — image bytes → caption + latency
│       └── utils/                               # Image decoding + content-type guards
│
├── configs/
│   ├── base.yaml                                # IEEE hyperparameters (cell 6 mirror)
│   └── train/debug.yaml                         # CI smoke override
│
├── scripts/
│   ├── train.py · evaluate.py · predict.py
│   ├── bootstrap_dev_artifacts.py               # Smoke-test artefacts so the API can boot pre-training
│   └── notebook_module_audit.py                 # Parity gate vs. notebook
│
├── tests/unit/
│   ├── test_caption_preprocessing.py · test_config.py · test_splits.py
│   ├── test_tokenizer.py · test_image_preprocessing.py
│   ├── test_evaluation.py · test_hashing.py
│   └── conftest.py
│
├── docs/
│   ├── restructure-plan.md · PHASE_0_NOTES.md · PHASE_1_NOTES.md
│
├── pyproject.toml · requirements*.txt · Makefile
├── .pre-commit-config.yaml · .python-version · .env.example
├── .paper-notebook.sha256                       # Locked notebook hash for CI freeze check
└── README.md
```

---

## Setup

Requires **Python 3.10–3.12** (TensorFlow 2.15 has no 3.13 wheels).

### PowerShell (Windows)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements-eval.txt
pip install -e ".[hf,mlflow]"
pre-commit install
```

### bash (Linux / macOS)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-eval.txt
pip install -e ".[hf,mlflow]"
pre-commit install
```

`make help` lists every available command (lint, format, type-check, test, train, serve, evaluate, predict, Docker, freeze-paper-notebook, …).

---

## Training

The training script consumes a YAML config validated by Pydantic:

```bash
python -m scripts.train --config configs/base.yaml
```

Override fields without editing YAML:

```bash
# CLI smoke run on a 64-caption subset (1 epoch, batch 8)
python -m scripts.train --config configs/base.yaml --override configs/train/debug.yaml

# Env-var override (double-underscore = nesting delimiter)
CAPTIONING__TRAIN__BATCH_SIZE=32 python -m scripts.train --config configs/base.yaml
```

Outputs (`weights.h5`, `vocab.pkl` + `vocab.json` sidecar, `history.json`, `training_log.csv`) land under `outputs/runs/latest/` by default.

The `Trainer` ([`training/trainer.py`](src/captioning/training/trainer.py)) wraps `model.compile + model.fit` with structured logging and history serialisation; everything else (loss, callbacks, optimizer choice) sits in dedicated modules so each piece can be unit-tested in isolation.

---

## Evaluation

```bash
python -m scripts.evaluate \
    --config configs/base.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --report docs/results/v1.0.0.md \
    --max-samples 500
```

Phase 1 ships **corpus BLEU-4 via sacrebleu** (deterministic, reproducible). CIDEr / METEOR / ROUGE-L slot into [`src/captioning/evaluation/`](src/captioning/evaluation/) in Phase 1b under the same runner interface.

---

## Inference

### Python API

```python
from captioning.config import load_config
from captioning.inference import CaptionPredictor

config    = load_config("configs/base.yaml")
predictor = CaptionPredictor.from_artifacts(
    weights_path="models/v1.0.0/model.h5",
    tokenizer_dir="models/v1.0.0",
    config=config,
)
predictor.warmup()                       # one dummy forward pass — kills first-request latency
caption = predictor.predict_path("photo.jpg")
print(caption)
```

### CLI

```bash
python -m scripts.predict \
    --config configs/base.yaml \
    --weights models/v1.0.0/model.h5 \
    --tokenizer-dir models/v1.0.0 \
    --image samples/photo.jpg
```

### REST API (Phase 2A — operational)

A FastAPI service under [`backend/app/`](backend/app/) is now live. The lifespan instantiates a single `CaptionPredictor`, runs `warmup()` once, and reuses it across every request — no per-request TF graph builds, no first-request latency cliff. The service currently boots against development bootstrap artefacts (see below); real Phase 1 weights drop in by replacing the files under `models/v1.0.0/`, no code changes required.

```bash
# Boot the API
uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port 8000

# Liveness + readiness (returns model_loaded + model_version + api_version)
curl http://localhost:8000/healthz

# Generate a caption from a multipart upload
curl -X POST http://localhost:8000/v1/captions \
    -F "image=@samples/photo.jpg"
```

Interactive Swagger UI is auto-generated at [`/docs`](http://localhost:8000/docs); the raw schema lives at [`/openapi.json`](http://localhost:8000/openapi.json).

---

## FastAPI backend

Phase 2A delivers a production-style inference service rather than a thin demo wrapper. The split mirrors how a real serving stack is laid out:

- **App factory + lifespan** — [`backend/app/main.py`](backend/app/main.py). `create_app()` builds the FastAPI instance; the lifespan loads the YAML `AppConfig`, instantiates a `CaptionPredictor`, calls `warmup()`, and stashes a `PredictorService` singleton on `app.state` so every request reuses one warm model.
- **Routes** — [`backend/app/api/routes.py`](backend/app/api/routes.py). Intentionally thin: validate inputs, delegate, shape the response. No TF imports leak into the HTTP layer.
- **Service layer** — [`backend/app/services/predictor_service.py`](backend/app/services/predictor_service.py). Wraps the predictor, decodes uploaded bytes, measures per-request latency, and returns `(caption, latency_ms)`.
- **Schemas** — [`backend/app/schemas/caption.py`](backend/app/schemas/caption.py). Pydantic v2 request/response models (`CaptionResponse`, `HealthResponse`, `ErrorResponse`) — every payload that crosses the wire is typed and OpenAPI-documented.
- **Backend settings** — [`backend/app/core/config.py`](backend/app/core/config.py). Separate `BackendSettings` (env-overridable: weights path, tokenizer dir, model version, warmup toggle) layered on top of the research-side `AppConfig`. The two are deliberately distinct: research hyperparameters and serving knobs change on different cadences.
- **Structured logging + request IDs** — [`backend/app/core/logging.py`](backend/app/core/logging.py). `RequestContextMiddleware` stamps each request with a UUID; `structlog` carries it through every log line so a single failed caption can be traced end-to-end.
- **Image safety** — [`backend/app/utils/image.py`](backend/app/utils/image.py). Content-type allow-list (JPEG / PNG / WebP / BMP), explicit `ImageDecodeError` so malformed bytes produce a clean `422` rather than a 500.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/healthz`      | Liveness + readiness — reports `model_loaded`, `model_version`, `api_version`. Always 200; readiness is conveyed in the body. |
| `POST` | `/v1/captions`  | Multipart image upload → generated caption + decode strategy + latency + request ID. |
| `GET`  | `/docs`         | Interactive Swagger UI, auto-generated from the Pydantic schemas. |
| `GET`  | `/openapi.json` | Raw OpenAPI 3.1 spec for client codegen. |

`POST /v1/captions` enforces input validation at the boundary: 415 on disallowed content types, 413 on oversized uploads (`serve.max_upload_bytes`), 422 on undecodable image bytes, 400 on empty uploads, 503 while the predictor is still loading during a rolling restart.

### Bootstrap dev artifacts

[`scripts/bootstrap_dev_artifacts.py`](scripts/bootstrap_dev_artifacts.py) generates a *valid but untrained* set of weights + tokenizer under `models/v1.0.0/` so the entire serving stack — lifespan, routes, multipart upload, predictor wiring — can be exercised end-to-end before Phase 1 training has been run on COCO. **The captions it produces are gibberish by design**: every weight is randomly initialised. The point is architectural smoke-testing, not prediction quality. Drop real Phase 1 outputs into the same directory and the backend serves them with zero code changes.

```bash
python -m scripts.bootstrap_dev_artifacts \
    --config configs/base.yaml \
    --output-dir models/v1.0.0
```

---

## Configuration system

Hyperparameters are not globals. They live in YAML files validated by Pydantic v2 `BaseSettings`:

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
2. **No silent typos.** `extra="forbid"` rejects unknown keys (e.g. `vocabularsy_size`) — typos in ML hyperparameters silently using defaults is the worst possible failure mode, and `extra="forbid"` eliminates it.
3. **Env overrides.** `CAPTIONING__TRAIN__BATCH_SIZE=32` overrides at any nesting depth — useful for CI smoke tests, ablations, and serve-time tuning without rebuilding images.

Schema lives in [`src/captioning/config/schema.py`](src/captioning/config/schema.py); loader in [`config/loader.py`](src/captioning/config/loader.py).

---

## Testing & code quality

```bash
make test            # pytest 37/37 (unit + integration)
make lint            # Ruff lint + format check
make typecheck       # mypy strict on src/captioning + scripts
make pre-commit      # All hooks across all files
make freeze-paper-notebook   # Asserts notebook SHA-256 unchanged
```

| Layer | Tool | Status |
|---|---|---|
| Lint + format | [Ruff](https://docs.astral.sh/ruff/) (replaces black + isort + flake8) | ✅ clean |
| Type-check | [mypy](https://mypy.readthedocs.io/) with `pandas-stubs`, `types-PyYAML`, `types-requests` | ✅ 0 errors / 34 files |
| Tests | pytest + pytest-cov + pytest-asyncio | ✅ 37 passing |
| Notebook hygiene | [`nbstripout`](https://github.com/kynan/nbstripout) (pre-commit) | ✅ outputs stripped on commit |
| Secret scanning | [`gitleaks`](https://github.com/gitleaks/gitleaks) (pre-commit) | ✅ enabled |
| Notebook integrity | SHA-256 freeze check via [`make freeze-paper-notebook`](Makefile) | ✅ locked |
| Parity audit | [`scripts/notebook_module_audit.py`](scripts/notebook_module_audit.py) — 4 stages | ✅ all passing |

The parity audit re-implements four notebook stages inline (caption preprocessing, tokenizer vocabulary + encoding, image preprocessing, decoder forward pass) and asserts the modular path produces byte-identical (or `tf.allclose`-identical) output. It is the contract that gates any behavioural improvement.

---

## Key engineering improvements

This is what separates this repository from a notebook conversion:

- **Modular package** with the `src/` layout — every test exercises the *installed* package the same way users will.
- **Strict Pydantic v2 configuration** — typed, validated, env-overridable, refuses unknown keys.
- **`CaptionTokenizer` wrapper** — stable interface for the model and inference; Phase 5 can swap it for HuggingFace `tokenizers` without touching the encoder, decoder, or generation loop.
- **Singleton-friendly inference** — `CaptionPredictor.from_artifacts(...)` + `warmup()` are designed for FastAPI lifespans, not just CLI calls.
- **Shared train/serve preprocessing** — the same `preprocess_image_tensor` runs in `tf.data` pipelines and at inference time, eliminating train/serve skew by construction.
- **Reproducibility** — seeded sampling, seeded splits, seeded RNGs (`utils.seed.set_global_seed`), pinned `tensorflow-cpu==2.15.0` (TF 2.16+ ships Keras 3 by default and silently breaks `TextVectorization` save/load).
- **Notebook freeze** — IEEE artefact protected by a SHA-256 check; published BLEU stays reproducible across the project's lifetime.
- **Optional dependency groups** (`[hf]`, `[eval]`, `[mlflow]`, `[dev]`) — slim production image stays lean; HF baselines and metric tooling are opt-in extras.
- **Decoupled experiment artefacts** — model weights live in HuggingFace Hub (planned), MLflow tracking on DagsHub free tier (planned). Git stays small.
- **Structured logging** — `structlog` emits JSON in production, pretty colourised logs in dev, switched by `APP_ENV`.
- **No silent rewrites** — every notebook → module move is documented with a cell mapping in [`docs/PHASE_1_NOTES.md`](docs/PHASE_1_NOTES.md); behavioural quirks (e.g. `compute_loss_and_acc` ignoring its `training` argument) are preserved verbatim with code comments referencing the doc.

---

## Limitations

- The model produces generic captions on cluttered or rare-object scenes — a known limitation of the IEEE-era architecture, addressed in Phase 3 by adding modern foundation-model baselines (BLIP, ViT-GPT2, GIT) for side-by-side comparison.
- Greedy decoding only; beam search is a Phase 1b addition.
- Validation pipeline includes a leftover `shuffle()` from the notebook (functionally harmless, removed in Phase 1b).
- BLEU is the only metric in v1; CIDEr / METEOR / ROUGE-L slot into the same runner interface in Phase 1b.

These are explicitly tracked rather than hidden; full list in [`docs/PHASE_1_NOTES.md` § Technical debt](docs/PHASE_1_NOTES.md#technical-debt-remaining).

---

## Roadmap

- **Phase 1b** — beam search, CIDEr / METEOR / ROUGE-L, masked accuracy parity-fix, label smoothing, warmup + cosine LR schedule.
- **Phase 2A** ✅ — FastAPI backend, lifespan-managed predictor singleton, multipart inference endpoint, structured logging + request IDs, Pydantic schemas, Swagger/OpenAPI docs, health/readiness probe.
- **Phase 2B** — React/Vite frontend with Tailwind UI, drag/drop image uploads, live API integration against `POST /v1/captions`, deployment integration (HuggingFace Spaces backend + Vercel-hosted frontend), GitHub Actions CI/CD.
- **Phase 3** — Tier-1 multimodal upgrades: BLIP-base / ViT-GPT2 / GIT-base-coco side-by-side comparison demo with per-model BLEU + latency.
- **Phase 4** — Sentry, Prometheus, DagsHub-hosted MLflow link, Architecture Decision Records (`docs/adr/`).
- **Future work** — ViT + Transformer fine-tune on COCO; VLM API integration (Anthropic Claude vision) behind a feature flag; VQA endpoint.

Detailed plan: [`docs/restructure-plan.md`](docs/restructure-plan.md).

### Current capabilities

- Notebook parity preserved — IEEE artefact frozen by SHA-256, four-stage parity audit gates every behavioural change.
- Typed modular ML package — Pydantic v2 configs, mypy-strict, 37 unit tests passing.
- Production-style inference API — FastAPI app factory, lifespan-managed `CaptionPredictor` singleton, warmup on boot.
- Swagger/OpenAPI testing — interactive `/docs` UI for hand-testing every endpoint, raw `/openapi.json` for client codegen.
- Structured logging — JSON in production, pretty in dev; per-request UUIDs threaded through every log line.
- End-to-end image upload → caption flow — multipart upload → content-type guard → image decode → predictor → typed response with latency + request ID.

---

## Citation

If you reference this work in academic writing, please cite the IEEE paper:

```bibtex
@inproceedings{ainarratives,
  title     = {AI Narratives: Bridging Visual Content and Linguistic Expression},
  booktitle = {Proceedings of the IEEE Conference},
  publisher = {IEEE},
  year      = {2024},
  url       = {https://ieeexplore.ieee.org/document/10675203},
}
```

---

## Acknowledgements

- The model architecture, hyperparameters, and BLEU baseline are from the IEEE-published paper *AI Narratives: Bridging Visual Content and Linguistic Expression*.
- COCO 2017 captions provided by the [Microsoft COCO project](https://cocodataset.org/).
- TensorFlow / Keras for the model layers; Pydantic for the configuration system; sacrebleu for evaluation; Ruff, mypy, and pytest for tooling.

---

## License

Released under the [MIT License](LICENSE). The IEEE paper itself is published under separate terms.

---

## Author

**Apoorv Raj** — AI / ML systems engineer.
Repository structured by phase; contributions and issues welcome.
