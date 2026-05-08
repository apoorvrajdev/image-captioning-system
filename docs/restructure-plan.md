# Production Restructuring Plan

> Public, in-repo copy of the engineering plan that drives the transition from
> a single-notebook research project into a deployable multimodal AI platform.
> The original (with internal exploration notes) lives in the developer's
> `~/.claude/plans/` directory; this version is the canonical public artefact.

## Context

This repository is the engineering home of an IEEE-published image-captioning
research project. The published artefact is a single Jupyter notebook
([`notebooks/01_ieee_inceptionv3_transformer.ipynb`](../notebooks/01_ieee_inceptionv3_transformer.ipynb))
implementing **InceptionV3 (frozen) + custom Keras Transformer decoder**
trained on **COCO 2017**, reporting **BLEU ~24**.

**Goal**: convert the repo into a recruiter-grade, production-style
multimodal AI platform with a live free-tier demo, while **preserving the
IEEE notebook byte-for-byte** as the canonical research artefact.

**Constraints**:

- Hosting budget: **$0/month** → HuggingFace Spaces (backend) + Vercel free
  (frontend) + HuggingFace Hub (model artefacts) + DagsHub free MLflow.
- Multimodal scope (v1): **Tier 1 only** — add three pretrained HuggingFace
  models (BLIP-base, ViT-GPT2, GIT-base-coco) for a side-by-side comparison
  demo. Tier 2/3/4 are listed under *Future work* only.

---

## 1. Folder Structure (target)

```
image-captioning-system/
├── notebooks/
│   └── 01_ieee_inceptionv3_transformer.ipynb   # FROZEN
├── src/captioning/                             # Installable Python package
│   ├── config/                                 # Pydantic settings + YAML loader
│   ├── data/                                   # COCO loaders, preprocess, splits
│   ├── tokenizer/                              # CaptionTokenizer (Keras TextVectorization wrapper)
│   ├── models/                                 # CNN encoder, Transformer decoder, factory
│   ├── training/                               # Trainer, losses, metrics, callbacks
│   ├── inference/                              # Greedy + beam search predictors
│   ├── evaluation/                             # BLEU, CIDEr, METEOR, ROUGE
│   ├── io/                                     # Checkpoints, image decoding, HF Hub I/O
│   └── utils/                                  # Logging, seeding, timing
├── configs/                                    # YAML hyperparameters (validated by Pydantic)
├── scripts/                                    # CLI entrypoints (train, eval, predict, upload)
├── models/                                     # Local checkpoint registry (gitignored content)
├── backend/                                    # FastAPI service (depends on src/captioning)
├── frontend/                                   # Next.js 14 + TypeScript + Tailwind + shadcn/ui
├── tests/                                      # ML-core tests (unit + integration)
├── docs/                                       # Architecture, ADRs, results, deployment
├── .github/workflows/                          # CI, CD, model-eval
├── docker-compose.yml                          # Local dev: backend + frontend + mlflow
├── pyproject.toml                              # Single source of truth for the package
└── Makefile                                    # Discoverable command index
```

**Key architectural rules**:

- `src/captioning/` is the ML core; `backend/app/` imports from it. Never
  reverse the dependency.
- The IEEE notebook is **frozen** — `make freeze-paper-notebook` is a CI
  check that fails on any byte change.
- Model weights are **never committed**; they live in HuggingFace Hub
  (`yourname/captioning-weights`) and are downloaded at backend startup.
- Configuration is **YAML files validated by Pydantic v2 BaseSettings**, not
  Hydra. Env vars override via `CAPTIONING__TRAIN__BATCH_SIZE=32` syntax.

---

## 2. Migration Strategy

**Approach: verbatim refactor first, improvements second.** Reproducibility
of the IEEE BLEU score is non-negotiable; behaviour parity must be proven
*before* any improvement is made.

### Phase 1a — "Lift and shift" (parity goal: BLEU within ±0.3 of notebook)

| Step | Notebook cell | Target module |
|---|---|---|
| 1 | Hyperparams | `configs/base.yaml` + `src/captioning/config/schema.py` |
| 2 | Caption preprocess | `data/preprocess.py::preprocess_caption` |
| 3 | COCO loader | `data/coco.py::load_coco_annotations` |
| 4 | Tokenizer | `tokenizer/vectorizer.py::CaptionTokenizer` |
| 5 | Splits | `data/splits.py::make_splits(seed=...)` |
| 6 | Image preprocess | `data/preprocess.py::preprocess_image` |
| 7 | tf.data pipeline | `data/pipeline.py::build_{train,val}_pipeline` |
| 8 | Augmentation | `data/augmentation.py::default_augmentation` |
| 9 | InceptionV3 encoder | `models/encoder_cnn.py` |
| 10 | Transformer encoder | `models/transformer_encoder.py` |
| 11 | Embeddings | `models/embeddings.py` |
| 12 | Transformer decoder | `models/transformer_decoder.py` |
| 13 | Captioning model | `models/captioning_model.py` |
| 14 | Wiring | `models/factory.py::build_caption_model(config)` |
| 15 | Loss + compile | `training/losses.py` + `training/trainer.py` |
| 16 | Fit | `training/trainer.py::Trainer.fit` |
| 17 | Inference | `inference/greedy.py`, `inference/predictor.py` |
| 18 | Save weights | `io/checkpoints.py` + `scripts/train.py` |

### Parity validation gate

`scripts/notebook_module_audit.py` runs both pipelines on a fixed 100-image
fixture and asserts:

- Tokenizer vocabulary identical (set equality).
- Image preprocessing tensor-equal (`np.allclose`, atol=1e-5).
- Model output logits equal at fixed weights (atol=1e-4).
- Captions on 20 fixed images byte-equal between notebook and module path.

### Phase 1b — Quality improvements (only after parity is green)

1. Masked accuracy metric (notebook tracks loss only).
2. Beam search inference.
3. Warmup + cosine LR schedule (replaces bare Adam).
4. CIDEr / METEOR / ROUGE-L (paper reports BLEU only).
5. `vocab.json` sidecar alongside `vocab.pkl`.
6. Label smoothing.

---

## 3. Implementation Roadmap

| Phase | Deliverable | Effort | Recruiter signal |
|---|---|---|---|
| **0** | Repo bootstrap (this phase) | 3 hrs | Clean repo, lint passes from commit 1 |
| **1** | Modular ML core + backend MVP | ~15 hrs | Working FastAPI for the IEEE model, runnable via `docker compose up` |
| **2** | CI/CD + first deploy (HF Space + Vercel) | ~12 hrs | Live demo URL on LinkedIn |
| **3** | Tier 1 multimodal: BLIP/ViT-GPT2/GIT comparison demo | ~20 hrs | The screenshot recruiters share |
| **4** | Polish + observability (Sentry, Prometheus, ADRs) | ~8 hrs | Reads as production-grade, not a research one-off |

### Future work (out of scope for v1)

- **Tier 2**: ViT + Transformer fine-tune on COCO via Kaggle GPU (BLEU 24 → 32+).
- **Tier 3**: Anthropic Claude vision endpoint as a "Frontier" tab.
- **Tier 4**: VQA "Ask the image" extension reusing Tier 3 infra.
- Self-hosted compose on a VPS with Caddy TLS and DVC dataset versioning.

---

## 4. Deployment Stack (free-tier)

| Layer | Service | Why |
|---|---|---|
| Backend hosting | HuggingFace Spaces (Docker SDK, free CPU) | 16 GB RAM, ML-native, recruiter-clickable |
| Frontend hosting | Vercel free | Next.js native; per-PR preview URLs |
| Model artefacts | HuggingFace Hub | Free, unlimited public, versioned, model cards |
| Experiment tracking | MLflow on DagsHub free | Public read-only tracking server |
| Errors | Sentry free (5k errors/mo) | |
| Uptime | UptimeRobot free | Doubles as HF Space wake-up keeper |
| Domain | None (use `*.hf.space` and `*.vercel.app`) | $0 budget |

---

## 5. Trade-offs Decided

| Decision | Alternative rejected | Reason |
|---|---|---|
| FastAPI | Flask | Async, OpenAPI, Pydantic, lifespan |
| Next.js 14 App Router | Streamlit | Streamlit screams "research demo" |
| TanStack Query | Redux | Server state belongs in a server-state lib |
| YAML + Pydantic | Hydra | Hydra is overkill for 1–3 active configs |
| MLflow on DagsHub | W&B | DagsHub public free; no recruiter login |
| Keep TextVectorization | HF tokenizer in v1 | Changes vocab → breaks paper parity |
| Verbatim refactor first | Clean rewrite | IEEE BLEU reproducibility non-negotiable |
| `tensorflow-cpu==2.15.0` pinned | Floating TF | TF 2.16 broke Keras 2 compat with notebook |
| HF Spaces backend | Fly.io paid | Free-tier-only constraint |
| Multipart uploads | Base64 in JSON | 33% overhead, no streaming |
| `--workers 1` uvicorn | Multi-worker | TF graph + InceptionV3 ×N OOMs |
| Tier 1 only (HF baselines) | Tier 2/3/4 in v1 | User selected Tier 1; others as future work |

---

## 6. Verification Plan

**Phase 1**:

- `pytest tests/ -v` → all green; coverage ≥ 70% on `src/captioning/`.
- `python scripts/notebook_module_audit.py` → parity assertions all pass.
- `docker compose up` → `curl -F "file=@sample.jpg" http://localhost:8000/v1/captions`
  returns valid caption JSON.

**Phase 2**:

- GitHub Actions `ci.yml` green on a PR.
- HF Space URL serves `/v1/model/info`.
- Vercel preview URL renders frontend; uploading a sample image returns a caption.

**Phase 3**:

- `GET /v1/models` returns 4 entries.
- `POST /v1/compare` returns 4 captions; total latency < 15s on HF Space CPU.
- `model-eval.yml` posts a BLEU comparison comment on a test PR.

**Phase 4**:

- `/metrics` exposes `caption_inference_seconds` histogram.
- DagsHub MLflow link shows ≥ 1 logged run with metrics.
- `make freeze-paper-notebook` fails when notebook bytes change; passes when restored.
