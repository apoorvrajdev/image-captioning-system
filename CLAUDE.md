# Project Conventions for Claude Code

## CRITICAL: Commit & Attribution Rules

**Claude Code MUST follow these rules without exception:**

1. **NEVER add `Co-Authored-By: Claude` or any AI co-author trailer to commit messages.**
2. **NEVER add `🤖 Generated with Claude Code` footers or any AI attribution.**
3. **NEVER mention Claude, Anthropic, OpenAI, Copilot, AI, LLMs, or any model/assistant name in commit messages, code comments, file headers, documentation, PR descriptions, or changelogs.**
4. **All commits must be authored solely by:**
   - Name: `apoorvrajdev`
   - Email: `apoorvrajmgr@gmail.com`
5. **NEVER stage or commit changes on your own.** Only suggest commit messages — the user runs `git commit` themselves.
6. **NEVER push to remote.** Only the user pushes.
7. **NEVER create branches, tags, or releases on your own.**

## Commit Message Format

Use Conventional Commits. Examples:
- `chore: initial repo scaffolding`
- `feat(backend): add /caption endpoint for image upload`
- `feat(inference): add beam search decoder`
- `fix(data): correct COCO split deduplication`
- `fix(training): stabilize loss scaling for mixed precision`
- `docs: update stabilized training runbook`
- `test(evaluation): add BLEU/CIDEr metric tests`
- `refactor(models): extract encoder CNN factory`
- `perf(inference): cache image features for batched predict`

Keep subject under 72 characters. Body optional but explains *why*, not *what*.

## Project Stack

- **Core ML:** Python 3.10+, TensorFlow / Keras, NumPy, Pillow
- **Model:** InceptionV3 encoder + Transformer decoder for image captioning
- **Backend:** FastAPI app under `backend/app/` (routes, services, schemas, utils)
- **Frontend:** React 18 + Vite (JSX) under `frontend/`, ESLint configured
- **Config:** YAML configs under `configs/` loaded via `src/captioning/config/`
- **Data:** MS COCO pipeline under `src/captioning/data/`
- **Evaluation:** BLEU, CIDEr, METEOR, ROUGE under `src/captioning/evaluation/`
- **Tooling:** `pyproject.toml`, `Makefile`, `pytest`, packaging as `captioning`

## Repository Layout (authoritative)

- `src/captioning/` — installable library (`config`, `data`, `models`, `preprocessing`, `training`, `inference`, `evaluation`, `utils`)
- `backend/app/` — FastAPI service (`api/routes.py`, `services/predictor_service.py`, `schemas/`, `core/`, `utils/`)
- `frontend/src/` — React UI (`components/`, `services/api.js`)
- `scripts/` — CLI entrypoints (`train.py`, `evaluate.py`, `predict.py`, etc.)
- `configs/` — YAML training/eval configs
- `models/vX.Y.Z/` — versioned model artifacts (`model.h5`, `vocab.json`)
- `tests/unit/` — pytest unit tests
- `notebooks/` — exploratory notebooks (not part of runtime)
- `docs/` — phase notes and runbooks

## Code Standards

- **Python:** type hints on all new/edited public functions; prefer `pathlib.Path` over string paths
- **Imports:** absolute imports from `captioning.*`; no relative imports across top-level packages
- **Determinism:** seed NumPy / TF / Python `random` whenever introducing stochastic code paths in training or evaluation
- **Configs:** never hardcode hyperparameters in scripts — extend `src/captioning/config/schema.py` and update the relevant YAML in `configs/`
- **Models / vocab:** never modify files under `models/vX.Y.Z/` in place; bump the version directory instead
- **Backend layering:** `api/routes.py` only orchestrates; inference logic stays in `backend/app/services/` and `src/captioning/inference/`
- **Schemas:** all FastAPI request/response bodies go through Pydantic schemas in `backend/app/schemas/`
- **Frontend:** functional components + hooks; keep API calls inside `frontend/src/services/api.js`
- **Tests:** new behavior gets a unit test under `tests/unit/`; keep tests CPU-only and offline (no network, no real model downloads)

## Working Style

- Plan before implementing for any non-trivial change (training loop, decoder, data pipeline, API contract)
- One module at a time, with tests
- Run `pytest` for touched areas before declaring a change done
- After making changes, summarize what you did so the user can review and commit
- If a change spans library + backend + frontend, list the affected files grouped by layer in the summary
