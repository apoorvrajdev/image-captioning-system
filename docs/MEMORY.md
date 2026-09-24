# Project memory — current state

> Living document: **current state only**. Permanent decisions → [`DECISIONS.md`](DECISIONS.md).
> Backlog → [`TASKS.md`](TASKS.md). Update at the end of every task.

_Last updated: 2026-09-24_

## Current phase

- **Completed:** Phase 0 (bootstrap), Phase 1 (modularisation), Phase 1b (training stabilisation +
  metric suite + stabilized checkpoint), Phase 2A (FastAPI), Phase 2B (SPA), Phase 2C (public deployment),
  Stage 0 evaluation-methodology gate (verdict: **reframe, do not retrain**).
- **Next:** Phase 3 — multimodal baselines (3A–3D in [`TASKS.md`](TASKS.md)). **Not started.**
  It must be decomposed into small tasks before any implementation.
- **Current task:** none in progress.

## System status (verified 2026-09-24, local Windows, Python 3.10.11)

| Check | Result |
|---|---|
| `pytest tests backend/app/tests` | 94 passed (1 pydantic `model_` namespace warning) |
| ruff lint + format check | clean |
| mypy (pyproject config, `strict = false`) | 0 errors, 71 files |
| Parity audit (`scripts/notebook_module_audit.py`) | 4/4 |
| Notebook SHA-256 freeze | OK (after the LF `.gitattributes` fix below) |
| Frontend `npm run lint` / `npm run build` | clean / builds |

Live: SPA on Vercel, API on HF Space (Docker, cpu-basic), weights from HF Hub
`apoorvrajdev/captioning-inceptionv3-transformer`. Headline results: `results/stabilized-greedy/`,
`results/stabilized-beam-w4-lp07-rp12/` (beam CIDEr 0.826; 5-ref BLEU-4 25.91).

## Recent changes

- Engineering-workflow bootstrap: added `docs/{MEMORY,TASKS,DECISIONS,TEST_PLAN,SECURITY}.md`,
  extended `CLAUDE.md` (commands, invariants, traps, definition of done), wired the parity audit into
  CI (`python-tests` job), corrected `docs/CI.md` drift, and pinned the frozen notebook to LF in `.gitattributes`
  (with `core.autocrlf=true` the Windows checkout was CRLF and the freeze check always failed locally).
- Stage 0 eval audit landed (`docs/EVAL_METHODOLOGY.md`, `results/stabilized-beam-w4-lp07-rp12/verdict.md`).

## Known issues / open debt

- **Doc drift:** README claims "mypy strict" (config is `strict = false`) and a 3.10/3.11/3.12 pytest
  matrix (CI runs 3.10/3.11). README's Sample outputs and Performance sections still describe "bootstrap
  weights" and Phase 1b re-training as pending. Tracked as TASK-003.
- **Model version labels disagree:** README says the stabilized checkpoint is Hub tag `v2.0.0` (1b-I),
  while the Live Demo table says "pinned to `v1.0.0`" and `BackendSettings.model_version` defaults to `v1.0.0`.
  The Space's actual `BACKEND_WEIGHTS_HUB_REVISION` needs confirming by the owner (TASK-004).
- `Makefile`: `docker-build*` targets point at the nonexistent `backend/Dockerfile`. `docker-up/down`
  reference a missing compose file. `eval` needs a `--weights` it doesn't pass (TASK-005).
- The upload route reads the whole body into memory before the size check (see [`SECURITY.md`](SECURITY.md)).
- No frontend tests / e2e, no coverage measured in CI, no dependency-vulnerability scanning.
- `deploy-backend.yml` has recently hit HF push rate limits (HTTP 429). That's a platform issue, not workflow logic.
- Pydantic warning: `BackendSettings.model_version` (`backend/app/core/config.py`) collides with the protected `model_` namespace (harmless; the response schemas already set `protected_namespaces=()`).

## Before coding, know this

- Windows + Git Bash, no `make`. Use `.venv/Scripts/*.exe` (see `CLAUDE.md` → Commands).
- The notebook is frozen, parity must stay 4/4, and `results/` + `models/vX.Y.Z/` are immutable.
- Retraining and deployments are owner-run (Kaggle / HF / Vercel). Code tasks prepare instructions only.
