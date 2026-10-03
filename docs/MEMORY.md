# Project memory — current state

> Living document: **current state only**. Permanent decisions → [`DECISIONS.md`](DECISIONS.md).
> Backlog → [`TASKS.md`](TASKS.md). Update at the end of every task.

_Last updated: 2026-10-03_

## Current phase

- **Completed:** Phase 0 (bootstrap), Phase 1 (modularisation), Phase 1b (training stabilisation +
  metric suite + stabilized checkpoint), Phase 2A (FastAPI), Phase 2B (SPA), Phase 2C (public deployment),
  Stage 0 evaluation-methodology gate (verdict: **reframe, do not retrain**), engineering-workflow setup.
- **Next:** Phase 3 — multimodal baselines (3A–3D in [`TASKS.md`](TASKS.md)). **Not started.**
  It must be decomposed into small tasks before any implementation.
- **Current task:** TASK-008 (backend deploy). The fix is implemented in `deploy-backend.yml` and awaiting
  review, commit, and a first `workflow_dispatch` run. TASK-007 (Playwright) is deferred to the start of Phase 3D.

## System status (verified 2026-10-03, local Windows, Python 3.10.11)

| Check | Result |
|---|---|
| `pytest tests backend/app/tests` | 94 passed (1 pydantic `model_` namespace warning) |
| ruff lint + format check | clean (84 files) |
| mypy (pyproject config, `strict = false`) | 0 errors, 71 files |
| Parity audit (`scripts/notebook_module_audit.py`) | 4/4 |
| Notebook SHA-256 freeze | OK |
| `SKIP=mypy pre-commit run --all-files` (clean clone, LF) | all hooks pass |
| Frontend `npm run lint` / `npm run build` | clean / builds |
| CI on `main` (last run, 2026-10-03) | green, all 6 jobs incl. `pre-commit` |
| `deploy-backend.yml` (last run, 2026-10-03) | **failed** (non-fast-forward to the Space, TASK-008) |
| Backend Space (HF runtime API, 2026-10-03) | **down: `CONFIG_ERROR`, "Missing configuration in README"** |

SPA on Vercel. The API's HF Space (Docker, cpu-basic) is **not serving**: it is in `CONFIG_ERROR` because every
commit pushed since `befac80` (2026-06-02) lacks the README YAML header, and later pushes were also rejected. Weights live
on HF Hub `apoorvrajdev/captioning-inceptionv3-transformer`. Headline results: `results/stabilized-greedy/`, `results/stabilized-beam-w4-lp07-rp12/`
(beam CIDEr 0.826; 5-ref BLEU-4 25.91).

## Recent changes

- 2026-10-03 deploy fix (TASK-008, uncommitted): `deploy-backend.yml` deploys the tested SHA (manual runs must
  prove the exact SHA passed CI), skips superseded commits, adds the Space config header to a force-pushed deploy commit, and gates on HF `RUNNING` +
  `/healthz` `model_loaded: true` (ADR-017).
- 2026-10-03 workflow upgrade (committed and pushed as `545c76b`..`9e8874c`):
  - `.claude/` config is now tracked (ADR-015), and `.claude/settings.json` enforces the invariants.
  - The code index rebuilds at session start.
  - CI gained a `pre-commit` job (ADR-016).
  - `CLAUDE.md` gained a debugging protocol and a review step.
  - `ship-task` gained per-change-type flows.
  - README drift fixed (TASK-003).
- 2026-09-24: living docs added, parity audit wired into CI, frozen notebook pinned to LF in `.gitattributes`.
- Stage 0 eval audit landed (`docs/EVAL_METHODOLOGY.md`, `results/stabilized-beam-w4-lp07-rp12/verdict.md`).

## Known issues / open debt

- **Backend down, auto-deploy broken (TASK-008, fix pending first run):**
  - The Space is in `CONFIG_ERROR` because `befac80` removed the README YAML header.
  - Pushes fail: 429s in June, then non-fast-forward since 2026-09-24. The Space's only extra commit, `302e907`,
    is tree-identical to GitHub's `64f80e8`.
  - The README's "Deployed / live" claims are stale until the redeploy succeeds. The README is currently
    staged by the owner, so it was left untouched.
- **Model version labels disagree (TASK-004):** README says the stabilized checkpoint is Hub tag `v2.0.0` (1b-I),
  while the Live Demo table says "pinned to `v1.0.0`" and `BackendSettings.model_version` defaults to `v1.0.0`.
  The owner needs to confirm the Space's `BACKEND_WEIGHTS_HUB_REVISION`.
- `Makefile`: `docker-build*` point at the nonexistent `backend/Dockerfile`. `docker-up/down` reference a
  missing compose file. `eval` lacks required `--weights`/`--tokenizer-dir` (TASK-005).
- The upload route reads the whole body into memory before the size check (TASK-006, [`SECURITY.md`](SECURITY.md)).
- No frontend tests / e2e (TASK-007), no coverage measured in CI, no dependency-vulnerability scanning,
  and no full-history secret scan in CI.
- Pydantic warning: `BackendSettings.model_version` (`backend/app/core/config.py`) collides with the protected
  `model_` namespace (harmless; the response schemas already set `protected_namespaces=()`).

## Before coding, know this

- Windows + Git Bash, no `make`. Use `.venv/Scripts/*.exe` (see `CLAUDE.md` → Commands).
- The notebook is frozen, parity must stay 4/4, and `results/` + `models/vX.Y.Z/` are immutable. Edits to
  them are blocked by `.claude/settings.json`.
- Retraining and deployments are owner-run (Kaggle / HF / Vercel). Code tasks prepare instructions only.
