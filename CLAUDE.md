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

## Commit Granularity

**Prefer many small, focused commits over a few large ones.** Atomic commits are
a widely defensible engineering practice — easier review, cleaner revert paths,
more legible history — and a portfolio project benefits from the richer
contribution graph as a byproduct. Split a batch of work so each logical change
lands as its own Conventional Commit.

### Rules

- **One reason per commit.** If you'd describe the work as "X *and* Y" with
  separable verbs (e.g. "fix tokenizer *and* add tests *and* update CHANGELOG"),
  that's three commits. If it's a single coherent action ("rename `foo` to `bar`
  across the codebase"), it's one commit — even if it touches twenty files.
  Granularity is logical, not per-file.

- **Indivisible commits stay indivisible.** Pre-registration blocks (which must
  land before any result), notebook SHA-256 freeze updates, atomic reverts —
  these exist as one commit on purpose and are NOT subject to the splitting
  rule. Do not break them apart to inflate the count.

- **Conventional Commits format applies to every split commit**, not just the
  combined one. `feat(eval): add rescore script` and `test(eval): cover error
  paths` are two valid commits; rolling them together loses scope clarity.

- **Always present the sequence, never execute it.** Per the existing rule that
  Claude does not stage, commit, or push: output the full intended commit
  sequence (each `git add <file>` + `git commit` pair) so the user can run
  them. Order matters — within a multi-commit sequence, prefer:
  schemas/types → implementation → tests → docs → CHANGELOG.

- **No padding.** Do not split a single indivisible change across artificial
  commits purely to inflate the count. Cohesive granularity, not noise.

### Example

For a change that adds a new evaluation script, its tests, and a Makefile target:

```
# Bad — one combined commit:
feat(eval): add rescore script, tests, and Makefile target

# Good — three commits, in order:
feat(eval): add scripts/rescore_nltk_bleu.py
test(eval): cover rescore script error paths
build(make): add rescore-5ref target
```

The "good" sequence is reviewable, revertable, and reads honestly as three
logical contributions.

## Project Stack

- **Core ML:** Python 3.10+, TensorFlow / Keras, NumPy, Pillow
- **Model:** InceptionV3 encoder + Transformer decoder for image captioning
- **Backend:** FastAPI app under `backend/app/` (routes, services, schemas, utils)
- **Frontend:** React 19 + Vite 8 + Tailwind v4 (JSX, no TypeScript) under `frontend/`, ESLint flat config
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
- `tests/unit/` — pytest unit tests; `backend/app/tests/` — route tests (fake predictor, no TF)
- `notebooks/` — frozen IEEE notebook + exploratory notebooks (not part of runtime)
- `results/<run_id>/` — committed evaluation artefact sets (append-only)
- `docs/` — phase notes, runbooks, and the living docs: `MEMORY.md` (current state), `TASKS.md` (backlog), `DECISIONS.md`, `TEST_PLAN.md`, `SECURITY.md`
- `.claude/` — local-only (gitignored) agent context: `context/repo-map.md` + generated index, `skills/`, `agents/`

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

## Commands

Local dev is Windows + Git Bash; `make` is **not** installed, so run the underlying commands.
Python is the repo venv: `.venv/Scripts/python.exe` (3.10). Tools: `.venv/Scripts/{pytest,ruff,mypy}.exe`.

| Task | Command |
|---|---|
| install | `pip install -r requirements-dev.txt -r requirements-eval.txt && pip install -e ".[hf,mlflow]"` |
| backend dev | `uvicorn app.main:app --app-dir backend --port 8000 --reload` (needs `models/v1.0.0/model.h5` or `BACKEND_WEIGHTS_HUB_REPO`) |
| frontend dev | `cd frontend && npm run dev` (http://localhost:5173) |
| all tests | `pytest tests backend/app/tests -q` (94 tests, ~30 s) |
| backend tests only | `pytest backend/app/tests -q` (<1 s, no TF) |
| single test | `pytest tests/unit/test_beam_decoder.py::test_name -q` |
| lint + format | `ruff check src/captioning backend scripts tests && ruff format --check src/captioning backend scripts tests` |
| typecheck | `MYPYPATH="src;backend" mypy --explicit-package-bases --namespace-packages src/captioning backend/app scripts` |
| parity audit | `python -m scripts.notebook_module_audit` (4 stages, must print `[OK] 4/4`) |
| notebook freeze | `python -c "import hashlib;print(hashlib.sha256(open('notebooks/01_ieee_inceptionv3_transformer.ipynb','rb').read()).hexdigest())"` must equal `.paper-notebook.sha256` |
| frontend checks | `cd frontend && npm run lint && npm run build` |
| refresh code index | `.venv/Scripts/python.exe .claude/context/build_index.py` |

CI (`.github/workflows/ci.yml`) runs exactly: ruff lint + format check, mypy, pytest on 3.10/3.11 + parity audit, notebook freeze, frontend lint + build. Local DoD = these.

## Invariants — never break silently

- **Frozen notebook.** `notebooks/01_ieee_inceptionv3_transformer.ipynb` is never edited or re-locked. Improvements go into `src/captioning/`.
- **Parity.** `scripts/notebook_module_audit.py` stays 4/4. Default config values in `schema.py` / `configs/base.yaml` reproduce the notebook; new behaviour is **opt-in** behind a flag whose default preserves parity.
- **Train/serve parity.** Serving decodes uploads through the same `preprocess_image_tensor` as training. Never add a second image or caption normalisation path.
- **Strict config.** Every config model keeps `extra="forbid"`. Research knobs → `AppConfig` (`CAPTIONING__*`); deployment knobs → `BackendSettings` (`BACKEND_*`). Don't mix them.
- **Inference lifecycle.** One `CaptionPredictor` loaded in the FastAPI lifespan, reused per request, TF work offloaded with `anyio.to_thread.run_sync`. No model loading or TF imports in routes.
- **API contract.** `/healthz` always 200 (readiness in body). `/v1/captions` status codes 200/400/413/415/422/503 and the `CaptionResponse` shape are consumed by `frontend/src/services/api.js`. Changing either means updating both sides in one task.
- **Artefacts.** `models/vX.Y.Z/` and published HF Hub tags are immutable; `results/<run_id>/` directories are append-only. New checkpoint or new eval → new version or new run dir.
- **Evaluation methodology.** Don't change tokenisation, reference count, or smoothing for an existing metric without documenting it (see `docs/EVAL_METHODOLOGY.md`). Compare runs only under an identical eval setup.
- **TF pin.** `tensorflow-cpu==2.15.0`, `numpy<2`. Keras 3 (TF ≥ 2.16) breaks `TextVectorization` save/load.

## Traps

- `core.autocrlf=true` on this machine: `.gitattributes` pins the frozen notebook to LF so the SHA-256 check passes. Don't remove that rule.
- Importing `captioning.models` / `inference` pulls in TensorFlow (~10 s). Backend route tests stay TF-free by using `FakePredictorService`. Keep it that way.
- `models/v1.0.0/model.h5` is untracked. Production pulls weights from HF Hub (`BACKEND_WEIGHTS_HUB_*`).
- README badges and prose drift from the config (e.g. "mypy strict" while `strict = false`, a Python 3.12 matrix that CI doesn't run). Trust the config files, not the README.
- `make docker-build` points at `backend/Dockerfile`, which doesn't exist (the Dockerfile is at the repo root).

## Retrieval protocol

1. Read `docs/MEMORY.md` (current state) and `.claude/context/repo-map.md` (which area).
2. Grep `.claude/context/symbols.tsv` for the symbol → file:line.
3. Check `.claude/context/deps.json` `imported_by` for blast radius.
4. Open **only** those files. Don't read the tree to get oriented. Rebuild the index after adding/moving modules.

## Working Style

- Default loop for every task: **locate → load the matching `.claude/skills/*/SKILL.md` → plan → smallest correct change → verify → update docs → report**
- Plan before implementing for any non-trivial change (training loop, decoder, data pipeline, API contract)
- One module at a time, with tests. Reuse existing abstractions; don't touch unrelated files; no new dependencies without saying why
- Never weaken an assertion or delete a test to make it pass. Fix the root cause
- Run `pytest` for touched areas before declaring a change done
- After making changes, summarize what you did so the user can review and commit
- If a change spans library + backend + frontend, list the affected files grouped by layer in the summary
- Work from `docs/TASKS.md` one task at a time. Never implement a whole phase in one pass

## Definition of done

A task is done only when every check from its skill's DoD ran green **in this session**, with the output quoted as proof:
the relevant tests, ruff lint + format, mypy (Python changes), parity audit + notebook freeze (any `src/captioning/` or `configs/` change),
frontend lint + build (any `frontend/` change). Then update `docs/MEMORY.md` (state) and `docs/TASKS.md` (status), add a `docs/DECISIONS.md` entry if a permanent decision was made,
and end with the proposed commit sequence. If a check can't run, say so. That box is not ticked.

## Orchestration (multi-lane tasks)

Lanes (`.claude/agents/`): `ml-backend-engineer` (`src/`, `backend/`, `scripts/`, `configs/`, `tests/`), `frontend-engineer` (`frontend/`), `docs-writer` (`docs/`, `README.md`).

1. Plan with a file-ownership map. No file is owned by two lanes.
2. Parallelise only independent work (e.g. frontend against an already-fixed API schema). Sequence anything with a dependency edge.
3. Dispatch independent lanes in one message, each with its goal, file boundary, skill, and verification commands.
4. The orchestrator integrates the seams itself (schema ↔ `api.js`, config ↔ YAML) and re-reads the merged diff.
5. Verify the merged tree against the full DoD. Lane reports are not proof.
