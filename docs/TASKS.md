# Tasks

Executable backlog. The phase roadmap and history live in the README ([Roadmap](../README.md#-roadmap));
this file breaks the *next* work into tasks small enough to ship and review one at a time.

**Task lifecycle:** acceptance criteria → implementation → tests → verification (see
[`TEST_PLAN.md`](TEST_PLAN.md)) → review → commit → status updated here + [`MEMORY.md`](MEMORY.md).

**Format:**

```
### TASK-NNN — <imperative title>            [status: todo | in-progress | done | blocked]
Area: ml-core | inference-api | frontend | evaluation | deployment | docs
Goal: one sentence.
Acceptance criteria:
- GIVEN … WHEN … THEN …
Verification: exact commands.
```

Rule: no task spans a whole phase. If a task needs more than ~one reviewable change set, split it.

---

## Maintenance (ready)

### TASK-001 — Record the engineering-workflow bootstrap            [status: done]
Area: docs · deployment
Goal: living docs, CI parity-audit step, LF pin for the frozen notebook, CI.md drift fixes.

### TASK-002 — Decide whether to track the agent context directory            [status: done]
Area: deployment
Goal: `.claude/` is gitignored, so the repo map, skills, and lane definitions exist only on the
development machine. Decide: keep local-only, or narrow the ignore to `settings.local.json` and generated index files.
Acceptance criteria: decision recorded in `DECISIONS.md`; `.gitignore` matches it.
Outcome: tracked, with guardrails (ADR-015). Local settings and the generated index stay ignored.

### TASK-003 — Fix README drift against config and CI            [status: done]
Area: docs
Goal: README reflects `pyproject.toml` (mypy not strict), `ci.yml` (3.10/3.11 matrix + parity audit), and the
shipped checkpoint (remove "bootstrap weights" / "pending re-training" wording).
Acceptance criteria: every tool claim in README § Testing and Tech Stack matches a config file.

### TASK-004 — Reconcile model-version labelling            [status: todo] (needs owner input)
Area: deployment · docs
Goal: one consistent version for the served checkpoint across README, `BackendSettings.model_version`
default, `.env.example`, and the Space's `BACKEND_WEIGHTS_HUB_REVISION` / `BACKEND_MODEL_VERSION`.
Acceptance criteria: `/healthz.model_version` equals the Hub tag actually served; README states the same tag.

### TASK-005 — Repair stale Makefile targets            [status: todo]
Area: deployment
Goal: `docker-build*` use the root `Dockerfile`; remove or fix `docker-up/down` (no compose file) and `eval` (missing required `--weights`/`--tokenizer-dir`).
Verification: `make -n docker-build eval` shows valid commands (dry run).

### TASK-006 — Bound upload reads in `/v1/captions`            [status: todo]
Area: inference-api
Goal: reject oversize uploads without reading the whole body into memory (check `Content-Length`
and/or read at most `max_upload_bytes + 1`).
Acceptance criteria: GIVEN a body over the limit THEN 413 and at most `limit+1` bytes read; existing 413 test still passes; new test added.

### TASK-007 — Committed browser E2E for the caption flow            [status: blocked] (awaiting approval to install)
Area: frontend · deployment
Goal: promote the manual browser check in `TEST_PLAN.md` into a committed Playwright spec that mocks
`/healthz` and `/v1/captions` with `page.route` (no backend, no TensorFlow), run in the CI `frontend` job.
Acceptance criteria: GIVEN a mocked healthy API WHEN a PNG is uploaded and Generate clicked THEN the caption card
renders; GIVEN a disallowed file THEN an inline error shows and no request is sent; GIVEN the API is unreachable THEN
"Cannot reach backend" shows; zero console errors in each case.
Needs: `@playwright/test` devDependency + Chromium download (local and CI).

### TASK-008 — Restore backend auto-deploy to the HF Space            [status: in-progress] (fix implemented, awaiting review + first deploy)
Area: deployment
Goal: no `deploy-backend.yml` run has succeeded since mid-June. The 2026-06-16/17 runs failed with HF HTTP 429
rate limits. The 2026-09-24 run was rejected as non-fast-forward because the Space's git history has commits that
aren't on GitHub `main`. Inspect the Space history, then either merge those commits into GitHub `main` or decide
that GitHub is the source of truth and force-push once.
Acceptance criteria: a green `deploy-backend.yml` run; Space `/healthz` reports `model_loaded: true`.
Findings: two faults.
- The Space's only extra commit, `302e907`, has a tree identical to GitHub's `64f80e8` (history rewritten after deploy).
- The Space has been in `CONFIG_ERROR` ("Missing configuration in README") because `befac80` removed the README
  YAML header.
Fix (ADR-017): `deploy-backend.yml` deploys the tested SHA, skips superseded commits, adds the header to a deploy
commit, force-pushes, and gates on HF runtime `RUNNING` + `/healthz` `model_loaded: true`.
Remaining: review and commit, then one `workflow_dispatch` run on `main`; then remove the status note in the
runbook § 1.

---

## Phase 3 — Multimodal baselines (next phase, NOT started)

Decompose each item into TASK-NNN entries (with acceptance criteria) **before** implementation.
Constraints already fixed: baselines live in the optional `[hf]` extra (`transformers==4.41.2`,
`torch==2.3.0`) and must not unpin the research pipeline (`tensorflow-cpu==2.15.0`). Every baseline
writes the standard `results/<run_id>/` artefact contract on the **same slice, reference count and
tokenisation** as the existing runs.

- [ ] **3A** — Side-by-side comparison harness: CNN+Transformer vs BLIP-base vs ViT-GPT2 vs GIT-base-coco
- [ ] **3B** — Per-model BLEU / CIDEr / METEOR / ROUGE-L on a shared COCO slice with deterministic tokenisation
- [ ] **3C** — Per-model latency benchmarking (single-image, batch, CPU vs GPU)
- [ ] **3D** — Comparison-result dashboard exposed through the existing SPA
