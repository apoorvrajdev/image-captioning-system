---
name: deployment
description: Acceptance criteria and definition of done for CI/CD, the Docker image, HuggingFace Space backend, Vercel frontend, weights versioning on HF Hub, and dependency pins. Use whenever changing Dockerfile, .dockerignore, .github/workflows/**, requirements*.txt, pyproject.toml dependencies, .env.example, or deploy runbooks.
---

# Deployment & CI — acceptance criteria

## Scope
`Dockerfile`, `.dockerignore`, `.github/workflows/{ci,deploy-backend,no-ai-attribution}.yml`,
`requirements*.txt`, `pyproject.toml` (deps + tool config), `.pre-commit-config.yaml`, `.env.example`,
`frontend/.env.example`, `docs/CI.md`, `docs/PHASE_2C_DEPLOYMENT_RUNBOOK.md`, `Makefile`.

## Topology (current)
GitHub `main` → `ci.yml` (ruff+mypy · pytest 3.10/3.11 + parity audit · notebook freeze · frontend lint+build+Playwright E2E)
→ on green `deploy-backend.yml` pushes to HF Space `apoorvrajdev/image-captioning-api` (Docker SDK, cpu-basic, port 7860, 1 worker)
→ lifespan pulls weights from HF Hub `apoorvrajdev/captioning-inceptionv3-transformer` at a pinned tag.
Vercel's Git integration builds `frontend/` with `VITE_API_BASE`. Prod CORS comes from the `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` Space variable.

## Expected behaviour
- A red CI never deploys. Deploy uses only the `HF_TOKEN` secret and fails loudly if it's unset.
- Container runs as UID 1000, `HEALTHCHECK` on `/healthz`, no weights baked into the Space git tree.
- Local DoD commands ≡ CI jobs (see CLAUDE.md Commands). A new gate is added to both.

## Definition of done — ALL must pass
- [ ] Workflow YAML parses (`python -c "import yaml,sys;yaml.safe_load(open(sys.argv[1], encoding='utf-8'))" <file>`), and no gate was removed or weakened.
- [ ] Every job names the pinned runner (`ubuntu-24.04`), never `ubuntu-latest`. Action majors run on Node 24, and the `frontend` job uses Node 24. Changing any of these updates `docs/CI.md` § Platform (ADR-024).
- [ ] `permissions: contents: read` kept. Secrets only via `${{ secrets.* }}`, never echoed.
- [ ] Dependency changes keep `tensorflow-cpu==2.15.0` + `numpy<2`. Runtime deps stay in `requirements.txt` (the Docker layer) *and* `pyproject.toml`.
- [ ] New env var ⇒ `.env.example` + runbook updated. No real values committed.
- [ ] `docs/CI.md` matches the workflows after the change.
- [ ] Production actions (Space variables, HF Hub uploads/tags, Vercel settings, pushes) are **prepared as instructions for the user**, never executed.

## Verification commands
```bash
.venv/Scripts/python.exe -c "import yaml;[yaml.safe_load(open(f, encoding='utf-8')) for f in ['.github/workflows/ci.yml','.github/workflows/deploy-backend.yml','.github/workflows/no-ai-attribution.yml']];print('ok')"
grep -rn "ubuntu-latest" .github/workflows   # must print nothing
.venv/Scripts/pytest.exe tests backend/app/tests -q
docker build -t captioning-backend:local .   # only if Docker is available; otherwise report "not run"
```

## Non-negotiables
- Never push, deploy, re-tag, or delete Hub revisions. Weights tags are immutable. New checkpoint → new tag.
- Weights promotion or rollback instructions must set `BACKEND_WEIGHTS_HUB_REVISION` **and** `BACKEND_MODEL_VERSION` to the same tag, and are done only when `/healthz` shows `model_loaded: true` with that `model_version` (ADR-018).
- Never add paid services or authenticated external integrations without the user's explicit approval.
