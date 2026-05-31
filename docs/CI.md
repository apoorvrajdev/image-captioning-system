# CI / CD

GitHub Actions runs two workflows out of [`.github/workflows/`](../.github/workflows/).

## `ci.yml` — quality + tests

Triggered on every push and pull request to `main`. Four parallel jobs:

| Job | What it runs | Why |
|---|---|---|
| `python-quality` | `ruff check`, `ruff format --check`, `mypy --strict` on `src/` and `backend/` | Catch style + typing regressions before they land |
| `python-tests` | `pytest` matrix on Python **3.10 / 3.11 / 3.12** | Confirm the package keeps working on every supported interpreter |
| `notebook-freeze` | `make freeze-paper-notebook` (SHA-256 check) | Fail if the IEEE notebook is mutated — it is the canonical research artefact |
| `frontend` | `npm ci`, `npm run lint`, `npm run build` on Node 20 | Catch ESLint + Vite build regressions in the SPA |

Caching:
- pip via `actions/setup-python` (key derived from `requirements*.txt` + `pyproject.toml`)
- npm via `actions/setup-node` (key derived from `frontend/package-lock.json`)

Concurrency: stacked runs on the same ref cancel each other so only the
newest commit's CI completes.

## `deploy-backend.yml` — push main to the HF Space

Triggered by:
- `workflow_run` on `CI` completion, only when conclusion is `success` and
  branch is `main` (so a failing CI never deploys)
- `workflow_dispatch` for manual redeploys from the Actions tab

The job:
1. Checks out the full git history (HF Space remote needs the parent
   commits to fast-forward)
2. Sets a fixed git identity (`apoorvrajdev <apoorvrajmgr@gmail.com>`)
3. Adds a `space` remote authenticated with the `HF_TOKEN` repository secret
4. Pushes `HEAD:main` to the Space

The Space then rebuilds its Docker image. See
[`PHASE_2C_DEPLOYMENT_RUNBOOK.md`](PHASE_2C_DEPLOYMENT_RUNBOOK.md) for the
end-to-end deployment topology and smoke tests.

## Required secrets

- `HF_TOKEN` — HuggingFace personal access token, **Write** scope. Used only
  by `deploy-backend.yml` to push to the Space remote.

Set under repo Settings → Secrets and variables → Actions → New repository
secret.

## Local equivalents

Everything CI does is reproducible locally:

```bash
make lint            # ruff check + format --check
make typecheck       # mypy strict
make test            # pytest (single Python version)
make freeze-paper-notebook   # SHA-256 freeze check

cd frontend
npm ci && npm run lint && npm run build
```
