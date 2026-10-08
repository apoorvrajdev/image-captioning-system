# CI / CD

GitHub Actions runs three workflows (`ci.yml`, `deploy-backend.yml`, and the commit-message policy gate `no-ai-attribution.yml`) out of [`.github/workflows/`](../.github/workflows/).

## Platform

Pinned explicitly ([ADR-024](DECISIONS.md)), so a GitHub default change can't move CI silently:

- Runner: every job runs on `ubuntu-24.04`. No job uses `ubuntu-latest`.
- Actions: major tags that run on the Node 24 runtime: `actions/checkout@v7`, `actions/setup-python@v7`,
  `actions/setup-node@v7`, `actions/cache@v6`, `actions/upload-artifact@v7`.
- Node: 24 (LTS) for the `frontend` job.
- Python: 3.11 for the quality, freeze and pre-commit jobs, and 3.10 + 3.11 for pytest. 3.10 is past its upstream end
  of life (2026-10-01) and stays for the reasons in ADR-024.

Changing any of these is a reviewed edit to the workflows and to this section.

## `ci.yml` — quality + tests

Triggered on every push and pull request to `main`. Five parallel jobs:

| Job | What it runs | Why |
|---|---|---|
| `python-quality` | `ruff check`, `ruff format --check`, `mypy` (config in `pyproject.toml`, `strict = false`) on `src/captioning`, `backend/app`, `scripts` | Catch style + typing regressions before they land |
| `python-tests` | `pytest` matrix on Python **3.10 / 3.11**, then the 4-stage notebook parity audit (`python -m scripts.notebook_module_audit`) | Confirm the package keeps working on every supported interpreter and still matches the notebook |
| `notebook-freeze` | `make freeze-paper-notebook` (SHA-256 check) | Fail if the IEEE notebook is mutated — it is the canonical research artefact |
| `pre-commit` | `pre-commit run --all-files` with the repo's pinned hooks (`SKIP=mypy`, which `python-quality` covers) | Enforce the same hygiene, nbstripout, prettier, and secret-scan hooks as local commits, including commits made without hooks installed |
| `frontend` | `npm install`, `npm run lint`, `npm run build`, then Playwright Chromium (`npx playwright install --with-deps --only-shell chromium`) and `npm run test:e2e` on Node 24; traces uploaded as `playwright-test-results` on failure | Catch ESLint + Vite build regressions, and break the caption flow or Phase 3 dashboard in a real browser against the production bundle with a mocked API (ADR-023) |

Caching:
- pip via `actions/setup-python` (key derived from `requirements*.txt` + `pyproject.toml`)
- npm via `actions/setup-node` (key derived from `frontend/package-lock.json`)

Concurrency: stacked runs on the same ref cancel each other so only the
newest commit's CI completes.

## `deploy-backend.yml` — deploy the tested commit to the HF Space

The Space is a deployment target that receives a generated commit, not a
mirror of GitHub history ([ADR-017](DECISIONS.md)).

Triggered by:
- `workflow_run` on `CI` completion, only when conclusion is `success` and
  branch is `main` (so a failing CI never deploys)
- `workflow_dispatch` from the Actions tab, on `main` only, to redeploy the tip of `main`.
  Allowed only if that exact commit has a completed, successful CI run on `main`

The job:
0. Manual runs only, before anything is checked out: queries the GitHub API
   (`/actions/workflows/ci.yml/runs?head_sha=<sha>`, with the built-in `github.token`
   and `actions: read`). It refuses unless a run matches the exact SHA, branch `main`,
   and `completed`/`success`. A success for another commit or an API error refuses too
1. Checks out the exact commit CI tested (`github.event.workflow_run.head_sha`,
   or `github.sha` for manual runs) with full history. The token isn't kept in
   `.git/config` (`persist-credentials: false`), because nothing pushes to GitHub
2. Skips the deploy if `main` has already moved past that commit (the newer
   commit's own CI run deploys it), so a slow older run can't roll the Space back
3. Decides whether the image changed (`python3 -m scripts.deploy_scope decide`,
   [ADR-027](DECISIONS.md)); see [When the Space is rebuilt](#when-the-space-is-rebuilt).
   A skip ends the run green with a "Space deploy skipped" notice and a job summary
   giving the reason. Every later step runs only when this step outputs `deploy=true`
4. Builds a deployment commit on top of it that prepends the Space's YAML
   config header (`sdk: docker`, `app_port: 7860`, …) to `README.md`. GitHub's
   README has no header, because GitHub renders it as a table
5. Force-pushes that commit to the Space with the `HF_TOKEN` secret
6. Polls the HF API (`/api/spaces/<id>` for the repo head, `/api/spaces/<id>/runtime`
   for the stage) until a rebuild of the new commit reaches `RUNNING`. Fails on
   `CONFIG_ERROR`, `BUILD_ERROR`, `RUNTIME_ERROR` and other error stages
7. Polls `https://<space-domain>/healthz` until it reports `model_loaded: true`
8. Records the deploy (`python3 -m scripts.deploy_scope record`): a GitHub deployment
   of the tested commit in the `huggingface-space` environment, with the Space's
   commit in its payload and a `success` status. This is the next run's baseline.
   A run that fails or is cancelled before this step records nothing

Timeouts: 10 min for a rebuild to start, 30 min to reach `RUNNING`, 10 min for
`/healthz`, 50 min for the whole job.

### When the Space is rebuilt

A run rebuilds the Space only if an **image input** differs between the last
successfully deployed commit and the commit being deployed. The image inputs are the
Dockerfile's `COPY` sources (`requirements.txt`, `pyproject.toml`, `README.md`, `src/`,
`backend/`, `configs/`, `models/`), plus `Dockerfile`, `.dockerignore`, `.gitattributes`,
this workflow and `scripts/deploy_scope.py`. Anything else (`docs/`, `tests/`,
`frontend/`, `results/`, `notebooks/`, `ci.yml`, other scripts) doesn't rebuild it.
`test_deploy_scope.py` fails if a Dockerfile `COPY` source is missing from the list.

- **Baseline.** The newest `huggingface-space` deployment that this workflow recorded,
  shown under the repository's Environments. It's never the parent commit, so an image
  change whose deploy was skipped, superseded, cancelled or failed is still in the
  range the next run compares.
- **The Space must still be on it.** A skip also needs the Space's repository head
  (public HF API) to equal the commit that deploy pushed. A deploy that pushed but then
  failed its gate, or a push made by hand, makes the next run deploy, even if the change
  was reverted.
- **Unknown means deploy.** No record yet, a record without `success`, an API error, a
  baseline missing from the checkout, or a failed diff all deploy.
- **`README.md` redeploys.** It's in the image (`pyproject.toml` `readme`), and its
  deployed copy, with the config header, is the Space's card.
- **Forcing a rebuild.** Run the workflow manually (`workflow_dispatch`). A manual run
  always deploys once its CI check passes. Use it after a Space breaks without a push,
  or to pick up a new base image.

See
[`PHASE_2C_DEPLOYMENT_RUNBOOK.md`](PHASE_2C_DEPLOYMENT_RUNBOOK.md) for the
end-to-end deployment topology and smoke tests.

## Required secrets

- `HF_TOKEN` — HuggingFace personal access token, **Write** scope. Used only
  by `deploy-backend.yml`, to push to the Space remote and to read the Space's
  status from the HF API. Never sent to the app itself.

No other secret is needed. `deploy-backend.yml` also uses the built-in `github.token`,
with `actions: read` for the manual-run CI check and `deployments: write` to read and
write the deploy record ([ADR-027](DECISIONS.md)).

Set under repo Settings → Secrets and variables → Actions → New repository
secret.

## Local equivalents

Everything CI does is reproducible locally:

```bash
make lint            # ruff check (CI also runs: ruff format --check src/captioning backend scripts tests)
make typecheck       # mypy (pyproject config)
make test            # pytest (single Python version)
python -m scripts.notebook_module_audit   # 4-stage notebook parity audit
make freeze-paper-notebook   # SHA-256 freeze check
SKIP=mypy pre-commit run --all-files   # same hooks as the pre-commit job

cd frontend                        # Node 24, as in CI
npm ci && npm run lint && npm run build
npx playwright install chromium   # once per machine
npm run test:e2e                  # builds, serves with vite preview, runs e2e/ on Chromium
```
