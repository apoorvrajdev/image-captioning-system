# Phase 2C — Public Deployment Runbook

This runbook captures every step needed to (re)deploy the Image Captioning System
to its public hosts: weights to the HuggingFace Hub, backend to a HuggingFace
Space, frontend to Vercel, and the CI/CD chain wiring it all together. It is
written so a future maintainer (or the author six months from now) can rebuild
the public deployment from a cold start without reading commit history.

## 0. Topology

```
GitHub (apoorvrajdev/image-captioning-system, main)
  ├── Actions: CI → Deploy backend to HuggingFace Space (workflow_run chained)
  └── Vercel Git Integration → image-captioning-system.vercel.app

HuggingFace Hub
  ├── Model repo: apoorvrajdev/captioning-inceptionv3-transformer  (weights + vocab; tag v2.0.0 served, v1.0.0 = dev scaffold)
  └── Space:     apoorvrajdev/image-captioning-api                  (Docker SDK, cpu-basic, port 7860)
```

The Space pulls weights from the model repo at lifespan startup via
`huggingface_hub.snapshot_download`, so the Space's git tree never contains
`model.h5` — only the code that knows how to fetch it.

---

## 1. Live URLs

> **Status (verified 2026-10-03):** the deploy path restored in TASK-008 was verified end to end.
>
> - `deploy-backend.yml` run `37140993110` deployed GitHub `915112b` (CI run `37140358717`) as Space
>   commit `123c5aa`.
> - The Space reached `RUNNING`, and `/healthz` returned HTTP 200 with `model_loaded: true`.
> - `/docs` and `/openapi.json` returned HTTP 200.
> - Served weights verified (TASK-004, 2026-10-03):
>   - The Space variables set `BACKEND_WEIGHTS_HUB_REVISION=v2.0.0` (Hub commit `59d93b4`) and
>     `BACKEND_MODEL_VERSION=v2.0.0`.
>   - `/healthz` reports `model_loaded: true`, `model_version: "v2.0.0"`.

| Component | URL |
|---|---|
| Frontend SPA | `https://image-captioning-system.vercel.app` |
| Backend API | `https://apoorvrajdev-image-captioning-api.hf.space` |
| Backend health | `https://apoorvrajdev-image-captioning-api.hf.space/healthz` |
| Backend docs (Swagger) | `https://apoorvrajdev-image-captioning-api.hf.space/docs` |
| Weights repo | `https://huggingface.co/apoorvrajdev/captioning-inceptionv3-transformer` |
| Space console | `https://huggingface.co/spaces/apoorvrajdev/image-captioning-api` |

---

## 2. Prerequisites

- Local git working tree on `main`, clean
- Python 3.11 venv with `requirements.txt` + `requirements-dev.txt` installed
- A HuggingFace account and a personal access token with **Write** scope
  (Settings → Access Tokens). Used both in the local shell (`huggingface-cli login`)
  and as a GitHub Actions secret named `HF_TOKEN`
- A Vercel account connected to the GitHub repo

---

## 3. Weights upload (WS-B) — only when shipping a new checkpoint

The Space's `BACKEND_WEIGHTS_HUB_REVISION` variable pins which Hub revision
the backend pulls at startup, so weights and code can be versioned
independently.

```bash
# 1. Login (token cached at ~/.cache/huggingface/token)
huggingface-cli login

# 2. Upload the contents of models/vX.Y.Z/ to the Hub repo
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id="apoorvrajdev/captioning-inceptionv3-transformer",
    folder_path="models/v1.0.0",
    path_in_repo=".",
    commit_message="upload v1.0.0 weights + vocab",
)
api.create_tag(
    repo_id="apoorvrajdev/captioning-inceptionv3-transformer",
    tag="v1.0.0",
    tag_message="v1.0.0 dev-scaffold weights",
)
PY

# 3. Verify the snapshot round-trips byte-for-byte
HF_HUB_DISABLE_SYMLINKS=1 python - <<'PY'
import hashlib, pathlib
from huggingface_hub import snapshot_download
local = snapshot_download(
    repo_id="apoorvrajdev/captioning-inceptionv3-transformer",
    revision="v1.0.0",
)
for f in ("model.h5", "vocab.json"):
    src = hashlib.sha256(pathlib.Path("models/v1.0.0", f).read_bytes()).hexdigest()
    dst = hashlib.sha256(pathlib.Path(local, f).read_bytes()).hexdigest()
    assert src == dst, f
    print(f, "OK", src)
PY
```

To promote a new checkpoint after this, set **both** Space variables to the new tag in the same change, e.g.
`BACKEND_WEIGHTS_HUB_REVISION=v2.0.0` and `BACKEND_MODEL_VERSION=v2.0.0`.

- **Why both:** they are independent settings in `BackendSettings` (`backend/app/core/config.py`). The code never
  derives `model_version` from the revision, so bumping only the revision serves new weights under the old label.
  That is exactly the mismatch TASK-004 found.
- **When they take effect:** settings are read once per process, so the new values apply when the Space restarts.
  No code change or deploy is required.
- **Then verify:** `/healthz` must report `model_loaded: true` and a `model_version` equal to the new tag.

---

## 4. Backend Space (WS-C) — one-time setup

1. Create the Space at https://huggingface.co/new-space
   - Owner: `apoorvrajdev` · Name: `image-captioning-api`
   - SDK: **Docker** (blank template) · Hardware: **cpu-basic (free)** · Public
2. In the Space's **Settings → Variables and secrets**, add **Variables**
   (not secrets — these are non-sensitive):

   | Name | Value |
   |---|---|
   | `BACKEND_WEIGHTS_HUB_REPO` | `apoorvrajdev/captioning-inceptionv3-transformer` |
   | `BACKEND_WEIGHTS_HUB_REVISION` | `v2.0.0` |
   | `BACKEND_MODEL_VERSION` | `v2.0.0`. Must always equal `BACKEND_WEIGHTS_HUB_REVISION` (§3) |
   | `BACKEND_WEIGHTS_HUB_FILENAME` | `model.h5` |
   | `BACKEND_WARMUP` | `true` |
   | `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` | `["https://image-captioning-system.vercel.app","http://localhost:5173","http://localhost:5174","http://127.0.0.1:5173","http://127.0.0.1:5174"]` |

3. Deploy by running **Deploy backend to HuggingFace Space** (`deploy-backend.yml`)
   from the Actions tab on `main` (`workflow_dispatch`). Don't push GitHub commits
   to the Space by hand: they lack the Space config header (see the note below).
4. Watch the Space's **Logs** tab. First build takes ~8–12 min (Docker base
   pull, `apt-get`, `pip install -r requirements.txt` with TensorFlow,
   weight download via `snapshot_download`, predictor warmup).
5. When the badge in the Space header turns **Running**, verify:
   ```bash
   curl https://apoorvrajdev-image-captioning-api.hf.space/healthz
   # {"status":"ok","model_loaded":true,"model_version":"v2.0.0",...}
   ```

The README YAML frontmatter (`title`, `emoji`, `sdk: docker`, `app_port: 7860`,
etc.) is what tells the Space how to build, and it must be at the literal top of
the Space's `README.md`. GitHub renders that block as a table, so it was removed
from the GitHub README (`befac80`). Without it the Space reports `CONFIG_ERROR`
("Missing configuration in README"). `deploy-backend.yml` now prepends the
original header to the deployment copy only (ADR-017).

---

## 5. Frontend (WS-E) — Vercel one-time setup

1. https://vercel.com/new → import `apoorvrajdev/image-captioning-system`
2. Configure:
   - Framework Preset: **Vite** (auto-detected from `frontend/package.json`)
   - Root Directory: `frontend`
   - Build / Output / Install commands: leave on defaults
3. Environment variable (Production + Preview):
   - `VITE_API_BASE` = `https://apoorvrajdev-image-captioning-api.hf.space`
4. Deploy. First build is ~90 sec. Production alias becomes
   `https://image-captioning-system.vercel.app`.

After the initial import every push to `main` triggers an automatic Vercel
build via the GitHub integration — no separate GitHub Action required.

---

## 6. CORS (WS-F)

`backend/app/main.py` registers `CORSMiddleware` with
`config.serve.cors_allowed_origins`. The defaults in
[`configs/base.yaml`](../configs/base.yaml) cover localhost dev. Production
origins are added via the Space's `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS`
variable (JSON array, see §4). To add a new origin (e.g. a custom domain):
edit that variable, save, and the Space restarts (~30 sec, no rebuild).

---

## 7. CI/CD (WS-G)

Two workflows under [`.github/workflows/`](../.github/workflows/):

- **`ci.yml`** — runs on every push and PR to `main`:
  - `python-quality`: ruff lint + format, mypy strict
  - `python-tests`: pytest matrix on 3.10 / 3.11 / 3.12
  - `notebook-freeze`: SHA-256 freeze check on the IEEE notebook
  - `frontend`: `npm ci && npm run lint && npm run build`
- **`deploy-backend.yml`** — chained via `workflow_run`, runs only after a
  successful `CI` run on `main`. It deploys the exact commit CI tested and skips
  commits `main` has moved past. It builds a single-commit snapshot of that
  commit's build context, with the Space config header and no GitHub history
  (ADR-030), force-pushes it to the Space with the `HF_TOKEN` repository secret, and
  passes only once the HF API reports `RUNNING`, `/healthz` reports
  `model_loaded: true`, and one real caption request succeeds (§ 8, ADR-028). `workflow_dispatch` (on `main`) redeploys the tip of
  `main`, but only if that exact commit has a successful CI run. Details: [`CI.md`](CI.md).
  - **It rebuilds the Space only when the image can change** (ADR-027). A commit that
    changes no image input (`Dockerfile` `COPY` sources, `Dockerfile`, `.dockerignore`,
    `.gitattributes`, the workflow, `scripts/deploy_scope.py`, `scripts/smoke_caption.py`) since the last successful
    deploy ends green with a "Space deploy skipped" notice. `README.md` counts as an image
    input. The last successful deploy is the newest `huggingface-space` deployment under
    the repository's Environments, written only after the health gate passes.
  - **After a failed deploy** nothing is recorded, so the next green commit deploys
    again, whatever it changed. That includes a docs-only commit, or a revert of the
    failed change.
  - **To force a rebuild**, for example after the Space broke without a push or to pick
    up a new base image, run the workflow manually.

### Required GitHub secret

`HF_TOKEN` (repo Settings → Secrets and variables → Actions → New repository
secret). Scope: **Write**. Used only for `git push` to the Space remote.

---

## 8. End-to-end smoke test

Every deploy already runs the backend half automatically (`scripts/smoke_caption.py`,
ADR-028). After the health gate it captions a generated PNG through the live
`POST /v1/captions` with the Vercel `Origin`. It checks HTTP 200, a non-empty caption,
the `model_version` that `/healthz` reports, the echoed `x-request-id` and the
`Access-Control-Allow-Origin` header, and a failure fails the deploy. To run the same
check by hand:

```bash
python -m scripts.smoke_caption --url https://apoorvrajdev-image-captioning-api.hf.space \
  --origin https://image-captioning-system.vercel.app
```

The HF Spaces proxy answers CORS itself and reflects any `Origin`, so the header shows
the SPA can read responses, not that the app's allow-list is enforced (`SECURITY.md`
§ Known gaps).

After a manual change (Space variables, weights), verify in this order:

```bash
# 1. Backend liveness + readiness
curl https://apoorvrajdev-image-captioning-api.hf.space/healthz

# 2. Backend caption round-trip (replace path with any local JPG/PNG)
curl -X POST https://apoorvrajdev-image-captioning-api.hf.space/v1/captions \
  -F "image=@assets/sample.jpg"

# 3. Frontend loads + status badge flips to green
open https://image-captioning-system.vercel.app  # macOS
# start https://image-captioning-system.vercel.app  # Windows

# 4. Frontend ↔ backend integration (in the browser)
#    Upload an image → expect a 200 caption response from /v1/captions
#    DevTools → Network → check no CORS errors
```

---

## 9. Known operational quirks

- **Status badge briefly flips to "offline"** while a `/v1/captions` request is
  in flight on the single uvicorn worker. The `/healthz` poll queues behind
  inference and the frontend's 3 s timeout expires. The next 10 s poll
  recovers. Cosmetic only — backend never actually goes down.
- **First request after Space idle is slow** (~5–10 s extra). HF Spaces
  sleep idle containers; the next call wakes the container, which then runs
  the lifespan startup (snapshot_download cache hit + predictor rewarmup).
- **Hub tag `v1.0.0` is a dev scaffold.** Its weights come from
  `scripts/bootstrap_dev_artifacts.py` and produce gibberish captions by design.
  Production serves the COCO-trained checkpoint at tag `v2.0.0`, promoted via
  the §3 variable change and verified 2026-10-03 (TASK-004).

---

## 10. Rollback

- **Bad code on the Space**: `git revert` the bad commit on GitHub `main` and
  push. CI runs and `deploy-backend.yml` redeploys the reverted tree with the
  config header and health checks. Don't force-push a raw GitHub SHA to the
  Space: it has no config header, so the Space would go to `CONFIG_ERROR`.
- **Bad weights on the Hub**: set the Space's
  `BACKEND_WEIGHTS_HUB_REVISION` **and** `BACKEND_MODEL_VERSION` back to the
  same previous tag (§3) and save. Space restarts in ~30 s with the previous
  weights. Verify that `/healthz` reports that tag.
- **Bad frontend on Vercel**: dashboard → Deployments → previous green
  deployment → "Promote to Production" (one click, no rebuild).
