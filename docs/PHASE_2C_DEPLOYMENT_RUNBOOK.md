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
  ├── Model repo: apoorvrajdev/captioning-inceptionv3-transformer  (weights + vocab, tag v1.0.0)
  └── Space:     apoorvrajdev/image-captioning-api                  (Docker SDK, cpu-basic, port 7860)
```

The Space pulls weights from the model repo at lifespan startup via
`huggingface_hub.snapshot_download`, so the Space's git tree never contains
`model.h5` — only the code that knows how to fetch it.

---

## 1. Live URLs

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

To promote a new checkpoint after this: bump the Space variable
`BACKEND_WEIGHTS_HUB_REVISION` from `v1.0.0` to the new tag (e.g. `v2.0.0`)
and the Space restarts with the new weights. No code change required.

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
   | `BACKEND_WEIGHTS_HUB_REVISION` | `v1.0.0` |
   | `BACKEND_WEIGHTS_HUB_FILENAME` | `model.h5` |
   | `BACKEND_WARMUP` | `true` |
   | `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` | `["https://image-captioning-system.vercel.app","http://localhost:5173","http://localhost:5174","http://127.0.0.1:5173","http://127.0.0.1:5174"]` |

3. Add a `space` git remote and push `main`:
   ```bash
   git remote add space https://huggingface.co/spaces/apoorvrajdev/image-captioning-api
   git push space main
   ```
4. Watch the Space's **Logs** tab. First build takes ~8–12 min (Docker base
   pull, `apt-get`, `pip install -r requirements.txt` with TensorFlow,
   weight download via `snapshot_download`, predictor warmup).
5. When the badge in the Space header turns **Running**, verify:
   ```bash
   curl https://apoorvrajdev-image-captioning-api.hf.space/healthz
   # {"status":"ok","model_loaded":true,"model_version":"v1.0.0",...}
   ```

The README YAML frontmatter (`title`, `emoji`, `sdk: docker`, `app_port: 7860`,
etc.) is what tells the Space how to build. It must remain at the literal top
of `README.md`. GitHub auto-hides the frontmatter when rendering the README, so
the same file serves both audiences.

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
  successful `CI` run on `main`. Pushes `HEAD:main` to the Space remote using
  the `HF_TOKEN` repository secret. Also supports `workflow_dispatch` for
  manual redeploys.

### Required GitHub secret

`HF_TOKEN` (repo Settings → Secrets and variables → Actions → New repository
secret). Scope: **Write**. Used only for `git push` to the Space remote.

---

## 8. End-to-end smoke test

After any redeploy, verify in this order:

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
- **Caption quality is gibberish** by design at `v1.0.0`. The shipped weights
  are dev scaffolds from `scripts/bootstrap_dev_artifacts.py`. A real trained
  checkpoint will be uploaded as `v2.0.0` and promoted via the Space variable
  bump described in §3.

---

## 10. Rollback

- **Bad code on the Space**: `git push space <known-good-sha>:main --force`
  (from a local checkout). Space rebuilds from that SHA.
- **Bad weights on the Hub**: bump the Space's
  `BACKEND_WEIGHTS_HUB_REVISION` back to the previous tag (e.g. `v1.0.0`)
  and save. Space restarts in ~30 s with the previous weights.
- **Bad frontend on Vercel**: dashboard → Deployments → previous green
  deployment → "Promote to Production" (one click, no rebuild).
