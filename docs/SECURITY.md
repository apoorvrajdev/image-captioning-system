# Security

Scope: a public, unauthenticated inference demo (no user accounts, no stored user data).
This file lists the requirements every change must keep, the controls that exist today, and
known gaps. It doesn't claim hardening that isn't implemented.

## Requirements for every change

- **Secrets:** never commit real values. Tokens live in GitHub Actions secrets (`HF_TOKEN`) or HF Space variables. `.env` is gitignored and `.env.example` holds placeholders only. Nothing secret goes in `VITE_*` (it ships to the browser).
- **Input validation at the boundary:** content-type allow-list, size limit, empty check, and safe decode mapped to 4xx. Every request/response body goes through a Pydantic schema.
- **CORS:** explicit origin list from config/env, `allow_credentials=False`, methods limited to GET/POST/OPTIONS. Never `*`.
- **Logging:** structured, with a request ID. Never log image bytes, tokens, or env values.
- **CI:** workflows keep `permissions: contents: read`. `deploy-backend.yml` adds `actions: read` (the manual-run CI check) and `deployments: write` (its deploy record, ADR-027), and its checkout doesn't persist the token. Secrets are referenced only through `${{ secrets.* }}` and never echoed.
- **Dependencies:** pinned. New dependencies need a stated reason. Starlette is pinned explicitly, because FastAPI's own range admits vulnerable releases.

## Controls in place

| Control | Where |
|---|---|
| Upload allow-list (JPEG/PNG/WebP/BMP) → 415 | `backend/app/utils/image.py`, `api/routes.py` |
| Empty → 400, oversize (`serve.max_upload_bytes`, 10 MB) → 413, undecodable → 422 | `api/routes.py` |
| Bounded upload read: the route reads at most `max_upload_bytes + 1` bytes, so an oversize upload is never loaded into memory in full (TASK-006) | `api/routes.py`, `backend/app/tests/test_captions.py` |
| Request-body cap: a body over `max_upload_bytes` + 64 KiB of multipart framing gets the same 413 before the multipart parser buffers it. With `Content-Length`, no byte is read. Without it, reading stops at the cap (TASK-020) | `core/body_limit.py`, `main.py`, `backend/app/tests/test_body_size_limit.py` |
| Serving dependencies with no known advisory in FastAPI, Starlette, `python-multipart`, Pillow or `anyio`, as of the 2026-10-08 audit (§ Dependency audit) | `requirements.txt`, `pyproject.toml` |
| Client-side type/size validation (mirrors backend) | `frontend/src/components/UploadZone.jsx` |
| Explicit CORS allow-list from config / `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` | `backend/app/main.py`, `configs/base.yaml` |
| Request-ID correlated structured logs | `backend/app/core/logging.py` |
| Non-root container (UID 1000), minimal slim image, HEALTHCHECK | `Dockerfile` |
| `detect-private-key`, large-file guard (pre-commit locally **and** the CI `pre-commit` job) | `.pre-commit-config.yaml`, `ci.yml` |
| gitleaks on staged changes (local pre-commit only; its `--staged` mode scans nothing in CI) | `.pre-commit-config.yaml` |
| Least-privilege CI permissions, secret-presence guard in deploy | `.github/workflows/*.yml` |
| Research-artefact integrity (SHA-256 notebook lock) | `.paper-notebook.sha256`, `ci.yml` |
| Development-tooling guardrails: no reads of `.env` files, no edits to the frozen notebook, `models/`, or `results/`, confirmation before commit/push/tag | `.claude/settings.json` |

## Known gaps (documented, not yet addressed)

| Gap | Risk | Suggested fix |
|---|---|---|
| A body up to the cap is still received: without `Content-Length`, up to about 10 MiB can be spooled to a temporary file before the 413. A client that keeps sending after the 413 still uses bandwidth, though uvicorn discards the bytes. The HF Spaces proxy in front of the app can't be configured | bandwidth, and bounded disk use per request on a small Space | rate limiting or a platform-level body cap, if abuse appears |
| No rate limiting | abuse can starve the single worker | platform-level limits or a lightweight limiter, if abuse appears |
| No security headers (CSP, HSTS, X-Content-Type-Options) | low for a JSON API; relevant for the SPA host | configure on Vercel (`vercel.json` headers) |
| No full-history secret scan in CI (the pre-commit job's gitleaks hook is staged-only) | a commit made without hooks isn't scanned | add a `gitleaks detect` CI step |
| No continuous dependency or container vulnerability scanning. The 2026-10-08 audit was a one-off, and three packages still have findings (§ Dependency audit) | new CVEs go unnoticed | `pip-audit` + `npm audit` in CI (TASK-021) |
| No authentication | by design (public demo) | revisit only if paid or expensive models are served |

## Dependency audit

Run on 2026-10-08 (TASK-020, ADR-025): `pip-audit -r requirements.txt` with pip-audit 2.10.1, in its own virtual
environment. It resolves the full transitive set, on Windows with Python 3.10. The image runs Linux with Python 3.11,
and CI scanning on Linux is TASK-021. Before the upgrade it found 89 vulnerabilities in 7 packages. After it, 26 in 3
packages, none of them in FastAPI, Starlette, `python-multipart`, Pillow or `anyio`.

Remediated:

| Package | Old → new | Advisories fixed | Exposure |
|---|---|---|---|
| Starlette | 0.37.2 → 1.3.1, now pinned | 7: CVE-2024-47874 and CVE-2025-54121 (multipart DoS), CVE-2026-54283 (form limits ignored), CVE-2026-48710 and CVE-2026-54282 (Host and authority poisoning of `request.url`), CVE-2026-48817 (`HTTPEndpoint` method dispatch), CVE-2026-48818 (`StaticFiles` UNC paths on Windows) | The multipart DoS on every `/v1/captions` request. The rest are in features the app doesn't use. |
| `python-multipart` | 0.0.9 → 0.0.31 | 8: CVE-2024-53981 (malformed boundary), CVE-2026-40347 (large preamble or epilogue), CVE-2026-42561 (unbounded part headers), CVE-2026-53540 (negative `Content-Length` buffers the body), CVE-2026-53537 (`Content-Disposition` parameter smuggling), CVE-2026-53538 and CVE-2026-53539 (querystring semicolons), CVE-2026-24486 (file write, non-default configuration only) | The parser runs on every upload. |
| Pillow | 10.3.0 → 12.3.0 | 17, fixed between 12.1.1 and 12.3.0, among them CVE-2026-25990 and CVE-2026-42311 (PSD out-of-bounds writes) and CVE-2026-40192 (FITS decompression bomb) | Not on the serving path: uploads decode through `tf.io.decode_image`. It ships in the image, and the `[hf]` baseline adapter uses it. |
| FastAPI | 0.111.0 → 0.133.0 | none of its own | The lowest release whose range admits Starlette 1.x. |
| `anyio` | 4.4.0 → 4.14.2 | 2: CVE-2026-64847 (process-pool workers' stderr), CVE-2026-63374 (IDNA 2003 host-name encoding in `TLSStream`) | Neither is reachable: serving uses only `anyio.to_thread.run_sync`. Upgraded because 4.14.2 also stops an idle worker thread from holding its last work item, which kept a refused upload's temporary file open. |

Remaining, each with its reason:

| Package | Version | Findings | Why it stays | Reachable from serving? |
|---|---|---|---|---|
| `keras` | 2.15.0 | 13: CVE-2024-55459, CVE-2025-9906, CVE-2025-12058, CVE-2025-12060, CVE-2026-1462, CVE-2026-9335, CVE-2026-11816, CVE-2026-12479 to CVE-2026-12482, CVE-2026-12484, CVE-2026-12570 | Fixed only in Keras 3. `tensorflow-cpu==2.15.0` requires Keras 2.15, and TF 2.16+ brings Keras 3, which breaks `TextVectorization` save/load. | No. Each needs an untrusted model file or archive (`load_model`, HDF5 links, `get_file`). Serving loads only its own weights, from the pinned, immutable Hub tag. |
| `protobuf` | 4.25.9 | CVE-2026-0994 | Fixed in 5.29.6 and 6.33.5. TF 2.15 requires protobuf < 5. | No. It needs untrusted JSON parsed with `json_format.ParseDict`, and serving parses none. |
| `click` | 8.1.7 | CVE-2026-7246 | Fixed in 8.3.3. Outside TASK-020's scope. | No. It's command injection through `click.edit()`, which nothing calls. |

TASK-021 decides whether its CI scan blocks on these or starts report-only.

## Reporting

Report suspected vulnerabilities privately to the maintainer (see README § License & Contact) rather
than opening a public issue.
