# Security

Scope: a public, unauthenticated inference demo (no user accounts, no stored user data).
This file lists the requirements every change must keep, the controls that exist today, and
known gaps. It doesn't claim hardening that isn't implemented.

## Requirements for every change

- **Secrets:** never commit real values. Tokens live in GitHub Actions secrets (`HF_TOKEN`) or HF Space variables. `.env` is gitignored and `.env.example` holds placeholders only. Nothing secret goes in `VITE_*` (it ships to the browser).
- **Input validation at the boundary:** content-type allow-list, size limit, empty check, and safe decode mapped to 4xx. Every request/response body goes through a Pydantic schema.
- **CORS:** explicit origin list from config/env, `allow_credentials=False`, methods limited to GET/POST/OPTIONS. Never `*`.
- **Logging:** structured, with a request ID. Never log image bytes, tokens, or env values.
- **CI:** workflows keep `permissions: contents: read`. Secrets are referenced only through `${{ secrets.* }}` and never echoed.
- **Dependencies:** pinned. New dependencies need a stated reason.

## Controls in place

| Control | Where |
|---|---|
| Upload allow-list (JPEG/PNG/WebP/BMP) → 415 | `backend/app/utils/image.py`, `api/routes.py` |
| Empty → 400, oversize (`serve.max_upload_bytes`, 10 MB) → 413, undecodable → 422 | `api/routes.py` |
| Client-side type/size validation (mirrors backend) | `frontend/src/components/UploadZone.jsx` |
| Explicit CORS allow-list from config / `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` | `backend/app/main.py`, `configs/base.yaml` |
| Request-ID correlated structured logs | `backend/app/core/logging.py` |
| Non-root container (UID 1000), minimal slim image, HEALTHCHECK | `Dockerfile` |
| gitleaks, `detect-private-key`, large-file guard (pre-commit, local only) | `.pre-commit-config.yaml` |
| Least-privilege CI permissions, secret-presence guard in deploy | `.github/workflows/*.yml` |
| Research-artefact integrity (SHA-256 notebook lock) | `.paper-notebook.sha256`, `ci.yml` |

## Known gaps (documented, not yet addressed)

| Gap | Risk | Suggested fix |
|---|---|---|
| `/v1/captions` reads the full upload into memory before the size check | memory pressure from very large bodies on a small Space | bound the read / check `Content-Length` (TASK-006) |
| No rate limiting | abuse can starve the single worker | platform-level limits or a lightweight limiter, if abuse appears |
| No security headers (CSP, HSTS, X-Content-Type-Options) | low for a JSON API; relevant for the SPA host | configure on Vercel (`vercel.json` headers) |
| gitleaks only runs in pre-commit, not CI | a commit made without hooks isn't scanned | add a gitleaks CI job |
| No dependency or container vulnerability scanning | stale CVEs go unnoticed | Dependabot and/or `pip-audit` + `npm audit` in CI |
| No authentication | by design (public demo) | revisit only if paid or expensive models are served |

## Reporting

Report suspected vulnerabilities privately to the maintainer (see README § License & Contact) rather
than opening a public issue.
