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
| CI dependency audit: `pip-audit` of `requirements.txt` against the reviewed baseline, and `npm audit --omit=dev`. Both block CI, and so the deploy (TASK-021, § CI scanning policy) | `ci.yml` (`security`, `frontend`), `scripts/check_pip_audit.py`, `.github/pip-audit-baseline.txt` |
| CI secret scan: gitleaks over the full committed history, with values redacted in the log (TASK-021) | `ci.yml` (`security`) |
| Least-privilege CI permissions, secret-presence guard in deploy | `.github/workflows/*.yml` |
| Research-artefact integrity (SHA-256 notebook lock) | `.paper-notebook.sha256`, `ci.yml` |
| Development-tooling guardrails: no reads of `.env` files, no edits to the frozen notebook, `models/`, or `results/`, confirmation before commit/push/tag | `.claude/settings.json` |

## Known gaps (documented, not yet addressed)

| Gap | Risk | Suggested fix |
|---|---|---|
| A body up to the cap is still received: without `Content-Length`, up to about 10 MiB can be spooled to a temporary file before the 413. A client that keeps sending after the 413 still uses bandwidth, though uvicorn discards the bytes. The HF Spaces proxy in front of the app can't be configured | bandwidth, and bounded disk use per request on a small Space | rate limiting or a platform-level body cap, if abuse appears |
| No rate limiting | abuse can starve the single worker | platform-level limits or a lightweight limiter, if abuse appears |
| No security headers (CSP, HSTS, X-Content-Type-Options) | low for a JSON API; relevant for the SPA host | configure on Vercel (`vercel.json` headers) |
| CI audits only what ships: not the dev and eval Python requirements (43 findings on 2026-10-08: 42 in `nltk` 3.8.1, 1 in `pytest` 8.2.2), and not dev npm dependencies (8 advisories in build tooling such as `vite` and `postcss`) | tooling that runs locally, in CI or on Kaggle, on trusted inputs | upgrade or gate them in a separate task |
| No scheduled scan and no container image scan | an advisory published while no one pushes isn't seen until the next push, and OS packages in the image aren't audited | a scheduled CI run, an image scanner |
| Production CORS isn't enforced by the app. The HF Spaces proxy answers CORS itself and reflects any `Origin`, preflights included. The Space's `CAPTIONING__SERVE__CORS_ALLOWED_ORIGINS` doesn't reach the app either: `load_config` passes `base.yaml` as constructor arguments, which outrank environment variables, so the app allows only the localhost origins. Found in TASK-023 (ADR-028). | low while the API is public and sends no credentials (`allow_credentials=False`), but the SPA works only through the proxy's reflection | make environment variables outrank the YAML in `load_config` (this changes every `CAPTIONING__*` override), then verify the app's allow-list locally; the proxy's reflection can't be configured |
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

`click` 8.1.7 was also on this list, for CVE-2026-7246 (PYSEC-2026-2132, command injection through `click.edit()`,
which nothing calls). OSV withdrew PYSEC-2026-2132 on 2026-10-07 and the CVE is now marked disputed, so pip-audit no
longer reports it. Its baseline entry went stale, and the first CI run of the gate failed on it, so it was removed in
TASK-021.

CI enforces this list through the pip-audit baseline (§ CI scanning policy).

## CI scanning policy

Three scans run on every push and pull request to `main`, and all three block (TASK-021, ADR-026). CI must be green
before anything deploys, so a failing scan also stops the backend deploy.

| Scan | Job | What it checks | Fails CI when |
|---|---|---|---|
| pip-audit 2.10.1, gated by `scripts/check_pip_audit.py` | `security` | `requirements.txt`, the image's dependency layer, resolved on Python 3.11 like the image | a finding isn't in `.github/pip-audit-baseline.txt`, a baseline entry matches no finding, or a dependency couldn't be audited |
| `npm audit --omit=dev` (npm 11.6.2) | `frontend` | the production dependencies in `frontend/package-lock.json`, which are what ships in the bundle | it reports any advisory, at any severity |
| gitleaks 8.18.4 `detect --redact` | `security` | every commit reachable from the tested commit (full-history checkout), with the default rules the pre-commit hook uses. It scans git history only, never the working tree | it finds anything |

- **Known findings.** The pip-audit baseline holds exactly the 14 findings TASK-020 reviewed that pip-audit still
  reports (§ Dependency audit): `keras` 2.15.0 (13) and `protobuf` 4.25.9 (1). pip-audit's summary line counts them
  as 25, because it lists some advisories more than once. Each entry names one package, one exact version
  and one vulnerability id, and matches only a finding with that exact id. Every run prints it as `[baseline]`, with
  its aliases (the CVE ids in § Dependency audit). Nothing is ignored by package, severity or class.
- **Version changes.** An entry is pinned to its version. `protobuf` floats within TensorFlow's `<5` range, so a new
  4.25.x patch that still carries CVE-2026-0994 fails the gate until the entry's version is updated in a reviewed
  change. A renamed or split advisory fails the same way.
- **New findings** fail CI, even when the code didn't change. Upgrade the package if a fixed release fits the pins.
  Otherwise, check whether serving can reach it, then add a baseline line with the reason and the removal condition,
  plus a row in § Dependency audit, in one reviewed change.
- **False positives.** For pip-audit, an advisory that doesn't apply is recorded like any accepted finding: one
  baseline line, with the reason. npm has no ignore list here; a production false positive needs a reviewed
  decision before anything is suppressed. For gitleaks, a false positive gets a `.gitleaksignore` entry for that
  single finding (commit, file, rule, line) with a comment. A real secret is rotated first, never only
  allowlisted.
- **Removing an exception.** When a baselined package is upgraded, for example at the TensorFlow / Keras migration,
  its entries stop matching and the gate fails until they and their § Dependency audit rows are removed. An
  exception can't outlive its reason.
- **Output.** gitleaks runs with `--redact`, so secret values never reach the log. pip-audit and npm print package
  names, versions and advisory ids only.
- **Reproducing it locally:** see `docs/CI.md` § Local equivalents.

## Reporting

Report suspected vulnerabilities privately to the maintainer (see README § License & Contact) rather
than opening a public issue.
