---
name: frontend
description: Acceptance criteria and definition of done for the React 19 + Vite 8 + Tailwind v4 SPA — upload flow, API service boundary, health badge, result/error rendering, and browser verification. Use whenever changing frontend/**.
---

# Frontend SPA — acceptance criteria

## Scope
`frontend/src/**` (`App.jsx`, `components/*`, `services/api.js`), `frontend/vite.config.js`,
`frontend/eslint.config.js`, `frontend/package.json`, `frontend/.env.example`.
Backend contract source of truth: `backend/app/schemas/caption.py`.

## Expected behaviour
- GIVEN no file WHEN the page loads THEN the upload zone is shown and Generate is disabled.
- GIVEN a JPEG/PNG/WebP ≤ 10 MB (drag/drop, click, or keyboard) THEN a preview renders. Other types or larger files → inline error, **no network call**.
- GIVEN Generate is clicked THEN a loading state shows until a `CaptionResponse` renders (caption, model version, decode strategy, latency, request ID, copy button) or `ErrorBanner` shows the `ApiError` message.
- Error classes: `timeout` (60 s caption / 3 s health), `network` (unreachable or CORS), `http` (shows backend `detail`), `unknown`.
- `StatusBadge` polls `/healthz` every 10 s and on window focus: `checking` → `online` / `offline`, and recovers by itself.
- Preview object URLs are revoked on change/unmount (no leaks).

## Edge cases
- Backend 503 during a cold start → readable message, badge offline or loading.
- Switching files mid-request doesn't render a stale result.
- `VITE_API_BASE` unset → `http://127.0.0.1:8000`. Trailing slash stripped.
- Keyboard-only use: upload zone focusable and activatable. Buttons have accessible names.

## Required checks
| Level | How | Must cover |
|---|---|---|
| static | `npm run lint` | hooks rules, unused vars |
| build | `npm run build` | Vite production build |
| browser | browser loop below | the changed user flow |
(No JS unit/e2e runner exists yet. Adding one is a deliberate task, not a side effect.)

## Browser verification loop (UI changes)
If the Playwright MCP is connected (`/mcp`), use it. Otherwise ask the user to run the flow manually and report back.
1. Start backend (`uvicorn app.main:app --app-dir backend --port 8000`) and `npm run dev`.
2. Navigate to http://localhost:5173 and take an accessibility snapshot (not screenshots).
3. Drive the changed flow: upload → Generate → result, plus one error path (bad type / backend stopped).
4. Console: zero errors, no new warnings. Network: no 4xx/5xx on the happy path.
5. Fail → fix source, reload, repeat. Keep a screenshot of the end state as evidence.
A UI change with no browser run is reported as **"not browser-verified"**, never as done.

## Definition of done — ALL must pass
- [ ] All HTTP calls live in `src/services/api.js`. Components never call `fetch`.
- [ ] Loading, error, and empty states handled for every new async surface.
- [ ] `cd frontend && npm run lint && npm run build` exit 0.
- [ ] Browser loop run, or explicitly reported as not run.
- [ ] Contract fields read here match `backend/app/schemas/caption.py`.
- [ ] No new dependency without justification. `package-lock.json` updated with it.

## Verification commands
```bash
cd frontend && npm run lint && npm run build
```

## Non-negotiables
- Functional components + hooks only. No global state library unless a task decides it (ADR).
- Client validation mirrors, never replaces, backend validation.
- No secrets in `VITE_*` vars (they ship to the browser).
