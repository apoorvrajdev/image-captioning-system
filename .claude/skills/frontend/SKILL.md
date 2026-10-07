---
name: frontend
description: Acceptance criteria and definition of done for the React 19 + Vite 8 + Tailwind v4 SPA — upload flow, API service boundary, health badge, result/error rendering, and browser verification. Use whenever changing frontend/**.
---

# Frontend SPA — acceptance criteria

## Scope
`frontend/src/**` (`App.jsx`, `components/*` incl. `Phase3Dashboard.jsx`, `services/api.js`), `frontend/vite.config.js`,
`frontend/eslint.config.js`, `frontend/package.json`, `frontend/.env.example`.
Backend contract source of truth: `backend/app/schemas/caption.py`.

## Expected behaviour
- GIVEN no file WHEN the page loads THEN the upload zone is shown and Generate is disabled.
- GIVEN a JPEG/PNG/WebP ≤ 10 MB (drag/drop, click, or keyboard) THEN a preview renders. Other types or larger files → inline error, **no network call**.
- GIVEN Generate is clicked THEN a loading state shows until a `CaptionResponse` renders (caption, model version, decode strategy, latency, request ID, copy button) or `ErrorBanner` shows the `ApiError` message.
- Error classes: `timeout` (60 s caption / 3 s health), `network` (unreachable or CORS), `http` (shows backend `detail`), `unknown`.
- `StatusBadge` polls `/healthz` every 10 s and on window focus: `checking` → `online` / `offline`, and recovers by itself.
- Preview object URLs are revoked on change/unmount (no leaks).
- GIVEN the view switch in `App.jsx` (`Caption an image` / `Phase 3 comparison`, `aria-pressed` buttons, no router) WHEN the dashboard is chosen THEN `components/Phase3Dashboard.jsx` renders from the build-time import of `src/generated/phase3-dashboard.json`. No request is made. The caption flow stays mounted behind `hidden`, so its file, result and in-flight request survive the switch.
- The dashboard shows every model's quality rows and its latency per device and batch size, each tied to its run id, plus the caveats (not live, not held-out, not a ranking, CPU/GPU from different hosts, sequential CNN batches) and the data file's notes. Values come from the JSON only: nothing is recomputed, ranked or colour-coded.

## Dashboard edge cases
- A missing metric, device run, batch or load time renders as "n/a"; the row stays, with its batch size.
- Display rounding follows the committed reports (metrics 2 decimals as in `comparison.md`; latency 0.0001 s and load 0.1 s as in `EVAL_METHODOLOGY.md` § 9.9). Every number keeps its exact value in `<data value>`, and "Show exact values" displays it.
- Wide tables scroll inside focusable `role="region"` containers; the page never scrolls sideways at 390 px. At 1280 px every column is visible.

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
| e2e | `npm run test:e2e` (Playwright, Chromium, mocked API; ADR-023) | `e2e/caption-flow.spec.js`, `e2e/phase3-dashboard.spec.js`, zero console errors |
| browser | browser loop below | a changed flow the specs don't cover yet |
New UI behaviour gets a spec in `frontend/e2e/`, using the `api` and `consoleErrors` fixtures from `e2e/support.js`. There are no JS unit tests.

## Browser verification loop (UI changes)
First choice: a spec in `frontend/e2e/`. For exploration, use the Playwright MCP if it is connected (`/mcp`). Otherwise ask the user to run the flow manually and report back.
1. Start backend (`uvicorn app.main:app --app-dir backend --port 8000`) and `npm run dev`.
2. Navigate to http://localhost:5173 and take an accessibility snapshot (not screenshots).
3. Drive the changed flow: upload → Generate → result, plus one error path (bad type / backend stopped). Dashboard changes: switch views, check the tables and caveats, and that the switch makes no request.
4. Console: zero errors, no new warnings. Network: no 4xx/5xx on the happy path.
5. Fail → fix source, reload, repeat. Keep a screenshot of the end state as evidence.
A UI change with no browser run is reported as **"not browser-verified"**, never as done.

## Definition of done — ALL must pass
- [ ] All HTTP calls live in `src/services/api.js`. Components never call `fetch`.
- [ ] Loading, error, and empty states handled for every new async surface.
- [ ] `cd frontend && npm run lint && npm run build && npm run test:e2e` exit 0.
- [ ] The changed flow is covered by a spec, or the browser loop was run, or it is explicitly reported as not run.
- [ ] Contract fields read here match `backend/app/schemas/caption.py`.
- [ ] No new dependency without justification. `package-lock.json` updated with it.

## Verification commands
```bash
cd frontend && npm run lint && npm run build
npx playwright install chromium   # once per machine
npm run test:e2e
```

## Non-negotiables
- `src/generated/phase3-dashboard.json` is generated from `results/` by `python -m scripts.export_dashboard_data` (ADR-021). Read it, never edit it: a hand edit fails `tests/unit/test_dashboard_export.py`.
- Functional components + hooks only. No global state library unless a task decides it (ADR).
- Client validation mirrors, never replaces, backend validation.
- No secrets in `VITE_*` vars (they ship to the browser).
