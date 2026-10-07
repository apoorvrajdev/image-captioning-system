import { test as base, expect } from "@playwright/test";

// A 1x1 PNG built in memory, so no binary fixture is committed.
export const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);

// Response bodies shaped like backend/app/schemas/caption.py.
export const HEALTH = {
  status: "ok",
  model_loaded: true,
  model_version: "v2.0.0",
  api_version: "0.1.0",
  timestamp: "2026-01-01T00:00:00Z",
};
export const CAPTION = {
  caption: "a small test image on a plain background",
  model_version: "v2.0.0",
  decode_strategy: "greedy",
  latency_ms: 123.4,
  request_id: "e2e-request-1",
};

const API_PATHS = ["/healthz", "/v1/captions"];
const REFUSED = "Failed to load resource: net::ERR_CONNECTION_REFUSED";
const apiPath = (url) => {
  const { pathname } = new URL(url);
  return API_PATHS.find((path) => pathname.endsWith(path));
};

export const test = base.extend({
  // Answers every request that leaves the app's origin, whatever VITE_API_BASE
  // the bundle was built with. Setting `api.down` makes /healthz and
  // /v1/captions refuse the connection. Any other off-origin request is
  // aborted and fails the test, so nothing reaches a real backend.
  api: [
    async ({ page, baseURL }, use) => {
      const appOrigin = new URL(baseURL).origin;
      const api = { down: false, calls: [], unexpected: [] };
      await page.route(
        (url) => url.origin !== appOrigin,
        async (route) => {
          const request = route.request();
          const endpoint = apiPath(request.url());
          if (!endpoint) {
            api.unexpected.push(request.url());
            return route.abort("blockedbyclient");
          }
          api.calls.push(`${request.method()} ${endpoint}`);
          if (api.down) return route.abort("connectionrefused");
          return route.fulfill({
            json: endpoint === "/healthz" ? HEALTH : CAPTION,
            headers: { "access-control-allow-origin": "*" },
          });
        },
      );
      await use(api);
      expect(api.unexpected, "off-origin requests outside the API").toEqual([]);
    },
    { auto: true },
  ],

  // Fails the test on any console error or uncaught page error. The one
  // exception: Chromium's network stack logs every refused request as a
  // console error even when the app handles the failure, so while `api.down`
  // is set that exact line is allowed for the two API URLs, and nothing else.
  consoleErrors: [
    async ({ page, api }, use) => {
      const errors = [];
      page.on("pageerror", (error) => errors.push(`pageerror: ${error}`));
      page.on("console", (message) => {
        if (message.type() !== "error") return;
        const { url } = message.location();
        const refusedApiCall =
          api.down &&
          message.text() === REFUSED &&
          Boolean(url) &&
          apiPath(url) !== undefined;
        if (!refusedApiCall) errors.push(`${message.text()} (${url})`);
      });
      await use(errors);
      expect(errors, "console errors and uncaught page errors").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
