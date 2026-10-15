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
// What the mock answers for a beam request (TASK-034).
export const BEAM_CAPTION = {
  ...CAPTION,
  caption: "a close up of a small test image",
  decode_strategy: "beam",
  latency_ms: 456.7,
  request_id: "e2e-request-beam",
};

// The decode_strategy form field of a multipart body, or null when absent.
export const strategyOf = (body) => {
  const match = body
    ?.toString("latin1")
    .match(/name="decode_strategy"\r\n\r\n([^\r]*)\r\n/);
  return match ? match[1] : null;
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
  // `api.captionBodies` holds each caption request's raw multipart body. A
  // caption request answers CAPTION, or BEAM_CAPTION when its body asks for
  // beam; `api.failStrategy` makes that one strategy answer 503 instead.
  // `api.allowedFailedResources` lists URL suffixes whose failed load a test
  // provokes on purpose (see consoleErrors).
  api: [
    async ({ page, baseURL }, use) => {
      const appOrigin = new URL(baseURL).origin;
      const api = {
        down: false,
        calls: [],
        unexpected: [],
        captionBodies: [],
        failStrategy: null,
        allowedFailedResources: [],
      };
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
          let json = HEALTH;
          if (endpoint === "/v1/captions") {
            const body = request.postDataBuffer();
            api.captionBodies.push(body);
            const strategy = strategyOf(body);
            if (api.failStrategy && strategy === api.failStrategy) {
              return route.fulfill({
                status: 503,
                json: { detail: `mock refused ${strategy}` },
                headers: { "access-control-allow-origin": "*" },
              });
            }
            json = strategy === "beam" ? BEAM_CAPTION : CAPTION;
          }
          if (api.down) return route.abort("connectionrefused");
          return route.fulfill({
            json,
            headers: { "access-control-allow-origin": "*" },
          });
        },
      );
      await use(api);
      expect(api.unexpected, "off-origin requests outside the API").toEqual([]);
    },
    { auto: true },
  ],

  // Fails the test on any console error or uncaught page error. Two
  // exceptions, because Chromium's network stack logs a failed load as a
  // console error even when the app handles it: while `api.down` is set the
  // refused line is allowed for the two API URLs, and a failed load is allowed
  // for a URL a test lists in `api.allowedFailedResources`. Nothing else.
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
        const provokedFailure =
          message.text().startsWith("Failed to load resource:") &&
          Boolean(url) &&
          api.allowedFailedResources.some((suffix) => url.endsWith(suffix));
        if (!refusedApiCall && !provokedFailure) {
          errors.push(`${message.text()} (${url})`);
        }
      });
      await use(errors);
      expect(errors, "console errors and uncaught page errors").toEqual([]);
    },
    { auto: true },
  ],
});

export { expect };
