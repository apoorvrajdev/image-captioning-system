import { test, expect, PNG_1X1, CAPTION } from "./support.js";

// TASK-007: the caption flow against a mocked API (TEST_PLAN.md, Frontend).

const png = { name: "e2e-pixel.png", mimeType: "image/png", buffer: PNG_1X1 };

test("a PNG uploaded against a healthy API renders the caption card", async ({
  page,
  api,
}) => {
  await page.goto("/");
  await expect(page.getByText("Backend online")).toBeVisible();
  const generate = page.getByRole("button", { name: "Generate caption" });
  await expect(generate).toBeDisabled();

  await page.locator('input[type="file"]').setInputFiles(png);
  await expect(page.getByRole("img", { name: png.name })).toBeVisible();
  await expect(generate).toBeEnabled();

  const captionRequest = page.waitForRequest("**/v1/captions");
  await generate.click();
  expect((await captionRequest).method()).toBe("POST");

  await expect(page.getByText("Generated caption")).toBeVisible();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
  for (const value of [
    CAPTION.model_version,
    CAPTION.decode_strategy,
    `${CAPTION.latency_ms.toFixed(1)} ms`,
    `req: ${CAPTION.request_id}`,
  ]) {
    await expect(page.getByText(value, { exact: true })).toBeVisible();
  }
  await expect(page.getByRole("button", { name: "Copy" })).toBeVisible();
  expect(api.calls.filter((call) => call.endsWith("/v1/captions"))).toEqual([
    "POST /v1/captions",
  ]);
});

test("a disallowed file shows an inline error and sends no request", async ({
  page,
  api,
}) => {
  await page.goto("/");
  await expect(page.getByText("Backend online")).toBeVisible();
  const input = page.locator('input[type="file"]');
  const generate = page.getByRole("button", { name: "Generate caption" });

  await input.setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("not an image"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Unsupported format. Use JPG, PNG, or WEBP.",
  );
  await expect(generate).toBeDisabled();

  await input.setInputFiles({
    name: "too-big.png",
    mimeType: "image/png",
    buffer: Buffer.alloc(10 * 1024 * 1024 + 1),
  });
  await expect(page.getByRole("alert")).toContainText(
    "File exceeds 10 MB limit.",
  );
  await expect(generate).toBeDisabled();

  expect(api.calls).not.toContain("POST /v1/captions");
});

test("an unreachable API shows 'Cannot reach backend'", async ({
  page,
  api,
}) => {
  api.down = true;
  await page.goto("/");
  await expect(page.getByText("Backend offline")).toBeVisible();

  await page.locator('input[type="file"]').setInputFiles(png);
  await page.getByRole("button", { name: "Generate caption" }).click();

  await expect(page.getByRole("alert")).toContainText("Cannot reach backend");
  await expect(page.getByText("Generated caption")).toBeHidden();
  expect(api.calls).toContain("POST /v1/captions");
});
