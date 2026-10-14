import { readFileSync } from "node:fs";
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
    "Unsupported format. Use JPG or PNG.",
  );
  await expect(generate).toBeDisabled();

  // The backend can't decode WebP, so the SPA refuses it before any upload.
  await input.setInputFiles({
    name: "photo.webp",
    mimeType: "image/webp",
    buffer: Buffer.from("RIFF\x1a\x00\x00\x00WEBPVP8 ", "latin1"),
  });
  await expect(page.getByRole("alert")).toContainText(
    "Unsupported format. Use JPG or PNG.",
  );
  await expect(page.getByRole("img", { name: "photo.webp" })).toHaveCount(0);
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

// TASK-033: the "Try an example" gallery.

const KITTEN = {
  label: "Kitten on a computer keyboard",
  file: "000000418799.jpg",
};
const HYDRANT = {
  label: "Red fire hydrant on a grass verge",
  file: "000000316059.jpg",
};
const sampleBytes = ({ file }) =>
  readFileSync(new URL(`../public/samples/${file}`, import.meta.url));
const gallery = (page) => page.getByRole("region", { name: "Try an example" });
const example = (page, sample) =>
  gallery(page).getByRole("button", { name: sample.label });

// Holds the app's fetch of one sample (not its thumbnail) until released.
const holdSampleFetch = async (page, sample) => {
  let release;
  const released = new Promise((resolve) => {
    release = resolve;
  });
  await page.route(`**/samples/${sample.file}`, async (route) => {
    if (route.request().resourceType() !== "fetch") return route.fallback();
    await released;
    return route.fallback();
  });
  return release;
};

const sampleFetchDone = (page, sample) =>
  page.waitForResponse(
    (response) =>
      response.url().endsWith(`/samples/${sample.file}`) &&
      response.request().resourceType() === "fetch",
  );

test("an example sends its exact bytes and its reference stays separate", async ({
  page,
  api,
}) => {
  await page.goto("/");
  await expect(gallery(page).getByRole("button")).toHaveCount(5);
  await expect(gallery(page)).toContainText("“Mango HaX0ring” by Slava");
  await expect(
    gallery(page).getByRole("link", { name: "CC BY 2.0" }),
  ).toHaveCount(5);

  const kitten = example(page, KITTEN);
  await expect(kitten).toHaveAttribute("aria-pressed", "false");
  await kitten.click();
  await expect(kitten).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("img", { name: KITTEN.file })).toBeVisible();
  const reference = page.getByRole("region", { name: "Dataset reference" });
  await expect(reference).toHaveCount(0);

  const generate = page.getByRole("button", { name: "Generate caption" });
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();

  expect(api.captionBodies).toHaveLength(1);
  const body = api.captionBodies[0];
  expect(body.includes(sampleBytes(KITTEN))).toBe(true);
  expect(body.toString("latin1")).toContain(`filename="${KITTEN.file}"`);
  expect(body.toString("latin1")).toContain("Content-Type: image/jpeg");

  await expect(reference).toContainText(
    "A kitten sitting on the keyboard of a computer.",
  );
  await expect(reference).toContainText("It isn't the model's output.");
  await expect(
    reference.getByRole("link", { name: "CC BY 4.0" }),
  ).toBeVisible();
  await expect(reference).not.toContainText(CAPTION.caption);
});

test("a second example replaces the first", async ({ page, api }) => {
  await page.goto("/");
  await example(page, KITTEN).click();
  await expect(page.getByRole("img", { name: KITTEN.file })).toBeVisible();

  await example(page, HYDRANT).click();
  await expect(example(page, HYDRANT)).toHaveAttribute("aria-pressed", "true");
  await expect(example(page, KITTEN)).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByRole("img", { name: HYDRANT.file })).toBeVisible();

  await page.getByRole("button", { name: "Generate caption" }).click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
  expect(api.captionBodies).toHaveLength(1);
  expect(api.captionBodies[0].includes(sampleBytes(HYDRANT))).toBe(true);
  expect(api.captionBodies[0].includes(sampleBytes(KITTEN))).toBe(false);
});

test("an upload after an example replaces it and clears the selection", async ({
  page,
  api,
}) => {
  await page.goto("/");
  await example(page, KITTEN).click();
  await expect(page.getByRole("img", { name: KITTEN.file })).toBeVisible();

  await page.getByRole("button", { name: "Remove" }).click();
  await page.locator('input[type="file"]').setInputFiles(png);
  await expect(page.getByRole("img", { name: png.name })).toBeVisible();
  await expect(
    gallery(page).getByRole("button", { pressed: true }),
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Generate caption" }).click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
  expect(api.captionBodies).toHaveLength(1);
  expect(api.captionBodies[0].includes(PNG_1X1)).toBe(true);
  expect(api.captionBodies[0].includes(sampleBytes(KITTEN))).toBe(false);
  await expect(
    page.getByRole("region", { name: "Dataset reference" }),
  ).toHaveCount(0);
});

test("a slow example can't replace an upload made while it loads", async ({
  page,
  api,
}) => {
  const release = await holdSampleFetch(page, KITTEN);
  await page.goto("/");
  await example(page, KITTEN).click();
  await expect(example(page, KITTEN)).toHaveAttribute("aria-busy", "true");

  await page.locator('input[type="file"]').setInputFiles(png);
  await expect(page.getByRole("img", { name: png.name })).toBeVisible();
  const done = sampleFetchDone(page, KITTEN);
  release();
  await done;

  await expect(page.getByRole("img", { name: png.name })).toBeVisible();
  await expect(example(page, KITTEN)).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "Generate caption" }).click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
  expect(api.captionBodies[0].includes(PNG_1X1)).toBe(true);
});

test("a slow example can't replace a newer example", async ({ page, api }) => {
  const release = await holdSampleFetch(page, KITTEN);
  await page.goto("/");
  await example(page, KITTEN).click();
  await example(page, HYDRANT).click();
  await expect(page.getByRole("img", { name: HYDRANT.file })).toBeVisible();
  const done = sampleFetchDone(page, KITTEN);
  release();
  await done;

  await expect(page.getByRole("img", { name: HYDRANT.file })).toBeVisible();
  await expect(example(page, HYDRANT)).toHaveAttribute("aria-pressed", "true");
  await expect(example(page, KITTEN)).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "Generate caption" }).click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
  expect(api.captionBodies[0].includes(sampleBytes(HYDRANT))).toBe(true);
});

test("examples respond to Enter and Space", async ({ page }) => {
  await page.goto("/");
  await example(page, KITTEN).focus();
  await page.keyboard.press("Enter");
  await expect(example(page, KITTEN)).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByRole("img", { name: KITTEN.file })).toBeVisible();

  await example(page, HYDRANT).focus();
  await page.keyboard.press("Space");
  await expect(example(page, HYDRANT)).toHaveAttribute("aria-pressed", "true");
  await expect(example(page, KITTEN)).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByRole("img", { name: HYDRANT.file })).toBeVisible();
});

test("an example that fails to load shows an error and sends nothing", async ({
  page,
  api,
}) => {
  api.allowedFailedResources.push(`/samples/${HYDRANT.file}`);
  await page.route(`**/samples/${HYDRANT.file}`, (route) =>
    route.request().resourceType() === "fetch"
      ? route.fulfill({ status: 404, body: "" })
      : route.fallback(),
  );
  await page.goto("/");
  await example(page, HYDRANT).click();

  await expect(page.getByRole("alert")).toContainText(
    "Couldn't load the example image",
  );
  await expect(example(page, HYDRANT)).toHaveAttribute("aria-pressed", "false");
  await expect(
    page.getByRole("button", { name: "Generate caption" }),
  ).toBeDisabled();
  expect(api.calls).not.toContain("POST /v1/captions");
});

test("the gallery is only on the caption view", async ({ page }) => {
  await page.goto("/");
  await expect(gallery(page)).toBeVisible();
  await page.getByRole("button", { name: "Phase 3 comparison" }).click();
  await expect(gallery(page)).toBeHidden();
  await page.getByRole("button", { name: "Caption an image" }).click();
  await expect(gallery(page)).toBeVisible();
});

test("the gallery fits a 390 px viewport in two columns", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  const buttons = gallery(page).getByRole("button");
  await expect(buttons).toHaveCount(5);
  await buttons.nth(4).scrollIntoViewIfNeeded();

  const boxes = [];
  for (let i = 0; i < 5; i += 1) boxes.push(await buttons.nth(i).boundingBox());
  expect(Math.round(boxes[0].y)).toBe(Math.round(boxes[1].y));
  expect(boxes[2].y).toBeGreaterThan(boxes[0].y + boxes[0].height / 2);
  for (const box of boxes) {
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(390);
  }
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
