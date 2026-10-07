import { readFileSync } from "node:fs";
import { test, expect, PNG_1X1, CAPTION } from "./support.js";

// TASK-018's dashboard spec, deferred until TASK-007 brought in Playwright.
// Expected values come from the same committed file the bundle imports;
// ADR-022 fixes how they are rounded for display.
const data = JSON.parse(
  readFileSync(
    new URL("../src/generated/phase3-dashboard.json", import.meta.url),
    "utf8",
  ),
);
const DEVICE_LABELS = { cpu: "CPU", cuda: "GPU (CUDA)" };
const NA = "n/a";
const exact = (value) => String(value);

const dashboardHeading = (page) =>
  page.getByRole("heading", { level: 1, name: "Phase 3 model comparison" });
const captionHeading = (page) =>
  page.getByRole("heading", { level: 1, name: /Describe any image/ });
const modelCard = (page, model) =>
  page.getByRole("article").filter({
    has: page.getByRole("heading", { name: model.display_name, exact: true }),
  });
// [value attribute, displayed text] of every <data> cell under `locator`.
const dataCells = (locator) =>
  locator
    .locator("data")
    .evaluateAll((cells) =>
      cells.map((cell) => [cell.value, cell.textContent]),
    );

async function openDashboard(page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Phase 3 comparison" }).click();
  await expect(dashboardHeading(page)).toBeVisible();
}

test("opens on the caption view and switches views without a request", async ({
  page,
  api,
}) => {
  await page.goto("/");
  const captionButton = page.getByRole("button", { name: "Caption an image" });
  const dashboardButton = page.getByRole("button", {
    name: "Phase 3 comparison",
  });
  await expect(captionButton).toHaveAttribute("aria-pressed", "true");
  await expect(dashboardButton).toHaveAttribute("aria-pressed", "false");
  await expect(captionHeading(page)).toBeVisible();
  await expect(page.getByText("Backend online")).toBeVisible();

  // The status badge polls /healthz on a timer in every view; any other
  // request after this point would come from switching views.
  const requests = [];
  page.on("request", (request) => {
    if (!request.url().endsWith("/healthz")) requests.push(request.url());
  });

  await dashboardButton.click();
  await expect(dashboardButton).toHaveAttribute("aria-pressed", "true");
  await expect(dashboardHeading(page)).toBeVisible();
  await expect(captionHeading(page)).toBeHidden();
  await expect(page.getByRole("table")).toHaveCount(3);

  await captionButton.click();
  await expect(captionHeading(page)).toBeVisible();
  await expect(page.getByRole("table")).toHaveCount(0);

  expect(requests).toEqual([]);
  expect(api.calls.filter((call) => call !== "GET /healthz")).toEqual([]);
});

test("shows every model's quality metrics from the generated data", async ({
  page,
}) => {
  await openDashboard(page);
  const table = page.getByRole("region", { name: "Caption quality table" });
  for (const { label } of data.quality.metrics) {
    await expect(
      table.getByRole("columnheader", { name: label, exact: true }),
    ).toBeVisible();
  }

  const runs = data.models.flatMap((model) => model.quality);
  await expect(table.getByRole("row")).toHaveCount(runs.length + 1);
  for (const model of data.models) {
    await expect(
      table.getByRole("rowheader", { name: model.display_name }),
    ).toBeVisible();
  }
  for (const run of runs) {
    const row = table
      .getByRole("row")
      .filter({ has: page.getByText(run.run_id, { exact: true }) });
    await expect(row).toContainText(run.kind);
    await expect(row).toContainText(run.decode_strategy);
    expect(await dataCells(row)).toEqual([
      [exact(run.n_samples), exact(run.n_samples)],
      ...data.quality.metrics.map(({ key }) => [
        exact(run.metrics[key]),
        run.metrics[key].toFixed(2),
      ]),
    ]);
  }
});

test("shows CPU and GPU latency for batch sizes 1 and 8", async ({ page }) => {
  await openDashboard(page);
  const { batch_sizes: batchSizes } = data.latency.settings;
  expect(batchSizes).toEqual([1, 8]);

  for (const [device, label] of Object.entries(DEVICE_LABELS)) {
    await expect(
      page.getByRole("heading", { name: label, exact: true }),
    ).toBeVisible();
    const table = page.getByRole("region", { name: `${label} latency table` });
    for (const model of data.models) {
      const run = model.latency.find((entry) => entry.device === device);
      expect(run, `${model.model_id} has a ${device} run`).toBeDefined();
      await expect(page.getByText(run.environment)).toBeVisible();

      const group = table.locator("tbody").filter({ hasText: run.run_id });
      await expect(group.getByRole("rowheader")).toContainText(
        model.display_name,
      );
      await expect(group).toContainText(run.batch_mode);
      if (run.batch_mode === "sequential") {
        await expect(group).toContainText("one image per call, not batched");
      }
      const rows = group.getByRole("row");
      await expect(rows).toHaveCount(batchSizes.length);
      for (const [i, size] of batchSizes.entries()) {
        await expect(
          rows.nth(i).getByRole("cell", { name: String(size), exact: true }),
        ).toBeVisible();
      }

      expect(await dataCells(group)).toEqual([
        [exact(run.load_seconds), run.load_seconds.toFixed(1)],
        ...batchSizes.flatMap((size) => {
          const batch = run.batches.find((entry) => entry.batch_size === size);
          return [
            [exact(batch.calls_per_pass), exact(batch.calls_per_pass)],
            ...data.latency.statistics.map((stat) => {
              const value = batch.summary_seconds[stat];
              return [
                exact(value),
                stat === "count" ? exact(value) : value.toFixed(4),
              ];
            }),
          ];
        }),
      ]);
    }
  }
});

test("shows each model's provenance and the evaluation slice", async ({
  page,
}) => {
  await openDashboard(page);
  for (const model of data.models) {
    const card = modelCard(page, model);
    await expect(
      card.getByRole("link", { name: model.hub_repo }),
    ).toHaveAttribute(
      "href",
      `https://huggingface.co/${model.hub_repo}/tree/${model.revision}`,
    );
    await expect(card).toContainText(model.model_id);
    await expect(card).toContainText(model.revision);
    for (const runId of model.source_run_ids) {
      await expect(
        card.getByText(runId, { exact: true }).first(),
      ).toBeVisible();
    }
  }
  for (const text of [
    data.slice.description,
    data.slice.source,
    data.slice.fingerprint_sha256,
  ]) {
    await expect(page.getByText(text, { exact: true })).toBeVisible();
  }
});

test("shows the caveats before any number, and every note", async ({
  page,
}) => {
  await openDashboard(page);
  const caveats = page.getByRole("region", { name: "Read before comparing" });
  const sequential = data.models
    .filter((model) =>
      model.latency.some((run) => run.batch_mode === "sequential"),
    )
    .map((model) => model.display_name);
  for (const text of [
    "An evaluation artefact, not live telemetry.",
    "Not a held-out comparison.",
    data.overlap_caveat,
    "Not a ranking.",
    "CPU and GPU aren't a controlled comparison.",
    `Batches are sequential for ${sequential.join(", ")}.`,
  ]) {
    await expect(caveats).toContainText(text);
  }

  const caveatsBox = await caveats.boundingBox();
  const tableBox = await page
    .getByRole("region", { name: "Caption quality table" })
    .boundingBox();
  expect(caveatsBox.y + caveatsBox.height).toBeLessThan(tableBox.y);

  for (const note of [...data.quality.notes, ...data.latency.notes]) {
    await expect(page.getByText(note, { exact: true })).toBeVisible();
  }
});

test("renders the data's missing values as n/a", async ({ page }) => {
  await openDashboard(page);
  const gaps = data.models.flatMap((model) =>
    model.quality
      .filter(
        (run) =>
          run.revision === null ||
          Object.values(run.decode_settings).includes(null),
      )
      .map((run) => ({ model, run })),
  );
  expect(gaps.length, "the data has null fields to render").toBeGreaterThan(0);

  for (const { model, run } of gaps) {
    const item = modelCard(page, model)
      .getByRole("listitem")
      .filter({
        hasText: `${run.run_id} (${run.kind}, ${run.decode_strategy})`,
      });
    await expect(item).toContainText(`revision ${run.revision ?? NA}`);
    await expect(item).toContainText(
      Object.entries(run.decode_settings)
        .map(([key, value]) => `${key} ${value ?? NA}`)
        .join(" · "),
    );
  }
});

test("'Show exact values' shows the unrounded numbers and switches back", async ({
  page,
}) => {
  await openDashboard(page);
  const bleu1 = data.models[0].quality[0].metrics.bleu1;
  const qualityTable = page.getByRole("region", {
    name: "Caption quality table",
  });

  await page.getByRole("button", { name: "Show exact values" }).click();
  const showRounded = page.getByRole("button", { name: "Show rounded values" });
  await expect(showRounded).toHaveAttribute("aria-pressed", "true");
  const cells = await dataCells(page.getByRole("main"));
  expect(cells.length).toBeGreaterThan(0);
  expect(cells.filter(([value, text]) => value !== text)).toEqual([]);
  await expect(
    qualityTable.getByText(exact(bleu1), { exact: true }),
  ).toBeVisible();

  await showRounded.click();
  await expect(
    page.getByRole("button", { name: "Show exact values" }),
  ).toHaveAttribute("aria-pressed", "false");
  await expect(
    qualityTable.getByText(bleu1.toFixed(2), { exact: true }),
  ).toBeVisible();
});

test("renders fully with the API unreachable", async ({ page, api }) => {
  api.down = true;
  await openDashboard(page);
  await expect(page.getByText("Backend offline")).toBeVisible();
  await expect(page.getByRole("table")).toHaveCount(3);
  await expect(page.getByText(data.overlap_caveat)).toBeVisible();
  expect(api.calls.filter((call) => call !== "GET /healthz")).toEqual([]);
});

test("switching views keeps the caption flow's file and result", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "e2e-pixel.png",
    mimeType: "image/png",
    buffer: PNG_1X1,
  });
  await page.getByRole("button", { name: "Generate caption" }).click();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();

  await page.getByRole("button", { name: "Phase 3 comparison" }).click();
  await expect(dashboardHeading(page)).toBeVisible();
  await expect(page.getByText(CAPTION.caption)).toBeHidden();

  await page.getByRole("button", { name: "Caption an image" }).click();
  await expect(page.getByRole("img", { name: "e2e-pixel.png" })).toBeVisible();
  await expect(page.getByText(CAPTION.caption)).toBeVisible();
});

// Whether every column fits at 1280 px depends on the platform's fonts, so
// these check what holds everywhere: the page never scrolls sideways, and a
// table wider than its region scrolls inside it instead of losing columns.
async function expectNoClippedTables(page, viewportWidth) {
  const pageOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(pageOverflow, "page scrolls sideways").toBeLessThanOrEqual(0);

  const regions = page.getByRole("region", { name: /table$/ });
  await expect(regions).toHaveCount(3);
  const layouts = [];
  for (const region of await regions.all()) {
    await expect(region).toHaveAttribute("tabindex", "0");
    const box = await region.boundingBox();
    expect(box.x).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width).toBeLessThanOrEqual(viewportWidth);
    layouts.push(
      await region.evaluate((element) => ({
        overflowX: getComputedStyle(element).overflowX,
        overflows: element.scrollWidth > element.clientWidth,
      })),
    );
  }
  for (const { overflowX } of layouts) expect(overflowX).toBe("auto");
  return layouts;
}

test.describe("at 390 px", () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test("wide tables scroll inside their regions, not the page", async ({
    page,
  }) => {
    await openDashboard(page);
    const layouts = await expectNoClippedTables(page, 390);
    expect(layouts.every(({ overflows }) => overflows)).toBe(true);
  });
});

test.describe("at 1280 px", () => {
  test.use({ viewport: { width: 1280, height: 720 } });

  test("no table is clipped and the page doesn't scroll sideways", async ({
    page,
  }) => {
    await openDashboard(page);
    await expectNoClippedTables(page, 1280);
  });
});
