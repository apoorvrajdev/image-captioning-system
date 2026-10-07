import { useId, useState } from "react";
import dashboard from "../generated/phase3-dashboard.json";

const NA = "n/a";
const DEVICE_LABELS = { cpu: "CPU", cuda: "GPU (CUDA)" };
const STAT_LABELS = {
  count: "Samples",
  mean: "Mean (s)",
  median: "Median (s)",
  min: "Min (s)",
  max: "Max (s)",
};
// Display rounding matches the committed reports: comparison.md rounds metrics
// to two decimals, and EVAL_METHODOLOGY.md § 9.9 rounds latency to 0.1 ms and
// load time to 0.1 s. Every cell keeps the exact value in its markup.
const METRIC_DIGITS = 2;
const LATENCY_DIGITS = 4;
const LOAD_DIGITS = 1;

const show = (value) =>
  value === null || value === undefined || value === "" ? NA : String(value);

function Value({ value, digits, exact }) {
  if (typeof value !== "number" || !Number.isFinite(value)) return NA;
  const full = String(value);
  return (
    <data value={full} title={full}>
      {exact || digits === undefined ? full : value.toFixed(digits)}
    </data>
  );
}

function Section({ title, children }) {
  const id = useId();
  return (
    <section
      aria-labelledby={id}
      className="rounded-2xl border border-white/10 bg-white/[0.02] p-5 md:p-6"
    >
      <h2
        id={id}
        className="text-sm font-medium text-white/80 uppercase tracking-wider mb-4"
      >
        {title}
      </h2>
      {children}
    </section>
  );
}

function Facts({ items }) {
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
      {items.map(([label, value, mono]) => (
        <div key={label} className="min-w-0">
          <dt className="text-xs text-white/40 uppercase tracking-wide">
            {label}
          </dt>
          <dd
            className={[
              "text-white/80 mt-0.5 break-words",
              mono ? "font-mono text-xs break-all" : "",
            ].join(" ")}
          >
            {show(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function Notes({ items }) {
  if (!items?.length) return null;
  return (
    <ul className="mt-4 space-y-1.5 list-disc pl-5 text-xs text-white/50 leading-relaxed">
      {items.map((note) => (
        <li key={note}>{note}</li>
      ))}
    </ul>
  );
}

function TableScroll({ label, children }) {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={0}
      className="mt-5 overflow-x-auto rounded-xl border border-white/10"
    >
      {children}
    </div>
  );
}

const th =
  "px-3 py-2 font-medium text-xs uppercase tracking-wider text-white/40 whitespace-nowrap";
const td = "px-3 py-2 whitespace-nowrap text-white/80";
const num = `${td} text-right tabular-nums`;
const rowHead = "px-3 py-2 text-left align-top font-medium text-white min-w-34";

function QualityTable({ exact }) {
  const { metrics } = dashboard.quality;
  return (
    <TableScroll label="Caption quality table">
      <table className="w-full text-sm">
        <caption className="sr-only">
          Caption quality per run, one row group per model, listed by model id
          (not a ranking)
        </caption>
        <thead className="bg-white/[0.03] text-left">
          <tr>
            <th scope="col" className={th}>
              Model
            </th>
            <th scope="col" className={th}>
              Run and kind
            </th>
            <th scope="col" className={th}>
              Decoding
            </th>
            <th scope="col" className={`${th} text-right`}>
              Samples
            </th>
            {metrics.map((metric) => (
              <th key={metric.key} scope="col" className={`${th} text-right`}>
                {metric.label}
              </th>
            ))}
          </tr>
        </thead>
        {dashboard.models.map((model) => {
          const rows = model.quality.length ? model.quality : [null];
          return (
            <tbody key={model.model_id} className="border-t border-white/10">
              {rows.map((row, i) => (
                <tr
                  key={row?.run_id ?? "missing"}
                  className={i ? "border-t border-white/5" : undefined}
                >
                  {i === 0 && (
                    <th
                      scope="rowgroup"
                      rowSpan={rows.length}
                      className={rowHead}
                    >
                      {model.display_name}
                    </th>
                  )}
                  <td className={td}>
                    <span className="block font-mono text-xs">
                      {show(row?.run_id)}
                    </span>
                    <span className="block text-[11px] text-white/40">
                      {show(row?.kind)}
                    </span>
                  </td>
                  <td className={td}>{show(row?.decode_strategy)}</td>
                  <td className={num}>
                    <Value value={row?.n_samples} />
                  </td>
                  {metrics.map((metric) => (
                    <td key={metric.key} className={num}>
                      <Value
                        value={row?.metrics?.[metric.key]}
                        digits={METRIC_DIGITS}
                        exact={exact}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          );
        })}
      </table>
    </TableScroll>
  );
}

function LatencyTable({ device, exact }) {
  const { statistics, settings } = dashboard.latency;
  const label = DEVICE_LABELS[device] ?? device;
  const runs = dashboard.models.map((model) => ({
    model,
    run: model.latency.find((entry) => entry.device === device) ?? null,
  }));
  const environments = [
    ...new Set(runs.map(({ run }) => run?.environment).filter(Boolean)),
  ];

  return (
    <div className="mt-6">
      <h3 className="text-base font-medium text-white">{label}</h3>
      {environments.map((environment) => (
        <p
          key={environment}
          className="mt-1 text-xs text-white/50 leading-relaxed break-words"
        >
          <span className="text-white/40 uppercase tracking-wide">
            Environment:{" "}
          </span>
          {environment}
        </p>
      ))}
      <TableScroll label={`${label} latency table`}>
        <table className="w-full text-sm">
          <caption className="sr-only">
            {label} latency in seconds per call, one row group per model, listed
            by model id (not a ranking)
          </caption>
          <thead className="bg-white/[0.03] text-left">
            <tr>
              <th scope="col" className={th}>
                Model and run
              </th>
              <th scope="col" className={th}>
                Batch mode
              </th>
              <th scope="col" className={`${th} text-right`}>
                Load (s)
              </th>
              <th scope="col" className={`${th} text-right`}>
                Batch size
              </th>
              <th scope="col" className={`${th} text-right`}>
                Calls per pass
              </th>
              {statistics.map((stat) => (
                <th key={stat} scope="col" className={`${th} text-right`}>
                  {STAT_LABELS[stat] ?? stat}
                </th>
              ))}
            </tr>
          </thead>
          {runs.map(({ model, run }) => (
            <tbody key={model.model_id} className="border-t border-white/10">
              {settings.batch_sizes.map((batchSize, i) => {
                const batch =
                  run?.batches.find(
                    (entry) => entry.batch_size === batchSize,
                  ) ?? null;
                const span = settings.batch_sizes.length;
                return (
                  <tr
                    key={batchSize}
                    className={i ? "border-t border-white/5" : undefined}
                  >
                    {i === 0 && (
                      <>
                        <th
                          scope="rowgroup"
                          rowSpan={span}
                          className={`${rowHead} max-w-64`}
                        >
                          {model.display_name}
                          <span className="block mt-0.5 font-mono text-[11px] font-normal text-white/40 break-all">
                            {show(run?.run_id)}
                          </span>
                        </th>
                        <td
                          rowSpan={span}
                          className="px-3 py-2 align-top text-white/80 min-w-28"
                        >
                          {show(run?.batch_mode)}
                          {run?.batch_mode === "sequential" && (
                            <span className="block text-[11px] text-white/40">
                              one image per call, not batched
                            </span>
                          )}
                        </td>
                        <td rowSpan={span} className={`${num} align-top`}>
                          <Value
                            value={run?.load_seconds}
                            digits={LOAD_DIGITS}
                            exact={exact}
                          />
                        </td>
                      </>
                    )}
                    <td className={num}>{batchSize}</td>
                    <td className={num}>
                      <Value value={batch?.calls_per_pass} />
                    </td>
                    {statistics.map((stat) => (
                      <td key={stat} className={num}>
                        <Value
                          value={batch?.summary_seconds?.[stat]}
                          digits={stat === "count" ? undefined : LATENCY_DIGITS}
                          exact={exact}
                        />
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          ))}
        </table>
      </TableScroll>
    </div>
  );
}

function DecodeSettings({ settings }) {
  const entries = Object.entries(settings ?? {});
  if (!entries.length) return NA;
  return entries.map(([key, value]) => `${key} ${show(value)}`).join(" · ");
}

function ModelCard({ model }) {
  const hubUrl =
    model.hub_repo &&
    `https://huggingface.co/${model.hub_repo}${
      model.revision ? `/tree/${model.revision}` : ""
    }`;
  return (
    <article className="rounded-xl border border-white/10 bg-white/[0.02] p-4 min-w-0">
      <h3 className="text-base font-medium text-white">{model.display_name}</h3>
      <p className="text-xs text-white/40 mt-0.5">
        <span className="font-mono">{model.model_id}</span> · backend{" "}
        {show(model.backend)}
      </p>

      <dl className="mt-4 space-y-3 text-sm">
        <div>
          <dt className="text-xs text-white/40 uppercase tracking-wide">
            Hub repository
          </dt>
          <dd className="mt-0.5 font-mono text-xs break-all">
            {hubUrl ? (
              <a
                href={hubUrl}
                target="_blank"
                rel="noreferrer"
                className="text-violet-300 hover:text-violet-200 underline underline-offset-2"
              >
                {model.hub_repo}
              </a>
            ) : (
              NA
            )}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-white/40 uppercase tracking-wide">
            Revision
          </dt>
          <dd className="mt-0.5 font-mono text-xs text-white/80 break-all">
            {show(model.revision)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-white/40 uppercase tracking-wide">
            Source runs
          </dt>
          <dd className="mt-0.5">
            {model.source_run_ids.length ? (
              <ul className="font-mono text-xs text-white/80 space-y-0.5 break-all">
                {model.source_run_ids.map((runId) => (
                  <li key={runId}>{runId}</li>
                ))}
              </ul>
            ) : (
              NA
            )}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-white/40 uppercase tracking-wide">
            Quality runs
          </dt>
          <dd className="mt-0.5">
            {!model.quality.length && NA}
            <ul className="space-y-2">
              {model.quality.map((row) => (
                <li key={row.run_id} className="text-xs text-white/60">
                  <span className="font-mono text-white/80 break-all">
                    {row.run_id}
                  </span>{" "}
                  ({show(row.kind)}, {show(row.decode_strategy)}) · revision{" "}
                  <span className="font-mono break-all">
                    {show(row.revision)}
                  </span>
                  <span className="block mt-0.5 font-mono text-white/40 break-words">
                    <DecodeSettings settings={row.decode_settings} />
                  </span>
                </li>
              ))}
            </ul>
          </dd>
        </div>
      </dl>
    </article>
  );
}

export default function Phase3Dashboard() {
  const [exact, setExact] = useState(false);
  const { slice, quality, latency, models } = dashboard;
  const devices = [
    ...new Set(
      models.flatMap((model) => model.latency.map((run) => run.device)),
    ),
  ];
  const sequentialModels = models
    .filter((model) =>
      model.latency.some((run) => run.batch_mode === "sequential"),
    )
    .map((model) => model.display_name);
  const multiImageBatches = latency.settings.batch_sizes.filter(
    (size) => size > 1,
  );

  return (
    <div className="space-y-6">
      <section className="mb-4">
        <h1 className="text-3xl md:text-5xl font-semibold tracking-tight leading-tight">
          Phase 3{" "}
          <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-indigo-300 bg-clip-text text-transparent">
            model comparison
          </span>
        </h1>
        <p className="text-white/50 mt-3 max-w-3xl">
          Caption quality and latency for {models.length} captioning models on
          one shared COCO slice. Every figure is copied from a committed
          evaluation run and was built into this page with the app: nothing here
          is measured live, and showing it needs no request to the API.
        </p>
      </section>

      <section
        aria-labelledby="phase3-caveats"
        className="rounded-2xl border border-amber-300/20 bg-amber-300/[0.04] p-5 md:p-6"
      >
        <div className="flex items-center gap-2 mb-3">
          <svg
            viewBox="0 0 24 24"
            className="w-5 h-5 text-amber-200 shrink-0"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="16" x2="12" y2="12" />
            <line x1="12" y1="8" x2="12.01" y2="8" />
          </svg>
          <h2
            id="phase3-caveats"
            className="text-sm font-medium text-amber-100 uppercase tracking-wider"
          >
            Read before comparing
          </h2>
        </div>
        <ul className="space-y-2 text-sm text-white/70 leading-relaxed list-disc pl-5">
          <li>
            <strong className="text-white">
              An evaluation artefact, not live telemetry.
            </strong>{" "}
            These are offline runs from <code>results/</code>, not measurements
            of the deployed API.
          </li>
          <li>
            <strong className="text-white">Not a held-out comparison.</strong>{" "}
            {dashboard.overlap_caveat}
          </li>
          <li>
            <strong className="text-white">Not a ranking.</strong> Models are
            listed by model id. No combined score or winner is computed.
          </li>
          {devices.length > 1 && (
            <li>
              <strong className="text-white">
                CPU and GPU aren&apos;t a controlled comparison.
              </strong>{" "}
              Each device&apos;s runs come from a different host, operating
              system and framework build. Compare figures only within one
              device.
            </li>
          )}
          {sequentialModels.length > 0 && multiImageBatches.length > 0 && (
            <li>
              <strong className="text-white">
                Batches are sequential for {sequentialModels.join(", ")}.
              </strong>{" "}
              Their batch-{multiImageBatches.join("/")} figures are single-image
              calls in a row, not batched inference, so they aren&apos;t
              like-for-like with the batched figures.
            </li>
          )}
        </ul>
      </section>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <p className="text-xs text-white/40 max-w-2xl leading-relaxed">
          Values are rounded for display as in the committed reports: metrics to
          two decimals, latency to 0.0001 s (0.1 ms) and load time to 0.1 s.
          Each cell holds the exact value from the data file.
        </p>
        <button
          type="button"
          aria-pressed={exact}
          onClick={() => setExact((value) => !value)}
          className="shrink-0 self-start sm:self-auto inline-flex items-center gap-1.5 text-xs text-white/70 hover:text-white px-2.5 py-1.5 rounded-md border border-white/10 hover:border-white/20 transition-colors"
        >
          {exact ? "Show rounded values" : "Show exact values"}
        </button>
      </div>

      <Section title="Caption quality">
        <p className="text-sm text-white/60 leading-relaxed mb-4">
          {slice.description}
        </p>
        <Facts
          items={[
            ["Images", slice.images],
            ["References", slice.references],
            ["References per image", slice.references_per_image],
            ["Slice source", slice.source, true],
            ["Slice fingerprint (SHA-256)", slice.fingerprint_sha256, true],
            ["Summary run", quality.summary_run_id, true],
            ["Protocol", quality.protocol],
            ["Normalisation", quality.normalisation, true],
          ]}
        />
        <QualityTable exact={exact} />
        <Notes items={quality.notes} />
      </Section>

      <Section title="Latency">
        <p className="text-sm text-white/60 leading-relaxed mb-4">
          Unit: {latency.unit}.
        </p>
        <Facts
          items={[
            ["Images timed", latency.settings.num_images],
            ["Batch sizes", latency.settings.batch_sizes.join(", ")],
            ["Warmup passes", latency.settings.warmup_passes],
            ["Measured passes", latency.settings.measured_passes],
            ["Clock", latency.timing.clock, true],
            ["Protocol", latency.protocol],
            ["One sample", latency.timing.sample],
            ["Load time", latency.timing.load],
          ]}
        />
        {devices.map((device) => (
          <LatencyTable key={device} device={device} exact={exact} />
        ))}
        <Notes items={latency.notes} />
      </Section>

      <Section title="Models and provenance">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model) => (
            <ModelCard key={model.model_id} model={model} />
          ))}
        </div>
      </Section>

      <p className="text-xs text-white/30 leading-relaxed">
        Data:{" "}
        <span className="font-mono">
          frontend/src/generated/phase3-dashboard.json
        </span>{" "}
        (schema {show(dashboard.schema_version)}), generated by{" "}
        <span className="font-mono">{show(dashboard.generated_by)}</span> from
        the committed results.
      </p>
    </div>
  );
}
