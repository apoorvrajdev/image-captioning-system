import { REFERENCE_CAPTION_CREDIT } from "../samples";

// A human-written COCO caption for the chosen example, shown beside the model's
// caption for comparison and never in its place (TASK-033).
export default function DatasetReference({ sample }) {
  const credit = REFERENCE_CAPTION_CREDIT;
  return (
    <section
      aria-labelledby="dataset-reference-heading"
      className="rounded-2xl border border-white/10 bg-white/[0.02] p-5"
    >
      <h2
        id="dataset-reference-heading"
        className="text-xs uppercase tracking-wider text-white/50"
      >
        Dataset reference
      </h2>
      <p className="text-sm text-white/75 mt-2 leading-relaxed">
        “{sample.reference}”
      </p>
      <p className="text-[11px] text-white/55 mt-3 leading-relaxed">
        One of the human-written captions for COCO image {sample.cocoId}, shown
        for comparison. It isn't the model's output. Caption by the{" "}
        <a
          href={credit.sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="underline decoration-white/20 hover:text-white/70"
        >
          {credit.source}
        </a>
        ,{" "}
        <a
          href={credit.licence.url}
          target="_blank"
          rel="noopener noreferrer"
          className="underline decoration-white/20 hover:text-white/70"
        >
          {credit.licence.name}
        </a>
        .
      </p>
    </section>
  );
}
