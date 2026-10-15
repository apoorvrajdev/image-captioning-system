// Both decoders' captions for the same image, side by side (TASK-034). Each
// side shows its own result or its own error, so one failing doesn't hide the
// other.

const LABELS = { greedy: "Greedy", beam: "Beam search" };

function Side({ strategy, outcome }) {
  return (
    <div
      role="group"
      aria-label={LABELS[strategy]}
      className="rounded-xl border border-white/10 bg-black/30 p-4 min-w-0"
    >
      <p className="text-xs uppercase tracking-wider text-white/50">
        {LABELS[strategy]}
      </p>
      {outcome?.result && (
        <>
          <p className="text-base text-white font-medium leading-snug mt-2">
            {outcome.result.caption}
          </p>
          <p className="text-xs text-white/50 mt-3">
            {outcome.result.latency_ms.toFixed(1)} ms ·{" "}
            {outcome.result.model_version}
          </p>
        </>
      )}
      {outcome?.error && (
        <p role="alert" className="text-sm text-red-200/80 mt-2 break-words">
          {outcome.error}
        </p>
      )}
      {!outcome && <p className="text-sm text-white/40 mt-2">Waiting…</p>}
    </div>
  );
}

export default function CaptionComparison({ comparison }) {
  if (!comparison) return null;
  return (
    <section
      aria-labelledby="caption-comparison-heading"
      className="rounded-2xl border border-white/10 bg-gradient-to-b from-white/[0.04] to-white/[0.01] p-5"
    >
      <h2
        id="caption-comparison-heading"
        className="text-xs uppercase tracking-wider text-white/40 mb-3"
      >
        Greedy vs beam search
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Side strategy="greedy" outcome={comparison.greedy} />
        <Side strategy="beam" outcome={comparison.beam} />
      </div>
      <p className="text-[11px] text-white/50 mt-3 leading-relaxed">
        Same image, same model, decoded twice in turn. On the committed
        500-image evaluation, beam search scored CIDEr 0.826 against greedy's
        0.789, with BLEU-4 slightly lower; one image proves nothing either way.
      </p>
    </section>
  );
}
