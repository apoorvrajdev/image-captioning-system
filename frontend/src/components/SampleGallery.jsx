// "Try an example" gallery (TASK-033): pick a licensed sample instead of
// uploading. Selecting one only reports it through `onSelect`; the caller turns
// it into the same file state an upload produces.

export default function SampleGallery({
  samples,
  selectedId = null,
  pendingId = null,
  onSelect,
  disabled = false,
}) {
  return (
    <section
      aria-labelledby="sample-gallery-heading"
      className="rounded-2xl border border-white/10 bg-white/[0.02] p-5"
    >
      <h2
        id="sample-gallery-heading"
        className="text-sm font-medium text-white/80"
      >
        Try an example
      </h2>
      <p className="text-xs text-white/50 mt-1">
        Pick a photo to caption it without uploading your own.
      </p>

      <ul className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mt-4">
        {samples.map((sample) => {
          const selected = sample.id === selectedId;
          const pending = sample.id === pendingId;
          return (
            <li key={sample.id}>
              <button
                type="button"
                aria-pressed={selected}
                aria-busy={pending}
                disabled={disabled}
                onClick={() => onSelect(sample)}
                className={[
                  "group relative w-full overflow-hidden rounded-xl border text-left transition-all",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fuchsia-300 focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0a0f]",
                  "disabled:cursor-not-allowed disabled:opacity-40",
                  selected
                    ? "border-violet-400 ring-2 ring-violet-400/60 bg-violet-500/10"
                    : "border-white/10 bg-black/30 hover:border-violet-400/50",
                ].join(" ")}
              >
                <img
                  src={sample.src}
                  alt=""
                  loading="lazy"
                  decoding="async"
                  className={[
                    "aspect-[4/3] w-full object-cover transition-opacity",
                    pending ? "opacity-50" : "group-hover:opacity-90",
                  ].join(" ")}
                />
                <span className="block px-2.5 py-2 text-xs leading-snug text-white/80">
                  {sample.label}
                </span>
                {selected && (
                  <span
                    aria-hidden="true"
                    className="absolute top-1.5 right-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-violet-500 text-white shadow"
                  >
                    <svg
                      viewBox="0 0 24 24"
                      className="h-3 w-3"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="3"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <p className="text-[11px] text-white/55 mt-4 leading-relaxed">
        Examples were picked by hand; captions for your own photos may be less
        accurate. Photos from COCO 2017:{" "}
        {samples.map((sample, index) => (
          <span key={sample.id}>
            <a
              href={sample.credit.sourceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="underline decoration-white/20 hover:text-white/70"
            >
              “{sample.credit.title}”
            </a>{" "}
            by {sample.credit.creator},{" "}
            <a
              href={sample.credit.licence.url}
              target="_blank"
              rel="noopener noreferrer"
              className="underline decoration-white/20 hover:text-white/70"
            >
              {sample.credit.licence.name}
            </a>
            {index < samples.length - 1 ? "; " : "."}
          </span>
        ))}
      </p>
    </section>
  );
}
