import { useState } from "react";

function MetaPill({ label, value }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/10 text-xs">
      <span className="text-white/40 uppercase tracking-wide">{label}</span>
      <span className="text-white/80 font-medium">{value}</span>
    </div>
  );
}

export default function CaptionResult({ result }) {
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(result.caption);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="relative rounded-2xl border border-white/10 bg-gradient-to-b from-white/[0.04] to-white/[0.01] p-6 overflow-hidden">
      <div className="absolute -top-24 -right-24 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="relative">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.7)]" />
          <span className="text-xs uppercase tracking-wider text-white/40">
            Generated caption
          </span>
        </div>

        <p className="text-xl md:text-2xl text-white font-medium leading-snug">
          {result.caption}
        </p>

        <div className="flex flex-wrap gap-2 mt-5">
          <MetaPill label="Model" value={result.model_version} />
          <MetaPill label="Decode" value={result.decode_strategy} />
          <MetaPill
            label="Latency"
            value={`${result.latency_ms.toFixed(1)} ms`}
          />
        </div>

        <div className="flex items-center justify-between mt-5 pt-4 border-t border-white/5">
          <span className="text-[11px] text-white/30 font-mono truncate">
            req: {result.request_id}
          </span>
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center gap-1.5 text-xs text-white/70 hover:text-white px-2.5 py-1.5 rounded-md border border-white/10 hover:border-white/20 transition-colors"
          >
            {copied ? (
              <>
                <svg
                  viewBox="0 0 24 24"
                  className="w-3.5 h-3.5 text-emerald-400"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Copied
              </>
            ) : (
              <>
                <svg
                  viewBox="0 0 24 24"
                  className="w-3.5 h-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="9" y="9" width="13" height="13" rx="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
                Copy
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
