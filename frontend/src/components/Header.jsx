import StatusBadge from "./StatusBadge";

export default function Header() {
  return (
    <header className="border-b border-white/5 bg-black/40 backdrop-blur-xl sticky top-0 z-30">
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 via-fuchsia-500 to-indigo-500 flex items-center justify-center shadow-lg shadow-violet-500/30">
              <svg
                viewBox="0 0 24 24"
                className="w-5 h-5 text-white"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="3" />
                <circle cx="9" cy="9" r="1.5" />
                <path d="m21 15-5-5L5 21" />
              </svg>
            </div>
            <div className="absolute inset-0 rounded-xl bg-violet-500/40 blur-xl -z-10" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-white font-semibold tracking-tight">
              Caption Studio
            </span>
            <span className="text-xs text-white/40">
              Vision-to-text inference
            </span>
          </div>
        </div>
        <StatusBadge />
      </div>
    </header>
  );
}
