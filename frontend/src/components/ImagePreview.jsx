function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export default function ImagePreview({ file, previewUrl, onClear, disabled }) {
  if (!file || !previewUrl) return null;

  return (
    <div className="relative rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
      <div className="aspect-video w-full bg-black flex items-center justify-center">
        <img
          src={previewUrl}
          alt={file.name}
          className="max-h-full max-w-full object-contain"
        />
      </div>
      <div className="flex items-center justify-between px-4 py-3 border-t border-white/10 bg-black/40">
        <div className="min-w-0 flex-1">
          <p className="text-sm text-white/90 truncate" title={file.name}>
            {file.name}
          </p>
          <p className="text-xs text-white/40 mt-0.5">
            {formatBytes(file.size)} ·{" "}
            {file.type.replace("image/", "").toUpperCase()}
          </p>
        </div>
        <button
          type="button"
          onClick={onClear}
          disabled={disabled}
          className="ml-3 inline-flex items-center gap-1.5 text-xs text-white/60 hover:text-white px-2.5 py-1.5 rounded-md border border-white/10 hover:border-white/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <svg
            viewBox="0 0 24 24"
            className="w-3.5 h-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
          Remove
        </button>
      </div>
    </div>
  );
}
