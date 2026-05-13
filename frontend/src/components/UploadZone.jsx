import { useRef, useState } from "react";

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;

export default function UploadZone({ onFileSelected, disabled = false }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const validate = (file) => {
    if (!file) return "No file selected.";
    if (!ACCEPTED.includes(file.type)) {
      return "Unsupported format. Use JPG, PNG, or WEBP.";
    }
    if (file.size > MAX_BYTES) {
      return "File exceeds 10 MB limit.";
    }
    return null;
  };

  const handleFile = (file) => {
    const error = validate(file);
    onFileSelected(file ?? null, error);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (disabled) return;
    const file = e.dataTransfer?.files?.[0];
    if (file) handleFile(file);
  };

  const onDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragActive(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const onChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      className={[
        "relative w-full rounded-2xl border border-dashed transition-all duration-200 select-none",
        "flex flex-col items-center justify-center text-center px-6 py-16",
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer",
        dragActive
          ? "border-violet-400/80 bg-violet-500/10 shadow-[0_0_0_4px_rgba(139,92,246,0.12)]"
          : "border-white/15 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.04]",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        onChange={onChange}
        className="hidden"
        disabled={disabled}
      />
      <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 border border-white/10 flex items-center justify-center mb-4">
        <svg
          viewBox="0 0 24 24"
          className="w-7 h-7 text-violet-300"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <p className="text-white/90 text-base font-medium">
        {dragActive ? "Drop to upload" : "Drag & drop an image"}
      </p>
      <p className="text-white/40 text-sm mt-1">
        or click to browse — JPG, PNG, WEBP · max 10 MB
      </p>
    </div>
  );
}
