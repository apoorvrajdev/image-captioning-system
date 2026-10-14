import { useEffect, useMemo, useRef, useState } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImagePreview from "./components/ImagePreview";
import CaptionResult from "./components/CaptionResult";
import DatasetReference from "./components/DatasetReference";
import ErrorBanner from "./components/ErrorBanner";
import SampleGallery from "./components/SampleGallery";
import Spinner from "./components/Spinner";
import Phase3Dashboard from "./components/Phase3Dashboard";
import { SAMPLES } from "./samples";
import { captionImage } from "./services/api";
import { validateImageFile } from "./utils/imageValidation";

const SAMPLE_LOAD_ERROR =
  "Couldn't load the example image. Try another, or upload your own.";

const VIEWS = [
  { id: "caption", label: "Caption an image" },
  { id: "phase3", label: "Phase 3 comparison" },
];

export default function App() {
  const [view, setView] = useState("caption");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedSampleId, setSelectedSampleId] = useState(null);
  const [pendingSampleId, setPendingSampleId] = useState(null);
  // Bumped by every selection, so a sample fetch that finishes after a newer
  // choice (another sample, an upload, or Remove) is dropped.
  const selectionRef = useRef(0);

  const selectedSample = SAMPLES.find(({ id }) => id === selectedSampleId);

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );

  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  // The one path for every chosen image: uploads call it directly, examples
  // after their fetch. `sampleId` is set only for an example.
  const handleFileSelected = (nextFile, validationError, sampleId = null) => {
    selectionRef.current += 1;
    setPendingSampleId(null);
    setResult(null);
    setSelectedSampleId(validationError ? null : sampleId);
    if (validationError) {
      setFile(null);
      setError(validationError);
      return;
    }
    setError(null);
    setFile(nextFile);
  };

  const handleSampleSelected = async (sample) => {
    if (loading) return;
    const token = ++selectionRef.current;
    setPendingSampleId(sample.id);
    let nextFile = null;
    try {
      const response = await fetch(sample.src);
      const blob = response.ok ? await response.blob() : null;
      if (blob?.type.startsWith("image/")) {
        nextFile = new File([blob], sample.fileName, { type: blob.type });
      }
    } catch {
      nextFile = null;
    }
    if (token !== selectionRef.current) return;
    if (!nextFile) {
      handleFileSelected(null, SAMPLE_LOAD_ERROR);
      return;
    }
    handleFileSelected(nextFile, validateImageFile(nextFile), sample.id);
  };

  const handleClear = () => {
    selectionRef.current += 1;
    setPendingSampleId(null);
    setSelectedSampleId(null);
    setFile(null);
    setResult(null);
    setError(null);
  };

  const handleGenerate = async () => {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await captionImage(file);
      setResult(data);
    } catch (err) {
      setError(err?.message || "Caption generation failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white antialiased">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 w-[480px] h-[480px] rounded-full bg-violet-600/20 blur-[120px]" />
        <div className="absolute top-1/3 -right-40 w-[520px] h-[520px] rounded-full bg-fuchsia-600/10 blur-[140px]" />
        <div className="absolute bottom-0 left-1/3 w-[420px] h-[420px] rounded-full bg-indigo-600/10 blur-[120px]" />
      </div>

      <div className="relative">
        <Header />

        <main className="max-w-6xl mx-auto px-6 py-10 md:py-16">
          <nav
            aria-label="Views"
            className="mb-8 md:mb-10 flex w-fit max-w-full gap-1 rounded-xl border border-white/10 bg-white/[0.02] p-1"
          >
            {VIEWS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                aria-pressed={view === id}
                onClick={() => setView(id)}
                className={[
                  "px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors",
                  view === id
                    ? "bg-white/10 text-white"
                    : "text-white/50 hover:text-white/80",
                ].join(" ")}
              >
                {label}
              </button>
            ))}
          </nav>

          {view === "phase3" && <Phase3Dashboard />}

          <section className="mb-10 md:mb-14" hidden={view !== "caption"}>
            <h1 className="text-3xl md:text-5xl font-semibold tracking-tight leading-tight">
              Describe any image{" "}
              <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-indigo-300 bg-clip-text text-transparent">
                in natural language
              </span>
            </h1>
            <p className="text-white/50 mt-3 max-w-2xl">
              Upload a photo and let the model generate a concise caption.
              Powered by a vision-encoder/text-decoder pipeline served over
              FastAPI.
            </p>
          </section>

          <div
            className="grid grid-cols-1 lg:grid-cols-5 gap-6"
            hidden={view !== "caption"}
          >
            <div className="lg:col-span-3 space-y-5">
              {file ? (
                <ImagePreview
                  file={file}
                  previewUrl={previewUrl}
                  onClear={handleClear}
                  disabled={loading}
                />
              ) : (
                <UploadZone
                  onFileSelected={handleFileSelected}
                  disabled={loading}
                />
              )}

              <ErrorBanner message={error} onDismiss={() => setError(null)} />

              <SampleGallery
                samples={SAMPLES}
                selectedId={selectedSampleId}
                pendingId={pendingSampleId}
                onSelect={handleSampleSelected}
                disabled={loading}
              />
            </div>

            <div className="lg:col-span-2 space-y-5">
              <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-medium text-white/80 uppercase tracking-wider">
                    Inference
                  </h2>
                  {loading && (
                    <span className="text-xs text-white/40">running…</span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={handleGenerate}
                  disabled={!file || loading}
                  className={[
                    "w-full inline-flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-sm font-medium transition-all",
                    !file || loading
                      ? "bg-white/5 text-white/40 cursor-not-allowed border border-white/5"
                      : "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white shadow-lg shadow-violet-500/30 hover:shadow-violet-500/50 hover:from-violet-400 hover:to-fuchsia-400",
                  ].join(" ")}
                >
                  {loading ? (
                    <>
                      <Spinner size="sm" />
                      Generating caption…
                    </>
                  ) : (
                    <>
                      <svg
                        viewBox="0 0 24 24"
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M5 12h14" />
                        <path d="m12 5 7 7-7 7" />
                      </svg>
                      Generate caption
                    </>
                  )}
                </button>

                <p className="text-xs text-white/40 mt-3 leading-relaxed">
                  {file
                    ? "Click generate to send the image to the inference endpoint."
                    : "Upload an image or pick an example to enable generation."}
                </p>
              </div>

              {result && <CaptionResult result={result} />}
              {result && selectedSample && (
                <DatasetReference sample={selectedSample} />
              )}
            </div>
          </div>
        </main>

        <footer className="max-w-6xl mx-auto px-6 py-8 text-xs text-white/30 border-t border-white/5 mt-10">
          POST <span className="font-mono text-white/50">/v1/captions</span> ·
          built with React + Vite + Tailwind
        </footer>
      </div>
    </div>
  );
}
