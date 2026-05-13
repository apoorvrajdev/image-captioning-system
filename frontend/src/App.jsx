import { useEffect, useMemo, useState } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImagePreview from "./components/ImagePreview";
import CaptionResult from "./components/CaptionResult";
import ErrorBanner from "./components/ErrorBanner";
import Spinner from "./components/Spinner";
import { captionImage } from "./services/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );

  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const handleFileSelected = (nextFile, validationError) => {
    setResult(null);
    if (validationError) {
      setFile(null);
      setError(validationError);
      return;
    }
    setError(null);
    setFile(nextFile);
  };

  const handleClear = () => {
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
          <section className="mb-10 md:mb-14">
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

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
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
                    : "Upload an image to enable generation."}
                </p>
              </div>

              {result && <CaptionResult result={result} />}
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
