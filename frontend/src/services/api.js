const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const HEALTH_TIMEOUT_MS = 3000;
const CAPTION_TIMEOUT_MS = 60000;

export const API_BASE = (
  import.meta.env?.VITE_API_BASE ?? DEFAULT_API_BASE
).replace(/\/$/, "");

class ApiError extends Error {
  constructor(message, { kind = "unknown", status = null, cause } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    if (cause) this.cause = cause;
  }
}

const isAbortError = (err) => err?.name === "AbortError" || err?.code === 20;

const classifyFetchError = (err) => {
  if (isAbortError(err)) {
    return new ApiError("Request timed out.", { kind: "timeout", cause: err });
  }
  // Browsers surface CORS denials and network failures as a generic TypeError.
  if (err instanceof TypeError) {
    return new ApiError(
      "Cannot reach backend. Check that the API is running and CORS allows this origin.",
      { kind: "network", cause: err },
    );
  }
  return new ApiError(err?.message || "Request failed.", {
    kind: "unknown",
    cause: err,
  });
};

const fetchWithTimeout = async (url, { timeoutMs, ...init } = {}) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetchWithTimeout(`${API_BASE}/healthz`, {
      timeoutMs: HEALTH_TIMEOUT_MS,
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
};

export const captionImage = async (imageFile) => {
  const formData = new FormData();
  formData.append("image", imageFile);

  let response;
  try {
    response = await fetchWithTimeout(`${API_BASE}/v1/captions`, {
      method: "POST",
      body: formData,
      timeoutMs: CAPTION_TIMEOUT_MS,
    });
  } catch (err) {
    throw classifyFetchError(err);
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData?.detail) detail = errorData.detail;
    } catch {
      /* response body was not JSON */
    }
    throw new ApiError(detail, { kind: "http", status: response.status });
  }

  return response.json();
};

export { ApiError };
