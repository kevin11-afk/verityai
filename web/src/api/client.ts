import type {
  AskResponse, EvaluationResponse, Health, KBActionResult, KBList, KBUploadResult, Report, ReportList,
} from "./types";

const OVERRIDE_KEY = "verityai.apiBase";

/** Where the API lives. Precedence: user override (Connection dialog) > build-time
 *  VITE_API_BASE_URL > derived from where the app is mounted (<mount root>/api). */
export function defaultApiBase(): string {
  const fromEnv = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (fromEnv) return fromEnv.replace(/\/$/, "");
  return new URL("../api", document.baseURI).pathname.replace(/\/$/, "");
}

export function getApiBase(): string {
  try {
    const o = localStorage.getItem(OVERRIDE_KEY);
    if (o) return o.replace(/\/$/, "");
  } catch { /* storage unavailable: fall through */ }
  return defaultApiBase();
}

export function setApiBaseOverride(value: string | null): void {
  try {
    if (value) localStorage.setItem(OVERRIDE_KEY, value.trim());
    else localStorage.removeItem(OVERRIDE_KEY);
  } catch { /* ignore */ }
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
    this.name = "ApiError";
  }
  /** True for failures that mean "could not reach the server" rather than a server response. */
  get isNetwork(): boolean { return this.status === 0; }
}

async function parseError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    if (body?.error?.message) return new ApiError(res.status, body.error.code ?? "error", body.error.message);
    if (body?.detail) return new ApiError(res.status, "http_error", String(body.detail));
  } catch { /* non-JSON error body */ }
  return new ApiError(res.status, "http_error", `The server responded with ${res.status} ${res.statusText}.`);
}

async function request<T>(path: string, init?: RequestInit, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(getApiBase() + path, { ...init, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network", "Could not reach the VerityAI server. Check your connection or the API address.");
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

/** multipart upload with progress (fetch cannot report upload progress). */
function upload<T>(path: string, form: FormData, onProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", getApiBase() + path);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total); };
    xhr.onload = () => {
      const body = xhr.response as { error?: { code?: string; message?: string }; detail?: string } | null;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body as T);
      reject(new ApiError(xhr.status, body?.error?.code ?? "http_error",
        body?.error?.message ?? body?.detail ?? `The server responded with ${xhr.status}.`));
    };
    xhr.onerror = () => reject(new ApiError(0, "network", "Could not reach the VerityAI server. Check your connection or the API address."));
    xhr.ontimeout = () => reject(new ApiError(0, "timeout", "The request timed out."));
    xhr.timeout = 5 * 60 * 1000;
    signal?.addEventListener("abort", () => { xhr.abort(); reject(new DOMException("Aborted", "AbortError")); });
    xhr.send(form);
  });
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", undefined, signal),
  evaluation: (signal?: AbortSignal) => request<EvaluationResponse>("/evaluation", undefined, signal),
  reports: (q = "", signal?: AbortSignal) => request<ReportList>(`/reports?q=${encodeURIComponent(q)}&limit=200`, undefined, signal),
  report: (id: string, signal?: AbortSignal) => request<Report>(`/reports/${encodeURIComponent(id)}`, undefined, signal),
  deleteReport: (id: string) => request<KBActionResult>(`/reports/${encodeURIComponent(id)}`, { method: "DELETE" }),
  reportDownloadUrl: (id: string, format: "json" | "md") => `${getApiBase()}/reports/${encodeURIComponent(id)}/download?format=${format}`,
  analyze(kind: "image" | "video", file: File, onProgress?: (f: number) => void, signal?: AbortSignal) {
    const form = new FormData();
    form.append("file", file);
    return upload<Report>(`/analyze/${kind}`, form, onProgress, signal);
  },
  ask: (question: string, topK = 5, signal?: AbortSignal) => request<AskResponse>("/research/ask", json({ question, top_k: topK }), signal),
  search: (query: string, topK = 5, signal?: AbortSignal) => request<AskResponse>("/kb/search", json({ query, top_k: topK }), signal),
  kbList: (signal?: AbortSignal) => request<KBList>("/kb/documents", undefined, signal),
  kbUpload(file: File, sourceUrl: string, onProgress?: (f: number) => void) {
    const form = new FormData();
    form.append("file", file);
    if (sourceUrl.trim()) form.append("source_url", sourceUrl.trim());
    return upload<KBUploadResult>("/kb/documents", form, onProgress);
  },
  kbRemove: (name: string) => request<KBActionResult>(`/kb/documents/${encodeURIComponent(name)}?confirm=${encodeURIComponent(name)}`, { method: "DELETE" }),
  kbReindex: () => request<KBActionResult>("/kb/reindex?confirm=true", { method: "POST" }),
};
