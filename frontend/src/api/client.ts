/**
 * Thin fetch-based client for the recognition backend.
 *
 * The base URL comes from `VITE_API_BASE_URL`; when it is empty (the default)
 * requests go to the same origin and the Vite dev server proxies `/api` to the
 * backend. Every non-2xx response is turned into an {@link ApiError} carrying
 * the server's `detail` message when present (FastAPI convention).
 */
import type { CardDetail, CardSearchResponse, HealthResponse } from "../types/cards";
import type { RecognitionResult } from "../types/recognition";

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

/** Error raised for network failures (`status === 0`) and non-2xx responses. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export interface RecognizeOptions {
  /** Ask the backend to also produce debug images (paths come back in `result.debug`). */
  debug?: boolean;
  signal?: AbortSignal;
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/** Best-effort human readable message for any thrown value. */
export function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message || error.name;
  }
  return typeof error === "string" ? error : "Unknown error";
}

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

/** FastAPI returns `detail` as a string, or as a list of validation errors. */
function detailToMessage(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item: unknown) => {
        if (item && typeof item === "object" && "msg" in item) {
          const { msg, loc } = item as { msg?: unknown; loc?: unknown };
          const where = Array.isArray(loc) ? loc.join(".") : "";
          return where ? `${where}: ${String(msg)}` : String(msg);
        }
        return typeof item === "string" ? item : JSON.stringify(item);
      })
      .filter(Boolean);
    return parts.length ? parts.join("; ") : null;
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return null;
}

async function toApiError(response: Response): Promise<ApiError> {
  const statusText = response.statusText ? ` ${response.statusText}` : "";
  let message = `Request failed with HTTP ${response.status}${statusText}`;
  let detail: unknown;
  const text = await response.text().catch(() => "");
  if (text) {
    try {
      const body = JSON.parse(text) as { detail?: unknown };
      detail = body.detail;
      message = detailToMessage(body.detail) ?? message;
    } catch {
      detail = text;
      message = text.slice(0, 300);
    }
  }
  return new ApiError(message, response.status, detail);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), {
      ...init,
      headers: { Accept: "application/json", ...init.headers },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(`Could not reach the API at ${apiUrl(path)} (${errorMessage(error)})`, 0);
  }
  if (!response.ok) {
    throw await toApiError(response);
  }
  return (await response.json()) as T;
}

/**
 * `POST /api/recognize` - multipart/form-data with the image in the field `image`.
 * The blob is sent as-is (JPEG from the camera, or the uploaded file with its own type).
 */
export async function recognizeImage(blob: Blob, opts: RecognizeOptions = {}): Promise<RecognitionResult> {
  const form = new FormData();
  form.append("image", blob, "capture.jpg");
  const query = opts.debug ? "?debug=true" : "";
  return request<RecognitionResult>(`/api/recognize${query}`, {
    method: "POST",
    body: form,
    signal: opts.signal,
  });
}

/** `GET /api/health` */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { signal });
}

/** `GET /api/search?q=` */
export function searchCards(q: string, signal?: AbortSignal): Promise<CardSearchResponse> {
  return request<CardSearchResponse>(`/api/search?q=${encodeURIComponent(q)}`, { signal });
}

/** `GET /api/cards/{id}` */
export function getCard(id: number, signal?: AbortSignal): Promise<CardDetail> {
  return request<CardDetail>(`/api/cards/${encodeURIComponent(String(id))}`, { signal });
}
