// Typed client for the DocFlow API. The API is the only contract the UI depends on.

export type FieldStatus = "auto_approved" | "needs_review" | "reviewed";

export interface Check {
  rule: string;
  passed: boolean;
  message: string | null;
  suggestion?: unknown;
}

export interface Field {
  value: unknown;
  raw: string | null;
  confidence: number;
  page: number | null;
  bbox: [number, number, number, number] | null;
  checks: Check[];
  source: "llm" | "review" | "rule";
  status: FieldStatus | null;
  signals: Record<string, unknown>;
}

export interface Extraction {
  id: string;
  document_id: string;
  status: string;
  revision: number;
  fields: Record<string, Field>;
  issues: { rule: string; field: string; message: string | null }[];
  counts: Record<FieldStatus, number>;
}

export interface Page {
  page_no: number;
  width: number | null;
  height: number | null;
}

export interface DocumentDetail {
  id: string;
  filename: string;
  status: string;
  page_count: number | null;
  failure_reason: string | null;
  created_at: string;
  pages: Page[];
  latest_extraction: { id: string; status: string } | null;
}

export interface QueueItem {
  document_id: string;
  filename: string;
  created_at: string;
  extraction_id: string | null;
  fields_to_review: number;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  status: string;
  created_at: string;
}

const KEY = "docflow.apiKey";

export const apiKey = {
  get: () => sessionStorage.getItem(KEY),
  set: (value: string) => sessionStorage.setItem(KEY, value),
  clear: () => sessionStorage.removeItem(KEY),
};

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("X-API-Key", apiKey.get() ?? "");
  const response = await fetch(path, { ...init, headers });
  const body = response.headers.get("content-type")?.includes("json") ? await response.json() : null;
  if (!response.ok) throw new ApiError(response.status, body?.detail ?? response.statusText);
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  queue: () => request<QueueItem[]>("/v1/review/queue"),
  documents: (status?: string) =>
    request<{ items: DocumentSummary[] }>(`/v1/documents?limit=200${status ? `&status=${status}` : ""}`),
  document: (id: string) => request<DocumentDetail>(`/v1/documents/${id}`),
  pageImage: (id: string, page: number) =>
    request<{ url: string }>(`/v1/documents/${id}/pages/${page}/image`),
  extraction: (id: string) => request<Extraction>(`/v1/extractions/${id}`),
  correct: (id: string, field: string, action: "accept" | "edit", value: unknown, seconds: number) =>
    request<Extraction>(
      `/v1/extractions/${id}/corrections`,
      json({ field, action, value, seconds_spent: Math.min(seconds, 3600) }),
    ),
  approve: (documentId: string, acknowledgeIssues = false) =>
    request<Extraction>(`/v1/documents/${documentId}/approve?acknowledge_issues=${acknowledgeIssues}`, {
      method: "POST",
    }),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ id: string; duplicate: boolean }>("/v1/documents", { method: "POST", body: form });
  },
};
