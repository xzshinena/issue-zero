export interface SearchRequest {
  query?: string;
  issue_url?: string;
  repo?: string;
  limit?: number;
}

export interface SimilarIssue {
  id: string;
  url: string;
  title: string;
  score: number;
  rerank_score: number;
  snippet: string;
}

export interface Predictions {
  urgency: string;
  urgency_confidence: number;
  issue_type: string;
  issue_type_confidence: number;
  action_recommendation: string;
  action_confidence: number;
  is_regression: boolean;
  regression_confidence: number;
}

export interface SearchResponse {
  query_text: string;
  query_issue_id: string | null;
  similar_issues: SimilarIssue[];
  predictions: Predictions;
  suggested_next_action: string;
  citation_issue_ids: string[];
}

async function apiFetch<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const body = await res.json();
  if (!res.ok) throw new Error((body as { detail?: string }).detail ?? res.statusText);
  return body as T;
}

export function search(req: SearchRequest): Promise<SearchResponse> {
  return apiFetch("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export function ingest(repo: string, index = true): Promise<{ status: string; repo: string }> {
  return apiFetch("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo, index }),
  });
}
