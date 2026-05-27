import { useState, FormEvent } from "react";
import { search, ingest, SearchResponse, Predictions } from "./api";

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------

const URGENCY_COLORS: Record<string, string> = {
  critical_bug: "badge-red",
  high: "badge-orange",
  medium: "badge-yellow",
  low: "badge-green",
  enhancement: "badge-blue",
  question: "badge-purple",
};

function UrgencyBadge({ value, conf }: { value: string; conf: number }) {
  const cls = URGENCY_COLORS[value] ?? "badge-gray";
  return (
    <span className={`badge ${cls}`}>
      {value.replace("_", " ")} <span className="conf">{pct(conf)}</span>
    </span>
  );
}

function Badge({ label, value, conf }: { label: string; value: string; conf: number }) {
  return (
    <span className="badge badge-gray">
      <span className="badge-label">{label}</span>{" "}
      {value.replace(/_/g, " ")} <span className="conf">{pct(conf)}</span>
    </span>
  );
}

function pct(n: number) {
  return `${Math.round(n * 100)}%`;
}

// ---------------------------------------------------------------------------
// Predictions panel
// ---------------------------------------------------------------------------

function PredictionsPanel({ p, action }: { p: Predictions; action: string }) {
  return (
    <div className="predictions">
      <div className="predictions-badges">
        <UrgencyBadge value={p.urgency} conf={p.urgency_confidence} />
        <Badge label="type" value={p.issue_type} conf={p.issue_type_confidence} />
        <Badge label="action" value={p.action_recommendation} conf={p.action_confidence} />
        {p.is_regression && (
          <span className="badge badge-orange">regression</span>
        )}
      </div>
      <p className="suggested-action">
        <strong>Suggested:</strong> {action.replace(/_/g, " ")}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Similar issue card
// ---------------------------------------------------------------------------

function IssueCard({
  issue,
  cited,
}: {
  issue: SearchResponse["similar_issues"][0];
  cited: boolean;
}) {
  const repoMatch = issue.url.match(/github\.com\/([^/]+\/[^/]+)\//);
  const repo = repoMatch ? repoMatch[1] : "";

  return (
    <div className={`issue-card ${cited ? "cited" : ""}`}>
      <div className="issue-header">
        <a href={issue.url} target="_blank" rel="noopener noreferrer" className="issue-title">
          {issue.title}
        </a>
        {cited && <span className="cited-dot" title="Cited in intelligence pack" />}
      </div>
      <div className="issue-meta">
        {repo && <span className="repo-tag">{repo}</span>}
        <span className="score-pill" title="Rerank score">
          ↑ {issue.rerank_score.toFixed(3)}
        </span>
        <span className="score-pill dim" title="Retrieval score">
          {issue.score.toFixed(3)}
        </span>
      </div>
      {issue.snippet && <p className="snippet">{issue.snippet}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ingest drawer
// ---------------------------------------------------------------------------

function IngestDrawer({ onClose }: { onClose: () => void }) {
  const [repo, setRepo] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!repo.trim()) return;
    setLoading(true);
    setStatus(null);
    try {
      const res = await ingest(repo.trim());
      setStatus(`✓ ${res.repo} — sync started in background`);
      setRepo("");
    } catch (err) {
      setStatus(`✗ ${err instanceof Error ? err.message : "failed"}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <h2>Ingest repo</h2>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>
        <p className="drawer-hint">
          Syncs GitHub issues and indexes embeddings in the background.
          The repo must be accessible with your <code>GITHUB_TOKEN</code>.
        </p>
        <form onSubmit={handleSubmit} className="ingest-form">
          <input
            className="input"
            placeholder="owner/repo"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            disabled={loading}
          />
          <button className="btn-primary" type="submit" disabled={loading || !repo.trim()}>
            {loading ? "Starting…" : "Sync"}
          </button>
        </form>
        {status && (
          <p className={`ingest-status ${status.startsWith("✓") ? "ok" : "err"}`}>{status}</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main app
// ---------------------------------------------------------------------------

export default function App() {
  const [query, setQuery] = useState("");
  const [repo, setRepo] = useState("");
  const [limit, setLimit] = useState(10);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showIngest, setShowIngest] = useState(false);

  const isUrl = query.trim().startsWith("https://github.com/");

  async function handleSearch(e: FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const req = isUrl
        ? { issue_url: query.trim(), repo: repo || undefined, limit }
        : { query: query.trim(), repo: repo || undefined, limit };
      const res = await search(req);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  const citedSet = new Set(result?.citation_issue_ids ?? []);

  return (
    <>
      <header className="app-header">
        <div className="header-inner">
          <span className="logo">
            <span className="logo-icon">◎</span> Issue Zero
          </span>
          <button className="btn-ghost" onClick={() => setShowIngest(true)}>
            + Ingest repo
          </button>
        </div>
      </header>

      <main className="app-main">
        <section className="search-section">
          <h1 className="headline">Find related issues instantly</h1>
          <p className="subline">
            Paste a bug report or GitHub issue URL — get similar past issues and triage predictions.
          </p>

          <form onSubmit={handleSearch} className="search-form">
            <textarea
              className="query-input"
              placeholder="Describe the bug, or paste a GitHub issue URL (https://github.com/owner/repo/issues/123)…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={5}
              disabled={loading}
            />
            <div className="search-controls">
              <input
                className="input repo-input"
                placeholder="Filter by repo (owner/name)"
                value={repo}
                onChange={(e) => setRepo(e.target.value)}
                disabled={loading}
              />
              <div className="limit-control">
                <label htmlFor="limit">Limit</label>
                <input
                  id="limit"
                  type="number"
                  min={1}
                  max={50}
                  className="input limit-input"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                  disabled={loading}
                />
              </div>
              <button
                className="btn-primary search-btn"
                type="submit"
                disabled={loading || !query.trim()}
              >
                {loading ? "Searching…" : isUrl ? "Resolve & Search" : "Search"}
              </button>
            </div>
          </form>

          {error && <div className="error-box">{error}</div>}
        </section>

        {result && (
          <section className="results-section">
            <div className="results-header">
              <h2 className="results-title">
                {result.similar_issues.length} similar issue
                {result.similar_issues.length !== 1 ? "s" : ""}
              </h2>
              <span className="query-echo">"{result.query_text.slice(0, 80)}{result.query_text.length > 80 ? "…" : ""}"</span>
            </div>

            <PredictionsPanel p={result.predictions} action={result.suggested_next_action} />

            <div className="issue-list">
              {result.similar_issues.length === 0 ? (
                <p className="empty">No similar issues found. Try ingesting some repos first.</p>
              ) : (
                result.similar_issues.map((issue) => (
                  <IssueCard key={issue.id} issue={issue} cited={citedSet.has(issue.id)} />
                ))
              )}
            </div>
          </section>
        )}
      </main>

      {showIngest && <IngestDrawer onClose={() => setShowIngest(false)} />}
    </>
  );
}
