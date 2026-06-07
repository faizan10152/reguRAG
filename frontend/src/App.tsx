import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  ExternalLink,
  FileText,
  Loader2,
  Search,
  ShieldCheck,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

type Retriever = "bm25" | "hybrid-rerank";

type Evidence = {
  rank: number;
  score: number;
  retriever: string;
  citation_label: string;
  source_id: string;
  title: string;
  url: string | null;
  section_heading: string | null;
  snippet: string;
};

type AnswerResponse = {
  mode: string;
  retriever: string;
  llm_model: string;
  supported: boolean;
  guardrail_triggered: boolean;
  answer: {
    answer: string;
    citations: string[];
    confidence: "low" | "medium" | "high";
    unsupported_claims: string[];
    should_refuse: boolean;
    refusal_reason: string | null;
  };
  retrieved_results: Evidence[];
  missing_citations: string[];
};

type EvaluationSnapshot = {
  answerable_supported_rate: number;
  citation_validity_rate: number;
  expected_refusal_success_rate: number;
  expected_citation_hit_rate: number;
  mean_latency_seconds: number;
  source_recall_at_k: number;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

const EXAMPLES = [
  "Is AI CV screening high-risk under the AI Act?",
  "Does the GDPR require a DPIA for every AI system that processes personal data?",
  "What is the exact penalty under Article 999 of the EU AI Act?",
  "Welche Datenschutzregeln sind relevant, wenn ein deutsches Unternehmen Beschäftigtendaten für ein KI-System verarbeitet?",
];

const DEFAULT_SNAPSHOT: EvaluationSnapshot = {
  answerable_supported_rate: 0.667,
  citation_validity_rate: 1.0,
  expected_refusal_success_rate: 1.0,
  expected_citation_hit_rate: 0.167,
  mean_latency_seconds: 75.36,
  source_recall_at_k: 0.556,
};

export function App() {
  const [question, setQuestion] = useState(EXAMPLES[0]);
  const [retriever, setRetriever] = useState<Retriever>("hybrid-rerank");
  const [showEvidence, setShowEvidence] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [snapshot, setSnapshot] = useState<EvaluationSnapshot>(DEFAULT_SNAPSHOT);

  useEffect(() => {
    fetch(`${API_BASE}/evaluation-snapshot`)
      .then((response) => (response.ok ? response.json() : DEFAULT_SNAPSHOT))
      .then((payload) => setSnapshot(payload))
      .catch(() => setSnapshot(DEFAULT_SNAPSHOT));
  }, []);

  const status = useMemo(() => getStatus(result), [result]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError("Enter a regulatory question first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          retriever,
          top_k: 5,
          candidate_k: 20,
          disable_json_mode: true,
          max_context_chars: 4500,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Request failed with status ${response.status}`);
      }

      setResult(await response.json());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown request error.");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <ShieldCheck aria-hidden="true" />
          <div>
            <h1>AI Regulation Evidence Workbench</h1>
            <p>Grounded answers for AI Act, GDPR, and German AI governance sources</p>
          </div>
        </div>
        <div className="system-status" title="Current local answer-evaluation snapshot">
          <Activity aria-hidden="true" />
          <span>Evaluated local RAG baseline</span>
        </div>
      </header>

      <main className="workspace">
        <section className="query-panel">
          <form onSubmit={submit} className="query-form">
            <div className="section-heading">
              <Search aria-hidden="true" />
              <h2>Ask a Regulatory Question</h2>
            </div>

            <label htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={5}
            />

            <div className="examples" aria-label="Example questions">
              {EXAMPLES.map((example) => (
                <button
                  className="example-button"
                  type="button"
                  key={example}
                  onClick={() => setQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>

            <div className="settings-row">
              <label className="setting">
                Retriever
                <select
                  value={retriever}
                  onChange={(event) => setRetriever(event.target.value as Retriever)}
                >
                  <option value="hybrid-rerank">Hybrid rerank</option>
                  <option value="bm25">BM25 baseline</option>
                </select>
              </label>

              <label className="toggle">
                <input
                  type="checkbox"
                  checked={showEvidence}
                  onChange={(event) => setShowEvidence(event.target.checked)}
                />
                Show evidence
              </label>

              <button className="primary-button" type="submit" disabled={isLoading}>
                {isLoading ? <Loader2 className="spin" aria-hidden="true" /> : <Search aria-hidden="true" />}
                <span>{isLoading ? "Running" : "Run"}</span>
              </button>
            </div>
          </form>

          {error && (
            <div className="error-box" role="alert">
              <AlertTriangle aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <section className="answer-surface">
            <div className="answer-header">
              <div className="section-heading">
                <FileText aria-hidden="true" />
                <h2>Answer</h2>
              </div>
              <StatusPill label={status.label} tone={status.tone} />
            </div>

            {result ? (
              <>
                <p className="answer-text">{result.answer.answer}</p>
                <div className="meta-grid">
                  <Metric label="Confidence" value={result.answer.confidence} />
                  <Metric label="Retriever" value={result.retriever} />
                  <Metric label="Model" value={shortModelName(result.llm_model)} />
                  <Metric label="Guardrail" value={String(result.guardrail_triggered)} />
                </div>

                {result.answer.refusal_reason && (
                  <div className="notice">
                    <AlertTriangle aria-hidden="true" />
                    <span>{result.answer.refusal_reason}</span>
                  </div>
                )}

                <CitationList citations={result.answer.citations} missing={result.missing_citations} />
              </>
            ) : (
              <div className="empty-state">
                <Database aria-hidden="true" />
                <p>Run a question to inspect the grounded answer and citation checks.</p>
              </div>
            )}
          </section>
        </section>

        <aside className="side-panel">
          <section className="snapshot-surface">
            <div className="section-heading">
              <Activity aria-hidden="true" />
              <h2>Evaluation Snapshot</h2>
            </div>
            <div className="snapshot-grid">
              <Metric label="Answerable supported" value={formatRate(snapshot.answerable_supported_rate)} />
              <Metric label="Citation validity" value={formatRate(snapshot.citation_validity_rate)} />
              <Metric label="Refusal success" value={formatRate(snapshot.expected_refusal_success_rate)} />
              <Metric label="Exact citation hit" value={formatRate(snapshot.expected_citation_hit_rate)} />
              <Metric label="Source recall@5" value={formatRate(snapshot.source_recall_at_k)} />
              <Metric label="Mean latency" value={`${snapshot.mean_latency_seconds.toFixed(1)}s`} />
            </div>
          </section>

          {showEvidence && (
            <section className="evidence-surface">
              <div className="section-heading">
                <Database aria-hidden="true" />
                <h2>Retrieved Evidence</h2>
              </div>
              {result?.retrieved_results.length ? (
                <div className="evidence-list">
                  {result.retrieved_results.map((item) => (
                    <EvidenceItem key={`${item.rank}-${item.citation_label}`} item={item} />
                  ))}
                </div>
              ) : (
                <div className="empty-state compact">
                  <p>No retrieved chunks yet.</p>
                </div>
              )}
            </section>
          )}
        </aside>
      </main>
    </div>
  );
}

function StatusPill({ label, tone }: { label: string; tone: "green" | "amber" | "red" | "neutral" }) {
  return <span className={`status-pill ${tone}`}>{label}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CitationList({ citations, missing }: { citations: string[]; missing: string[] }) {
  if (!citations.length && !missing.length) {
    return <p className="muted">No citations returned.</p>;
  }

  return (
    <div className="citation-block">
      <h3>Citations</h3>
      <div className="citation-list">
        {citations.map((citation) => (
          <span className="citation-token" key={citation}>
            <CheckCircle2 aria-hidden="true" />
            {citation}
          </span>
        ))}
        {missing.map((citation) => (
          <span className="citation-token missing" key={citation}>
            <AlertTriangle aria-hidden="true" />
            {citation}
          </span>
        ))}
      </div>
    </div>
  );
}

function EvidenceItem({ item }: { item: Evidence }) {
  return (
    <article className="evidence-item">
      <div className="evidence-top">
        <span className="rank">#{item.rank}</span>
        <span className="source">{item.source_id}</span>
        <span className="score">{item.score.toFixed(3)}</span>
      </div>
      <h3>{item.title}</h3>
      <p className="section">{item.section_heading || "No section heading"}</p>
      <p className="snippet">{item.snippet}</p>
      <div className="evidence-footer">
        <code>{item.citation_label}</code>
        {item.url && (
          <a href={item.url} target="_blank" rel="noreferrer" aria-label={`Open source for ${item.title}`}>
            <ExternalLink aria-hidden="true" />
          </a>
        )}
      </div>
    </article>
  );
}

function getStatus(result: AnswerResponse | null): { label: string; tone: "green" | "amber" | "red" | "neutral" } {
  if (!result) {
    return { label: "Idle", tone: "neutral" };
  }
  if (result.supported) {
    return { label: "Supported", tone: "green" };
  }
  if (result.answer.should_refuse) {
    return { label: "Refused", tone: "amber" };
  }
  if (result.guardrail_triggered) {
    return { label: "Guardrail", tone: "red" };
  }
  return { label: "Unsupported", tone: "red" };
}

function formatRate(value: number) {
  return value.toFixed(3);
}

function shortModelName(model: string) {
  return model.replace("ollama/", "");
}
