# Architecture Notes

## Main Design Principle

The system should make unsupported answers difficult to produce.

That means retrieval, generation, citation validation, and evaluation are separate components. This is more work than a demo, but it makes failure modes easier to inspect.

## Components

### Source Manifest

`configs/source_manifest.json` is the registry of official sources. Each source has:

- `source_id`
- title
- URL
- language
- jurisdiction
- source type
- domain tags

Best practice: never silently scrape random pages. A production RAG system needs to know what it ingested, when it ingested it, and where the text came from.

### Ingestion

The downloader stores raw source files and sidecar metadata with a SHA-256 checksum.

Why this matters:

- You can reproduce the index.
- You can detect source changes.
- You can debug bad answers against the exact indexed text.

### Chunking

Chunking is not just splitting every 650 words. For legal and regulatory text, headings, articles, annexes, and sections are part of the meaning.

Current MVP:

- Detects article/section-like headings.
- Adds source metadata to every chunk.
- Uses overlap to avoid cutting context at chunk boundaries.

Possible issue:

- HTML pages can include navigation text, cookie banners, or footer noise. We will improve extraction after seeing the real downloaded files.

### Retrieval

Current MVP:

- BM25 lexical retrieval.
- Qdrant-backed dense retrieval.
- Reciprocal rank fusion utility.
- Source-level retrieval evaluation for BM25, dense, and hybrid runs.

Planned:

- Hybrid retrieval endpoint.
- Cross-encoder reranking.

Do not skip the baseline. Without a baseline, you cannot prove that embeddings improved the system.

### Grounding

The answer generator will eventually output structured JSON:

```json
{
  "answer": "...",
  "claims": [],
  "citations": [],
  "confidence": "medium",
  "unsupported_claims": [],
  "refusal_reason": null
}
```

The citation validator should reject answers that cite chunks not retrieved for the question.

### Evaluation

Evaluation has two layers:

- Retrieval quality: did we retrieve the right chunks?
- Answer quality: did the generated answer stay faithful to those chunks?

Initial metrics:

- Recall@K
- Precision@K
- Hit@K
- MRR
- citation accuracy
- refusal accuracy

Current retrieval evaluation has source-level labels for all answerable questions and exact citation labels for the first 12 high-value questions. Source-level scoring gives a fast signal on whether the retriever found the right regulation or regulator source. Citation-level scoring is stricter and checks whether the exact supporting chunk reached the top K.

Later:

- Ragas faithfulness
- answer relevancy
- context precision
- context recall

### Observability

Production RAG needs traces:

- user query
- query rewrite
- retrieved chunks
- reranked chunks
- prompt version
- model
- token count
- latency
- answer
- citations
- refusal decision

This is why Langfuse is on the roadmap.
