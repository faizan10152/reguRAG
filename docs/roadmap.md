# Roadmap

## Week 1 Goal

Create a credible version that can go on a CV and GitHub profile:

- local API works
- ingestion is reproducible
- retrieval works on official sources
- citations are visible
- evaluation plan exists
- README explains business value

## Day 1: Repo and Foundations

- Create standalone repo.
- Add Python project structure.
- Add official source manifest.
- Add downloader, chunker, BM25 baseline, tests, Docker, CI.

Deliverable:

- `uv run pytest` passes.
- `uv run regurag search ...` returns sample retrieval results.

## Day 2: Real Corpus Ingestion

- Download sources.
- Inspect raw extracted text.
- Remove obvious boilerplate/noise.
- Generate `data/processed/chunks.jsonl`.

Deliverable:

- 500+ source-aware chunks.
- Each chunk has source URL, language, jurisdiction, heading, and chunk id.

## Day 3: Dense Retrieval

- Add Qdrant collection.
- Use multilingual embeddings such as `BAAI/bge-m3`.
- Compare BM25 vs dense retrieval on 20 questions.

Deliverable:

- Retrieval report with examples where BM25 wins and where dense retrieval wins.

## Day 4: Hybrid Retrieval and Reranking

- Combine BM25 and dense retrieval with reciprocal rank fusion.
- Add cross-encoder reranker.
- Log before/after top-k quality.

Deliverable:

- Ablation table: BM25, dense, hybrid, hybrid plus rerank.

## Day 5: Answer Generation and Citation Enforcement

- Add LiteLLM provider adapter.
- Force structured JSON answer output.
- Validate citations against retrieved chunks.
- Refuse unsupported answers.

Deliverable:

- Demo questions with supported answers and refusal examples.

## Day 6: Evaluation

- Build 50-question initial golden set.
- Measure Recall@5, MRR, citation accuracy, refusal accuracy.
- Add CI quality gate for retrieval metrics.

Deliverable:

- `uv run regurag eval ...` produces a report.

## Day 7: Portfolio Packaging

- Add simple UI.
- Polish README.
- Add architecture diagram.
- Record 3-minute demo.
- Write CV bullets.

Deliverable:

- GitHub repo and demo-ready local app.

## After Week 1

- Expand golden set to 100-150 questions.
- Add Langfuse tracing.
- Add Ragas faithfulness scoring.
- Add German/English query rewriting.
- Add source version monitoring.
- Add public deployment only after API key and rate-limit strategy are clear.
