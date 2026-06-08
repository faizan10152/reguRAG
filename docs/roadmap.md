# Roadmap

This document tracks what has already been built and what is still planned. The API, grounded answer generation, and React workbench are complete for the current local MVP; they are no longer future milestones.

## Completed MVP

- Standalone Python repository with tests, linting, pre-commit checks, Docker Compose, and reproducible commands.
- Official EU/German source manifest, downloader, text extraction, and source-aware chunking.
- BM25 lexical retrieval baseline.
- Dense Qdrant retrieval with multilingual embedding models.
- Hybrid retrieval with reciprocal rank fusion.
- Cross-encoder reranking.
- Golden-question retrieval evaluation with source-level and exact-citation metrics.
- Grounded answer generation through LiteLLM and local Ollama models.
- Citation aliasing, citation validation, unsupported-claim checks, and refusal behavior.
- Answer-level evaluation for structured output, refusal behavior, citation validity, and latency.
- FastAPI endpoints for retrieval-only search, grounded answers, and evaluation snapshots.
- React/TypeScript evidence workbench with answer, citation, evidence, and metric panels.
- GitHub-facing README with screenshots and a short local demo video.

## Current Next Milestones

1. Improve legal chunking with article-aware and annex-aware splits.
2. Add parent-child retrieval so precise chunks can retrieve exact evidence while larger parent spans provide enough context for generation.
3. Expand exact citation labels from 12 high-value questions to all answerable golden questions.
4. Try a stronger multilingual reranker such as `BAAI/bge-reranker-v2-m3`.
5. Add Langfuse tracing for retrieval candidates, prompt context, model responses, citation validation, and latency.
6. Expand answer-level evaluation and add manual semantic correctness labels.

## Longer-Term Options

- Add German/English query rewriting for better cross-lingual retrieval.
- Add source version monitoring for updated AI Act, GDPR, and German regulator pages.
- Add a hosted frontend-only demo page if the repo needs a public link.
- Add a hosted backend only after compute, model, API-key, and rate-limit strategy are clear.
