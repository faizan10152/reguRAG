# ReguRAG

Production-style bilingual RAG assistant for AI Act, GDPR, and German AI governance research.

The goal is not to build another "chat with PDF" demo. ReguRAG is designed as a portfolio project for junior AI engineer roles in Germany: reproducible ingestion, source-aware chunking, hybrid retrieval, reranking, citation enforcement, refusal behavior, evaluation, observability, and CI quality gates.

## Problem Statement

German companies adopting AI in HR, finance, healthcare, and business operations need reliable answers to questions like:

- Does this AI use case look high-risk under the EU AI Act?
- What GDPR issues appear if we use customer support data for model training?
- What documentation or AI literacy expectations apply to employees using AI tools?
- Which official source supports the answer?
- When should the system refuse because the evidence is not in the corpus?

ReguRAG answers from official EU and German sources and must cite the retrieved evidence. It is decision support, not legal advice.

## Current MVP

This first version contains the production-shaped foundation:

- Standalone repository outside the parent workspace.
- Source manifest for official EU/German regulatory sources.
- Reproducible downloader for raw source files.
- Text extraction and article-aware chunking utilities.
- Transparent BM25 retrieval baseline.
- Dense retrieval commands backed by Qdrant and sentence-transformers.
- Reciprocal rank fusion utility for future hybrid retrieval.
- Citation extraction and validation helpers.
- Retrieval evaluation over an approved 38-question golden set.
- Source-level and citation-level retrieval metrics: Recall@K, Precision@K, Hit@K, MRR.
- FastAPI shell with retrieval-only `/query` endpoint.
- Docker Compose with API and Qdrant.
- GitHub Actions CI for tests and linting.

## Target Architecture

```mermaid
flowchart LR
  A["Official EU/German sources"] --> B["Downloader + source manifest"]
  B --> C["Text extraction"]
  C --> D["Article-aware chunking"]
  D --> E["BM25 baseline"]
  D --> F["Dense embeddings in Qdrant"]
  E --> G["Hybrid retrieval + RRF"]
  F --> G
  G --> H["Cross-encoder reranker"]
  H --> I["LLM answer generator"]
  I --> J["Citation validator + refusal gate"]
  J --> K["FastAPI / UI"]
  J --> L["Evaluation + CI gate"]
  J --> M["Langfuse traces"]
```

## Quick Start

Recommended local setup:

```bash
cd /Users/faizan/Desktop/regurag
uv venv --python 3.11
uv pip install -e ".[api,dev,ingestion]"
uv run pre-commit install
uv run pytest
```

The pre-commit hook blocks commits when project requirements fail:

- no staged `.env` files
- no staged generated `data/raw/*` or `data/processed/*` artifacts, except `.gitkeep`
- no obvious API keys or private keys in text files
- `ruff check .` must pass
- `pytest` must pass

Run the retrieval-only API:

```bash
uv run uvicorn regurag.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

Run local search before real ingestion:

```bash
uv run regurag search --chunks data/processed/chunks.jsonl --query "AI systems for employment screening"
```

Download and chunk official sources:

```bash
uv run regurag download --manifest configs/source_manifest.json --raw-dir data/raw
uv run regurag chunk --manifest configs/source_manifest.json --raw-dir data/raw --out data/processed/chunks.jsonl
```

Docker:

```bash
docker compose up --build
```

Dense retrieval:

```bash
docker compose up -d qdrant
uv run --extra rag regurag dense-index --chunks data/processed/chunks.jsonl
uv run --extra rag regurag dense-search --query "Can we train a model on customer support tickets?"
uv run --extra rag regurag compare-retrieval --chunks data/processed/chunks.jsonl --query "Can we train a model on customer support tickets?"
```

The default embedding model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` because it is fast enough for local iteration. For a stronger portfolio run, switch to:

```bash
uv run --extra rag regurag dense-index \
  --chunks data/processed/chunks.jsonl \
  --model BAAI/bge-m3 \
  --batch-size 4
```

Retrieval evaluation:

```bash
uv run regurag eval-retrieval \
  --chunks data/processed/chunks.jsonl \
  --questions data/eval/golden_questions_v1.jsonl \
  --retrievers bm25
```

Full local BM25 + dense + hybrid evaluation without Docker:

```bash
uv run --extra rag regurag eval-retrieval \
  --chunks data/processed/chunks.jsonl \
  --questions data/eval/golden_questions_v1.jsonl \
  --retrievers bm25,dense,hybrid \
  --qdrant-path .qdrant/local \
  --output-md reports/retrieval_eval_latest.md \
  --output-json reports/retrieval_eval_latest.json
```

See [docs/retrieval_evaluation.md](docs/retrieval_evaluation.md) for metric definitions and the current baseline snapshot.

## Why BM25 First?

BM25 is a keyword retrieval algorithm. It is not "old-fashioned"; it is still useful in legal and regulatory RAG because exact terms matter:

- Article numbers
- Acronyms
- German legal phrases
- Regulator names
- Domain terms like "high-risk", "AI literacy", "legal basis"

Dense vector search is better at semantic similarity, but it can miss exact terms. The production version will combine BM25 and dense vectors, then rerank.

## Next Milestones

1. Tune dense retrieval and hybrid fusion using the golden set.
2. Add cross-encoder reranking.
3. Add LLM answer generation through LiteLLM.
4. Enforce structured answer JSON with citations.
5. Build a 100-question golden evaluation set with exact citation labels.
6. Add Ragas and custom citation metrics.
7. Add Langfuse tracing.
8. Build a simple bilingual UI.
9. Record a 3-minute demo and publish a project page.

## Portfolio Pitch

Built a bilingual production-style RAG assistant for AI Act/GDPR compliance in German regulated industries, with source-aware ingestion, BM25 baseline retrieval, planned hybrid dense retrieval, reranking, citation validation, refusal handling, evaluation metrics, CI, and Dockerized deployment.
