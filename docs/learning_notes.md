# Learning Notes

## RAG Is More Than Generation

RAG has three separate jobs:

1. Find the right evidence.
2. Generate an answer from that evidence.
3. Prove the answer is grounded.

Many demos only do step 2. Production systems spend most of their effort on steps 1 and 3.

## Do

- Keep raw sources separate from processed chunks.
- Store metadata on every chunk.
- Build a lexical baseline before embeddings.
- Evaluate retrieval before judging answer quality.
- Include refusal examples in your test set.
- Version prompts like code.
- Log the retrieved chunks for every answer.
- Write down known limitations.

## Do Not

- Do not chunk legal text without preserving source and section metadata.
- Do not claim legal certainty from a RAG answer.
- Do not use only semantic search for article-number-heavy questions.
- Do not evaluate only with nice examples.
- Do not hide failures. Good portfolio projects explain tradeoffs.

## Common Failure Modes

### Bad Extraction

The index contains navigation menus, cookie notices, or footer text.

Fix:

- Inspect random chunks.
- Add boilerplate filters.
- Prefer structured source formats when available.

### Bad Chunking

The answer needs two adjacent paragraphs, but they were split apart.

Fix:

- Use overlap.
- Preserve section headings.
- Try parent-child retrieval later.

### Retrieval Misses

The answer exists, but top-k does not include it.

Fix:

- Increase candidate pool.
- Add BM25.
- Add dense retrieval.
- Use hybrid fusion.
- Add reranking.

### Hallucinated Answer

The LLM gives a plausible but unsupported answer.

Fix:

- Structured output.
- Citation validation.
- Refusal gate.
- Faithfulness evaluation.

### Evaluation Blind Spot

The app works on your favorite examples but fails on realistic ones.

Fix:

- Build a golden set with factual, scenario, multi-hop, bilingual, and unanswerable questions.

## Pre-Commit Checks

Pre-commit checks run before a commit is created. They are useful because they catch mistakes while the context is still fresh.

This project uses the `pre-commit` framework with local hooks:

- project guard script
- Ruff linting
- pytest

The custom guard blocks common portfolio-repo mistakes:

- committing `.env`
- committing generated raw/processed data artifacts
- committing obvious API keys or private keys

Install once:

```bash
uv run pre-commit install
```

Run manually:

```bash
uv run pre-commit run --all-files
```

Best practice: keep fast checks in pre-commit. Expensive model downloads, dense indexing, and long RAG evaluations should run in CI or as explicit commands, not before every local commit.

## Dense Retrieval

Dense retrieval uses embeddings. An embedding model converts text into a vector of numbers. Texts with similar meanings should have vectors that are close together.

Pros:

- handles paraphrases better than BM25
- helps with semantic queries
- can improve cross-lingual retrieval

Cons:

- needs model downloads
- needs a vector database or vector index
- can miss exact legal terms
- model changes require reindexing

For this project, dense retrieval is implemented with sentence-transformers and Qdrant. BM25 remains in place because legal retrieval often needs exact terms.
