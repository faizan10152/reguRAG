# Dense Retrieval

Dense retrieval turns text into vectors. Similar meanings should land near each other in vector space, even when the words are not identical.

BM25 answers this question:

> Which chunks contain the same important words as the query?

Dense retrieval answers this question:

> Which chunks are semantically close to the query?

Both are useful. Legal and regulatory RAG needs exact terms and semantic matching.

## Current Implementation

Files:

- `src/regurag/embeddings/encoder.py`
- `src/regurag/retrieval/dense_qdrant.py`
- `src/regurag/cli.py`

The flow:

1. Load chunks from `data/processed/chunks.jsonl`.
2. Encode each chunk with a sentence-transformers model.
3. Store vectors in Qdrant.
4. Encode the user query with the same model.
5. Ask Qdrant for nearest vectors.
6. Convert Qdrant payloads back into `Chunk` and `RetrievalResult`.

## Commands

Start Qdrant:

```bash
docker compose up -d qdrant
```

Index chunks:

```bash
uv run --extra rag regurag dense-index \
  --chunks data/processed/chunks.jsonl
```

Search dense index:

```bash
uv run --extra rag regurag dense-search \
  --query "Can we train a model on customer support tickets?"
```

Compare BM25, dense, and RRF fusion:

```bash
uv run --extra rag regurag compare-retrieval \
  --chunks data/processed/chunks.jsonl \
  --query "Can we train a model on customer support tickets?"
```

For one-process smoke tests, Qdrant also supports local in-memory mode through `--qdrant-location :memory:`. Do not use that for normal CLI indexing/search across separate commands, because the index disappears when the process exits.

If Docker Desktop is not running, use persistent local Qdrant storage:

```bash
uv run --extra rag regurag dense-index \
  --chunks data/processed/chunks.jsonl \
  --qdrant-path .qdrant/local

uv run --extra rag regurag dense-search \
  --qdrant-path .qdrant/local \
  --query "Can we train a model on customer support tickets?"
```

`.qdrant/` is ignored by git because it is generated local index state.

## Model Choice

Default:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This is small and fast for local iteration.

Portfolio target:

```text
BAAI/bge-m3
```

This is stronger for multilingual retrieval, but heavier. Use a smaller batch size:

```bash
uv run --extra rag regurag dense-index \
  --chunks data/processed/chunks.jsonl \
  --model BAAI/bge-m3 \
  --batch-size 4
```

## Failure Modes

Dense retrieval can still fail:

- The embedding model may not understand legal nuance.
- Long chunks can blur multiple topics into one vector.
- Cross-lingual queries can retrieve the right topic in the wrong jurisdiction or language.
- If the query is about an exact article number, BM25 may beat dense retrieval.
- Embedding model changes require reindexing.

## Best Practices

- Keep BM25 as a baseline.
- Evaluate dense retrieval separately before fusing it.
- Store source metadata in Qdrant payloads.
- Version the embedding model name.
- Rebuild the index whenever chunks or embedding model change.
- Compare examples where BM25 wins and where dense retrieval wins.

## Next Improvement

Add an evaluation command that measures:

- BM25 source recall
- dense source recall
- RRF fused source recall

Then we can show whether dense retrieval improves the weak baseline questions.
