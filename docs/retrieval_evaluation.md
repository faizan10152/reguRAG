# Retrieval Evaluation

ReguRAG evaluates retrieval before answer generation. This is important because a polished LLM answer is still unreliable if the retriever failed to bring the right evidence into context.

## What The Current Benchmark Measures

The approved benchmark is `data/eval/golden_questions_v1.jsonl`.

It contains 38 questions:

- 32 answerable questions with expected source documents
- 6 refusal questions where no corpus source is expected to fully answer the question
- 12 answerable questions with exact expected citation labels
- English and German questions
- HR, finance, healthcare, business operations, legal basics, cross-lingual, and refusal domains
- structural difficulty labels such as semantic paraphrase, multi-source, source tension, scope ambiguity, negative factual, and out-of-corpus

The current evaluation has two layers:

- source-level metrics for all answerable questions
- citation-level metrics for the 12 manually labeled questions

Source-level evaluation gives credit when the top K results contain expected source IDs such as `gdpr_en` or `eu_ai_act_en`. Citation-level evaluation is stricter and gives credit only when the exact supporting `source_id:chunk_id` label appears in the top K.

## Metrics

| Metric | Meaning |
| --- | --- |
| Recall@K | Fraction of expected source documents found in the top K retrieved chunks. |
| Precision@K | Fraction of top K retrieved chunks whose source is expected. |
| Hit@K | Whether at least one expected source appears in the top K. |
| MRR | Reciprocal rank of the first expected source. Higher means the first useful source appears earlier. |
| Citation Recall@K | Fraction of exact expected citation labels found in the top K retrieved chunks. |
| Citation Hit@K | Whether at least one exact expected citation appears in the top K. |
| Citation MRR | Reciprocal rank of the first exact expected citation. |

Refusal questions are excluded from source recall because they intentionally have no expected source. For now, the report shows which distractor sources each retriever returned. Later, refusal quality will be measured at the answer layer with confidence thresholds and citation validation.

## Running The Evaluation

BM25-only evaluation:

```bash
make eval-retrieval
```

Full local evaluation with the persistent Qdrant index:

```bash
make eval-retrieval-local
```

Equivalent direct command:

```bash
uv run --extra rag regurag eval-retrieval \
  --chunks data/processed/chunks.jsonl \
  --questions data/eval/golden_questions_v1.jsonl \
  --retrievers bm25,dense,hybrid \
  --qdrant-path .qdrant/local \
  --top-k 5 \
  --candidate-k 20 \
  --output-md reports/retrieval_eval_latest.md \
  --output-json reports/retrieval_eval_latest.json
```

`top-k` is the result window used for metrics. `candidate-k` is the larger retrieval window used before fusion.

## Current Local Baseline

Run date: 2026-05-31

Corpus: 843 chunks from the official EU/German source manifest.

Dense model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Retriever | Answerable Questions | Source Recall@5 | Source Hit@5 | Source MRR | Citation-Labeled Questions | Citation Recall@5 | Citation Hit@5 | Citation MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 32 | 0.641 | 0.938 | 0.769 | 12 | 0.083 | 0.083 | 0.017 |

Previous MiniLM dense and hybrid source-level snapshot:

| Retriever | Answerable Questions | Source Recall@5 | Source Hit@5 | Source MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense Qdrant | 32 | 0.469 | 0.750 | 0.555 |
| Hybrid RRF | 32 | 0.625 | 0.906 | 0.724 |

BM25 is currently strongest. That is plausible for this corpus because legal and regulatory queries often contain exact terms, article numbers, acronyms, and domain phrases. Dense retrieval is still useful, but the first local model is intentionally small and not yet tuned.

## What This Tells Us

The strongest current gaps are:

- German and cross-lingual retrieval need improvement.
- Dense retrieval misses some exact GDPR and German legal terms.
- Hybrid fusion needs tuning because simple reciprocal rank fusion does not yet beat BM25.
- Source-level labels are useful but too coarse; exact citation labels reveal much harder retrieval misses.
- Refusal questions need answer-level scoring, not only retrieval inspection.

## Next Improvements

1. Expand exact citation labels from 12 to all answerable questions.
2. Compare stronger embedding models such as `BAAI/bge-m3` when Wi-Fi is available.
3. Tune hybrid retrieval with larger candidate windows and source diversity.
4. Add a cross-encoder reranker after hybrid retrieval.
5. Add confidence thresholds and answer-level refusal evaluation.
