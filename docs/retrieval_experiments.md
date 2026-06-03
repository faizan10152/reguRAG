# Retrieval Experiments

This log records retrieval experiments in a compact format so later portfolio reports can cite what was tried, what changed, and what we learned.

## Benchmark Setup

- Corpus: 843 chunks from official EU/German sources.
- Golden set: 38 questions in `data/eval/golden_questions_v1.jsonl`.
- Answerable questions: 32.
- Refusal questions: 6, excluded from source/citation recall.
- Citation-labeled questions: 12.
- Default metric window: top 5.

## Results

| ID | Date | Retriever | Model / Index | Parameters | Source Recall@5 | Source Hit@5 | Source MRR | Citation Recall@5 | Citation Hit@5 | Citation MRR | Finding |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E001 | 2026-05-30 | BM25 | local lexical index | seed set, top_k=5 | 0.625 | n/a | n/a | n/a | n/a | n/a | Early smoke test showed BM25 can catch exact-term questions but misses broader privacy scenarios. |
| E002 | 2026-05-31 | BM25 | local lexical index | full set, top_k=5, candidate_k=20 | 0.641 | 0.938 | 0.769 | 0.083 | 0.083 | 0.017 | BM25 is a decent source finder, but weak at exact evidence retrieval. |
| E003 | 2026-05-31 | Dense Qdrant | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `.qdrant/local` | full set, top_k=5, candidate_k=20 | 0.469 | 0.750 | 0.555 | 0.111 | 0.167 | 0.125 | MiniLM underperforms BM25 at source retrieval but slightly improves exact citation hits. |
| E004 | 2026-05-31 | Hybrid RRF | BM25 + MiniLM dense | full set, top_k=5, candidate_k=20, RRF k=60 | 0.625 | 0.906 | 0.724 | 0.111 | 0.167 | 0.111 | Simple RRF does not beat BM25 yet; dense noise likely dilutes lexical strengths. |
| E005 | 2026-05-31 | Dense Qdrant | `BAAI/bge-m3` | attempted indexing | n/a | n/a | n/a | n/a | n/a | n/a | Deferred because first download needs a multi-GB model file and no Wi-Fi was available. |
| E006 | 2026-06-03 | Dense Qdrant | `BAAI/bge-m3`, `.qdrant/bge-m3` | full set, top_k=5, candidate_k=20 | 0.531 | 0.781 | 0.677 | 0.194 | 0.333 | 0.225 | BGE-M3 improves over MiniLM, especially exact citation retrieval, but still trails BM25 on source recall. |
| E007 | 2026-06-03 | Hybrid RRF | BM25 + BGE-M3 dense | full set, top_k=5, candidate_k=20, RRF k=60 | 0.568 | 0.812 | 0.683 | 0.125 | 0.167 | 0.125 | BGE-M3 hybrid is worse than BM25 and worse than BGE-M3 dense for exact citations; simple RRF needs tuning. |
| E008 | 2026-06-03 | BM25 | local lexical index | full set, top_k=20, candidate_k=100 | 0.755 | 0.969 | 0.771 | 0.194 | 0.333 | 0.038 | Wider top K improves source recall, but exact citations remain low and often rank late. |
| E009 | 2026-06-03 | Dense Qdrant | `BAAI/bge-m3`, `.qdrant/bge-m3` | full set, top_k=20, candidate_k=100 | 0.599 | 0.875 | 0.684 | 0.236 | 0.417 | 0.237 | BGE-M3 finds more exact citations at top 20, so reranking has useful dense candidates to work with. |
| E010 | 2026-06-03 | Hybrid RRF | BM25 + BGE-M3 dense | full set, top_k=20, candidate_k=100, RRF k=60 | 0.604 | 0.844 | 0.677 | 0.278 | 0.500 | 0.176 | Hybrid has the best top-20 citation hit rate, suggesting fusion helps recall but not ranking quality yet. |

## Current Interpretation

- BM25 has the best source-level retrieval so far.
- BGE-M3 is a better dense model than MiniLM for this corpus.
- BM25 still has the best source-level retrieval at top 5 and top 20.
- BGE-M3 improves exact citation retrieval, especially compared with MiniLM.
- Hybrid retrieval is not automatically better; simple RRF improves top-20 citation hit rate but does not rank exact citations high enough at top 5.
- Citation-level metrics expose the real bottleneck: finding and ranking the exact supporting chunk, not merely the right document.

## Planned Experiments

| ID | Experiment | Why it matters | Expected signal |
| --- | --- | --- | --- |
| P001 | Evaluate top_k=10 and top_k=20 | Checks whether correct citations are nearby but not in the first five results. | Completed for BGE-M3; citation hit improves at top 20, so reranking is likely valuable. |
| P002 | Increase dense/hybrid candidate_k to 50 or 100 | Gives fusion/reranking a larger candidate pool. | Higher source/citation recall before reranking. |
| P003 | Add source diversity after retrieval | Prevents top results from being dominated by one long source. | Better multi-source recall for GDPR + AI Act questions. |
| P004 | Try `BAAI/bge-m3` | Stronger multilingual retrieval model, especially for cross-lingual and German queries. | Completed; dense source recall and citation recall improved over MiniLM. |
| P005 | Add cross-encoder reranking | Reorders candidates using query-chunk relevance instead of independent vector similarity. | Citation MRR and Citation Recall@5 should improve if correct chunks appear in top 20/50. |
| P006 | Improve legal chunking | Article/annex-aware chunks reduce evidence dilution and boundary misses. | Better exact citation retrieval and cleaner citations. |
