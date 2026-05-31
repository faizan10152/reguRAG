# Evaluation Baseline

Date: 2026-05-30

This is an intentionally small seed evaluation. It is not the final benchmark. Its purpose is to catch obvious retrieval failures early and create a habit of measuring changes.

## Corpus

Generated from `configs/source_manifest.json`.

Current local chunk build:

- EU AI Act EN/DE
- GDPR EN/DE
- BDSG DE
- Bundesnetzagentur AI pages
- BfDI AI questions page
- DSK AI and data protection PDF

Chunk count after PDF extraction: 843.

## Method

Command:

```bash
uv run regurag eval-retrieval \
  --chunks data/processed/chunks.jsonl \
  --questions data/eval/golden_questions_seed.jsonl \
  --top-k 5
```

Metric:

- Source-level Recall@5.

This checks whether the top 5 retrieved chunks come from at least one expected source. It is weaker than chunk-level citation evaluation, but useful as an early smoke test.

## Result

```text
questions=5
answerable_questions=4
mean_source_recall_at_5=0.625
q001: recall=1.000 expected=eu_ai_act_en
q002: recall=1.000 expected=bnetza_ai_literacy_de,eu_ai_act_de
q003: recall=0.000 expected=bfdi_ki_fragen_de,gdpr_en
q004: recall=0.500 expected=dsk_ki_datenschutz_de,gdpr_de
q005: recall=0.000 expected=unanswerable
```

## Interpretation

BM25 works well for exact-term questions such as employment screening and KI-Kompetenz.

BM25 struggles when the user asks a broader semantic privacy question, for example whether customer support tickets containing personal data can be used for model training. This is expected: keyword retrieval alone does not understand that "customer support tickets", "personal data", "training", and "legal basis" are connected in a privacy scenario.

## Next Fixes

1. Add dense multilingual embeddings.
2. Fuse BM25 and dense results.
3. Add reranking.
4. Move from source-level relevance to chunk-level expected citations.
5. Add refusal-specific metrics for unanswerable questions.
