# Answer Generation Experiments

This log records local answer-generation experiments so model, prompt, and context-window choices can be compared later.

## Benchmark Setup

- Corpus: official EU/German regulatory chunks in `data/processed/chunks.jsonl`.
- Retrieval path: BM25 baseline or BGE-M3 dense retrieval plus hybrid RRF and cross-encoder reranking.
- Local LLM runtime: Ollama.
- Local LLM tested: `qwen3:14b`.
- Example query: `Is AI CV screening high-risk under the AI Act?`

## Results

| ID | Date | Model | Retrieval | Parameters | Result | Finding |
| --- | --- | --- | --- | --- | --- | --- |
| A001 | 2026-06-07 | `qwen3:14b` | BM25 top-5 | Ollama JSON mode, `max_context_chars=1500` | Supported, but answer was weak/misleading | BM25 retrieved broad AI Act chunks and missed the strongest employment evidence. Generation cannot fix missing or weak retrieval. |
| A002 | 2026-06-07 | `qwen3:14b` | Hybrid rerank top-5 | Ollama JSON mode, `max_context_chars=1500` | Guarded refusal, raw response `{}` | Forced Ollama JSON mode can fail on longer reranked contexts. The guardrail correctly blocked invalid structured output. |
| A003 | 2026-06-07 | `qwen3:14b` | Hybrid rerank top-5 | Prompt-only JSON, `max_context_chars=1500` | Supported, but answer was wrong | Retrieval found the correct chunk, but the prompt truncated the chunk before the employment/recruitment sentence. |
| A004 | 2026-06-07 | `qwen3:14b` | Hybrid rerank top-5 | Prompt-only JSON, `max_context_chars=4500` | Supported and correct | The model cited the AI Act employment/recruitment chunk and answered that AI systems used for recruitment and selection are high-risk. |
| A005 | 2026-06-07 | `qwen3:14b` | Hybrid rerank top-5 | 8-question answer eval smoke set, prompt-only JSON, `max_context_chars=4500`, timeout 120s | Supported rate 0.500, answerable supported 0.667, valid structured output 0.875, refusal accuracy 0.750, citation validity 1.000, source recall@5 0.556, expected citation hit 0.167, mean latency 75.36s | Refusal behavior is promising, citation validity is strong, but source recall and exact evidence usage are weak. GQ036 refused incorrectly because DPIA evidence was not retrieved clearly enough. GQ001 returned invalid JSON once under the batch run, showing local generation variance. |

## Current Interpretation

- Local Qwen can produce useful grounded answers when the relevant evidence is actually present in the prompt.
- Exact retrieval labels are too error-prone for local models to copy directly; short evidence aliases such as `[E1]` are more robust.
- Provider-enforced JSON mode is not always better. For `qwen3:14b` through Ollama, prompt-only JSON worked better on the reranked context.
- Per-chunk context limits are a real RAG parameter. Too small a value can hide the relevant sentence inside an otherwise correctly retrieved chunk.
- Local answer evaluation needs progress output and request timeouts. A single slow Ollama call can otherwise make a batch run look stuck.
- The next improvement should be article-aware chunking or parent-child retrieval so the answer context contains the most relevant legal span without sending long chunks every time.
