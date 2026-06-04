# Answer Generation

ReguRAG now has a guarded answer-generation path. The goal is not just to produce fluent text. The goal is to produce an answer that can be checked against retrieved evidence.

## Flow

```text
question
  -> retrieve candidates
  -> optionally fuse and rerank
  -> build evidence context with exact citation labels
  -> call an LLM through LiteLLM
  -> parse structured JSON
  -> validate cited labels against retrieved chunks
  -> mark unsupported/refused if validation fails
```

## Why Structured JSON?

Free-form answers are hard to validate. The answer command asks the model for this JSON object:

```json
{
  "answer": "short grounded answer with inline citations, or a refusal",
  "citations": ["source_id:chunk_id"],
  "confidence": "low|medium|high",
  "unsupported_claims": ["claim that could not be supported"],
  "should_refuse": false,
  "refusal_reason": null
}
```

The important fields are `answer`, `citations`, and `should_refuse`.

The `citations` field is not trusted automatically. ReguRAG validates every citation label against the retrieved chunks. If the model cites a chunk that was not retrieved, or cites nothing when citations are required, the answer is marked unsupported.

## Dry Run Without API Calls

Use this to inspect retrieval and the prompt before spending API credits:

```bash
make answer-dry-run q="Is AI CV screening high-risk under the AI Act?"
```

This prints:

- the selected retriever
- the exact system and user messages
- the retrieved evidence chunks
- the exact citation labels available to the model

This is a useful debugging habit. If the retrieved evidence is wrong, the answer should not be trusted.

## Real LLM Run

Set a LiteLLM model name through an environment variable or pass `--llm-model` directly:

```bash
REGURAG_LLM_MODEL="<provider>/<model>" make answer-local \
  q="Is AI CV screening high-risk under the AI Act?"
```

Equivalent direct command:

```bash
HF_HUB_DISABLE_XET=1 uv run --extra rag regurag answer \
  --chunks data/processed/chunks.jsonl \
  --query "Is AI CV screening high-risk under the AI Act?" \
  --retriever hybrid-rerank \
  --qdrant-path .qdrant/bge-m3 \
  --model BAAI/bge-m3 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --top-k 5 \
  --candidate-k 20 \
  --llm-model "<provider>/<model>"
```

If a provider does not support JSON mode, add:

```bash
--disable-json-mode
```

The prompt still demands JSON, but the provider will not receive an explicit JSON-mode request.

## Guardrail Behavior

The guardrail triggers when:

- no evidence chunks are retrieved
- the model cites no chunks
- the model cites labels that were not retrieved
- the number of valid citations is below `--min-citations`

When the guardrail triggers, ReguRAG marks the response as unsupported and refuses instead of passing through an ungrounded answer.

## Current Limitation

Citation validation checks whether cited labels were retrieved. It does not yet prove every sentence is semantically entailed by the cited text. That is the next evaluation layer: answer faithfulness, refusal accuracy, and possibly Ragas-style metrics.
