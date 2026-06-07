# Answer Evaluation

Answer evaluation checks the full RAG path, not only retrieval.

The evaluator runs golden questions through:

```text
retrieve evidence
  -> build answer prompt
  -> call LLM
  -> parse structured output
  -> validate citations
  -> score refusal behavior and support signals
```

## Local Command

```bash
make eval-answers-ollama
```

The command prints per-question progress, for example:

```text
evaluating 1/8 GQ001
evaluating 2/8 GQ003
```

Local LLM evaluation can be slow. The CLI uses an LLM timeout for answer evaluation so one stalled generation call becomes a scored row instead of blocking the whole benchmark.

The default smoke set is:

```text
GQ001,GQ003,GQ004,GQ008,GQ021,GQ029,GQ036,GQ038
```

This set covers HR, GDPR/AI Act overlap, German language, multi-hop finance, AI literacy, refusal, negative factual reasoning, and out-of-corpus refusal.

## Metrics

- `supported_rate`: share of rows where the final answer passed citation validation and did not refuse.
- `answerable_supported_rate`: share of answerable rows that produced a supported answer.
- `valid_structured_output_rate`: share of rows where the LLM returned parseable structured output rather than malformed JSON or an empty object.
- `refusal_accuracy`: share of rows where `should_refuse` matched the golden question's `must_refuse` flag.
- `expected_refusal_success_rate`: share of intentionally unanswerable rows where the system refused.
- `citation_validity_rate`: among rows with generated citations, share where all cited chunks were actually retrieved.
- `source_recall@K`: source-level retrieval recall for answerable rows inside the answer run.
- `expected_citation_hit_rate`: for rows with exact expected citations, share where the final answer cited at least one expected evidence chunk.
- `mean_latency_seconds`: average end-to-end latency per question.

## What This Does Not Prove Yet

These metrics prove structure, citation validity, refusal behavior, and whether expected evidence appeared. They do not automatically prove that every generated sentence is semantically correct.

For that reason, each answer row is marked:

```text
manual_label=unreviewed
```

The next layer is manual correctness labeling:

```text
correct | partially_correct | wrong | unclear
```

That manual review is important for regulatory RAG because an answer can cite a valid chunk and still make a weak legal interpretation.
