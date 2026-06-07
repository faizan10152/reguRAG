from __future__ import annotations

from regurag.schemas import RetrievalResult

SYSTEM_PROMPT = """You are ReguRAG, a grounded regulatory intelligence system.

You answer questions about AI Act, GDPR, and German AI/data-protection guidance only from the provided evidence.
You are not a lawyer and must not present the answer as binding legal advice.

Rules:
- Use only the evidence chunks in the user message.
- Cite every factual claim with short evidence IDs like [E1], [E2], or [E3].
- Never invent evidence IDs. Use only IDs that are present in the evidence.
- If the evidence is insufficient, set should_refuse to true and explain why.
- Return strict JSON only. Do not wrap it in markdown.
"""

JSON_CONTRACT = """Return this JSON object:
{
  "answer": "short grounded answer with inline citations, or a refusal",
  "citations": ["E1"],
  "confidence": "low|medium|high",
  "unsupported_claims": ["claim that could not be supported"],
  "should_refuse": false,
  "refusal_reason": null
}
"""


def format_evidence_context(
    results: list[RetrievalResult],
    *,
    max_chars_per_chunk: int = 1500,
) -> str:
    if not results:
        return "No evidence chunks were retrieved."

    blocks = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        evidence_id = f"E{index}"
        title = str(chunk.metadata.get("title") or chunk.source_id)
        heading = str(chunk.metadata.get("section_heading") or "no section")
        url = chunk.metadata.get("url")
        text = chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = f"{text[:max_chars_per_chunk].rstrip()}..."

        blocks.append(
            "\n".join(
                [
                    f"[{evidence_id}]",
                    f"Citation label: {chunk.citation_label}",
                    f"Title: {title}",
                    f"Section: {heading}",
                    f"URL: {url or 'unknown'}",
                    f"Retriever: {result.retriever}",
                    f"Rank: {result.rank}",
                    f"Score: {result.score:.6f}",
                    "Text:",
                    text,
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def build_answer_messages(
    question: str,
    results: list[RetrievalResult],
    *,
    max_chars_per_chunk: int = 1500,
) -> list[dict[str, str]]:
    evidence = format_evidence_context(results, max_chars_per_chunk=max_chars_per_chunk)
    user_prompt = "\n\n".join(
        [
            f"Question:\n{question}",
            f"Evidence:\n{evidence}",
            JSON_CONTRACT,
        ]
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
