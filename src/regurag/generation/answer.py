from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from regurag.generation.litellm_client import LLMClient
from regurag.generation.prompts import build_answer_messages
from regurag.grounding.citations import (
    extract_citation_labels,
    should_refuse_answer,
    validate_citation_labels,
)
from regurag.schemas import CitationValidation, RetrievalResult

CONFIDENCE_VALUES = {"low", "medium", "high"}
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
EVIDENCE_ALIAS_RE = re.compile(r"\[(E[1-9][0-9]*)\]")
DEFAULT_MAX_CHARS_PER_CHUNK = 4500


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[str]
    confidence: str
    unsupported_claims: list[str]
    should_refuse: bool
    refusal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerGenerationResult:
    question: str
    answer: GroundedAnswer
    retrieved_results: list[RetrievalResult]
    citation_validation: CitationValidation
    supported: bool
    guardrail_triggered: bool
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer.to_dict(),
            "supported": self.supported,
            "guardrail_triggered": self.guardrail_triggered,
            "citation_validation": {
                "cited_labels": sorted(self.citation_validation.cited_labels),
                "available_labels": sorted(self.citation_validation.available_labels),
                "missing_labels": sorted(self.citation_validation.missing_labels),
                "is_supported": self.citation_validation.is_supported,
            },
            "retrieved_results": [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "retriever": result.retriever,
                    "citation_label": result.citation_label,
                    "source_id": result.chunk.source_id,
                    "chunk_id": result.chunk.chunk_id,
                    "title": result.chunk.metadata.get("title", result.chunk.source_id),
                    "section_heading": result.chunk.metadata.get("section_heading"),
                    "url": result.chunk.metadata.get("url"),
                    "snippet": result.chunk.text[:500],
                }
                for result in self.retrieved_results
            ],
            "raw_response": self.raw_response,
        }


def parse_grounded_answer(raw_response: str) -> GroundedAnswer:
    payload = _parse_json_object(raw_response)

    answer = _required_str(payload, "answer")
    citations = _str_list(payload.get("citations", []), "citations")
    confidence = str(payload.get("confidence", "low")).lower()
    if confidence not in CONFIDENCE_VALUES:
        confidence = "low"

    unsupported_claims = _str_list(payload.get("unsupported_claims", []), "unsupported_claims")
    should_refuse = bool(payload.get("should_refuse", False))
    refusal_reason = payload.get("refusal_reason")
    if refusal_reason is not None:
        refusal_reason = str(refusal_reason)

    return GroundedAnswer(
        answer=answer,
        citations=citations,
        confidence=confidence,
        unsupported_claims=unsupported_claims,
        should_refuse=should_refuse,
        refusal_reason=refusal_reason,
    )


def generate_grounded_answer(
    *,
    question: str,
    retrieved_results: list[RetrievalResult],
    llm: LLMClient,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    min_citations: int = 1,
) -> AnswerGenerationResult:
    if not retrieved_results:
        answer = GroundedAnswer(
            answer="I cannot answer from the indexed corpus because no supporting evidence was retrieved.",
            citations=[],
            confidence="low",
            unsupported_claims=[question],
            should_refuse=True,
            refusal_reason="No evidence chunks were retrieved.",
        )
        validation = validate_citation_labels(set(), [])
        return AnswerGenerationResult(
            question=question,
            answer=answer,
            retrieved_results=[],
            citation_validation=validation,
            supported=False,
            guardrail_triggered=True,
            raw_response=None,
        )

    messages = build_answer_messages(
        question,
        retrieved_results,
        max_chars_per_chunk=max_chars_per_chunk,
    )
    raw_response = llm.generate(messages)
    try:
        parsed = parse_grounded_answer(raw_response)
    except ValueError as exc:
        validation = validate_citation_labels(set(), retrieved_results)
        answer = GroundedAnswer(
            answer="I cannot answer because the language model returned invalid structured output.",
            citations=[],
            confidence="low",
            unsupported_claims=[raw_response[:500]],
            should_refuse=True,
            refusal_reason=f"Invalid LLM response: {exc}",
        )
        return AnswerGenerationResult(
            question=question,
            answer=answer,
            retrieved_results=retrieved_results,
            citation_validation=validation,
            supported=False,
            guardrail_triggered=True,
            raw_response=raw_response,
        )

    cited_labels = _normalize_generated_citations(parsed, retrieved_results)
    validation = validate_citation_labels(cited_labels, retrieved_results)
    parsed = _with_citations(
        _replace_answer_aliases(parsed, retrieved_results),
        sorted(cited_labels),
    )
    guardrail_triggered = should_refuse_answer(validation, min_citations=min_citations)

    if guardrail_triggered and not parsed.should_refuse:
        parsed = GroundedAnswer(
            answer=(
                "I cannot answer from the retrieved evidence because the generated answer "
                "did not cite enough valid supporting chunks."
            ),
            citations=sorted(validation.cited_labels),
            confidence="low",
            unsupported_claims=parsed.unsupported_claims or [parsed.answer],
            should_refuse=True,
            refusal_reason="Citation validation failed.",
        )

    supported = not parsed.should_refuse and validation.is_supported and not guardrail_triggered
    return AnswerGenerationResult(
        question=question,
        answer=parsed,
        retrieved_results=retrieved_results,
        citation_validation=validation,
        supported=supported,
        guardrail_triggered=guardrail_triggered,
        raw_response=raw_response,
    )


def _with_citations(answer: GroundedAnswer, citations: list[str]) -> GroundedAnswer:
    if answer.citations == citations:
        return answer
    return GroundedAnswer(
        answer=answer.answer,
        citations=citations,
        confidence=answer.confidence,
        unsupported_claims=answer.unsupported_claims,
        should_refuse=answer.should_refuse,
        refusal_reason=answer.refusal_reason,
    )


def _normalize_generated_citations(
    answer: GroundedAnswer,
    retrieved_results: list[RetrievalResult],
) -> set[str]:
    alias_to_label = _evidence_alias_to_label(retrieved_results)
    raw_citations = set(answer.citations)
    raw_citations.update(extract_citation_labels(answer.answer))
    raw_citations.update(_extract_evidence_aliases(answer.answer))

    normalized = set()
    for citation in raw_citations:
        normalized.add(alias_to_label.get(citation, citation))
    return normalized


def _replace_answer_aliases(
    answer: GroundedAnswer,
    retrieved_results: list[RetrievalResult],
) -> GroundedAnswer:
    alias_to_label = _evidence_alias_to_label(retrieved_results)

    def replace(match: re.Match[str]) -> str:
        alias = match.group(1)
        label = alias_to_label.get(alias)
        if not label:
            return match.group(0)
        return f"[{label}]"

    return GroundedAnswer(
        answer=EVIDENCE_ALIAS_RE.sub(replace, answer.answer),
        citations=answer.citations,
        confidence=answer.confidence,
        unsupported_claims=answer.unsupported_claims,
        should_refuse=answer.should_refuse,
        refusal_reason=answer.refusal_reason,
    )


def _extract_evidence_aliases(answer_text: str) -> set[str]:
    return set(EVIDENCE_ALIAS_RE.findall(answer_text))


def _evidence_alias_to_label(retrieved_results: list[RetrievalResult]) -> dict[str, str]:
    return {f"E{index}": result.citation_label for index, result in enumerate(retrieved_results, start=1)}


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    stripped = raw_response.strip()
    fence_match = JSON_FENCE_RE.match(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"LLM response field '{key}' must be a non-empty string.")
    return value.strip()


def _str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"LLM response field '{field_name}' must be a list.")
    return [str(item).strip() for item in value if str(item).strip()]
