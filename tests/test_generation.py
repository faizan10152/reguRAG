import json

import pytest

from regurag.generation.answer import generate_grounded_answer, parse_grounded_answer
from regurag.generation.prompts import build_answer_messages, format_evidence_context
from regurag.grounding.citations import validate_citation_labels
from regurag.schemas import Chunk, RetrievalResult


class FakeLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return json.dumps(self.response)


def _result() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="aaaabbbbccccdddd",
        source_id="eu_ai_act_en",
        text="AI systems used for recruitment or selection are high-risk in the cited annex.",
        metadata={
            "title": "EU AI Act",
            "section_heading": "Annex III",
            "url": "https://example.test/ai-act",
        },
    )
    return RetrievalResult(chunk=chunk, score=0.95, rank=1, retriever="hybrid_rerank")


def test_parse_grounded_answer_accepts_json_fence() -> None:
    raw = """```json
{"answer": "Supported [eu_ai_act_en:aaaabbbbccccdddd].", "citations": ["eu_ai_act_en:aaaabbbbccccdddd"], "confidence": "high", "unsupported_claims": [], "should_refuse": false, "refusal_reason": null}
```"""

    parsed = parse_grounded_answer(raw)

    assert parsed.answer.startswith("Supported")
    assert parsed.citations == ["eu_ai_act_en:aaaabbbbccccdddd"]
    assert parsed.confidence == "high"
    assert not parsed.should_refuse


def test_parse_grounded_answer_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_grounded_answer("not json")


def test_format_evidence_context_includes_exact_citation_label() -> None:
    context = format_evidence_context([_result()])

    assert "[eu_ai_act_en:aaaabbbbccccdddd]" in context
    assert "EU AI Act" in context
    assert "Annex III" in context


def test_build_answer_messages_contains_question_and_contract() -> None:
    messages = build_answer_messages("Is CV screening high-risk?", [_result()])

    assert messages[0]["role"] == "system"
    assert "strict JSON" in messages[0]["content"]
    assert "Is CV screening high-risk?" in messages[1]["content"]
    assert '"citations"' in messages[1]["content"]


def test_generate_grounded_answer_marks_supported_when_citations_are_valid() -> None:
    llm = FakeLLM(
        {
            "answer": "Recruitment systems can be high-risk [eu_ai_act_en:aaaabbbbccccdddd].",
            "citations": ["eu_ai_act_en:aaaabbbbccccdddd"],
            "confidence": "medium",
            "unsupported_claims": [],
            "should_refuse": False,
            "refusal_reason": None,
        }
    )

    result = generate_grounded_answer(
        question="Is CV screening high-risk?",
        retrieved_results=[_result()],
        llm=llm,
    )

    assert result.supported
    assert not result.guardrail_triggered
    assert result.citation_validation.is_supported
    assert len(llm.calls) == 1


def test_generate_grounded_answer_normalizes_inline_citations() -> None:
    llm = FakeLLM(
        {
            "answer": "Recruitment systems can be high-risk [eu_ai_act_en:aaaabbbbccccdddd].",
            "citations": [],
            "confidence": "medium",
            "unsupported_claims": [],
            "should_refuse": False,
            "refusal_reason": None,
        }
    )

    result = generate_grounded_answer(
        question="Is CV screening high-risk?",
        retrieved_results=[_result()],
        llm=llm,
    )

    assert result.supported
    assert result.answer.citations == ["eu_ai_act_en:aaaabbbbccccdddd"]


def test_generate_grounded_answer_triggers_guardrail_for_missing_citation() -> None:
    llm = FakeLLM(
        {
            "answer": "This is supported [gdpr_en:eeeeffff00001111].",
            "citations": ["gdpr_en:eeeeffff00001111"],
            "confidence": "high",
            "unsupported_claims": [],
            "should_refuse": False,
            "refusal_reason": None,
        }
    )

    result = generate_grounded_answer(
        question="Is CV screening high-risk?",
        retrieved_results=[_result()],
        llm=llm,
    )

    assert not result.supported
    assert result.guardrail_triggered
    assert result.answer.should_refuse
    assert result.answer.refusal_reason == "Citation validation failed."
    assert result.citation_validation.missing_labels == {"gdpr_en:eeeeffff00001111"}


def test_generate_grounded_answer_refuses_without_retrieved_evidence() -> None:
    llm = FakeLLM({})

    result = generate_grounded_answer(
        question="What does Article 999 say?",
        retrieved_results=[],
        llm=llm,
    )

    assert not result.supported
    assert result.guardrail_triggered
    assert result.answer.should_refuse
    assert result.answer.refusal_reason == "No evidence chunks were retrieved."
    assert llm.calls == []


def test_validate_citation_labels_accepts_structured_citations() -> None:
    validation = validate_citation_labels({"eu_ai_act_en:aaaabbbbccccdddd"}, [_result()])

    assert validation.is_supported
