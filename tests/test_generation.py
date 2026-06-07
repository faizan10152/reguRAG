import json

import pytest

from regurag.generation.answer import generate_grounded_answer, parse_grounded_answer
from regurag.generation.litellm_client import LiteLLMClient
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


class RawFakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self.response


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

    assert "[E1]" in context
    assert "Citation label: eu_ai_act_en:aaaabbbbccccdddd" in context
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
            "answer": "Recruitment systems can be high-risk [E1].",
            "citations": ["E1"],
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
    assert result.answer.citations == ["eu_ai_act_en:aaaabbbbccccdddd"]
    assert "[eu_ai_act_en:aaaabbbbccccdddd]" in result.answer.answer
    assert len(llm.calls) == 1


def test_generate_grounded_answer_normalizes_inline_citations() -> None:
    llm = FakeLLM(
        {
            "answer": "Recruitment systems can be high-risk [E1].",
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
    assert "[eu_ai_act_en:aaaabbbbccccdddd]" in result.answer.answer


def test_generate_grounded_answer_accepts_exact_labels_for_backward_compatibility() -> None:
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
    assert result.answer.citations == ["eu_ai_act_en:aaaabbbbccccdddd"]


def test_generate_grounded_answer_triggers_guardrail_for_missing_citation() -> None:
    llm = FakeLLM(
        {
            "answer": "This is supported [E9].",
            "citations": ["E9"],
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
    assert result.citation_validation.missing_labels == {"E9"}


def test_generate_grounded_answer_refuses_invalid_structured_output() -> None:
    llm = RawFakeLLM(
        json.dumps(
            {
                "answer": "",
                "citations": [],
                "confidence": "low",
                "unsupported_claims": [],
                "should_refuse": False,
                "refusal_reason": None,
            }
        )
    )

    result = generate_grounded_answer(
        question="Is CV screening high-risk?",
        retrieved_results=[_result()],
        llm=llm,
    )

    assert not result.supported
    assert result.guardrail_triggered
    assert result.answer.should_refuse
    assert result.answer.refusal_reason
    assert result.answer.refusal_reason.startswith("Invalid LLM response")
    assert result.raw_response == llm.response


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


def test_litellm_client_builds_ollama_kwargs() -> None:
    client = LiteLLMClient(
        model="ollama/qwen3:14b",
        temperature=0.1,
        max_tokens=256,
        api_base="http://localhost:11434",
        timeout=30.0,
    )

    kwargs = client.completion_kwargs([{"role": "user", "content": "test"}])

    assert kwargs["model"] == "ollama/qwen3:14b"
    assert kwargs["api_base"] == "http://localhost:11434"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 256
    assert kwargs["timeout"] == 30.0
    assert kwargs["response_format"] == {"type": "json_object"}


def test_litellm_client_can_disable_json_mode() -> None:
    client = LiteLLMClient(model="ollama/qwen3:8b", json_mode=False)

    kwargs = client.completion_kwargs([{"role": "user", "content": "test"}])

    assert "response_format" not in kwargs
