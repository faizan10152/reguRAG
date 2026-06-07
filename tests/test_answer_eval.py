import json

from regurag.evaluation.answers import (
    evaluate_answer_run,
    format_answer_markdown_report,
)
from regurag.evaluation.runner import GoldenQuestion
from regurag.schemas import Chunk, RetrievalResult


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self.responses[self.calls]
        self.calls += 1
        return json.dumps(response)


def _result() -> RetrievalResult:
    chunk = Chunk(
        chunk_id="aaaabbbbccccdddd",
        source_id="eu_ai_act_en",
        text="AI systems used for recruitment and selection are high-risk.",
        metadata={"title": "EU AI Act"},
    )
    return RetrievalResult(chunk=chunk, score=1.0, rank=1, retriever="test")


def test_evaluate_answer_run_summarizes_supported_and_refusal_rows() -> None:
    questions = [
        GoldenQuestion(
            question_id="GQ001",
            question="Is CV screening high-risk?",
            language="en",
            expected_sources={"eu_ai_act_en"},
            expected_citations={"eu_ai_act_en:aaaabbbbccccdddd"},
            must_refuse=False,
        ),
        GoldenQuestion(
            question_id="GQ029",
            question="What is Article 999?",
            language="en",
            expected_sources=set(),
            must_refuse=True,
        ),
    ]
    llm = FakeLLM(
        [
            {
                "answer": "Recruitment systems are high-risk [E1].",
                "citations": ["E1"],
                "confidence": "high",
                "unsupported_claims": [],
                "should_refuse": False,
                "refusal_reason": None,
            }
        ]
    )

    def search(query: str) -> list[RetrievalResult]:
        if "Article 999" in query:
            return []
        return [_result()]

    report = evaluate_answer_run(
        questions=questions,
        search=search,
        llm=llm,
        run_name="test-run",
        retriever="test",
        llm_model="fake",
        top_k=1,
        candidate_k=1,
    )
    summary = report.summary()

    assert summary.rows == 2
    assert summary.answerable_rows == 1
    assert summary.refusal_rows == 1
    assert summary.supported_rate == 0.5
    assert summary.answerable_supported_rate == 1.0
    assert summary.refusal_accuracy == 1.0
    assert summary.expected_refusal_success_rate == 1.0
    assert summary.citation_validity_rate == 1.0
    assert summary.source_recall_at_k == 1.0
    assert summary.expected_citation_hit_rate == 1.0
    assert report.rows[0].manual_label == "unreviewed"
    assert "Answer Evaluation Report" in format_answer_markdown_report(report)


def test_evaluate_answer_run_tracks_invalid_structured_output() -> None:
    question = GoldenQuestion(
        question_id="GQ001",
        question="Is CV screening high-risk?",
        language="en",
        expected_sources={"eu_ai_act_en"},
        must_refuse=False,
    )
    llm = FakeLLM(
        [
            {
                "answer": "",
                "citations": [],
                "confidence": "low",
                "unsupported_claims": [],
                "should_refuse": False,
                "refusal_reason": None,
            }
        ]
    )

    report = evaluate_answer_run(
        questions=[question],
        search=lambda _query: [_result()],
        llm=llm,
        run_name="test-run",
        retriever="test",
        llm_model="fake",
        top_k=1,
        candidate_k=1,
    )
    row = report.rows[0]
    summary = report.summary()

    assert row.guardrail_triggered
    assert not row.valid_structured_output
    assert row.refusal_reason
    assert row.refusal_reason.startswith("Invalid LLM response")
    assert summary.valid_structured_output_rate == 0.0
    assert summary.error_rows == 0
