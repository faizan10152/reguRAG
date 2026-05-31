import json

from regurag.evaluation.runner import (
    GoldenQuestion,
    evaluate_bm25_source_recall,
    evaluate_retrieval_runs,
    format_markdown_report,
    load_golden_questions,
)
from regurag.schemas import Chunk, RetrievalResult


def test_load_golden_questions_accepts_approved_schema(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    payload = {
        "id": "GQ001",
        "domain": "hr",
        "type": "scenario",
        "difficulty": "medium",
        "language": "en",
        "question": "Is AI CV screening high-risk?",
        "expected_sources": ["eu_ai_act_en"],
        "must_refuse": False,
        "review_status": "approved",
        "structural_difficulty": "semantic_paraphrase",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    questions = load_golden_questions(path)

    assert questions[0].question_id == "GQ001"
    assert questions[0].domain == "hr"
    assert questions[0].question_type == "scenario"
    assert questions[0].expected_sources == {"eu_ai_act_en"}
    assert questions[0].relevant_sources == {"eu_ai_act_en"}


def test_evaluate_bm25_source_recall() -> None:
    chunks = [
        Chunk("aaaabbbbccccdddd", "ai_act", "employment screening high risk", {}),
        Chunk("eeeeffff00001111", "gdpr", "personal data legal basis", {}),
    ]
    questions = [
        GoldenQuestion(
            question_id="q1",
            question="employment high risk",
            language="en",
            expected_sources={"ai_act"},
            must_refuse=False,
        )
    ]

    report = evaluate_bm25_source_recall(chunks, questions, top_k=1)

    assert report.mean_source_recall_at_k == 1.0
    assert report.summary_for_retriever("bm25").source_mrr == 1.0


def test_evaluate_retrieval_runs_scores_multiple_retrievers() -> None:
    ai_act = Chunk("aaaabbbbccccdddd", "ai_act", "employment screening high risk", {})
    gdpr = Chunk("eeeeffff00001111", "gdpr", "personal data legal basis", {})
    questions = [
        GoldenQuestion(
            question_id="q1",
            question="employment high risk",
            language="en",
            expected_sources={"ai_act", "gdpr"},
            must_refuse=False,
            domain="hr",
            structural_difficulty="multi_source",
        )
    ]

    report = evaluate_retrieval_runs(
        questions=questions,
        retriever_runs={
            "weak": lambda _query: [
                RetrievalResult(chunk=ai_act, score=1.0, rank=1, retriever="weak")
            ],
            "strong": lambda _query: [
                RetrievalResult(chunk=gdpr, score=2.0, rank=1, retriever="strong"),
                RetrievalResult(chunk=ai_act, score=1.0, rank=2, retriever="strong"),
            ],
        },
        top_k=2,
        candidate_k=10,
    )

    weak_summary = report.summary_for_retriever("weak")
    strong_summary = report.summary_for_retriever("strong")

    assert weak_summary.source_recall_at_k == 0.5
    assert strong_summary.source_recall_at_k == 1.0
    assert report.breakdown("domain")[0]["domain"] == "hr"
    assert "Retrieval Evaluation Report" in format_markdown_report(report)
