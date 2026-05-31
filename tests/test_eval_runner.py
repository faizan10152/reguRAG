from regurag.evaluation.runner import GoldenQuestion, evaluate_bm25_source_recall
from regurag.schemas import Chunk


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
            relevant_sources={"ai_act"},
            must_refuse=False,
        )
    ]

    report = evaluate_bm25_source_recall(chunks, questions, top_k=1)

    assert report.mean_source_recall_at_k == 1.0
