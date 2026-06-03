import pytest

from regurag.reranking.cross_encoder import (
    dedupe_results,
    format_rerank_passage,
    rerank_results,
)
from regurag.schemas import Chunk, RetrievalResult


def _result(chunk_id: str, score: float, rank: int) -> RetrievalResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        source_id="eu_ai_act_en",
        text=f"chunk text {chunk_id}",
        metadata={"title": "EU AI Act", "section_heading": "Article 6"},
    )
    return RetrievalResult(chunk=chunk, score=score, rank=rank, retriever="hybrid_rrf")


def test_dedupe_results_preserves_first_candidate() -> None:
    first = _result("aaaabbbbccccdddd", score=0.1, rank=1)
    duplicate = _result("aaaabbbbccccdddd", score=0.9, rank=2)
    second = _result("eeeeffff00001111", score=0.8, rank=3)

    deduped = dedupe_results([first, duplicate, second])

    assert deduped == [first, second]


def test_format_rerank_passage_includes_source_context() -> None:
    result = _result("aaaabbbbccccdddd", score=0.1, rank=1)

    passage = format_rerank_passage(result.chunk)

    assert passage.startswith("EU AI Act\nArticle 6\n")
    assert "chunk text aaaabbbbccccdddd" in passage


def test_rerank_results_orders_by_cross_encoder_score() -> None:
    low = _result("aaaabbbbccccdddd", score=0.1, rank=1)
    high = _result("eeeeffff00001111", score=0.2, rank=2)

    reranked = rerank_results(
        [low, high],
        [0.25, 0.95],
        top_k=2,
        retriever_name="hybrid_rerank",
    )

    assert [result.chunk_id for result in reranked] == [
        "eeeeffff00001111",
        "aaaabbbbccccdddd",
    ]
    assert [result.rank for result in reranked] == [1, 2]
    assert [result.score for result in reranked] == [0.95, 0.25]
    assert {result.retriever for result in reranked} == {"hybrid_rerank"}


def test_rerank_results_rejects_score_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        rerank_results([_result("aaaabbbbccccdddd", score=0.1, rank=1)], [])
