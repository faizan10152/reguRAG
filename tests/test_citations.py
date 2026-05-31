from regurag.grounding.citations import (
    extract_citation_labels,
    should_refuse_answer,
    validate_citations,
)
from regurag.schemas import Chunk, RetrievalResult


def test_validate_citations_flags_missing_sources() -> None:
    chunk = Chunk("aaaabbbbccccdddd", "doc1", "supported text", {})
    results = [RetrievalResult(chunk=chunk, score=1.0, rank=1, retriever="bm25")]
    answer = "Supported claim [doc1:aaaabbbbccccdddd]. Unsupported [doc2:eeeeffff00001111]."

    validation = validate_citations(answer, results)

    assert "doc1:aaaabbbbccccdddd" in extract_citation_labels(answer)
    assert validation.missing_labels == {"doc2:eeeeffff00001111"}
    assert should_refuse_answer(validation)
