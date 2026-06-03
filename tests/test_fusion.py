from regurag.retrieval.fusion import reciprocal_rank_fusion
from regurag.schemas import Chunk, RetrievalResult


def test_reciprocal_rank_fusion_promotes_results_seen_by_multiple_retrievers() -> None:
    shared = Chunk("aaaabbbbccccdddd", "doc1", "shared", {})
    lexical_only = Chunk("eeeeffff00001111", "doc2", "lexical", {})
    dense_only = Chunk("2222333344445555", "doc3", "dense", {})

    bm25 = [
        RetrievalResult(lexical_only, score=20.0, rank=1, retriever="bm25"),
        RetrievalResult(shared, score=10.0, rank=2, retriever="bm25"),
    ]
    dense = [
        RetrievalResult(dense_only, score=0.9, rank=1, retriever="dense"),
        RetrievalResult(shared, score=0.8, rank=2, retriever="dense"),
    ]

    fused = reciprocal_rank_fusion([bm25, dense], top_k=3)

    assert fused[0].chunk == shared
    assert fused[0].retriever == "rrf"
