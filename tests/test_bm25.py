from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.schemas import Chunk


def test_bm25_retrieves_exact_term_match() -> None:
    chunks = [
        Chunk("aaaabbbbccccdddd", "doc1", "Employment screening and AI risk", {}),
        Chunk("eeeeffff00001111", "doc2", "Weather forecast and gardening", {}),
    ]
    retriever = SimpleBM25Retriever(chunks)

    results = retriever.search("employment screening", top_k=2)

    assert results
    assert results[0].chunk.source_id == "doc1"
    assert results[0].rank == 1
