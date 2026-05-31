from regurag.ingestion.chunking import chunk_text
from regurag.schemas import SourceRecord


def test_chunk_text_preserves_source_metadata() -> None:
    source = SourceRecord(
        source_id="test_source",
        title="Test Source",
        url="https://example.com",
        language="en",
        jurisdiction="EU",
        source_type="regulation",
        domain_tags=["test"],
    )
    text = "Article 1\n" + " ".join(f"word{i}" for i in range(30))

    chunks = chunk_text(source, text, max_words=10, overlap_words=2)

    assert len(chunks) >= 3
    assert chunks[0].source_id == "test_source"
    assert chunks[0].metadata["language"] == "en"
    assert chunks[0].metadata["word_count"] <= 10
