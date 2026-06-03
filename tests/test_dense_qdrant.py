from regurag.retrieval.dense_qdrant import chunk_from_payload, chunk_to_payload, point_id_for_chunk
from regurag.schemas import Chunk


def test_qdrant_payload_round_trip() -> None:
    chunk = Chunk(
        chunk_id="aaaabbbbccccdddd",
        source_id="doc",
        text="AI Act evidence",
        metadata={"language": "en", "url": "https://example.com"},
    )

    payload = chunk_to_payload(chunk)
    restored = chunk_from_payload(payload)

    assert payload["citation_label"] == "doc:aaaabbbbccccdddd"
    assert restored == chunk


def test_point_id_for_chunk_is_stable_uuid() -> None:
    chunk = Chunk("aaaabbbbccccdddd", "doc", "text", {})

    assert point_id_for_chunk(chunk) == point_id_for_chunk(chunk)
    assert len(point_id_for_chunk(chunk)) == 36
