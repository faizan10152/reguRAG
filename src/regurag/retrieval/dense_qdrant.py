from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from regurag.schemas import Chunk, RetrievalResult

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "regurag_chunks"


def point_id_for_chunk(chunk: Chunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.citation_label))


def chunk_to_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "text": chunk.text,
        "metadata": chunk.metadata,
        "citation_label": chunk.citation_label,
    }


def chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        text=str(payload["text"]),
        metadata=dict(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class DenseSearchHit:
    chunk: Chunk
    score: float


class QdrantDenseRetriever:
    """Qdrant-backed dense vector retrieval."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_COLLECTION,
        location: str | None = None,
        path: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant dense retrieval needs the optional rag dependencies. "
                'Install with: uv pip install -e ".[rag]"'
            ) from exc

        if path:
            self.client = QdrantClient(path=path)
        elif location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def recreate_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        batch_size: int = 64,
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        total = 0
        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start : batch_start + batch_size]
            batch_vectors = vectors[batch_start : batch_start + batch_size]
            points = [
                self._make_point(chunk, vector)
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)
        return total

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievalResult]:
        points = self._query_points(query_vector=query_vector, top_k=top_k)
        results: list[RetrievalResult] = []

        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            chunk = chunk_from_payload(payload)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(point.score),
                    rank=rank,
                    retriever="dense_qdrant",
                )
            )

        return results

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _query_points(self, query_vector: list[float], top_k: int):
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            return response.points

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    @staticmethod
    def _make_point(chunk: Chunk, vector: list[float]):
        from qdrant_client.models import PointStruct

        return PointStruct(
            id=point_id_for_chunk(chunk),
            vector=vector,
            payload=chunk_to_payload(chunk),
        )


def source_ids(results: Iterable[RetrievalResult]) -> list[str]:
    return [result.chunk.source_id for result in results]
