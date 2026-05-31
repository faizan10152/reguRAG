from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TextEmbedder(Protocol):
    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, query: str) -> list[float]: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 16
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""


def with_prefix(prefix: str, text: str) -> str:
    if not prefix:
        return text
    return f"{prefix}{text}"


def l2_normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Thin adapter around sentence-transformers.

    The import is intentionally lazy so tests and BM25-only development do not
    require heavy ML dependencies.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval needs the optional rag dependencies. "
                'Install with: uv pip install -e ".[rag]"'
            ) from exc

        self._model = SentenceTransformer(self.config.model_name)

    @property
    def dimension(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            sample = self.encode_query("dimension probe")
            return len(sample)
        return int(dimension)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = [with_prefix(self.config.document_prefix, text) for text in texts]
        embeddings = self._model.encode(
            prepared,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=True,
        )
        return [embedding.tolist() for embedding in embeddings]

    def encode_query(self, query: str) -> list[float]:
        prepared = with_prefix(self.config.query_prefix, query)
        embedding = self._model.encode(
            prepared,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
        )
        return embedding.tolist()

