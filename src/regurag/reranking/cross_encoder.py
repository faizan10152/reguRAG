from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from regurag.schemas import Chunk, RetrievalResult

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def dedupe_results(results: Sequence[RetrievalResult]) -> list[RetrievalResult]:
    """Remove duplicate chunks while preserving the first candidate ranking."""

    seen: set[str] = set()
    deduped: list[RetrievalResult] = []
    for result in results:
        if result.citation_label in seen:
            continue
        seen.add(result.citation_label)
        deduped.append(result)
    return deduped


def format_rerank_passage(chunk: Chunk) -> str:
    """Add source context to chunk text before cross-encoder scoring."""

    title = str(chunk.metadata.get("title") or chunk.source_id)
    heading = str(chunk.metadata.get("section_heading") or "").strip()
    parts = [title]
    if heading:
        parts.append(heading)
    parts.append(chunk.text)
    return "\n".join(parts)


def rerank_results(
    results: Sequence[RetrievalResult],
    scores: Sequence[float],
    *,
    top_k: int | None = None,
    retriever_name: str = "rerank",
) -> list[RetrievalResult]:
    """Return candidates ordered by cross-encoder relevance score."""

    if len(results) != len(scores):
        raise ValueError("results and scores must have the same length")

    ranked = sorted(
        zip(results, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    if top_k is not None:
        ranked = ranked[:top_k]

    return [
        RetrievalResult(
            chunk=result.chunk,
            score=float(score),
            rank=rank,
            retriever=retriever_name,
        )
        for rank, (result, score) in enumerate(ranked, start=1)
    ]


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    model_name: str = DEFAULT_RERANKER_MODEL
    batch_size: int = 4
    max_length: int = 512
    retriever_name: str = "hybrid_rerank"


class CrossEncoderReranker:
    """Cross-encoder reranker for query-candidate pairs.

    A bi-encoder retrieves candidates quickly by comparing independent vectors.
    This reranker is slower because it reads the query and each candidate
    together, but that pairwise scoring is usually better for top-k ordering.
    """

    def __init__(self, config: CrossEncoderRerankerConfig | None = None) -> None:
        self.config = config or CrossEncoderRerankerConfig()

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking needs the optional rag dependencies. "
                'Install with: uv pip install -e ".[rag]"'
            ) from exc

        self._model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
        )

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        candidates = dedupe_results(results)
        if not candidates:
            return []

        pairs = [(query, format_rerank_passage(result.chunk)) for result in candidates]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        scores = [_coerce_score(score) for score in raw_scores]
        return rerank_results(
            candidates,
            scores,
            top_k=top_k,
            retriever_name=self.config.retriever_name,
        )


def _coerce_score(score: Any) -> float:
    if hasattr(score, "item"):
        return float(score.item())
    if isinstance(score, Sequence) and not isinstance(score, str):
        if len(score) != 1:
            raise ValueError(f"Expected scalar reranker score, got {score!r}")
        return float(score[0])
    return float(score)
