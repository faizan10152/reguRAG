from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from regurag.schemas import RetrievalResult


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[RetrievalResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Fuse ranked lists while depending on ranks rather than raw scores."""

    scores: dict[str, float] = defaultdict(float)
    best_result: dict[str, RetrievalResult] = {}

    for run in runs:
        for result in run:
            label = result.citation_label
            scores[label] += 1.0 / (k + result.rank)
            if label not in best_result or result.score > best_result[label].score:
                best_result[label] = result

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]

    return [
        RetrievalResult(
            chunk=best_result[label].chunk,
            score=score,
            rank=rank,
            retriever="rrf",
        )
        for rank, (label, score) in enumerate(fused, start=1)
    ]
