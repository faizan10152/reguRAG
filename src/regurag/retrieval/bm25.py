from __future__ import annotations

import math
import re
from collections import Counter

from regurag.schemas import Chunk, RetrievalResult

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


class SimpleBM25Retriever:
    """Small BM25 implementation for a transparent lexical baseline.

    In production we can replace this with bm25s, OpenSearch, Elasticsearch,
    or Qdrant sparse vectors. Keeping this baseline visible is useful because
    interviews often ask why vector search alone is not enough.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs = self._compute_doc_freqs()
        self.idf = self._compute_idf()

    def _compute_doc_freqs(self) -> Counter[str]:
        doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freqs.update(set(tokens))
        return doc_freqs

    def _compute_idf(self) -> dict[str, float]:
        document_count = len(self.chunks)
        if document_count == 0:
            return {}

        return {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in self.doc_freqs.items()
        }

    def _score_document(self, query_tokens: list[str], doc_index: int) -> float:
        if not self.avg_doc_length:
            return 0.0

        score = 0.0
        doc_length = self.doc_lengths[doc_index]
        term_freqs = self.term_freqs[doc_index]

        for term in query_tokens:
            if term not in term_freqs:
                continue

            term_frequency = term_freqs[term]
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += self.idf.get(term, 0.0) * (
                term_frequency * (self.k1 + 1) / denominator
            )

        return score

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored = [
            (self._score_document(query_tokens, doc_index), chunk)
            for doc_index, chunk in enumerate(self.chunks)
        ]
        scored = [(score, chunk) for score, chunk in scored if score > 0]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank, retriever="bm25")
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
