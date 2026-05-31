from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    url: str
    language: str
    jurisdiction: str
    source_type: str
    domain_tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> SourceRecord:
        return cls(
            source_id=str(payload["source_id"]),
            title=str(payload["title"]),
            url=str(payload["url"]),
            language=str(payload["language"]),
            jurisdiction=str(payload["jurisdiction"]),
            source_type=str(payload["source_type"]),
            domain_tags=[str(tag) for tag in payload.get("domain_tags", [])],
        )


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: JsonDict) -> Chunk:
        return cls(
            chunk_id=str(payload["chunk_id"]),
            source_id=str(payload["source_id"]),
            text=str(payload["text"]),
            metadata=dict(payload.get("metadata", {})),
        )

    @property
    def citation_label(self) -> str:
        return f"{self.source_id}:{self.chunk_id}"


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def citation_label(self) -> str:
        return self.chunk.citation_label


@dataclass(frozen=True)
class CitationValidation:
    cited_labels: set[str]
    available_labels: set[str]
    missing_labels: set[str]

    @property
    def is_supported(self) -> bool:
        return bool(self.cited_labels) and not self.missing_labels
