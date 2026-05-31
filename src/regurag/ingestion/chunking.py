from __future__ import annotations

import hashlib
import re

from regurag.schemas import Chunk, SourceRecord


def _word_count(text: str) -> int:
    return len(text.split())


def _looks_like_section_heading(line: str) -> bool:
    normalized = line.strip().lower()
    if not normalized:
        return False
    if normalized.startswith("\u00a7"):
        return True
    return normalized.startswith(
        (
            "article ",
            "artikel ",
            "annex ",
            "anhang ",
            "chapter ",
            "kapitel ",
            "section ",
            "abschnitt ",
            "recital ",
        )
    )


def split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _looks_like_section_heading(line) and current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line
            current_lines = []
            continue

        if _looks_like_section_heading(line) and current_heading is None:
            current_heading = line
            continue

        if line:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections or [(None, text.strip())]


def _chunk_words(words: list[str], max_words: int, overlap_words: int) -> list[list[str]]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    stride = max_words - overlap_words
    windows: list[list[str]] = []
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        if window:
            windows.append(window)
        start += stride
    return windows


def make_chunk_id(source_id: str, text: str, index: int) -> str:
    digest = hashlib.sha256(f"{source_id}:{index}:{text}".encode()).hexdigest()
    return digest[:16]


def chunk_text(
    source: SourceRecord,
    text: str,
    max_words: int = 650,
    overlap_words: int = 100,
) -> list[Chunk]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0

    for section_index, (heading, section_text) in enumerate(split_sections(text)):
        words = re.sub(r"\s+", " ", section_text).strip().split()
        if not words:
            continue

        for window in _chunk_words(words, max_words=max_words, overlap_words=overlap_words):
            chunk_body = " ".join(window).strip()
            metadata = {
                "title": source.title,
                "url": source.url,
                "language": source.language,
                "jurisdiction": source.jurisdiction,
                "source_type": source.source_type,
                "domain_tags": source.domain_tags,
                "section_heading": heading,
                "section_index": section_index,
                "chunk_index": chunk_index,
                "word_count": _word_count(chunk_body),
            }
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(source.source_id, chunk_body, chunk_index),
                    source_id=source.source_id,
                    text=chunk_body,
                    metadata=metadata,
                )
            )
            chunk_index += 1

    return chunks
