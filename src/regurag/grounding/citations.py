from __future__ import annotations

import re
from collections.abc import Iterable

from regurag.schemas import CitationValidation, RetrievalResult

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.-]+:[A-Fa-f0-9]{16})\]")


def extract_citation_labels(answer_text: str) -> set[str]:
    return set(CITATION_RE.findall(answer_text))


def validate_citations(
    answer_text: str,
    retrieved_results: Iterable[RetrievalResult],
) -> CitationValidation:
    cited = extract_citation_labels(answer_text)
    return validate_citation_labels(cited, retrieved_results)


def validate_citation_labels(
    cited_labels: Iterable[str],
    retrieved_results: Iterable[RetrievalResult],
) -> CitationValidation:
    cited = set(cited_labels)
    available = {result.citation_label for result in retrieved_results}
    missing = cited - available
    return CitationValidation(cited_labels=cited, available_labels=available, missing_labels=missing)


def should_refuse_answer(validation: CitationValidation, min_citations: int = 1) -> bool:
    return len(validation.cited_labels) < min_citations or bool(validation.missing_labels)
