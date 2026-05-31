from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from regurag.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.schemas import Chunk, RetrievalResult

RetrieverSearch = Callable[[str], list[RetrievalResult]]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    language: str
    expected_sources: set[str]
    must_refuse: bool
    domain: str = "unknown"
    question_type: str = "unknown"
    difficulty: str = "unknown"
    structural_difficulty: str = "unknown"
    review_status: str = "unknown"
    notes: str = ""

    @property
    def relevant_sources(self) -> set[str]:
        """Backward-compatible alias for the initial seed benchmark schema."""

        return self.expected_sources

    @property
    def is_answerable(self) -> bool:
        return bool(self.expected_sources)


@dataclass(frozen=True)
class RetrievalEvalRow:
    retriever: str
    question_id: str
    question: str
    domain: str
    question_type: str
    difficulty: str
    language: str
    structural_difficulty: str
    must_refuse: bool
    expected_sources: set[str]
    retrieved_citations: list[str]
    retrieved_sources: list[str]
    unique_retrieved_sources: list[str]
    source_recall_at_k: float
    source_precision_at_k: float
    source_mrr: float

    @property
    def is_answerable(self) -> bool:
        return bool(self.expected_sources)

    @property
    def source_hit_at_k(self) -> float:
        return 1.0 if self.source_mrr > 0 else 0.0

    @property
    def missing_sources(self) -> set[str]:
        return self.expected_sources - set(self.retrieved_sources)


@dataclass(frozen=True)
class RetrievalEvalSummary:
    retriever: str
    rows: int
    answerable_rows: int
    source_recall_at_k: float
    source_precision_at_k: float
    source_hit_rate_at_k: float
    source_mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "retriever": self.retriever,
            "rows": self.rows,
            "answerable_rows": self.answerable_rows,
            "source_recall_at_k": self.source_recall_at_k,
            "source_precision_at_k": self.source_precision_at_k,
            "source_hit_rate_at_k": self.source_hit_rate_at_k,
            "source_mrr": self.source_mrr,
        }


@dataclass(frozen=True)
class RetrievalEvalReport:
    rows: list[RetrievalEvalRow]
    top_k: int
    candidate_k: int

    @property
    def retrievers(self) -> list[str]:
        return sorted({row.retriever for row in self.rows})

    @property
    def answerable_rows(self) -> list[RetrievalEvalRow]:
        return [row for row in self.rows if row.is_answerable]

    @property
    def mean_source_recall_at_k(self) -> float:
        rows = self.answerable_rows
        if not rows:
            return 0.0
        return mean(row.source_recall_at_k for row in rows)

    def rows_for_retriever(self, retriever: str) -> list[RetrievalEvalRow]:
        return [row for row in self.rows if row.retriever == retriever]

    def summary_for_retriever(self, retriever: str) -> RetrievalEvalSummary:
        rows = self.rows_for_retriever(retriever)
        answerable_rows = [row for row in rows if row.is_answerable]
        return _summarize_rows(retriever, rows, answerable_rows)

    def summaries(self) -> list[RetrievalEvalSummary]:
        return [self.summary_for_retriever(retriever) for retriever in self.retrievers]

    def breakdown(self, group_by: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[RetrievalEvalRow]] = defaultdict(list)
        for row in self.rows:
            grouped[(row.retriever, str(getattr(row, group_by)))].append(row)

        output: list[dict[str, Any]] = []
        for (retriever, group_value), rows in sorted(grouped.items()):
            answerable_rows = [row for row in rows if row.is_answerable]
            summary = _summarize_rows(retriever, rows, answerable_rows)
            payload = summary.to_dict()
            payload[group_by] = group_value
            output.append(payload)
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "summaries": [summary.to_dict() for summary in self.summaries()],
            "breakdowns": {
                "domain": self.breakdown("domain"),
                "difficulty": self.breakdown("difficulty"),
                "language": self.breakdown("language"),
                "structural_difficulty": self.breakdown("structural_difficulty"),
            },
            "rows": [
                {
                    "retriever": row.retriever,
                    "question_id": row.question_id,
                    "question": row.question,
                    "domain": row.domain,
                    "question_type": row.question_type,
                    "difficulty": row.difficulty,
                    "language": row.language,
                    "structural_difficulty": row.structural_difficulty,
                    "must_refuse": row.must_refuse,
                    "expected_sources": sorted(row.expected_sources),
                    "retrieved_citations": row.retrieved_citations,
                    "retrieved_sources": row.retrieved_sources,
                    "unique_retrieved_sources": row.unique_retrieved_sources,
                    "source_recall_at_k": row.source_recall_at_k,
                    "source_precision_at_k": row.source_precision_at_k,
                    "source_hit_at_k": row.source_hit_at_k,
                    "source_mrr": row.source_mrr,
                    "missing_sources": sorted(row.missing_sources),
                }
                for row in self.rows
            ],
        }


def _summarize_rows(
    retriever: str,
    rows: list[RetrievalEvalRow],
    answerable_rows: list[RetrievalEvalRow],
) -> RetrievalEvalSummary:
    if not answerable_rows:
        return RetrievalEvalSummary(
            retriever=retriever,
            rows=len(rows),
            answerable_rows=0,
            source_recall_at_k=0.0,
            source_precision_at_k=0.0,
            source_hit_rate_at_k=0.0,
            source_mrr=0.0,
        )

    return RetrievalEvalSummary(
        retriever=retriever,
        rows=len(rows),
        answerable_rows=len(answerable_rows),
        source_recall_at_k=mean(row.source_recall_at_k for row in answerable_rows),
        source_precision_at_k=mean(row.source_precision_at_k for row in answerable_rows),
        source_hit_rate_at_k=mean(row.source_hit_at_k for row in answerable_rows),
        source_mrr=mean(row.source_mrr for row in answerable_rows),
    )


def load_golden_questions(path: str | Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            expected_sources = payload.get("expected_sources", payload.get("relevant_sources", []))
            questions.append(
                GoldenQuestion(
                    question_id=str(payload["id"]),
                    question=str(payload["question"]),
                    language=str(payload["language"]),
                    expected_sources={str(source) for source in expected_sources},
                    must_refuse=bool(payload.get("must_refuse", False)),
                    domain=str(payload.get("domain", "unknown")),
                    question_type=str(payload.get("type", payload.get("answer_type", "unknown"))),
                    difficulty=str(payload.get("difficulty", "unknown")),
                    structural_difficulty=str(payload.get("structural_difficulty", "unknown")),
                    review_status=str(payload.get("review_status", "unknown")),
                    notes=str(payload.get("notes", "")),
                )
            )
    return questions


def evaluate_retrieval_runs(
    questions: list[GoldenQuestion],
    retriever_runs: Mapping[str, RetrieverSearch],
    top_k: int = 5,
    candidate_k: int | None = None,
) -> RetrievalEvalReport:
    candidate_k = candidate_k or top_k
    rows: list[RetrievalEvalRow] = []

    for retriever_name, search in retriever_runs.items():
        for question in questions:
            results = search(question.question)[:top_k]
            retrieved_citations = [result.citation_label for result in results]
            retrieved_sources = [result.chunk.source_id for result in results]
            unique_sources = _ordered_unique(retrieved_sources)
            rows.append(
                RetrievalEvalRow(
                    retriever=retriever_name,
                    question_id=question.question_id,
                    question=question.question,
                    domain=question.domain,
                    question_type=question.question_type,
                    difficulty=question.difficulty,
                    language=question.language,
                    structural_difficulty=question.structural_difficulty,
                    must_refuse=question.must_refuse,
                    expected_sources=question.expected_sources,
                    retrieved_citations=retrieved_citations,
                    retrieved_sources=retrieved_sources,
                    unique_retrieved_sources=unique_sources,
                    source_recall_at_k=recall_at_k(
                        retrieved_sources,
                        question.expected_sources,
                        top_k,
                    ),
                    source_precision_at_k=precision_at_k(
                        retrieved_sources,
                        question.expected_sources,
                        top_k,
                    ),
                    source_mrr=mean_reciprocal_rank(
                        retrieved_sources,
                        question.expected_sources,
                    ),
                )
            )

    return RetrievalEvalReport(rows=rows, top_k=top_k, candidate_k=candidate_k)


def evaluate_bm25_source_recall(
    chunks: list[Chunk],
    questions: list[GoldenQuestion],
    top_k: int = 5,
) -> RetrievalEvalReport:
    retriever = SimpleBM25Retriever(chunks)
    return evaluate_retrieval_runs(
        questions=questions,
        retriever_runs={"bm25": lambda query: retriever.search(query, top_k=top_k)},
        top_k=top_k,
        candidate_k=top_k,
    )


def write_json_report(report: RetrievalEvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_markdown_report(report: RetrievalEvalReport) -> str:
    lines = [
        "# Retrieval Evaluation Report",
        "",
        "This report evaluates retrieval only. It checks whether the retriever finds the expected source documents before answer generation.",
        "",
        f"- Top K: {report.top_k}",
        f"- Candidate K: {report.candidate_k}",
        "",
        "## Summary",
        "",
        "| Retriever | Rows | Answerable | Recall@K | Precision@K | Hit@K | MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in report.summaries():
        lines.append(
            "| "
            f"{summary.retriever} | "
            f"{summary.rows} | "
            f"{summary.answerable_rows} | "
            f"{summary.source_recall_at_k:.3f} | "
            f"{summary.source_precision_at_k:.3f} | "
            f"{summary.source_hit_rate_at_k:.3f} | "
            f"{summary.source_mrr:.3f} |"
        )

    lines.extend(_format_breakdown_table(report, "domain", "Domain"))
    lines.extend(_format_breakdown_table(report, "structural_difficulty", "Structural Difficulty"))
    lines.extend(_format_worst_misses(report))
    lines.extend(_format_refusal_rows(report))
    return "\n".join(lines) + "\n"


def write_markdown_report(report: RetrievalEvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_report(report), encoding="utf-8")


def _format_breakdown_table(
    report: RetrievalEvalReport,
    group_by: str,
    title: str,
) -> list[str]:
    lines = [
        "",
        f"## Breakdown By {title}",
        "",
        f"| Retriever | {title} | Answerable | Recall@K | Hit@K | MRR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report.breakdown(group_by):
        lines.append(
            "| "
            f"{row['retriever']} | "
            f"{row[group_by]} | "
            f"{row['answerable_rows']} | "
            f"{row['source_recall_at_k']:.3f} | "
            f"{row['source_hit_rate_at_k']:.3f} | "
            f"{row['source_mrr']:.3f} |"
        )
    return lines


def _format_worst_misses(report: RetrievalEvalReport, limit: int = 12) -> list[str]:
    misses = [
        row
        for row in report.rows
        if row.is_answerable and (row.source_recall_at_k < 1.0 or row.source_mrr == 0.0)
    ]
    misses.sort(key=lambda row: (row.source_recall_at_k, row.source_mrr, row.question_id))

    lines = [
        "",
        "## Worst Misses",
        "",
        "| Retriever | Question | Missing Sources | Retrieved Sources | Recall@K |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for row in misses[:limit]:
        lines.append(
            "| "
            f"{row.retriever} | "
            f"{row.question_id} | "
            f"{', '.join(sorted(row.missing_sources)) or '-'} | "
            f"{', '.join(row.unique_retrieved_sources) or '-'} | "
            f"{row.source_recall_at_k:.3f} |"
        )
    return lines


def _format_refusal_rows(report: RetrievalEvalReport) -> list[str]:
    refusal_rows = [row for row in report.rows if row.must_refuse]
    if not refusal_rows:
        return []

    lines = [
        "",
        "## Refusal Questions",
        "",
        "These rows are intentionally excluded from source recall because no source is expected to fully answer them. Retrieved sources are treated as distractors until we add confidence thresholds and answer-level refusal evaluation.",
        "",
        "| Retriever | Question | Retrieved Sources |",
        "| --- | --- | --- |",
    ]
    for row in refusal_rows:
        lines.append(
            "| "
            f"{row.retriever} | "
            f"{row.question_id} | "
            f"{', '.join(row.unique_retrieved_sources) or '-'} |"
        )
    return lines
