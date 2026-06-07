from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from regurag.evaluation.metrics import recall_at_k
from regurag.evaluation.runner import GoldenQuestion
from regurag.generation.answer import AnswerGenerationResult, generate_grounded_answer
from regurag.generation.litellm_client import LLMClient
from regurag.schemas import RetrievalResult

AnswerSearch = Callable[[str], list[RetrievalResult]]
ProgressCallback = Callable[[int, int, GoldenQuestion], None]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


@dataclass(frozen=True)
class AnswerEvalRow:
    run_name: str
    question_id: str
    question: str
    domain: str
    question_type: str
    difficulty: str
    language: str
    structural_difficulty: str
    must_refuse: bool
    expected_sources: set[str]
    expected_citations: set[str]
    retrieved_citations: list[str]
    retrieved_sources: list[str]
    unique_retrieved_sources: list[str]
    source_recall_at_k: float
    answer: str
    answer_citations: list[str]
    confidence: str
    unsupported_claims: list[str]
    should_refuse: bool
    refusal_reason: str | None
    supported: bool
    guardrail_triggered: bool
    citation_validation_supported: bool
    missing_generated_citations: set[str]
    valid_structured_output: bool
    latency_seconds: float
    error: str | None = None
    manual_label: str = "unreviewed"
    manual_notes: str = ""

    @property
    def is_answerable(self) -> bool:
        return bool(self.expected_sources)

    @property
    def has_expected_citations(self) -> bool:
        return bool(self.expected_citations)

    @property
    def refusal_correct(self) -> bool:
        return self.should_refuse == self.must_refuse

    @property
    def has_answer_citations(self) -> bool:
        return bool(self.answer_citations)

    @property
    def expected_citation_hit(self) -> float:
        if not self.expected_citations:
            return 0.0
        return 1.0 if set(self.answer_citations) & self.expected_citations else 0.0

    @property
    def answer_preview(self) -> str:
        return self.answer.replace("\n", " ")[:240]


@dataclass(frozen=True)
class AnswerEvalSummary:
    run_name: str
    rows: int
    answerable_rows: int
    refusal_rows: int
    supported_rate: float
    answerable_supported_rate: float
    guardrail_rate: float
    valid_structured_output_rate: float
    refusal_accuracy: float
    expected_refusal_success_rate: float
    cited_answer_rows: int
    citation_validity_rate: float
    source_recall_at_k: float
    citation_labeled_rows: int
    expected_citation_hit_rate: float
    mean_latency_seconds: float
    error_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "rows": self.rows,
            "answerable_rows": self.answerable_rows,
            "refusal_rows": self.refusal_rows,
            "supported_rate": self.supported_rate,
            "answerable_supported_rate": self.answerable_supported_rate,
            "guardrail_rate": self.guardrail_rate,
            "valid_structured_output_rate": self.valid_structured_output_rate,
            "refusal_accuracy": self.refusal_accuracy,
            "expected_refusal_success_rate": self.expected_refusal_success_rate,
            "cited_answer_rows": self.cited_answer_rows,
            "citation_validity_rate": self.citation_validity_rate,
            "source_recall_at_k": self.source_recall_at_k,
            "citation_labeled_rows": self.citation_labeled_rows,
            "expected_citation_hit_rate": self.expected_citation_hit_rate,
            "mean_latency_seconds": self.mean_latency_seconds,
            "error_rows": self.error_rows,
        }


@dataclass(frozen=True)
class AnswerEvalReport:
    rows: list[AnswerEvalRow]
    run_name: str
    retriever: str
    llm_model: str
    top_k: int
    candidate_k: int
    max_context_chars: int

    def summary(self) -> AnswerEvalSummary:
        return _summarize_answer_rows(self.run_name, self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "retriever": self.retriever,
            "llm_model": self.llm_model,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "max_context_chars": self.max_context_chars,
            "summary": self.summary().to_dict(),
            "rows": [
                {
                    "run_name": row.run_name,
                    "question_id": row.question_id,
                    "question": row.question,
                    "domain": row.domain,
                    "question_type": row.question_type,
                    "difficulty": row.difficulty,
                    "language": row.language,
                    "structural_difficulty": row.structural_difficulty,
                    "must_refuse": row.must_refuse,
                    "expected_sources": sorted(row.expected_sources),
                    "expected_citations": sorted(row.expected_citations),
                    "retrieved_citations": row.retrieved_citations,
                    "retrieved_sources": row.retrieved_sources,
                    "unique_retrieved_sources": row.unique_retrieved_sources,
                    "source_recall_at_k": row.source_recall_at_k,
                    "answer": row.answer,
                    "answer_citations": row.answer_citations,
                    "confidence": row.confidence,
                    "unsupported_claims": row.unsupported_claims,
                    "should_refuse": row.should_refuse,
                    "refusal_reason": row.refusal_reason,
                    "supported": row.supported,
                    "guardrail_triggered": row.guardrail_triggered,
                    "citation_validation_supported": row.citation_validation_supported,
                    "missing_generated_citations": sorted(row.missing_generated_citations),
                    "valid_structured_output": row.valid_structured_output,
                    "refusal_correct": row.refusal_correct,
                    "expected_citation_hit": row.expected_citation_hit,
                    "latency_seconds": row.latency_seconds,
                    "error": row.error,
                    "manual_label": row.manual_label,
                    "manual_notes": row.manual_notes,
                }
                for row in self.rows
            ],
        }


def evaluate_answer_run(
    *,
    questions: list[GoldenQuestion],
    search: AnswerSearch,
    llm: LLMClient,
    run_name: str,
    retriever: str,
    llm_model: str,
    top_k: int = 5,
    candidate_k: int = 20,
    max_context_chars: int = 4500,
    min_citations: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> AnswerEvalReport:
    rows: list[AnswerEvalRow] = []

    total_questions = len(questions)
    for index, question in enumerate(questions, start=1):
        if progress_callback:
            progress_callback(index, total_questions, question)
        started_at = time.perf_counter()
        results: list[RetrievalResult] = []
        try:
            results = search(question.question)[:top_k]
            answer_result = generate_grounded_answer(
                question=question.question,
                retrieved_results=results,
                llm=llm,
                max_chars_per_chunk=max_context_chars,
                min_citations=min_citations,
            )
            rows.append(
                _row_from_answer_result(
                    run_name=run_name,
                    question=question,
                    results=results,
                    answer_result=answer_result,
                    top_k=top_k,
                    latency_seconds=time.perf_counter() - started_at,
                )
            )
        except Exception as exc:
            rows.append(
                _error_row(
                    run_name=run_name,
                    question=question,
                    results=results,
                    top_k=top_k,
                    latency_seconds=time.perf_counter() - started_at,
                    error=exc,
                )
            )

    return AnswerEvalReport(
        rows=rows,
        run_name=run_name,
        retriever=retriever,
        llm_model=llm_model,
        top_k=top_k,
        candidate_k=candidate_k,
        max_context_chars=max_context_chars,
    )


def _row_from_answer_result(
    *,
    run_name: str,
    question: GoldenQuestion,
    results: list[RetrievalResult],
    answer_result: AnswerGenerationResult,
    top_k: int,
    latency_seconds: float,
) -> AnswerEvalRow:
    retrieved_citations = [result.citation_label for result in results]
    retrieved_sources = [result.chunk.source_id for result in results]
    refusal_reason = answer_result.answer.refusal_reason
    return AnswerEvalRow(
        run_name=run_name,
        question_id=question.question_id,
        question=question.question,
        domain=question.domain,
        question_type=question.question_type,
        difficulty=question.difficulty,
        language=question.language,
        structural_difficulty=question.structural_difficulty,
        must_refuse=question.must_refuse,
        expected_sources=question.expected_sources,
        expected_citations=question.expected_citations,
        retrieved_citations=retrieved_citations,
        retrieved_sources=retrieved_sources,
        unique_retrieved_sources=_ordered_unique(retrieved_sources),
        source_recall_at_k=recall_at_k(retrieved_sources, question.expected_sources, top_k),
        answer=answer_result.answer.answer,
        answer_citations=answer_result.answer.citations,
        confidence=answer_result.answer.confidence,
        unsupported_claims=answer_result.answer.unsupported_claims,
        should_refuse=answer_result.answer.should_refuse,
        refusal_reason=refusal_reason,
        supported=answer_result.supported,
        guardrail_triggered=answer_result.guardrail_triggered,
        citation_validation_supported=answer_result.citation_validation.is_supported,
        missing_generated_citations=answer_result.citation_validation.missing_labels,
        valid_structured_output=not _is_invalid_structured_output(refusal_reason),
        latency_seconds=latency_seconds,
    )


def _error_row(
    *,
    run_name: str,
    question: GoldenQuestion,
    results: list[RetrievalResult],
    top_k: int,
    latency_seconds: float,
    error: Exception,
) -> AnswerEvalRow:
    retrieved_citations = [result.citation_label for result in results]
    retrieved_sources = [result.chunk.source_id for result in results]
    return AnswerEvalRow(
        run_name=run_name,
        question_id=question.question_id,
        question=question.question,
        domain=question.domain,
        question_type=question.question_type,
        difficulty=question.difficulty,
        language=question.language,
        structural_difficulty=question.structural_difficulty,
        must_refuse=question.must_refuse,
        expected_sources=question.expected_sources,
        expected_citations=question.expected_citations,
        retrieved_citations=retrieved_citations,
        retrieved_sources=retrieved_sources,
        unique_retrieved_sources=_ordered_unique(retrieved_sources),
        source_recall_at_k=recall_at_k(retrieved_sources, question.expected_sources, top_k),
        answer="",
        answer_citations=[],
        confidence="low",
        unsupported_claims=[],
        should_refuse=False,
        refusal_reason=None,
        supported=False,
        guardrail_triggered=True,
        citation_validation_supported=False,
        missing_generated_citations=set(),
        valid_structured_output=False,
        latency_seconds=latency_seconds,
        error=str(error),
    )


def _is_invalid_structured_output(refusal_reason: str | None) -> bool:
    return bool(refusal_reason and refusal_reason.startswith("Invalid LLM response:"))


def _summarize_answer_rows(run_name: str, rows: list[AnswerEvalRow]) -> AnswerEvalSummary:
    answerable_rows = [row for row in rows if row.is_answerable]
    refusal_rows = [row for row in rows if row.must_refuse]
    cited_rows = [row for row in rows if row.has_answer_citations]
    citation_labeled_rows = [row for row in rows if row.has_expected_citations]

    return AnswerEvalSummary(
        run_name=run_name,
        rows=len(rows),
        answerable_rows=len(answerable_rows),
        refusal_rows=len(refusal_rows),
        supported_rate=_mean_bool(row.supported for row in rows),
        answerable_supported_rate=_mean_bool(row.supported for row in answerable_rows),
        guardrail_rate=_mean_bool(row.guardrail_triggered for row in rows),
        valid_structured_output_rate=_mean_bool(row.valid_structured_output for row in rows),
        refusal_accuracy=_mean_bool(row.refusal_correct for row in rows),
        expected_refusal_success_rate=_mean_bool(row.should_refuse for row in refusal_rows),
        cited_answer_rows=len(cited_rows),
        citation_validity_rate=_mean_bool(row.citation_validation_supported for row in cited_rows),
        source_recall_at_k=_mean_float(row.source_recall_at_k for row in answerable_rows),
        citation_labeled_rows=len(citation_labeled_rows),
        expected_citation_hit_rate=_mean_float(
            row.expected_citation_hit for row in citation_labeled_rows
        ),
        mean_latency_seconds=_mean_float(row.latency_seconds for row in rows),
        error_rows=sum(1 for row in rows if row.error),
    )


def _mean_bool(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return mean(1.0 if item else 0.0 for item in items)


def _mean_float(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return mean(float(item) for item in items)


def write_answer_json_report(report: AnswerEvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_answer_markdown_report(report: AnswerEvalReport) -> str:
    summary = report.summary()
    lines = [
        "# Answer Evaluation Report",
        "",
        "This report evaluates the full answer path: retrieval, prompt construction, generation, citation validation, and refusal behavior.",
        "",
        f"- Run: {report.run_name}",
        f"- Retriever: {report.retriever}",
        f"- LLM model: {report.llm_model}",
        f"- Top K: {report.top_k}",
        f"- Candidate K: {report.candidate_k}",
        f"- Max context chars per chunk: {report.max_context_chars}",
        "",
        "## Summary",
        "",
        "| Rows | Answerable | Refusal Rows | Supported Rate | Answerable Supported | Valid JSON | Refusal Accuracy | Expected Refusal Success | Citation Validity | Source Recall@K | Expected Citation Hit | Mean Latency | Errors |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        f"{summary.rows} | "
        f"{summary.answerable_rows} | "
        f"{summary.refusal_rows} | "
        f"{summary.supported_rate:.3f} | "
        f"{summary.answerable_supported_rate:.3f} | "
        f"{summary.valid_structured_output_rate:.3f} | "
        f"{summary.refusal_accuracy:.3f} | "
        f"{summary.expected_refusal_success_rate:.3f} | "
        f"{summary.citation_validity_rate:.3f} | "
        f"{summary.source_recall_at_k:.3f} | "
        f"{summary.expected_citation_hit_rate:.3f} | "
        f"{summary.mean_latency_seconds:.2f}s | "
        f"{summary.error_rows} |",
    ]
    lines.extend(_format_answer_rows_needing_review(report))
    lines.extend(_format_manual_review_table(report))
    return "\n".join(lines) + "\n"


def write_answer_markdown_report(report: AnswerEvalReport, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_answer_markdown_report(report), encoding="utf-8")


def _format_answer_rows_needing_review(report: AnswerEvalReport, limit: int = 15) -> list[str]:
    rows = [
        row
        for row in report.rows
        if row.error
        or row.guardrail_triggered
        or not row.refusal_correct
        or (row.is_answerable and not row.supported)
    ]
    rows.sort(key=lambda row: (not row.error, not row.guardrail_triggered, row.question_id))

    lines = [
        "",
        "## Rows Needing Review",
        "",
        "| Question | Expected Refusal | Actual Refusal | Supported | Guardrail | Reason / Error | Retrieved Sources |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        reason = row.error or row.refusal_reason or "-"
        lines.append(
            "| "
            f"{row.question_id} | "
            f"{row.must_refuse} | "
            f"{row.should_refuse} | "
            f"{row.supported} | "
            f"{row.guardrail_triggered} | "
            f"{_escape_table_cell(reason[:160])} | "
            f"{', '.join(row.unique_retrieved_sources) or '-'} |"
        )
    return lines


def _format_manual_review_table(report: AnswerEvalReport, limit: int = 20) -> list[str]:
    lines = [
        "",
        "## Manual Review Queue",
        "",
        "Rows are marked `unreviewed` until a human checks semantic correctness. Automatic metrics validate structure, support, citations, and refusal behavior, but not full legal interpretation quality.",
        "",
        "| Question | Manual Label | Confidence | Answer Preview | Citations |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report.rows[:limit]:
        lines.append(
            "| "
            f"{row.question_id} | "
            f"{row.manual_label} | "
            f"{row.confidence} | "
            f"{_escape_table_cell(row.answer_preview)} | "
            f"{', '.join(row.answer_citations) or '-'} |"
        )
    return lines


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
