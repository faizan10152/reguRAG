from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from regurag.evaluation.metrics import recall_at_k
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.schemas import Chunk


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    language: str
    relevant_sources: set[str]
    must_refuse: bool


@dataclass(frozen=True)
class RetrievalEvalRow:
    question_id: str
    question: str
    retrieved_sources: list[str]
    relevant_sources: set[str]
    source_recall_at_k: float


@dataclass(frozen=True)
class RetrievalEvalReport:
    rows: list[RetrievalEvalRow]

    @property
    def answerable_rows(self) -> list[RetrievalEvalRow]:
        return [row for row in self.rows if row.relevant_sources]

    @property
    def mean_source_recall_at_k(self) -> float:
        rows = self.answerable_rows
        if not rows:
            return 0.0
        return mean(row.source_recall_at_k for row in rows)


def load_golden_questions(path: str | Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(
                GoldenQuestion(
                    question_id=str(payload["id"]),
                    question=str(payload["question"]),
                    language=str(payload["language"]),
                    relevant_sources=set(payload.get("relevant_sources", [])),
                    must_refuse=bool(payload.get("must_refuse", False)),
                )
            )
    return questions


def evaluate_bm25_source_recall(
    chunks: list[Chunk],
    questions: list[GoldenQuestion],
    top_k: int = 5,
) -> RetrievalEvalReport:
    retriever = SimpleBM25Retriever(chunks)
    rows: list[RetrievalEvalRow] = []

    for question in questions:
        results = retriever.search(question.question, top_k=top_k)
        retrieved_sources = [result.chunk.source_id for result in results]
        rows.append(
            RetrievalEvalRow(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retrieved_sources,
                relevant_sources=question.relevant_sources,
                source_recall_at_k=recall_at_k(
                    retrieved_sources,
                    question.relevant_sources,
                    top_k,
                ),
            )
        )

    return RetrievalEvalReport(rows=rows)
