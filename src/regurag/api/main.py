from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from regurag.grounding.citations import validate_citations
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.sample_data import sample_chunks
from regurag.storage.jsonl import read_chunks_jsonl


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    label: str
    title: str
    url: str | None = None
    section_heading: str | None = None
    snippet: str
    score: float


class QueryResponse(BaseModel):
    mode: str
    answer: str
    citations: list[Citation]
    supported: bool


def _load_chunks() -> list:
    chunks_path = Path(os.getenv("REGURAG_CHUNKS_PATH", "data/processed/chunks.jsonl"))
    chunks = read_chunks_jsonl(chunks_path)
    return chunks or sample_chunks()


app = FastAPI(
    title="ReguRAG",
    version="0.1.0",
    description="Bilingual production-style RAG assistant for AI Act/GDPR research.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    chunks = _load_chunks()
    retriever = SimpleBM25Retriever(chunks)
    results = retriever.search(request.question, top_k=request.top_k)

    citations = [
        Citation(
            label=result.citation_label,
            title=str(result.chunk.metadata.get("title", result.chunk.source_id)),
            url=result.chunk.metadata.get("url"),
            section_heading=result.chunk.metadata.get("section_heading"),
            snippet=result.chunk.text[:500],
            score=result.score,
        )
        for result in results
    ]

    if not results:
        return QueryResponse(
            mode="retrieval_only",
            answer=(
                "I cannot answer from the indexed corpus yet because no supporting "
                "chunks were retrieved."
            ),
            citations=[],
            supported=False,
        )

    citation_text = " ".join(f"[{result.citation_label}]" for result in results[:2])
    answer = (
        "Retrieval-only MVP: these are the most relevant chunks I found. "
        "The next milestone will add an LLM answer generator that must cite retrieved "
        f"evidence. {citation_text}"
    )
    validation = validate_citations(answer, results)

    return QueryResponse(
        mode="retrieval_only",
        answer=answer,
        citations=citations,
        supported=validation.is_supported,
    )
