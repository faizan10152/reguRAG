from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from regurag.embeddings.encoder import EmbeddingConfig, SentenceTransformerEmbedder
from regurag.generation.answer import generate_grounded_answer
from regurag.generation.litellm_client import LiteLLMClient
from regurag.grounding.citations import validate_citations
from regurag.reranking.cross_encoder import CrossEncoderReranker, CrossEncoderRerankerConfig
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.retrieval.dense_qdrant import DEFAULT_COLLECTION, QdrantDenseRetriever
from regurag.retrieval.fusion import reciprocal_rank_fusion
from regurag.sample_data import sample_chunks
from regurag.schemas import Chunk, RetrievalResult
from regurag.storage.jsonl import read_chunks_jsonl

DEFAULT_LLM_MODEL = "ollama/qwen3:14b"
DEFAULT_LLM_API_BASE = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_QDRANT_PATH = ".qdrant/bge-m3"


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


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=3)
    retriever: Literal["bm25", "hybrid-rerank"] = "hybrid-rerank"
    top_k: int = Field(default=5, ge=1, le=10)
    candidate_k: int = Field(default=20, ge=1, le=100)
    max_context_chars: int = Field(default=4500, ge=500, le=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=900, ge=128, le=2000)
    disable_json_mode: bool = True


class Evidence(BaseModel):
    rank: int
    score: float
    retriever: str
    citation_label: str
    source_id: str
    title: str
    url: str | None = None
    section_heading: str | None = None
    snippet: str


class AnswerPayload(BaseModel):
    answer: str
    citations: list[str]
    confidence: str
    unsupported_claims: list[str]
    should_refuse: bool
    refusal_reason: str | None = None


class AnswerResponse(BaseModel):
    mode: str
    retriever: str
    llm_model: str
    supported: bool
    guardrail_triggered: bool
    answer: AnswerPayload
    retrieved_results: list[Evidence]
    missing_citations: list[str]


class EvaluationSnapshot(BaseModel):
    answerable_supported_rate: float
    citation_validity_rate: float
    expected_refusal_success_rate: float
    expected_citation_hit_rate: float
    mean_latency_seconds: float
    source_recall_at_k: float


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[Chunk, ...]:
    chunks_path = Path(os.getenv("REGURAG_CHUNKS_PATH", "data/processed/chunks.jsonl"))
    chunks = read_chunks_jsonl(chunks_path)
    return tuple(chunks or sample_chunks())


@lru_cache(maxsize=1)
def _bm25_retriever() -> SimpleBM25Retriever:
    return SimpleBM25Retriever(list(_load_chunks()))


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=os.getenv("REGURAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            batch_size=int(os.getenv("REGURAG_EMBEDDING_BATCH_SIZE", "4")),
        )
    )


@lru_cache(maxsize=1)
def _dense_retriever() -> QdrantDenseRetriever:
    return QdrantDenseRetriever(
        collection_name=os.getenv("REGURAG_QDRANT_COLLECTION", DEFAULT_COLLECTION),
        path=os.getenv("REGURAG_QDRANT_PATH", DEFAULT_QDRANT_PATH),
        url=os.getenv("REGURAG_QDRANT_URL", "http://localhost:6333"),
    )


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        CrossEncoderRerankerConfig(
            model_name=os.getenv("REGURAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            batch_size=int(os.getenv("REGURAG_RERANKER_BATCH_SIZE", "4")),
            max_length=int(os.getenv("REGURAG_RERANKER_MAX_LENGTH", "512")),
        )
    )


def _bm25_search(query: str, top_k: int) -> list[RetrievalResult]:
    return _bm25_retriever().search(query, top_k=top_k)


def _hybrid_rerank_search(query: str, top_k: int, candidate_k: int) -> list[RetrievalResult]:
    candidate_k = max(candidate_k, top_k)
    bm25_results = _bm25_search(query, top_k=candidate_k)
    query_vector = _embedder().encode_query(query)
    dense_results = _dense_retriever().search(query_vector=query_vector, top_k=candidate_k)
    fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=candidate_k)
    return _reranker().rerank(query, fused_results, top_k=top_k)


def _search_for_answer(request: AnswerRequest) -> tuple[str, list[RetrievalResult]]:
    if request.retriever == "bm25":
        return "bm25", _bm25_search(request.question, top_k=request.top_k)
    return (
        "hybrid_rerank",
        _hybrid_rerank_search(
            request.question,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
        ),
    )


def _evidence_from_results(results: list[RetrievalResult]) -> list[Evidence]:
    return [
        Evidence(
            rank=result.rank,
            score=result.score,
            retriever=result.retriever,
            citation_label=result.citation_label,
            source_id=result.chunk.source_id,
            title=str(result.chunk.metadata.get("title", result.chunk.source_id)),
            url=result.chunk.metadata.get("url"),
            section_heading=result.chunk.metadata.get("section_heading"),
            snippet=result.chunk.text[:700],
        )
        for result in results
    ]


def _evaluation_snapshot() -> EvaluationSnapshot:
    report_path = Path(os.getenv("REGURAG_ANSWER_EVAL_REPORT", "reports/answer_eval_ollama_latest.json"))
    if report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        return EvaluationSnapshot(
            answerable_supported_rate=float(summary.get("answerable_supported_rate", 0.667)),
            citation_validity_rate=float(summary.get("citation_validity_rate", 1.0)),
            expected_refusal_success_rate=float(
                summary.get("expected_refusal_success_rate", 1.0)
            ),
            expected_citation_hit_rate=float(summary.get("expected_citation_hit_rate", 0.167)),
            mean_latency_seconds=float(summary.get("mean_latency_seconds", 75.36)),
            source_recall_at_k=float(summary.get("source_recall_at_k", 0.556)),
        )
    return EvaluationSnapshot(
        answerable_supported_rate=0.667,
        citation_validity_rate=1.0,
        expected_refusal_success_rate=1.0,
        expected_citation_hit_rate=0.167,
        mean_latency_seconds=75.36,
        source_recall_at_k=0.556,
    )


app = FastAPI(
    title="AI Regulation Evidence API",
    version="0.1.0",
    description="Bilingual regulatory intelligence RAG system for AI Act, GDPR, and German AI governance sources.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/evaluation-snapshot", response_model=EvaluationSnapshot)
def evaluation_snapshot() -> EvaluationSnapshot:
    return _evaluation_snapshot()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = _bm25_search(request.question, top_k=request.top_k)

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


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    retriever_name, results = _search_for_answer(request)
    llm_model = os.getenv("REGURAG_LLM_MODEL", DEFAULT_LLM_MODEL)
    llm = LiteLLMClient(
        model=llm_model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        json_mode=not request.disable_json_mode,
        api_base=os.getenv("REGURAG_LLM_API_BASE", DEFAULT_LLM_API_BASE),
        api_key=os.getenv("REGURAG_LLM_API_KEY"),
        timeout=float(os.getenv("REGURAG_LLM_TIMEOUT", "120")),
    )
    result = generate_grounded_answer(
        question=request.question,
        retrieved_results=results,
        llm=llm,
        max_chars_per_chunk=request.max_context_chars,
    )
    return AnswerResponse(
        mode="grounded_answer",
        retriever=retriever_name,
        llm_model=llm_model,
        supported=result.supported,
        guardrail_triggered=result.guardrail_triggered,
        answer=AnswerPayload(
            answer=result.answer.answer,
            citations=result.answer.citations,
            confidence=result.answer.confidence,
            unsupported_claims=result.answer.unsupported_claims,
            should_refuse=result.answer.should_refuse,
            refusal_reason=result.answer.refusal_reason,
        ),
        retrieved_results=_evidence_from_results(result.retrieved_results),
        missing_citations=sorted(result.citation_validation.missing_labels),
    )
