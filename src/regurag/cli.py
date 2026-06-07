from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path

from regurag.embeddings.encoder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from regurag.evaluation.answers import (
    evaluate_answer_run,
    write_answer_json_report,
    write_answer_markdown_report,
)
from regurag.evaluation.runner import (
    evaluate_retrieval_runs,
    load_golden_questions,
    write_json_report,
    write_markdown_report,
)
from regurag.generation.answer import AnswerGenerationResult, generate_grounded_answer
from regurag.generation.litellm_client import LiteLLMClient
from regurag.generation.prompts import build_answer_messages
from regurag.ingestion.chunking import chunk_text
from regurag.ingestion.download import download_sources
from regurag.ingestion.manifest import load_source_manifest
from regurag.ingestion.text_extract import extract_text_from_path
from regurag.reranking.cross_encoder import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
)
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.retrieval.dense_qdrant import (
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_URL,
    QdrantDenseRetriever,
)
from regurag.retrieval.fusion import reciprocal_rank_fusion
from regurag.sample_data import sample_chunks
from regurag.schemas import Chunk, RetrievalResult
from regurag.storage.jsonl import read_chunks_jsonl, write_chunks_jsonl

EVAL_RETRIEVERS = {"bm25", "dense", "hybrid", "hybrid-rerank"}
LLM_MODEL_ENV = "REGURAG_LLM_MODEL"
LLM_API_BASE_ENV = "REGURAG_LLM_API_BASE"
LLM_API_KEY_ENV = "REGURAG_LLM_API_KEY"


def _find_raw_source(raw_dir: Path, source_id: str) -> Path | None:
    for candidate in sorted(raw_dir.glob(f"{source_id}.*")):
        if not candidate.name.endswith(".meta.json"):
            return candidate
    return None


def _download(args: argparse.Namespace) -> None:
    sources = load_source_manifest(args.manifest)
    downloaded = download_sources(sources, args.raw_dir)
    for path in downloaded:
        print(f"downloaded {path}")


def _chunk(args: argparse.Namespace) -> None:
    sources = load_source_manifest(args.manifest)
    raw_dir = Path(args.raw_dir)
    all_chunks = []

    for source in sources:
        raw_path = _find_raw_source(raw_dir, source.source_id)
        if raw_path is None:
            print(f"skipping {source.source_id}: no raw file found")
            continue

        try:
            text = extract_text_from_path(raw_path)
        except Exception as exc:
            print(f"skipping {source.source_id}: {exc}")
            continue

        chunks = chunk_text(
            source=source,
            text=text,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
        all_chunks.extend(chunks)
        print(f"chunked {source.source_id}: {len(chunks)} chunks")

    count = write_chunks_jsonl(all_chunks, args.out)
    print(f"wrote {count} chunks to {args.out}")


def _search(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        print("No chunks found; using synthetic sample chunks.")
        chunks = sample_chunks()

    retriever = SimpleBM25Retriever(chunks)
    results = retriever.search(args.query, top_k=args.top_k)
    _print_results(results)


def _print_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("No matching chunks found.")
        return

    for result in results:
        chunk = result.chunk
        title = chunk.metadata.get("title", chunk.source_id)
        heading = chunk.metadata.get("section_heading") or "no section"
        snippet = chunk.text[:300].replace("\n", " ")
        print(f"\n#{result.rank} score={result.score:.4f} citation=[{chunk.citation_label}]")
        print(f"{title} | {heading}")
        print(snippet)


def _build_embedder(args: argparse.Namespace) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        EmbeddingConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
        )
    )


def _dense_index(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    embedder = _build_embedder(args)
    retriever = QdrantDenseRetriever(
        url=args.qdrant_url,
        collection_name=args.collection,
        location=args.qdrant_location,
        path=args.qdrant_path,
    )

    print(f"embedding_model={args.model}")
    print(f"chunks={len(chunks)}")
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])

    vector_size = len(vectors[0])
    retriever.recreate_collection(vector_size=vector_size)
    indexed = retriever.upsert_chunks(
        chunks=chunks, vectors=vectors, batch_size=args.upsert_batch_size
    )

    print(f"collection={args.collection}")
    print(f"vector_size={vector_size}")
    print(f"indexed={indexed}")


def _dense_search(args: argparse.Namespace) -> None:
    embedder = _build_embedder(args)
    retriever = QdrantDenseRetriever(
        url=args.qdrant_url,
        collection_name=args.collection,
        location=args.qdrant_location,
        path=args.qdrant_path,
    )
    query_vector = embedder.encode_query(args.query)
    results = retriever.search(query_vector=query_vector, top_k=args.top_k)
    _print_results(results)


def _compare_retrieval(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    bm25_results = SimpleBM25Retriever(chunks).search(args.query, top_k=args.top_k)
    embedder = _build_embedder(args)
    dense_results = QdrantDenseRetriever(
        url=args.qdrant_url,
        collection_name=args.collection,
        location=args.qdrant_location,
        path=args.qdrant_path,
    ).search(query_vector=embedder.encode_query(args.query), top_k=args.top_k)
    fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=args.top_k)

    print("\n=== BM25 ===")
    _print_results(bm25_results)
    print("\n=== Dense ===")
    _print_results(dense_results)
    print("\n=== Fused RRF ===")
    _print_results(fused_results)


def _parse_eval_retrievers(raw_value: str) -> list[str]:
    retrievers = [value.strip() for value in raw_value.split(",") if value.strip()]
    unknown = sorted(set(retrievers) - EVAL_RETRIEVERS)
    if unknown:
        raise SystemExit(
            f"Unknown retriever(s): {', '.join(unknown)}. "
            f"Use one or more of: {', '.join(sorted(EVAL_RETRIEVERS))}."
        )
    if not retrievers:
        raise SystemExit("At least one retriever must be selected.")
    return retrievers


def _cached_search(
    search: Callable[[str], list[RetrievalResult]],
) -> Callable[[str], list[RetrievalResult]]:
    cache: dict[str, list[RetrievalResult]] = {}

    def wrapped(query: str) -> list[RetrievalResult]:
        if query not in cache:
            cache[query] = search(query)
        return cache[query]

    return wrapped


def _register_cleanup(args: argparse.Namespace, callback: Callable[[], None]) -> None:
    callbacks = getattr(args, "_cleanup_callbacks", None)
    if callbacks is not None:
        callbacks.append(callback)


def _run_cleanups(args: argparse.Namespace) -> None:
    callbacks = getattr(args, "_cleanup_callbacks", [])
    for callback in reversed(callbacks):
        callback()


def _build_eval_retriever_runs(
    args: argparse.Namespace,
    chunks: list[Chunk],
) -> dict[str, Callable[[str], list[RetrievalResult]]]:
    selected = _parse_eval_retrievers(args.retrievers)
    candidate_k = max(args.candidate_k, args.top_k)
    runs: dict[str, Callable[[str], list[RetrievalResult]]] = {}

    bm25_retriever: SimpleBM25Retriever | None = None
    bm25_search: Callable[[str], list[RetrievalResult]] | None = None

    def get_bm25_search() -> Callable[[str], list[RetrievalResult]]:
        nonlocal bm25_retriever, bm25_search
        if bm25_search is None:
            bm25_retriever = SimpleBM25Retriever(chunks)
            bm25_search = _cached_search(
                lambda query: bm25_retriever.search(query, top_k=candidate_k)
            )
        return bm25_search

    dense_search: Callable[[str], list[RetrievalResult]] | None = None
    needs_dense = any(retriever in selected for retriever in {"dense", "hybrid", "hybrid-rerank"})
    if needs_dense:
        embedder = _build_embedder(args)
        dense_retriever = QdrantDenseRetriever(
            url=args.qdrant_url,
            collection_name=args.collection,
            location=args.qdrant_location,
            path=args.qdrant_path,
        )
        _register_cleanup(args, dense_retriever.close)
        dense_search = _cached_search(
            lambda query: dense_retriever.search(
                query_vector=embedder.encode_query(query),
                top_k=candidate_k,
            )
        )

    hybrid_search: Callable[[str], list[RetrievalResult]] | None = None

    def get_hybrid_search() -> Callable[[str], list[RetrievalResult]]:
        nonlocal hybrid_search
        if dense_search is None:
            raise SystemExit("Hybrid retrieval needs dense retrieval.")
        if hybrid_search is None:
            bm25 = get_bm25_search()
            hybrid_search = _cached_search(
                lambda query, bm25_search=bm25, dense_search=dense_search: reciprocal_rank_fusion(
                    [bm25_search(query), dense_search(query)],
                    top_k=candidate_k,
                )
            )
        return hybrid_search

    for retriever in selected:
        if retriever == "bm25":
            runs["bm25"] = get_bm25_search()
        elif retriever == "dense":
            if dense_search is None:
                raise SystemExit("Dense retrieval could not be initialized.")
            runs["dense_qdrant"] = dense_search
        elif retriever == "hybrid":
            runs["hybrid_rrf"] = get_hybrid_search()
        elif retriever == "hybrid-rerank":
            reranker = CrossEncoderReranker(
                CrossEncoderRerankerConfig(
                    model_name=args.reranker_model,
                    batch_size=args.reranker_batch_size,
                    max_length=args.reranker_max_length,
                    retriever_name="hybrid_rerank",
                )
            )
            hybrid = get_hybrid_search()
            runs["hybrid_rerank"] = _cached_search(
                lambda query, hybrid_search=hybrid, reranker=reranker: reranker.rerank(
                    query,
                    hybrid_search(query),
                    top_k=candidate_k,
                )
            )

    return runs


def _eval_retrieval(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    args._cleanup_callbacks = []
    try:
        questions = load_golden_questions(args.questions)
        candidate_k = max(args.candidate_k, args.top_k)
        runs = _build_eval_retriever_runs(args, chunks)
        report = evaluate_retrieval_runs(
            questions=questions,
            retriever_runs=runs,
            top_k=args.top_k,
            candidate_k=candidate_k,
        )

        print(f"questions={len(questions)}")
        print(f"top_k={args.top_k}")
        print(f"candidate_k={candidate_k}")
        for summary in report.summaries():
            print(
                f"{summary.retriever}: "
                f"answerable={summary.answerable_rows} "
                f"recall@{args.top_k}={summary.source_recall_at_k:.3f} "
                f"hit@{args.top_k}={summary.source_hit_rate_at_k:.3f} "
                f"mrr={summary.source_mrr:.3f}"
            )
            if summary.citation_labeled_rows:
                print(
                    f"{summary.retriever}: "
                    f"citation_labeled={summary.citation_labeled_rows} "
                    f"citation_recall@{args.top_k}={summary.citation_recall_at_k:.3f} "
                    f"citation_hit@{args.top_k}={summary.citation_hit_rate_at_k:.3f} "
                    f"citation_mrr={summary.citation_mrr:.3f}"
                )

        if args.output_json:
            write_json_report(report, args.output_json)
            print(f"wrote JSON report to {args.output_json}")
        if args.output_md:
            write_markdown_report(report, args.output_md)
            print(f"wrote Markdown report to {args.output_md}")
    finally:
        _run_cleanups(args)


def _build_single_retriever_run(
    args: argparse.Namespace,
    chunks: list[Chunk],
) -> tuple[str, Callable[[str], list[RetrievalResult]]]:
    retrieval_args = argparse.Namespace(**vars(args), retrievers=args.retriever)
    runs = _build_eval_retriever_runs(retrieval_args, chunks)
    if len(runs) != 1:
        raise SystemExit(f"Expected one retriever run, got {len(runs)}.")
    return next(iter(runs.items()))


def _filter_questions(
    questions: list,
    *,
    question_ids: str | None = None,
    limit: int | None = None,
) -> list:
    selected = questions
    if question_ids:
        ids = {question_id.strip() for question_id in question_ids.split(",") if question_id.strip()}
        selected = [question for question in selected if question.question_id in ids]
        missing = sorted(ids - {question.question_id for question in selected})
        if missing:
            raise SystemExit(f"Unknown question id(s): {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise SystemExit("No questions selected for evaluation.")
    return selected


def _answer(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    args._cleanup_callbacks = []
    try:
        retriever_name, search = _build_single_retriever_run(args, chunks)
        results = search(args.query)[: args.top_k]

        if args.dry_run_prompt:
            messages = build_answer_messages(
                args.query,
                results,
                max_chars_per_chunk=args.max_context_chars,
            )
            print(
                json.dumps(
                    {
                        "question": args.query,
                        "retriever": retriever_name,
                        "messages": messages,
                        "retrieved_results": _retrieval_results_payload(results),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        llm_model = args.llm_model or os.getenv(LLM_MODEL_ENV)
        if not llm_model:
            raise SystemExit(
                f"Set --llm-model or {LLM_MODEL_ENV} before running answer generation. "
                "Use --dry-run-prompt to inspect retrieval and prompting without an LLM call."
            )

        llm = LiteLLMClient(
            model=llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=not args.disable_json_mode,
            api_base=args.llm_api_base or os.getenv(LLM_API_BASE_ENV),
            api_key=args.llm_api_key or os.getenv(LLM_API_KEY_ENV),
            timeout=args.llm_timeout,
        )
        answer_result = generate_grounded_answer(
            question=args.query,
            retrieved_results=results,
            llm=llm,
            max_chars_per_chunk=args.max_context_chars,
            min_citations=args.min_citations,
        )
        llm_api_base = args.llm_api_base or os.getenv(LLM_API_BASE_ENV)
        _print_answer_result(
            answer_result,
            retriever_name=retriever_name,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
        )

        if args.output_json:
            payload = answer_result.to_dict()
            payload["retriever"] = retriever_name
            payload["llm_model"] = llm_model
            payload["llm_api_base"] = llm_api_base
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"wrote JSON answer report to {args.output_json}")
    finally:
        _run_cleanups(args)


def _eval_answers(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    llm_model = args.llm_model or os.getenv(LLM_MODEL_ENV)
    if not llm_model:
        raise SystemExit(f"Set --llm-model or {LLM_MODEL_ENV} before answer evaluation.")

    args._cleanup_callbacks = []
    try:
        questions = _filter_questions(
            load_golden_questions(args.questions),
            question_ids=args.question_ids,
            limit=args.limit,
        )
        retriever_name, search = _build_single_retriever_run(args, chunks)
        llm = LiteLLMClient(
            model=llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            json_mode=not args.disable_json_mode,
            api_base=args.llm_api_base or os.getenv(LLM_API_BASE_ENV),
            api_key=args.llm_api_key or os.getenv(LLM_API_KEY_ENV),
            timeout=args.llm_timeout,
        )
        run_name = args.run_name or f"{retriever_name}:{llm_model}"
        report = evaluate_answer_run(
            questions=questions,
            search=search,
            llm=llm,
            run_name=run_name,
            retriever=retriever_name,
            llm_model=llm_model,
            top_k=args.top_k,
            candidate_k=max(args.candidate_k, args.top_k),
            max_context_chars=args.max_context_chars,
            min_citations=args.min_citations,
            progress_callback=None
            if args.quiet
            else lambda index, total, question: print(
                f"evaluating {index}/{total} {question.question_id}",
                flush=True,
            ),
        )
        summary = report.summary()
        print(f"run={run_name}")
        print(f"questions={summary.rows}")
        print(f"retriever={retriever_name}")
        print(f"llm_model={llm_model}")
        print(f"answerable={summary.answerable_rows}")
        print(f"refusal_rows={summary.refusal_rows}")
        print(f"supported_rate={summary.supported_rate:.3f}")
        print(f"answerable_supported_rate={summary.answerable_supported_rate:.3f}")
        print(f"valid_structured_output_rate={summary.valid_structured_output_rate:.3f}")
        print(f"refusal_accuracy={summary.refusal_accuracy:.3f}")
        print(f"expected_refusal_success_rate={summary.expected_refusal_success_rate:.3f}")
        print(f"citation_validity_rate={summary.citation_validity_rate:.3f}")
        print(f"source_recall@{args.top_k}={summary.source_recall_at_k:.3f}")
        print(f"expected_citation_hit_rate={summary.expected_citation_hit_rate:.3f}")
        print(f"mean_latency_seconds={summary.mean_latency_seconds:.2f}")
        print(f"errors={summary.error_rows}")

        if args.output_json:
            write_answer_json_report(report, args.output_json)
            print(f"wrote JSON answer report to {args.output_json}")
        if args.output_md:
            write_answer_markdown_report(report, args.output_md)
            print(f"wrote Markdown answer report to {args.output_md}")
    finally:
        _run_cleanups(args)


def _print_answer_result(
    result: AnswerGenerationResult,
    *,
    retriever_name: str,
    llm_model: str,
    llm_api_base: str | None = None,
) -> None:
    print(f"retriever={retriever_name}")
    print(f"llm_model={llm_model}")
    if llm_api_base:
        print(f"llm_api_base={llm_api_base}")
    print(f"supported={result.supported}")
    print(f"guardrail_triggered={result.guardrail_triggered}")
    print(f"confidence={result.answer.confidence}")
    if result.answer.refusal_reason:
        print(f"refusal_reason={result.answer.refusal_reason}")
    print("\nAnswer:")
    print(result.answer.answer)
    print("\nCitations:")
    for label in result.answer.citations:
        print(f"- [{label}]")
    if result.citation_validation.missing_labels:
        print("\nMissing citations:")
        for label in sorted(result.citation_validation.missing_labels):
            print(f"- [{label}]")
    print("\nRetrieved evidence:")
    for item in _retrieval_results_payload(result.retrieved_results):
        print(
            f"- #{item['rank']} [{item['citation_label']}] "
            f"{item['title']} | {item['section_heading'] or 'no section'}"
        )


def _retrieval_results_payload(results: list[RetrievalResult]) -> list[dict[str, object]]:
    return [
        {
            "rank": result.rank,
            "score": result.score,
            "retriever": result.retriever,
            "citation_label": result.citation_label,
            "source_id": result.chunk.source_id,
            "chunk_id": result.chunk.chunk_id,
            "title": result.chunk.metadata.get("title", result.chunk.source_id),
            "section_heading": result.chunk.metadata.get("section_heading"),
            "url": result.chunk.metadata.get("url"),
            "snippet": result.chunk.text[:500],
        }
        for result in results
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regurag",
        description="ReguRAG local research and retrieval commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official sources.")
    download.add_argument("--manifest", required=True)
    download.add_argument("--raw-dir", required=True)
    download.set_defaults(func=_download)

    chunk = subparsers.add_parser("chunk", help="Extract text and create JSONL chunks.")
    chunk.add_argument("--manifest", required=True)
    chunk.add_argument("--raw-dir", required=True)
    chunk.add_argument("--out", required=True)
    chunk.add_argument("--max-words", type=int, default=650)
    chunk.add_argument("--overlap-words", type=int, default=100)
    chunk.set_defaults(func=_chunk)

    search = subparsers.add_parser("search", help="Run local BM25 search over chunks.")
    search.add_argument("--chunks", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.set_defaults(func=_search)

    dense_index = subparsers.add_parser(
        "dense-index",
        help="Embed chunks and index them in Qdrant.",
    )
    dense_index.add_argument("--chunks", required=True)
    dense_index.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    dense_index.add_argument("--qdrant-location", default=None)
    dense_index.add_argument("--qdrant-path", default=None)
    dense_index.add_argument("--collection", default=DEFAULT_COLLECTION)
    dense_index.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    dense_index.add_argument("--batch-size", type=int, default=16)
    dense_index.add_argument("--upsert-batch-size", type=int, default=64)
    dense_index.add_argument("--query-prefix", default="")
    dense_index.add_argument("--document-prefix", default="")
    dense_index.set_defaults(func=_dense_index)

    dense_search = subparsers.add_parser(
        "dense-search",
        help="Search an existing Qdrant dense index.",
    )
    dense_search.add_argument("--query", required=True)
    dense_search.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    dense_search.add_argument("--qdrant-location", default=None)
    dense_search.add_argument("--qdrant-path", default=None)
    dense_search.add_argument("--collection", default=DEFAULT_COLLECTION)
    dense_search.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    dense_search.add_argument("--batch-size", type=int, default=16)
    dense_search.add_argument("--query-prefix", default="")
    dense_search.add_argument("--document-prefix", default="")
    dense_search.add_argument("--top-k", type=int, default=5)
    dense_search.set_defaults(func=_dense_search)

    compare_retrieval = subparsers.add_parser(
        "compare-retrieval",
        help="Compare BM25, dense Qdrant, and RRF-fused results for one query.",
    )
    compare_retrieval.add_argument("--chunks", required=True)
    compare_retrieval.add_argument("--query", required=True)
    compare_retrieval.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    compare_retrieval.add_argument("--qdrant-location", default=None)
    compare_retrieval.add_argument("--qdrant-path", default=None)
    compare_retrieval.add_argument("--collection", default=DEFAULT_COLLECTION)
    compare_retrieval.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    compare_retrieval.add_argument("--batch-size", type=int, default=16)
    compare_retrieval.add_argument("--query-prefix", default="")
    compare_retrieval.add_argument("--document-prefix", default="")
    compare_retrieval.add_argument("--top-k", type=int, default=5)
    compare_retrieval.set_defaults(func=_compare_retrieval)

    eval_retrieval = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate source-level retrieval quality on a JSONL golden set.",
    )
    eval_retrieval.add_argument("--chunks", required=True)
    eval_retrieval.add_argument("--questions", required=True)
    eval_retrieval.add_argument(
        "--retrievers",
        default="bm25",
        help="Comma-separated retrievers to evaluate: bm25,dense,hybrid,hybrid-rerank.",
    )
    eval_retrieval.add_argument("--top-k", type=int, default=5)
    eval_retrieval.add_argument("--candidate-k", type=int, default=20)
    eval_retrieval.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    eval_retrieval.add_argument("--qdrant-location", default=None)
    eval_retrieval.add_argument("--qdrant-path", default=None)
    eval_retrieval.add_argument("--collection", default=DEFAULT_COLLECTION)
    eval_retrieval.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    eval_retrieval.add_argument("--batch-size", type=int, default=16)
    eval_retrieval.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    eval_retrieval.add_argument("--reranker-batch-size", type=int, default=4)
    eval_retrieval.add_argument("--reranker-max-length", type=int, default=512)
    eval_retrieval.add_argument("--query-prefix", default="")
    eval_retrieval.add_argument("--document-prefix", default="")
    eval_retrieval.add_argument("--output-json", default=None)
    eval_retrieval.add_argument("--output-md", default=None)
    eval_retrieval.set_defaults(func=_eval_retrieval)

    eval_answers = subparsers.add_parser(
        "eval-answers",
        help="Evaluate grounded answer generation on a JSONL golden set.",
    )
    eval_answers.add_argument("--chunks", required=True)
    eval_answers.add_argument("--questions", required=True)
    eval_answers.add_argument(
        "--retriever",
        choices=sorted(EVAL_RETRIEVERS),
        default="hybrid-rerank",
        help="Retrieval pipeline to use before answer generation.",
    )
    eval_answers.add_argument("--top-k", type=int, default=5)
    eval_answers.add_argument("--candidate-k", type=int, default=20)
    eval_answers.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    eval_answers.add_argument("--qdrant-location", default=None)
    eval_answers.add_argument("--qdrant-path", default=None)
    eval_answers.add_argument("--collection", default=DEFAULT_COLLECTION)
    eval_answers.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    eval_answers.add_argument("--batch-size", type=int, default=16)
    eval_answers.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    eval_answers.add_argument("--reranker-batch-size", type=int, default=4)
    eval_answers.add_argument("--reranker-max-length", type=int, default=512)
    eval_answers.add_argument("--query-prefix", default="")
    eval_answers.add_argument("--document-prefix", default="")
    eval_answers.add_argument("--llm-model", default=None)
    eval_answers.add_argument("--llm-api-base", default=None)
    eval_answers.add_argument("--llm-api-key", default=None)
    eval_answers.add_argument("--temperature", type=float, default=0.0)
    eval_answers.add_argument("--max-tokens", type=int, default=900)
    eval_answers.add_argument("--llm-timeout", type=float, default=120.0)
    eval_answers.add_argument("--max-context-chars", type=int, default=4500)
    eval_answers.add_argument("--min-citations", type=int, default=1)
    eval_answers.add_argument("--disable-json-mode", action="store_true")
    eval_answers.add_argument("--question-ids", default=None)
    eval_answers.add_argument("--limit", type=int, default=None)
    eval_answers.add_argument("--run-name", default=None)
    eval_answers.add_argument("--quiet", action="store_true")
    eval_answers.add_argument("--output-json", default=None)
    eval_answers.add_argument("--output-md", default=None)
    eval_answers.set_defaults(func=_eval_answers)

    answer = subparsers.add_parser(
        "answer",
        help="Generate a grounded answer with retrieved citations.",
    )
    answer.add_argument("--chunks", required=True)
    answer.add_argument("--query", required=True)
    answer.add_argument(
        "--retriever",
        choices=sorted(EVAL_RETRIEVERS),
        default="hybrid-rerank",
        help="Retrieval pipeline to use before answer generation.",
    )
    answer.add_argument("--top-k", type=int, default=5)
    answer.add_argument("--candidate-k", type=int, default=20)
    answer.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    answer.add_argument("--qdrant-location", default=None)
    answer.add_argument("--qdrant-path", default=None)
    answer.add_argument("--collection", default=DEFAULT_COLLECTION)
    answer.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    answer.add_argument("--batch-size", type=int, default=16)
    answer.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    answer.add_argument("--reranker-batch-size", type=int, default=4)
    answer.add_argument("--reranker-max-length", type=int, default=512)
    answer.add_argument("--query-prefix", default="")
    answer.add_argument("--document-prefix", default="")
    answer.add_argument("--llm-model", default=None)
    answer.add_argument("--llm-api-base", default=None)
    answer.add_argument("--llm-api-key", default=None)
    answer.add_argument("--temperature", type=float, default=0.0)
    answer.add_argument("--max-tokens", type=int, default=900)
    answer.add_argument("--llm-timeout", type=float, default=None)
    answer.add_argument("--max-context-chars", type=int, default=4500)
    answer.add_argument("--min-citations", type=int, default=1)
    answer.add_argument("--disable-json-mode", action="store_true")
    answer.add_argument("--dry-run-prompt", action="store_true")
    answer.add_argument("--output-json", default=None)
    answer.set_defaults(func=_answer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
