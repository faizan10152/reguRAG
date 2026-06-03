from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from regurag.embeddings.encoder import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    SentenceTransformerEmbedder,
)
from regurag.evaluation.runner import (
    evaluate_retrieval_runs,
    load_golden_questions,
    write_json_report,
    write_markdown_report,
)
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
