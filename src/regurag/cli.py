from __future__ import annotations

import argparse
from pathlib import Path

from regurag.evaluation.runner import evaluate_bm25_source_recall, load_golden_questions
from regurag.ingestion.chunking import chunk_text
from regurag.ingestion.download import download_sources
from regurag.ingestion.manifest import load_source_manifest
from regurag.ingestion.text_extract import extract_text_from_path
from regurag.retrieval.bm25 import SimpleBM25Retriever
from regurag.sample_data import sample_chunks
from regurag.storage.jsonl import read_chunks_jsonl, write_chunks_jsonl


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


def _eval_retrieval(args: argparse.Namespace) -> None:
    chunks = read_chunks_jsonl(args.chunks)
    if not chunks:
        raise SystemExit(f"No chunks found at {args.chunks}. Run the chunk command first.")

    questions = load_golden_questions(args.questions)
    report = evaluate_bm25_source_recall(chunks, questions, top_k=args.top_k)

    print(f"questions={len(report.rows)}")
    print(f"answerable_questions={len(report.answerable_rows)}")
    print(f"mean_source_recall_at_{args.top_k}={report.mean_source_recall_at_k:.3f}")

    for row in report.rows:
        expected = ",".join(sorted(row.relevant_sources)) or "unanswerable"
        retrieved = ",".join(row.retrieved_sources) or "none"
        print(
            f"{row.question_id}: recall={row.source_recall_at_k:.3f} "
            f"expected={expected} retrieved={retrieved}"
        )


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

    eval_retrieval = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate BM25 source-level recall on a JSONL golden set.",
    )
    eval_retrieval.add_argument("--chunks", required=True)
    eval_retrieval.add_argument("--questions", required=True)
    eval_retrieval.add_argument("--top-k", type=int, default=5)
    eval_retrieval.set_defaults(func=_eval_retrieval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
