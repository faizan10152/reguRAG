from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from regurag.schemas import Chunk


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_chunks_jsonl(path: str | Path) -> list[Chunk]:
    in_path = Path(path)
    if not in_path.exists():
        return []

    chunks: list[Chunk] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
