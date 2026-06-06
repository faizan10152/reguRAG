from __future__ import annotations

import hashlib
import json
import mimetypes
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from regurag.schemas import SourceRecord


def guess_extension(url: str, content_type: str | None) -> str:
    path_suffix = Path(urlparse(url).path).suffix.lower()
    if path_suffix in {".html", ".htm", ".pdf", ".txt", ".xml"}:
        return path_suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed in {".html", ".htm", ".pdf", ".txt", ".xml"}:
            return guessed

    return ".html"


def download_source(source: SourceRecord, raw_dir: str | Path) -> Path:
    out_dir = Path(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        source.url,
        headers={
            "User-Agent": "ReguRAG regulatory research crawler; contact: local development",
            "Accept": "text/html,application/pdf,text/plain,*/*",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            content_type = response.headers.get("content-type")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download {source.source_id}: {exc}") from exc

    extension = guess_extension(source.url, content_type)
    out_path = out_dir / f"{source.source_id}{extension}"
    out_path.write_bytes(body)

    metadata = {
        **asdict(source),
        "content_type": content_type,
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }
    out_path.with_suffix(out_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out_path


def download_sources(sources: list[SourceRecord], raw_dir: str | Path) -> list[Path]:
    return [download_source(source, raw_dir) for source in sources]
