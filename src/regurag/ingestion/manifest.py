from __future__ import annotations

import json
from pathlib import Path

from regurag.schemas import SourceRecord


def load_source_manifest(path: str | Path) -> list[SourceRecord]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [SourceRecord.from_dict(source) for source in payload["sources"]]
