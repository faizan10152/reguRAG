from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path


def _slice_from_marker(raw_html: str, marker: str) -> str | None:
    lower = raw_html.lower()
    index = lower.find(marker.lower())
    if index < 0:
        return None

    clipped = raw_html[index:]
    end_candidates = [
        clipped.lower().find("<footer"),
        clipped.lower().find('id="footer'),
        clipped.lower().find('class="footer'),
    ]
    valid_ends = [end for end in end_candidates if end > 0]
    if valid_ends:
        clipped = clipped[: min(valid_ends)]
    return clipped


def clip_to_main_content(raw_html: str) -> str:
    main_match = re.search(r"<main\b[^>]*>(?P<body>.*?)</main>", raw_html, flags=re.I | re.S)
    if main_match:
        return main_match.group("body")

    for marker in (
        '<div class="eli-main-title"',
        '<div class="jnnorm"',
        '<div id="content"',
        '<article',
    ):
        clipped = _slice_from_marker(raw_html, marker)
        if clipped:
            return clipped

    return raw_html


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {
            "p",
            "br",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return normalize_text(" ".join(self._parts))


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def extract_html_text(raw_html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(clip_to_main_content(raw_html))
    return parser.get_text()


def extract_text_from_path(path: str | Path) -> str:
    source_path = Path(path)
    suffix = source_path.suffix.lower()

    if suffix in {".html", ".htm", ".xml"}:
        return extract_html_text(source_path.read_text(encoding="utf-8", errors="ignore"))

    if suffix == ".txt":
        return normalize_text(source_path.read_text(encoding="utf-8", errors="ignore"))

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction needs the optional package pypdf. "
                "Install it or convert this source to text first."
            ) from exc

        reader = PdfReader(str(source_path))
        return normalize_text("\n\n".join(page.extract_text() or "" for page in reader.pages))

    raise ValueError(f"Unsupported source file type: {source_path}")
