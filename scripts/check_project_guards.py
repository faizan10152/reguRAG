from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BLOCKED_STAGED_PATHS = (
    re.compile(r"^\.env$"),
    re.compile(r"^\.env\.(?!example$).+"),
    re.compile(r"^data/raw/.+"),
    re.compile(r"^data/processed/.+"),
)

ALLOWED_DATA_PLACEHOLDERS = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
}

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?m)^[ \t]*OPENAI_API_KEY[ \t]*=[ \t]*['\"]?[^#\s'\"]+"),
    re.compile(r"(?m)^[ \t]*ANTHROPIC_API_KEY[ \t]*=[ \t]*['\"]?[^#\s'\"]+"),
    re.compile(r"(?m)^[ \t]*GEMINI_API_KEY[ \t]*=[ \t]*['\"]?[^#\s'\"]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)

TEXT_FILE_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def staged_files() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def tracked_files() -> list[str]:
    output = run_git(["ls-files"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def should_scan_text(path: str) -> bool:
    file_path = Path(path)
    return file_path.suffix in TEXT_FILE_SUFFIXES or file_path.name in {
        ".env",
        ".env.example",
        "Dockerfile",
        "Makefile",
    }


def read_file_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def check_blocked_paths(paths: list[str]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path in ALLOWED_DATA_PLACEHOLDERS:
            continue
        if any(pattern.search(path) for pattern in BLOCKED_STAGED_PATHS):
            errors.append(
                f"blocked path staged: {path}. Keep secrets and generated raw/processed data out of git."
            )
    return errors


def check_secret_patterns(paths: list[str]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not should_scan_text(path):
            continue
        text = read_file_text(path)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible secret in {path}: matched {pattern.pattern}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Project-specific commit guards.")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Scan tracked files instead of only staged files. Used by CI.",
    )
    parser.add_argument("paths", nargs="*", help="Paths passed by pre-commit.")
    args = parser.parse_args()

    if args.all_files:
        paths = tracked_files()
    elif args.paths:
        paths = args.paths
    else:
        paths = staged_files()

    errors = check_blocked_paths(paths)
    errors.extend(check_secret_patterns(paths))

    if errors:
        print("Project guard failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
