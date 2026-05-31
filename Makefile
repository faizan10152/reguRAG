.PHONY: install test lint pre-commit-install api docker-up download chunk search

install:
	uv venv --python 3.11
	uv pip install -e ".[api,dev,ingestion]"

pre-commit-install:
	uv run --extra dev pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .

api:
	uv run uvicorn regurag.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

download:
	uv run regurag download --manifest configs/source_manifest.json --raw-dir data/raw

chunk:
	uv run regurag chunk --manifest configs/source_manifest.json --raw-dir data/raw --out data/processed/chunks.jsonl

search:
	uv run regurag search --chunks data/processed/chunks.jsonl --query "$(q)"
