.PHONY: install test lint pre-commit-install api docker-up qdrant-up download chunk search dense-index dense-search compare eval-retrieval eval-retrieval-local

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

qdrant-up:
	docker compose up -d qdrant

download:
	uv run regurag download --manifest configs/source_manifest.json --raw-dir data/raw

chunk:
	uv run regurag chunk --manifest configs/source_manifest.json --raw-dir data/raw --out data/processed/chunks.jsonl

search:
	uv run regurag search --chunks data/processed/chunks.jsonl --query "$(q)"

dense-index:
	uv run --extra rag regurag dense-index --chunks data/processed/chunks.jsonl

dense-search:
	uv run --extra rag regurag dense-search --query "$(q)"

compare:
	uv run --extra rag regurag compare-retrieval --chunks data/processed/chunks.jsonl --query "$(q)"

eval-retrieval:
	uv run regurag eval-retrieval --chunks data/processed/chunks.jsonl --questions data/eval/golden_questions_v1.jsonl --retrievers bm25 --output-md reports/retrieval_eval_latest.md --output-json reports/retrieval_eval_latest.json

eval-retrieval-local:
	uv run --extra rag regurag eval-retrieval --chunks data/processed/chunks.jsonl --questions data/eval/golden_questions_v1.jsonl --retrievers bm25,dense,hybrid --qdrant-path .qdrant/local --output-md reports/retrieval_eval_latest.md --output-json reports/retrieval_eval_latest.json
