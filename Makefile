OLLAMA_MODEL ?= ollama/qwen3:8b
OLLAMA_API_BASE ?= http://localhost:11434

.PHONY: install test lint pre-commit-install api docker-up qdrant-up download chunk search dense-index dense-search compare eval-retrieval eval-retrieval-local eval-rerank-local answer-dry-run answer-local ollama-pull-8b ollama-pull-14b answer-ollama-local answer-ollama-rerank

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

eval-rerank-local:
	HF_HUB_DISABLE_XET=1 uv run --extra rag regurag eval-retrieval --chunks data/processed/chunks.jsonl --questions data/eval/golden_questions_v1.jsonl --retrievers bm25,dense,hybrid,hybrid-rerank --qdrant-path .qdrant/bge-m3 --model BAAI/bge-m3 --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --top-k 5 --candidate-k 20 --reranker-batch-size 4 --output-md reports/retrieval_eval_rerank_latest.md --output-json reports/retrieval_eval_rerank_latest.json

answer-dry-run:
	uv run regurag answer --chunks data/processed/chunks.jsonl --query "$(q)" --retriever bm25 --top-k 5 --dry-run-prompt

answer-local:
	HF_HUB_DISABLE_XET=1 uv run --extra rag regurag answer --chunks data/processed/chunks.jsonl --query "$(q)" --retriever hybrid-rerank --qdrant-path .qdrant/bge-m3 --model BAAI/bge-m3 --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --top-k 5 --candidate-k 20

ollama-pull-8b:
	ollama pull qwen3:8b

ollama-pull-14b:
	ollama pull qwen3:14b

answer-ollama-local:
	uv run --extra rag regurag answer --chunks data/processed/chunks.jsonl --query "$(q)" --retriever bm25 --top-k 5 --llm-model "$(OLLAMA_MODEL)" --llm-api-base "$(OLLAMA_API_BASE)"

answer-ollama-rerank:
	HF_HUB_DISABLE_XET=1 uv run --extra rag regurag answer --chunks data/processed/chunks.jsonl --query "$(q)" --retriever hybrid-rerank --qdrant-path .qdrant/bge-m3 --model BAAI/bge-m3 --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --top-k 5 --candidate-k 20 --llm-model "$(OLLAMA_MODEL)" --llm-api-base "$(OLLAMA_API_BASE)"
