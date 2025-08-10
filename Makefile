.PHONY: setup fetch index serve eval eval-retrieval test lint up down

setup:            ## install deps into .venv
	uv sync

fetch:            ## download the pinned RFC corpus (~20MB)
	uv run specsage fetch

index: fetch      ## embed + index the corpus (needs qdrant: `make up` or docker run)
	uv run specsage index

serve:            ## run the API + UI on :8000
	uv run specsage serve

eval:             ## full evaluation (retrieval + generation; needs ollama)
	uv run specsage eval

eval-retrieval:   ## retrieval metrics only (no LLM needed)
	uv run specsage eval --no-generation

test:
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run mypy src

up:               ## start qdrant + api via docker compose
	docker compose up -d --build

down:
	docker compose down
