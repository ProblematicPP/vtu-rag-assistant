.PHONY: help env up down logs ingest reindex test lint format pull-model

help:
	@echo "env         create .env from .env.example"
	@echo "up          build and start the stack"
	@echo "down        stop the stack"
	@echo "logs        follow api logs"
	@echo "ingest      index new/changed notes"
	@echo "reindex     re-index all notes"
	@echo "pull-model  pull the configured Ollama model"
	@echo "test/lint/format"

env:
	test -f .env || cp .env.example .env

up: env
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

ingest:
	docker compose exec api python scripts/ingest.py

reindex:
	docker compose exec api python scripts/ingest.py --force

pull-model:
	docker compose run --rm ollama-init

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
