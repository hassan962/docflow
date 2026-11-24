# Convenience targets (Linux/macOS/CI). On Windows, run the commands after each colon directly.
.PHONY: up down reset test test-unit lint typecheck migrate seed

up:            ## start the local stack
	docker compose up --build -d

down:          ## stop the stack, keep data
	docker compose down

reset:         ## stop the stack and delete the database volume
	docker compose down -v

migrate:
	uv run alembic upgrade head

seed:
	uv run python scripts/seed.py

test:          ## all tests (database tests need `make up`)
	uv run pytest

test-unit:     ## tests that need no database
	uv run pytest -m "not db"

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy packages/docflow_core/src apps/api/src
