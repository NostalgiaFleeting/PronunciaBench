.PHONY: install dev test lint typecheck clean benchmark dashboard up down

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

install:
	$(PIP) install -e ".[dev]"

dev: install
	$(PIP) install -e ".[dashboard]"

test:
	$(PYTEST) tests/ -v --tb=short --cov=src/pronunciabench --cov-report=term-missing

test-ci:
	$(PYTEST) tests/ -v --tb=short -x

lint:
	ruff check src/ tests/

typecheck:
	mypy src/pronunciabench/

benchmark:
	$(PYTHON) -m pronunciabench.cli.main benchmark --dataset data/samples/test.jsonl --output reports/benchmark.json

dashboard:
	$(PYTHON) dashboard/app.py

api:
	uvicorn pronunciabench.api.app:app --reload --port 8000

clean:
	rm -rf artifacts/ reports/ __pycache__/ src/**/*.pyc .pytest_cache/ .ruff_cache/

help:
	@echo "Targets:"
	@echo "  install   - Install package with all dependencies"
	@echo "  dev       - Install package with dev+dashboard dependencies"
	@echo "  test      - Run tests with coverage"
	@echo "  test-ci   - Run tests (CI mode, exit on first failure)"
	@echo "  lint      - Run ruff linter"
	@echo "  typecheck - Run mypy type checker"
	@echo "  benchmark - Run benchmark on sample dataset"
	@echo "  dashboard - Launch Gradio dashboard"
	@echo "  api       - Launch FastAPI server"
	@echo "  clean     - Remove build artifacts"