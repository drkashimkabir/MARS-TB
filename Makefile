.PHONY: install test test-slow lint format typecheck cov audit validate security all

install:
	pip install -e ".[dev]"

test:
	python -m pytest -m "not slow"

test-slow:
	python -m pytest -m slow

lint:
	ruff check .

format:
	ruff format .

typecheck:
	python -m mypy

cov:
	python -m pytest -m "not slow" --cov --cov-report=term-missing

audit:
	python scripts/audit_manuscript.py --allow-fail
	python scripts/parameter_accounting.py
	python reporting/build_checklists.py

validate:
	python scripts/validate_artifacts.py

security:
	pip-audit -r requirements.txt
	bandit -q -r src scripts -x tests

all: lint typecheck test validate
