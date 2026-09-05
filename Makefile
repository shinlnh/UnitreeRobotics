PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup demo test lint doctor clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e '.[dev]'

demo:
	PYTHONPATH=src $(PYTHON) -m unitree_gr00t.cli demo

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

doctor:
	PYTHONPATH=src $(PYTHON) -m unitree_gr00t.cli doctor --profile mock

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
