# Makefile for LangGraph Cloud deployment using uv and pyproject.toml

PYTHON = uv
LINTER = ruff
FORMATTER = ruff
TYPECHECKER = mypy

.PHONY: install lint format typecheck check clean

# Install all dependencies (main + dev) from pyproject.toml
install:
	$(PYTHON) pip install . --dev

# Lint code
lint:
	$(LINTER) check .

# Format code
format:
	$(FORMATTER) format .

# Type check code
typecheck:
	$(TYPECHECKER) src/

# Run all quality checks
check: lint typecheck

# Clean pyc files and caches
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Deploy LangGraph to LangGraph Cloud
deploy:
	langgraph deploy
