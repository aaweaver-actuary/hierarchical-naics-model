SHELL := /bin/sh

PACKAGE := hierarchical_naics_model
SRC_DIR := src/$(PACKAGE)

COV_MIN := 95

lint: 
	uv run ruff check .
	uv run ruff format .
	uv run ty check .

.PHONY: test
test:
	make lint
	uv run pytest -x \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=lcov \
		--cov-fail-under=$(COV_MIN)
