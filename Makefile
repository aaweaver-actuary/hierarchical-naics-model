SHELL := /bin/sh

PACKAGE := hierarchical_naics_model
SRC_DIR := src/$(PACKAGE)

COV_MIN := 95

.PHONY: test
test:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check .
	uv run pytest \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=lcov \
		--cov-fail-under=$(COV_MIN)
