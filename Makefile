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
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -x -p pytest_cov \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=lcov \
		--cov-fail-under=$(COV_MIN)

check: lint test

test-rec:
	RUN_RECOVERY=1 make test

test-rec-only:
	make lint
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_RECOVERY=1 uv run pytest -p pytest_cov tests/test_param_recovery_full.py -x