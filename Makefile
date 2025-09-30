SHELL := /bin/sh

PACKAGE := hierarchical_naics_model
SRC_DIR := src/$(PACKAGE)

COV_MIN := 95

lint: 
	uv run ruff check .
	uv run ruff format .
	uv run ty check .

.PHONY: test


TESTFILE ?=
test:
	if [ -z "$(TESTFILE)" ]; then \
		PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -x -p pytest_cov \
			--cov=$(SRC_DIR) \
			--cov-report=term-missing \
			--cov-report=html \
			--cov-report=lcov \
			--cov-fail-under=$(COV_MIN); \
	else \
		PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest $(TESTFILE) -x; \
	fi

cc:
	RADONFILESENCODING="UTF-8" uv run radon cc . \
		--min "B" \
		--total-average \
		--show-complexity \
		--order "SCORE" \
		--no-assert

mi:
	RADONFILESENCODING="UTF-8" uv run radon mi .
	
hal: 
	uv run radon hal .

radon: cc mi hal

xenon:
	uv run xenon -b B -m A -a A ./src

check:
	make lint
	make cc
	make test

test-rec:
	RUN_RECOVERY=1 make test

test-rec-only:
	make lint
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_RECOVERY=1 uv run pytest -p pytest_cov tests/test_param_recovery_full.py -x