SHELL := /bin/sh

PACKAGE := hierarchical_naics_model
SRC_DIR := src/$(PACKAGE)

COV_MIN := 95

lint: 
	uv run ruff check --fix src/
	uv run ruff format src/
	uv run ty check src/

CHECKFILE ?=
check:
	make lint
	if [ -n "$(CHECKFILE)" ]; then \
		uv run pytest --ignore=__OLD $(CHECKFILE); \
	else \
		uv run pytest --ignore=__OLD; \
	fi


.PHONY: test
TESTFILE ?=
test:
	if [ -z "$(TESTFILE)" ]; then \
		PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -x \
			-p pytest_cov \
			--cov=$(SRC_DIR) \
			--cov-report=term-missing \
			--cov-report=html \
			--cov-report=lcov \
			--cov-fail-under=$(COV_MIN) \
			src/; \
	else \
		PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest $(TESTFILE) -x; \
	fi

cc:
	uv run radon cc src/ \
		--min "B" \
		--total-average \
		--show-complexity \
		--order "SCORE" \
		--no-assert

mi:
	uv run radon mi src/
	
hal: 
	uv run radon hal src/

radon: cc mi hal

xenon:
	uv run xenon -b B -m A -a A src/


test-rec:
	RUN_RECOVERY=1 make test

test-rec-only:
	make lint
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_RECOVERY=1 uv run pytest -p pytest_cov tests/test_param_recovery_full.py -x