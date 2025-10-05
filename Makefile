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
		uv run pytest $(CHECKFILE); \
	else \
		uv run pytest; \
	fi


.PHONY: test
TESTFILE ?=
test:
	make lint
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

codecheck: lint test xenon



test-rec:
	RUN_RECOVERY=1 make test

test-rec-only:
	make lint
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RUN_RECOVERY=1 uv run pytest -p pytest_cov tests/test_param_recovery_full.py -x

.PHONY: e2e-test
e2e-test:
	mkdir -p artifacts/e2e
	uv run synthgen --n 200 --seed 123 --out artifacts/e2e/raw.parquet
	uv run python -c "import polars as pl; df = pl.read_parquet('artifacts/e2e/raw.parquet'); df = df.rename({'naics_code': 'NAICS', 'zip_code': 'ZIP', 'y': 'is_written'}); df.write_parquet('artifacts/e2e/synthetic.parquet')"
	uv run model fit \
		--train artifacts/e2e/synthetic.parquet \
		--artifacts artifacts/e2e/model \
		--dashboard artifacts/e2e/dashboard \
		--naics-cuts 2 4 \
		--zip-cuts 3 5 \
		--draws 300 \
		--tune 300 \
		--chains 1 \
		--cores 1 \
		--target-accept 0.9 \
		--no-progressbar
	@printf '\nDashboard written to %s\n' "artifacts/e2e/dashboard/model_dashboard.html"
