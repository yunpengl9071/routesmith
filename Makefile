# Makefile — RouteSmith paper build targets

.PHONY: lint type-check test test-api verify-results figures exp1 exp2 ablations paper-pdf all

lint:
	.venv/bin/ruff check benchmark/ src/routesmith/predictor/

type-check:
	.venv/bin/mypy benchmark/ src/routesmith/predictor/lints.py src/routesmith/predictor/linucb.py --ignore-missing-imports

test:
	.venv/bin/pytest tests/ -m "not api" -v

test-api:
	.venv/bin/pytest tests/integration/ -v

verify-results:
	python3 -m benchmark.resume

figures:
	python3 -m benchmark.plot

exp1:
	python3 -m benchmark.experiments.exp1_binary

exp2:
	python3 -m benchmark.experiments.exp2_multimodel

ablations:
	python3 -m benchmark.experiments.ablations

paper-pdf:
	bash paper/build.sh

all: lint type-check test verify-results figures paper-pdf
