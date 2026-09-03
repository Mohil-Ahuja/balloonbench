.PHONY: env check test lint bench clean

PY ?= python

env:
	pip install -e ".[dev,vision]"

check:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check balloonbench scripts tests

# Regenerates the dataset and reproduces the results table from a fresh clone.
# Wired up milestone by milestone.
bench: check test
	@echo "bench: pipeline stages land with milestones M1-M11"

clean:
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
