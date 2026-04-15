.ONESHELL:
SHELL = /bin/bash

# Get conda root directory
CONDA_ROOT := $(shell conda info --json 2>/dev/null | python3 -c "import json, sys; print(json.load(sys.stdin)['root_prefix'])")

# Define the activation command
ACTIVATE_ENV := \
	source $(CONDA_ROOT)/etc/profile.d/conda.sh && \
	(conda activate myenv 2>/dev/null || ( \
		echo "Creating env myenv"; \
		conda create -y -n myenv python=3.10 && \
		conda activate myenv && \
		pip install -r requirements.txt \
	)) && \
	export PYTHONPATH=.

format:
	@$(ACTIVATE_ENV)
	python3 -m black .
	python3 -m flake8
	python3 -m mypy

clean:
	@echo "Cleaning Python cache..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".black_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

test: _test_run format

_test_run:
	@$(ACTIVATE_ENV)
	python3 -m pytest

pull:
	@if [ -z "$(SERVER)" ]; then echo "Error: SERVER variable is required"; exit 1; fi
	@$(ACTIVATE_ENV)
	python3 ./scripts/download-remote.py --url $(SERVER) && \
	python3 ./scripts/preprocess-remote.py && \
	python3 ./scripts/create-test-dataset.py 

COMMON_TRAIN_ARGS = \
	--batch-size 32 \
	--epochs 250 \
	--patience 3 \
	--sampling oversample \
	--teacher-scale 8

train:
	@$(ACTIVATE_ENV)
	python3 ./scripts/train.py $(COMMON_TRAIN_ARGS) --trainer teacher
	python3 ./scripts/train.py $(COMMON_TRAIN_ARGS) --trainer student --teacher-weights best
