.PHONY: install test lint typecheck run-example clean

# Provision the venv and install the package (with dev extras) using uv.
install:
	uv venv --python 3.11
	uv pip install --python .venv -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

run-example:
	uv run agent-eval run \
		--suite examples/suites/refund_support.yaml \
		--agent echo \
		--trials 3 \
		--output reports/refund_support

clean:
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache
