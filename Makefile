.PHONY: install test lint typecheck run-example baseline compare clean

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

# Regenerate the committed CI baseline from the deterministic echo agent.
baseline:
	uv run agent-eval run \
		--suite examples/suites/refund_support.yaml \
		--agent echo \
		--trials 3 \
		--output reports/refund_support
	cp reports/refund_support/results.json baselines/refund_support.json

# Gate the latest run against the committed baseline (exits non-zero on regression).
compare:
	uv run agent-eval compare \
		--baseline baselines/refund_support.json \
		--current reports/refund_support/results.json \
		--tolerance 0.0

clean:
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache
