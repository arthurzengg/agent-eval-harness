from pathlib import Path

import pytest

from agent_eval.suite_loader import SuiteLoadError, load_suite

SUITE = Path("examples/suites/refund_support.yaml")


def test_loads_example_suite() -> None:
    suite = load_suite(SUITE)
    assert suite.suite.id == "refund_support"
    assert len(suite.tasks) == 2
    assert suite.defaults.trials == 3
    assert suite.task_trials(suite.tasks[0]) == 3


def test_missing_file_raises() -> None:
    with pytest.raises(SuiteLoadError, match="not found"):
        load_suite("does/not/exist.yaml")


def test_invalid_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("suite: {id: x}\n")  # missing required name + tasks
    with pytest.raises(SuiteLoadError, match="failed validation"):
        load_suite(bad)
