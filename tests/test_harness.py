from pathlib import Path

import agent_eval.graders  # noqa: F401 - register graders
from agent_eval.harness import RunConfig, apply_overrides, run_suite_to_disk
from agent_eval.schemas import EvalSuite, ScoringMode


def _suite() -> EvalSuite:
    return EvalSuite.model_validate(
        {
            "suite": {"id": "s", "name": "S"},
            "defaults": {"trials": 1},
            "tasks": [
                {
                    "id": "t1",
                    "expected_outcome": {"final_output": "hello"},
                    "graders": [{"type": "exact_match", "expected": "hello"}],
                }
            ],
        }
    )


def test_apply_overrides_mutates_defaults() -> None:
    suite = _suite()
    apply_overrides(suite, RunConfig(trials=5, scoring_mode=ScoringMode.binary))
    assert suite.defaults.trials == 5
    assert suite.defaults.scoring.mode == ScoringMode.binary


def test_run_suite_to_disk_writes_reports(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    artifacts = run_suite_to_disk(_suite(), out, RunConfig(agent="echo", trials=2))

    assert artifacts.result.metrics.total_trials == 2
    assert artifacts.json_path is not None and artifacts.json_path.exists()
    assert artifacts.html_path.exists()
    assert (out / "results.json").exists()
