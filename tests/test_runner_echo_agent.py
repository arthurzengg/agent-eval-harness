from pathlib import Path

import pytest

import agent_eval.graders  # noqa: F401 - register graders
from agent_eval.adapters.echo import EchoAgentAdapter
from agent_eval.reporters.html_reporter import HTMLReporter
from agent_eval.reporters.json_reporter import JSONReporter
from agent_eval.runner import Runner
from agent_eval.suite_loader import load_suite

SUITE = Path("examples/suites/refund_support.yaml")


@pytest.fixture
async def suite_result():
    suite = load_suite(SUITE)
    return await Runner(EchoAgentAdapter()).run_suite(suite)


async def test_echo_agent_passes_example_suite(suite_result) -> None:
    m = suite_result.metrics
    assert m.total_tasks == 2
    assert m.total_trials == 6
    # The echo agent fulfils each task's declared criteria deterministically.
    assert m.pass_at_k == 1.0
    assert m.pass_caret_k == 1.0
    assert m.error_rate == 0.0


async def test_json_and_html_reports_written(tmp_path: Path, suite_result) -> None:
    json_path = JSONReporter().render(suite_result, tmp_path)
    html_path = HTMLReporter().render(suite_result, tmp_path)

    assert json_path.exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "transcripts" / "refund_allowed_under_30_days" / "trial_0.json").exists()
    assert html_path.exists()
    assert "Refund Support Agent Eval" in html_path.read_text(encoding="utf-8")


async def test_tool_calling_suite_passes(tmp_path: Path) -> None:
    suite = load_suite(Path("examples/suites/tool_calling_agent.yaml"))
    result = await Runner(EchoAgentAdapter()).run_suite(suite)
    assert result.metrics.pass_caret_k == 1.0


async def test_runner_emits_progress_events() -> None:
    from agent_eval.runner import ProgressEvent, TrialFinished, TrialStarted

    events: list[ProgressEvent] = []
    suite = load_suite(SUITE)
    result = await Runner(EchoAgentAdapter(), on_event=events.append).run_suite(suite)

    started = [e for e in events if isinstance(e, TrialStarted)]
    finished = [e for e in events if isinstance(e, TrialFinished)]
    assert len(started) == len(finished) == result.metrics.total_trials
    # Every trial starts before it finishes, keyed by (task_id, index).
    for fin in finished:
        start_pos = events.index(TrialStarted(task_id=fin.task_id, index=fin.index))
        assert start_pos < events.index(fin)
    # Finished events carry the graded result.
    assert all(f.result.passed for f in finished)


async def test_runner_survives_observer_errors() -> None:
    def boom(_event: object) -> None:
        raise RuntimeError("observer bug")

    suite = load_suite(SUITE)
    result = await Runner(EchoAgentAdapter(), on_event=boom).run_suite(suite)
    assert result.metrics.error_rate == 0.0
