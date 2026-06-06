"""Interactive terminal UI for browsing eval results."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from agent_eval.harness import RunConfig
    from agent_eval.schemas import EvalSuite, SuiteResult
    from agent_eval.storage import RunInfo

__all__ = ["pick_run", "run_compare_ui", "run_live", "run_ui"]


def run_ui(results_path: str) -> None:
    """Launch the results browser. Imports Textual lazily so the core
    install works without the ``ui`` extra."""
    from agent_eval.ui.app import EvalBrowserApp

    EvalBrowserApp(results_path).run()


def run_live(suite: "EvalSuite", config: "RunConfig") -> "SuiteResult | None":
    """Run ``suite`` under the live TUI; returns the result, or ``None`` if
    aborted. Imports Textual lazily like :func:`run_ui`."""
    from agent_eval.ui.live import LiveRunApp

    return LiveRunApp(suite, config).run()


def pick_run(runs: "list[RunInfo]") -> "Path | None":
    """Show the run picker; returns the chosen results path, or ``None`` if
    the user quit. Imports Textual lazily like :func:`run_ui`."""
    from agent_eval.ui.picker import RunPickerApp

    return RunPickerApp(runs).run()


def run_compare_ui(baseline_path: str, current_path: str) -> None:
    """Launch the run-comparison browser. Imports Textual lazily like
    :func:`run_ui`."""
    from agent_eval.ui.compare_view import CompareApp

    CompareApp(baseline_path, current_path).run()
