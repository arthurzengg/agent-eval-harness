"""Interactive terminal UI for browsing eval results."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_eval.harness import RunConfig
    from agent_eval.schemas import EvalSuite, SuiteResult

__all__ = ["run_live", "run_ui"]


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
