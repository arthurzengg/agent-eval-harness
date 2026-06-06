"""Interactive terminal UI for browsing eval results."""

__all__ = ["run_ui"]


def run_ui(results_path: str) -> None:
    """Launch the results browser. Imports Textual lazily so the core
    install works without the ``ui`` extra."""
    from agent_eval.ui.app import EvalBrowserApp

    EvalBrowserApp(results_path).run()
