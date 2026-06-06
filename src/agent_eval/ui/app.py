"""Textual application: an interactive browser over a stored ``results.json``."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

from agent_eval.schemas import SuiteResult, TrialResult
from agent_eval.storage import load_results
from agent_eval.ui import presenter


class EvalBrowserApp(App[None]):
    """Browse tasks, trials, transcripts, and grader verdicts."""

    TITLE = "agent-eval"

    CSS = """
    #sidebar { width: 42; border-right: solid $panel; }
    #nav { height: 1fr; }
    #metrics { height: auto; padding: 1 1; border-top: solid $panel; }
    #detail { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Switch pane", show=True),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, results_path: str | Path) -> None:
        super().__init__()
        self._results_path = Path(results_path)
        self._result: SuiteResult = load_results(self._results_path)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Tree("suite", id="nav")
                yield Static(presenter.metrics_summary(self._result.metrics), id="metrics")
            yield VerticalScroll(Static(id="detail"))
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = presenter.suite_title(self._result)
        tree: Tree[TrialResult] = self.query_one("#nav", Tree)
        tree.show_root = False
        tree.guide_depth = 2
        for task in self._result.task_results:
            node: TreeNode[TrialResult] = tree.root.add(presenter.task_label(task), expand=True)
            for trial in task.trials:
                node.add_leaf(presenter.trial_label(trial), data=trial)
        tree.focus()
        # Select the first trial so the detail pane is never empty.
        for task_node in tree.root.children:
            if task_node.children:
                tree.select_node(task_node.children[0])
                break

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[TrialResult]) -> None:
        self._show_trial(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected[TrialResult]) -> None:
        self._show_trial(event.node.data)

    def _show_trial(self, trial: TrialResult | None) -> None:
        if trial is None:
            return
        self.query_one("#detail", Static).update(presenter.format_trial_detail(trial))
        self.query_one(VerticalScroll).scroll_home(animate=False)

    def action_scroll_home(self) -> None:
        self.query_one(VerticalScroll).scroll_home()

    def action_scroll_end(self) -> None:
        self.query_one(VerticalScroll).scroll_end()
