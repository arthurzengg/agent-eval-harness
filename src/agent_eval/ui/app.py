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
        Binding("f", "toggle_failures", "Failures only"),
        Binding("n", "next_failure", "Next fail"),
        Binding("p", "prev_failure", "Prev fail", show=False),
        Binding("e", "toggle_expand", "Expand"),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, results_path: str | Path) -> None:
        super().__init__()
        self._results_path = Path(results_path)
        self._result: SuiteResult = load_results(self._results_path)
        self._failures_only = False
        self._expand = False
        self._current: TrialResult | None = None
        self._trial_nodes: list[TreeNode[TrialResult]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Tree("suite", id="nav")
                yield Static(presenter.metrics_summary(self._result.metrics), id="metrics")
            yield VerticalScroll(Static(id="detail"))
        yield Footer()

    def on_mount(self) -> None:
        self._update_sub_title()
        self._build_tree()
        self.query_one("#nav", Tree).focus()

    def _update_sub_title(self) -> None:
        title = presenter.suite_title(self._result)
        if self._failures_only:
            title += "  ·  failures only"
        self.sub_title = title

    def _build_tree(self) -> None:
        """(Re)populate the navigation tree, honoring the failures filter."""
        tree: Tree[TrialResult] = self.query_one("#nav", Tree)
        tree.clear()
        tree.show_root = False
        tree.guide_depth = 2
        self._trial_nodes = []
        for task in self._result.task_results:
            trials = [t for t in task.trials if not (self._failures_only and t.passed)]
            if not trials:
                continue
            node: TreeNode[TrialResult] = tree.root.add(presenter.task_label(task), expand=True)
            for trial in trials:
                self._trial_nodes.append(node.add_leaf(presenter.trial_label(trial), data=trial))
        self._select_initial()

    def _select_initial(self) -> None:
        """Select the first failed trial, falling back to the first trial."""
        tree: Tree[TrialResult] = self.query_one("#nav", Tree)
        target = next(
            (n for n in self._trial_nodes if n.data is not None and not n.data.passed),
            self._trial_nodes[0] if self._trial_nodes else None,
        )
        if target is not None:
            tree.select_node(target)
        else:
            self.query_one("#detail", Static).update(
                "[grey62]No trials match the current filter.[/grey62]"
            )

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[TrialResult]) -> None:
        self._show_trial(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected[TrialResult]) -> None:
        self._show_trial(event.node.data)

    def _show_trial(self, trial: TrialResult | None) -> None:
        if trial is None:
            return
        self._current = trial
        self.query_one("#detail", Static).update(
            presenter.format_trial_detail(trial, expand=self._expand)
        )
        self.query_one(VerticalScroll).scroll_home(animate=False)

    def _jump_failure(self, direction: int) -> None:
        """Move selection to the next/previous failed trial, wrapping around."""
        failures = [n for n in self._trial_nodes if n.data is not None and not n.data.passed]
        if not failures:
            self.notify("No failed trials.", severity="information")
            return
        tree: Tree[TrialResult] = self.query_one("#nav", Tree)
        try:
            pos = failures.index(tree.cursor_node) if tree.cursor_node in failures else -direction
        except ValueError:  # pragma: no cover - defensive
            pos = -direction
        tree.select_node(failures[(pos + direction) % len(failures)])

    def action_next_failure(self) -> None:
        self._jump_failure(1)

    def action_prev_failure(self) -> None:
        self._jump_failure(-1)

    def action_toggle_failures(self) -> None:
        self._failures_only = not self._failures_only
        self._update_sub_title()
        self._build_tree()

    def action_toggle_expand(self) -> None:
        self._expand = not self._expand
        if self._current is not None:
            self.query_one("#detail", Static).update(
                presenter.format_trial_detail(self._current, expand=self._expand)
            )

    def action_scroll_home(self) -> None:
        self.query_one(VerticalScroll).scroll_home()

    def action_scroll_end(self) -> None:
        self.query_one(VerticalScroll).scroll_end()
