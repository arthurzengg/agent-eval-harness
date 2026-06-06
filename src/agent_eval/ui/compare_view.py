"""Textual application: compare two stored runs of the same suite."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

from agent_eval.compare import TaskDelta, compare_results
from agent_eval.schemas import TaskResult
from agent_eval.storage import load_results
from agent_eval.ui import presenter


class CompareApp(App[None]):
    """Browse per-task deltas and tool-sequence diffs between two runs."""

    TITLE = "agent-eval · compare"

    CSS = """
    #sidebar { width: 52; border-right: solid $panel; }
    #nav { height: 1fr; }
    #summary { height: auto; padding: 1 1; border-top: solid $panel; }
    #detail { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Switch pane", show=True),
        Binding("n", "next_regression", "Next regression"),
        Binding("g", "scroll_home", "Top", show=False),
        Binding("G", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, baseline_path: str | Path, current_path: str | Path) -> None:
        super().__init__()
        self._baseline = load_results(Path(baseline_path))
        self._current = load_results(Path(current_path))
        self._report = compare_results(self._baseline, self._current)
        self._base_tasks = {t.task_id: t for t in self._baseline.task_results}
        self._cur_tasks = {t.task_id: t for t in self._current.task_results}
        self._task_nodes: list[TreeNode[TaskDelta]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Tree("tasks", id="nav")
                yield Static(
                    presenter.compare_summary(self._report, self._baseline, self._current),
                    id="summary",
                )
            yield VerticalScroll(Static(id="detail"))
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = (
            f"{self._baseline.suite.name}: baseline vs current"
            if self._baseline.suite.id == self._current.suite.id
            else f"{self._baseline.suite.id} vs {self._current.suite.id}"
        )
        tree: Tree[TaskDelta] = self.query_one("#nav", Tree)
        tree.show_root = False
        tree.guide_depth = 2
        for delta in self._report.tasks:
            self._task_nodes.append(
                tree.root.add_leaf(presenter.compare_task_label(delta), data=delta)
            )
        self._select_initial()
        tree.focus()

    def _select_initial(self) -> None:
        """Select the first regressed task, falling back to the first task."""
        tree: Tree[TaskDelta] = self.query_one("#nav", Tree)
        target = next(
            (n for n in self._task_nodes if n.data is not None and n.data.regressed),
            self._task_nodes[0] if self._task_nodes else None,
        )
        if target is not None:
            tree.select_node(target)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[TaskDelta]) -> None:
        self._show_task(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected[TaskDelta]) -> None:
        self._show_task(event.node.data)

    def _show_task(self, delta: TaskDelta | None) -> None:
        if delta is None:
            return
        base: TaskResult | None = self._base_tasks.get(delta.task_id)
        cur: TaskResult | None = self._cur_tasks.get(delta.task_id)
        self.query_one("#detail", Static).update(presenter.format_task_compare(delta, base, cur))
        self.query_one(VerticalScroll).scroll_home(animate=False)

    def action_next_regression(self) -> None:
        regressions = [n for n in self._task_nodes if n.data is not None and n.data.regressed]
        if not regressions:
            self.notify("No regressed tasks.", severity="information")
            return
        tree: Tree[TaskDelta] = self.query_one("#nav", Tree)
        pos = regressions.index(tree.cursor_node) if tree.cursor_node in regressions else -1
        tree.select_node(regressions[(pos + 1) % len(regressions)])

    def action_scroll_home(self) -> None:
        self.query_one(VerticalScroll).scroll_home()

    def action_scroll_end(self) -> None:
        self.query_one(VerticalScroll).scroll_end()
