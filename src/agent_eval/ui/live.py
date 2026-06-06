"""Textual application: watch a suite run live, trial by trial."""

from __future__ import annotations

import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

from agent_eval.harness import RunConfig, apply_overrides, run_suite
from agent_eval.runner import ProgressEvent, TrialFinished, TrialStarted
from agent_eval.schemas import EvalSuite, SuiteResult
from agent_eval.ui import presenter


class LiveRunApp(App[SuiteResult | None]):
    """Run a suite with live per-trial progress; exits with the result.

    Returns the finished ``SuiteResult`` (the caller persists reports and
    hands off to the browser), or ``None`` if the run was aborted.
    """

    TITLE = "agent-eval · running"

    CSS = """
    #sidebar { width: 42; border-right: solid $panel; }
    #nav { height: 1fr; }
    #progress { height: auto; padding: 1 1; border-top: solid $panel; }
    """

    BINDINGS = [Binding("q", "abort", "Abort run")]

    def __init__(self, suite: EvalSuite, config: RunConfig) -> None:
        super().__init__()
        self._suite = suite
        self._config = config
        # Overrides are applied up front (idempotently re-applied by
        # run_suite) so trial counts shown in the tree match the run.
        apply_overrides(self._suite, self._config)
        self._trial_nodes: dict[tuple[str, int], TreeNode[None]] = {}
        self._total = sum(suite.task_trials(t) for t in suite.tasks)
        self._done = 0
        self._passed = 0
        self._start = time.perf_counter()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Tree("suite", id="nav")
                yield Static(id="progress")
            yield Static(
                f"Running [b]{self._suite.suite.name}[/b] with the "
                f"[b]{self._config.agent}[/b] agent "
                f"(concurrency {self._config.concurrency})…\n\n"
                "[grey62]The results browser opens when the run finishes. "
                "Press q to abort.[/grey62]",
                id="banner",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self._suite.suite.name} ({self._suite.suite.id})"
        tree: Tree[None] = self.query_one("#nav", Tree)
        tree.show_root = False
        tree.guide_depth = 2
        for task in self._suite.tasks:
            node = tree.root.add(task.id, expand=True)
            for index in range(self._suite.task_trials(task)):
                self._trial_nodes[(task.id, index)] = node.add_leaf(
                    presenter.live_trial_label(index, "pending")
                )
        self._refresh_progress()
        self.run_worker(self._execute(), exclusive=True)

    async def _execute(self) -> None:
        result = await run_suite(self._suite, self._config, on_event=self._on_event)
        self.exit(result)

    def _on_event(self, event: ProgressEvent) -> None:
        # The runner shares the app's event loop (the run lives in an async
        # worker), so it is safe to touch widgets directly here.
        node = self._trial_nodes.get((event.task_id, event.index))
        if node is None:  # pragma: no cover - defensive
            return
        if isinstance(event, TrialStarted):
            node.set_label(presenter.live_trial_label(event.index, "running"))
        elif isinstance(event, TrialFinished):
            node.set_label(presenter.live_trial_label(event.index, "done", event.result))
            self._done += 1
            self._passed += int(event.result.passed)
        self._refresh_progress()

    def _refresh_progress(self) -> None:
        elapsed = time.perf_counter() - self._start
        self.query_one("#progress", Static).update(
            presenter.live_progress(self._done, self._total, self._passed, elapsed)
        )

    def action_abort(self) -> None:
        self.workers.cancel_all()
        self.exit(None)
