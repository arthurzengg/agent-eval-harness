"""Textual application: pick one stored run from a list of discovered runs."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, OptionList
from textual.widgets.option_list import Option

from agent_eval.storage import RunInfo
from agent_eval.ui import presenter


class RunPickerApp(App[Path | None]):
    """List discovered runs (newest first); exits with the chosen path.

    Returns ``None`` if the user quits without choosing.
    """

    TITLE = "agent-eval"
    SUB_TITLE = "pick a run"

    CSS = """
    OptionList { padding: 1 1; }
    """

    BINDINGS = [Binding("q", "quit_picker", "Quit")]

    def __init__(self, runs: list[RunInfo]) -> None:
        super().__init__()
        self._runs = runs

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield OptionList(
            *(Option(presenter.run_label(run), id=str(i)) for i, run in enumerate(self._runs)),
            id="runs",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#runs", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id is not None:
            self.exit(self._runs[int(event.option_id)].path)

    def action_quit_picker(self) -> None:
        self.exit(None)
