"""Reporters and their registry registration."""

from agent_eval.registry import reporter_registry
from agent_eval.reporters.base import Reporter
from agent_eval.reporters.console_reporter import ConsoleReporter
from agent_eval.reporters.html_reporter import HTMLReporter, render_html
from agent_eval.reporters.json_reporter import JSONReporter


@reporter_registry.register("json")
def _make_json(**_: object) -> JSONReporter:
    return JSONReporter()


@reporter_registry.register("html")
def _make_html(**_: object) -> HTMLReporter:
    return HTMLReporter()


@reporter_registry.register("console")
def _make_console(**_: object) -> ConsoleReporter:
    return ConsoleReporter()


__all__ = [
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
    "Reporter",
    "render_html",
]
