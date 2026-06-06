"""HTML reporter using Jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_eval.schemas import SuiteResult

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _tojson(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_html(result: SuiteResult) -> str:
    """Render a ``SuiteResult`` to an HTML string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.globals["tojson"] = _tojson
    env.globals["pct"] = _pct
    template = env.get_template("report.html.j2")
    return template.render(
        suite=result.suite,
        scoring_mode=str(result.scoring_mode),
        metrics=result.metrics,
        task_results=result.task_results,
    )


class HTMLReporter:
    """Writes ``index.html`` (or a chosen path) for a suite run."""

    name = "html"

    def render(self, result: SuiteResult, output_dir: Path, filename: str = "index.html") -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        path.write_text(render_html(result), encoding="utf-8")
        return path

    def render_to_file(self, result: SuiteResult, output_file: Path) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(render_html(result), encoding="utf-8")
        return output_file
