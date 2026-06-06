"""HTML reporter using Jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_eval.reliability import flaky_tasks, suite_reliability_curve
from agent_eval.schemas import SuiteResult

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def _tojson(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _per_grader_detail(result: SuiteResult) -> list[dict[str, Any]]:
    """Aggregate pass/total counts per grader type across all trials."""
    passed: dict[str, int] = {}
    total: dict[str, int] = {}
    for tr in result.task_results:
        for trial in tr.trials:
            for gr in trial.grader_results:
                total[gr.grader_type] = total.get(gr.grader_type, 0) + 1
                passed[gr.grader_type] = passed.get(gr.grader_type, 0) + (1 if gr.passed else 0)
    return [
        {
            "type": g,
            "passed": passed[g],
            "total": total[g],
            "rate": passed[g] / total[g] if total[g] else 0.0,
        }
        for g in sorted(total)
    ]


def _task_tokens(result: SuiteResult) -> dict[str, int]:
    """Total tokens per task id (0 when no usage was reported)."""
    return {
        tr.task_id: sum(t.trial.transcript.total_tokens() for t in tr.trials)
        for tr in result.task_results
    }


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
        per_grader_detail=_per_grader_detail(result),
        task_tokens=_task_tokens(result),
        reliability=suite_reliability_curve(result.task_results),
        flaky=flaky_tasks(result.task_results),
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
