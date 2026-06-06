"""End-to-end CLI tests via Typer's in-process runner (also drives coverage)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_eval.cli import app

runner = CliRunner()

_SUITE = """
suite: { id: s, name: S }
defaults: { trials: 2 }
tasks:
  - id: t1
    expected_outcome: { final_output: hello }
    graders:
      - { type: exact_match, expected: hello }
"""

_BAD_SUITE = """
suite: { id: s, name: S }
tasks:
  - id: t1
    graders:
      - { type: not_a_real_grader }
"""


def _write(tmp_path: Path, text: str, name: str = "suite.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_validate_ok(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(_write(tmp_path, _SUITE))])
    assert result.exit_code == 0
    assert "OK" in result.stdout


def test_validate_unknown_grader(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(_write(tmp_path, _BAD_SUITE))])
    assert result.exit_code == 1
    assert "unknown grader type" in result.stdout


def test_validate_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "Invalid suite" in result.stdout


def test_run_writes_reports(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    result = runner.invoke(
        app,
        ["run", "--suite", str(_write(tmp_path, _SUITE)), "--agent", "echo", "--output", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "results.json").exists()
    assert (out / "index.html").exists()


def test_run_with_concurrency_and_trials(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(_write(tmp_path, _SUITE)),
            "--agent",
            "echo",
            "--trials",
            "3",
            "--concurrency",
            "2",
            "--scoring-mode",
            "binary",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads((out / "results.json").read_text())
    assert data["metrics"]["total_trials"] == 3


def test_run_invalid_suite(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["run", "--suite", str(tmp_path / "nope.yaml"), "--output", str(tmp_path / "o")]
    )
    assert result.exit_code == 1


def test_report_regenerates_html(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    runner.invoke(
        app,
        ["run", "--suite", str(_write(tmp_path, _SUITE)), "--agent", "echo", "--output", str(out)],
    )
    html = tmp_path / "fresh.html"
    result = runner.invoke(
        app, ["report", "--results", str(out / "results.json"), "--output", str(html)]
    )
    assert result.exit_code == 0
    assert html.exists()


def test_report_missing_results(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", "--results", str(tmp_path / "x.json"), "--output", str(tmp_path / "o.html")]
    )
    assert result.exit_code == 1


def test_compare_no_regression(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    runner.invoke(
        app,
        ["run", "--suite", str(_write(tmp_path, _SUITE)), "--agent", "echo", "--output", str(out)],
    )
    results = out / "results.json"
    result = runner.invoke(app, ["compare", "--baseline", str(results), "--current", str(results)])
    assert result.exit_code == 0
    assert "no regressions" in result.stdout


def test_compare_detects_regression(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    runner.invoke(
        app,
        ["run", "--suite", str(_write(tmp_path, _SUITE)), "--agent", "echo", "--output", str(out)],
    )
    baseline = out / "results.json"
    current = tmp_path / "current.json"
    data = json.loads(baseline.read_text())
    data["metrics"]["pass_rate"] = data["metrics"]["pass_rate"] - 0.5
    data["task_results"][0]["pass_rate"] = 0.0
    current.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["compare", "--baseline", str(baseline), "--current", str(current)])
    assert result.exit_code == 1
    assert "REGRESSED" in result.stdout


def test_compare_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["compare", "--baseline", str(tmp_path / "a.json"), "--current", str(tmp_path / "b.json")],
    )
    assert result.exit_code == 1


def test_run_enforces_consistent_trials(tmp_path: Path) -> None:
    suite = """
suite: { id: s, name: S }
defaults: { trials: 2, enforce_consistent_trials: true }
tasks:
  - id: a
    graders: [{ type: exact_match, expected: "" }]
  - id: b
    trials: 4
    graders: [{ type: exact_match, expected: "" }]
"""
    result = runner.invoke(
        app, ["run", "--suite", str(_write(tmp_path, suite)), "--output", str(tmp_path / "o")]
    )
    assert result.exit_code == 1
    assert "different trial counts" in result.stdout
