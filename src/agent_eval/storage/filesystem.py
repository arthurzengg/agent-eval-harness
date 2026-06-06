"""Filesystem storage for run artifacts.

Layout written under the output directory::

    results.json                 # full SuiteResult
    summary.json                 # suite metadata + metrics only
    transcripts/<task_id>/trial_<n>.json
    index.html                   # HTML report (written by the HTML reporter)
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_eval.schemas import SuiteResult


def _dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_results(result: SuiteResult, output_dir: Path) -> Path:
    """Write ``results.json`` (the full SuiteResult) and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.json"
    _dump(path, result.model_dump(mode="json"))
    return path


def write_summary(result: SuiteResult, output_dir: Path) -> Path:
    """Write ``summary.json`` (suite metadata + metrics) and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    summary = {
        "suite": result.suite.model_dump(mode="json"),
        "scoring_mode": str(result.scoring_mode),
        "metrics": result.metrics.model_dump(mode="json"),
    }
    _dump(path, summary)
    return path


def write_transcripts(result: SuiteResult, output_dir: Path) -> None:
    """Write per-trial transcript JSON files under ``transcripts/``."""
    base = output_dir / "transcripts"
    for task_result in result.task_results:
        task_dir = base / task_result.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        for trial_result in task_result.trials:
            path = task_dir / f"trial_{trial_result.trial.index}.json"
            _dump(path, trial_result.model_dump(mode="json"))


def load_results(path: Path) -> SuiteResult:
    """Load a previously written ``results.json`` into a ``SuiteResult``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return SuiteResult.model_validate(data)
