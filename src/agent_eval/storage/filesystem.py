"""Filesystem storage for run artifacts.

Layout written under the output directory::

    results.json                 # full SuiteResult
    summary.json                 # suite metadata + metrics only
    transcripts/<task_id>/trial_<n>.json
    index.html                   # HTML report (written by the HTML reporter)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RunInfo:
    """Lightweight summary of one stored run, for listings and pickers."""

    path: Path
    suite_id: str
    suite_name: str
    pass_rate: float
    total_trials: int
    mtime: float


def discover_runs(root: Path) -> list[RunInfo]:
    """Scan ``root`` recursively for ``results.json`` files, newest first.

    Reads only the fields a listing needs (no full schema validation), and
    skips files that are not parseable results.
    """
    runs: list[RunInfo] = []
    for path in sorted(root.rglob("results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            suite = data["suite"]
            metrics = data.get("metrics", {})
            runs.append(
                RunInfo(
                    path=path,
                    suite_id=str(suite["id"]),
                    suite_name=str(suite.get("name") or suite["id"]),
                    pass_rate=float(metrics.get("pass_rate", 0.0)),
                    total_trials=int(metrics.get("total_trials", 0)),
                    mtime=path.stat().st_mtime,
                )
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # not a results file we understand
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs
