"""JSON reporter: persists full results, summary, and per-trial transcripts."""

from __future__ import annotations

from pathlib import Path

from agent_eval.schemas import SuiteResult
from agent_eval.storage import write_results, write_summary, write_transcripts


class JSONReporter:
    """Writes ``results.json``, ``summary.json``, and transcript files."""

    name = "json"

    def render(self, result: SuiteResult, output_dir: Path) -> Path | None:
        path = write_results(result, output_dir)
        write_summary(result, output_dir)
        write_transcripts(result, output_dir)
        return path
