"""Load and validate eval suites from YAML or JSONL files.

Two on-disk formats are supported:

- **YAML/JSON** (``.yaml``/``.yml``/``.json``) — a single document with
  ``suite``, ``defaults``, and ``tasks`` keys.
- **JSONL** (``.jsonl``/``.ndjson``) — one JSON object per line, where each
  line is a task. An optional first "header" line carrying a ``suite`` (and
  optional ``defaults``) key supplies suite metadata; without it, metadata is
  synthesized from the filename. This makes it trivial to generate eval suites
  from logs, datasets, or production traces (emit one task per line).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval.schemas import EvalSuite

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class SuiteLoadError(Exception):
    """Raised when a suite file cannot be read or validated."""


def load_suite(path: str | Path) -> EvalSuite:
    """Read a suite file and return a validated ``EvalSuite``.

    Dispatches on file extension: ``.jsonl``/``.ndjson`` use the line-oriented
    loader; everything else is parsed as YAML/JSON. Raises ``SuiteLoadError``
    with a friendly message on any failure.
    """
    p = Path(path)
    if not p.exists():
        raise SuiteLoadError(f"Suite file not found: {p}")
    if p.suffix.lower() in _JSONL_SUFFIXES:
        return _load_jsonl_suite(p)
    return _load_yaml_suite(p)


def _load_yaml_suite(p: Path) -> EvalSuite:
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SuiteLoadError(f"Failed to parse YAML in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise SuiteLoadError(f"Suite file {p} must contain a mapping at the top level.")

    try:
        return EvalSuite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteLoadError(f"Suite {p} failed validation:\n{exc}") from exc


def _load_jsonl_suite(p: Path) -> EvalSuite:
    """Build an ``EvalSuite`` from a one-object-per-line JSONL file."""
    records = _read_jsonl_records(p)
    if not records:
        raise SuiteLoadError(f"JSONL suite {p} is empty (no tasks).")

    header: dict[str, object] = {}
    first_line, first = records[0]
    # A first record containing a "suite" key (and no "id") is a header, not a task.
    if "suite" in first and "id" not in first:
        header = first
        task_records = records[1:]
    else:
        task_records = records

    if not task_records:
        raise SuiteLoadError(f"JSONL suite {p} has a header but no task lines.")

    suite_meta = header.get("suite") or {"id": p.stem, "name": p.stem}
    payload: dict[str, object] = {"suite": suite_meta, "tasks": [t for _, t in task_records]}
    if "defaults" in header:
        payload["defaults"] = header["defaults"]

    try:
        return EvalSuite.model_validate(payload)
    except ValidationError as exc:
        raise SuiteLoadError(f"JSONL suite {p} failed validation:\n{exc}") from exc


def _read_jsonl_records(p: Path) -> list[tuple[int, dict[str, object]]]:
    """Parse non-blank lines into (line_number, object) pairs."""
    records: list[tuple[int, dict[str, object]]] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue  # allow blank lines and // comments
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SuiteLoadError(f"{p} line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise SuiteLoadError(f"{p} line {lineno}: each line must be a JSON object.")
        records.append((lineno, obj))
    return records
