"""Load and validate eval suites from YAML (or JSON) files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agent_eval.schemas import EvalSuite


class SuiteLoadError(Exception):
    """Raised when a suite file cannot be read or validated."""


def load_suite(path: str | Path) -> EvalSuite:
    """Read a suite file and return a validated ``EvalSuite``.

    Raises ``SuiteLoadError`` with a friendly message on any failure.
    """
    p = Path(path)
    if not p.exists():
        raise SuiteLoadError(f"Suite file not found: {p}")

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
