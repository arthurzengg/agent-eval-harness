"""Tests for JSONL/dataset suite loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_eval.suite_loader import SuiteLoadError, load_suite

_TASK = '{"id": "%s", "graders": [{"type": "exact_match", "expected": "x"}]}'


def _write(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_jsonl_with_header(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "suite.jsonl",
        [
            '{"suite": {"id": "s1", "name": "S1"}, "defaults": {"trials": 4}}',
            _TASK % "t1",
            _TASK % "t2",
        ],
    )
    suite = load_suite(p)
    assert suite.suite.id == "s1"
    assert suite.defaults.trials == 4
    assert [t.id for t in suite.tasks] == ["t1", "t2"]


def test_jsonl_without_header_synthesizes_metadata(tmp_path: Path) -> None:
    p = _write(tmp_path, "my_dataset.jsonl", [_TASK % "t1"])
    suite = load_suite(p)
    assert suite.suite.id == "my_dataset"
    assert suite.suite.name == "my_dataset"
    assert len(suite.tasks) == 1


def test_blank_lines_and_comments_ignored(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.jsonl", ["", "// a comment", _TASK % "t1", ""])
    suite = load_suite(p)
    assert [t.id for t in suite.tasks] == ["t1"]


def test_ndjson_extension_supported(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.ndjson", [_TASK % "t1"])
    assert load_suite(p).tasks[0].id == "t1"


def test_bad_json_reports_line_number(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.jsonl", [_TASK % "t1", "{not json}"])
    with pytest.raises(SuiteLoadError, match="line 2"):
        load_suite(p)


def test_non_object_line_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.jsonl", ["[1, 2, 3]"])
    with pytest.raises(SuiteLoadError, match="must be a JSON object"):
        load_suite(p)


def test_empty_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="empty"):
        load_suite(p)


def test_header_only_no_tasks_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "s.jsonl", ['{"suite": {"id": "s", "name": "S"}}'])
    with pytest.raises(SuiteLoadError, match="no task lines"):
        load_suite(p)


def test_example_jsonl_suite_loads() -> None:
    suite = load_suite("examples/suites/refund_support.jsonl")
    assert suite.suite.id == "refund_support_jsonl"
    assert len(suite.tasks) == 2
