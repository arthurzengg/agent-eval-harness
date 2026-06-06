"""Service layer: turn a loaded suite into results on disk.

This wraps the run pipeline (build adapter -> apply overrides -> run -> persist)
behind small, Typer-free functions so the CLI, the upcoming ``compare`` command,
and programmatic callers all share one code path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import agent_eval.adapters  # noqa: F401 - register adapters
from agent_eval.adapters.base import AgentAdapter
from agent_eval.environments.local_tempdir import LocalTempDirEnvironment
from agent_eval.registry import adapter_registry
from agent_eval.reporters.html_reporter import HTMLReporter
from agent_eval.reporters.json_reporter import JSONReporter
from agent_eval.runner import ProgressCallback, Runner
from agent_eval.schemas import EvalSuite, ScoringMode, SuiteResult


@dataclass
class RunConfig:
    """Options controlling how a suite is run."""

    agent: str = "echo"
    agent_url: str = ""
    trials: int | None = None
    scoring_mode: ScoringMode | None = None
    concurrency: int = 1
    keep_workdirs: bool = False
    retry_attempts: int = 3
    retry_backoff: float = 0.2


@dataclass
class RunArtifacts:
    """The result of a run plus where its reports were written."""

    result: SuiteResult
    json_path: Path | None
    html_path: Path


def apply_overrides(suite: EvalSuite, config: RunConfig) -> None:
    """Mutate ``suite`` in place with any CLI/config overrides."""
    if config.trials is not None:
        suite.defaults.trials = config.trials
    if config.scoring_mode is not None:
        suite.defaults.scoring.mode = config.scoring_mode


def build_adapter(suite: EvalSuite, config: RunConfig) -> AgentAdapter:
    """Instantiate the configured agent adapter for ``suite``."""
    return adapter_registry.create(
        config.agent,
        agent_url=config.agent_url,
        timeout=suite.defaults.timeout_seconds,
        retry_attempts=config.retry_attempts,
        retry_backoff=config.retry_backoff,
    )


def build_runner(
    adapter: AgentAdapter, config: RunConfig, on_event: ProgressCallback | None = None
) -> Runner:
    """Construct a Runner wired with the env factory and concurrency cap."""
    return Runner(
        adapter,
        env_factory=lambda: LocalTempDirEnvironment(config.keep_workdirs),
        concurrency=config.concurrency,
        on_event=on_event,
    )


async def run_suite(
    suite: EvalSuite, config: RunConfig, on_event: ProgressCallback | None = None
) -> SuiteResult:
    """Apply overrides, build the adapter/runner, and run the suite."""
    apply_overrides(suite, config)
    adapter = build_adapter(suite, config)
    return await build_runner(adapter, config, on_event).run_suite(suite)


def write_reports(result: SuiteResult, output: Path) -> RunArtifacts:
    """Write the JSON + HTML reports for ``result`` under ``output``."""
    json_path = JSONReporter().render(result, output)
    html_path = HTMLReporter().render(result, output)
    return RunArtifacts(result=result, json_path=json_path, html_path=html_path)


def run_suite_to_disk(suite: EvalSuite, output: Path, config: RunConfig) -> RunArtifacts:
    """Run ``suite`` and write the JSON + HTML reports under ``output``."""
    return write_reports(asyncio.run(run_suite(suite, config)), output)
