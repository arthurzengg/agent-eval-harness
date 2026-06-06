# agent-eval-harness

An extensible, CLI-first evaluation harness for tool-calling / agentic systems.
It models the core concepts from Anthropic's "Demystifying evals for AI agents"
and turns them into a small, typed, testable framework that runs local YAML eval
suites and produces JSON and HTML reports.

## 1. What this project is

`agent-eval-harness` runs automated evaluations against agents that call tools.
It is deliberately minimal but production-shaped: clear schemas, deterministic
code-based graders by default, an optional LLM judge behind a plugin, and a CLI
that produces inspectable artifacts. There is no web app — the surface is the
`agent-eval` command and the static reports it writes.

## 2. Why agent evals need transcripts, outcomes, graders, and multiple trials

Agent behavior is non-deterministic and multi-step, so a single string-match on
the final answer is rarely enough:

- **Transcripts** record the whole trace (assistant messages, tool calls, tool
  results, timings, token usage, errors) so you can grade *how* the agent
  worked, not just its last line — and inspect the run when a grader is wrong.
- **Outcomes** capture the final observable state (e.g. "refund processed"),
  which is often the thing you actually care about.
- **Graders** score transcripts and outcomes. Different tasks need different
  checks: required/forbidden tools, argument validity, final-state assertions,
  budgets, or (optionally) an LLM rubric. Tool-call *sequence* matching is
  supported but never the only way to pass.
- **Multiple trials** expose flakiness. The harness aggregates `pass@k` (at
  least one of k trials passed) and `pass^k` (all k trials passed), which tell
  very different stories about reliability.

## 3. How this maps to Anthropic's eval concepts

| Concept | Where it lives |
| --- | --- |
| Task | `schemas.Task` (input + success criteria) |
| Trial | `schemas.Trial` (one attempt; k per task) |
| Transcript / trace | `schemas.Transcript`, `TranscriptStep`, `ToolCall`, `ToolResult` |
| Outcome | `schemas.Outcome` (final observable state) |
| Grader | `graders/` (`BaseGrader` + 8 implementations) |
| Evaluation harness | `runner.Runner` (isolate, run, record, grade, aggregate) |
| Eval suite | `schemas.EvalSuite` + `suite_loader` (YAML) |
| Environment | `environments/` (`LocalTempDirEnvironment` for trial isolation) |
| Metrics | `metrics.py` (`pass@k`, `pass^k`, per-task / per-grader) |

## 4. Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # provision Python 3.11 and install the package
make test           # run the pytest suite
make lint           # ruff check + format check
make typecheck      # mypy (strict)
make run-example    # run the refund_support suite with the echo agent
```

Direct CLI usage:

```bash
# Validate a suite
agent-eval validate examples/suites/refund_support.yaml

# Run a suite against the deterministic echo agent, 3 trials each.
# --concurrency runs up to N trials at once (default 1 = serial); raise it to
# parallelize slow network/LLM-backed agents while keeping results ordered.
agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent echo \
  --trials 3 \
  --concurrency 8 \
  --output reports/refund_support

# Regenerate the HTML report from stored JSON results
agent-eval report \
  --results reports/refund_support/results.json \
  --output reports/refund_support/index.html

# Gate a run against a baseline: exits non-zero if pass rate / score
# regresses beyond --tolerance. Use in CI to block quality drops.
agent-eval compare \
  --baseline baselines/refund_support.json \
  --current reports/refund_support/results.json \
  --tolerance 0.0

# Browse stored results in an interactive terminal UI
# (requires the optional extra: pip install 'agent-eval-harness[ui]')
agent-eval ui --results reports/refund_support/results.json

# Without --results, a picker lists every run found under --dir
# (default: reports/) with suite name, timestamp, and pass rate.
agent-eval ui
agent-eval ui --dir reports/

# Compare two runs interactively: per-task pass-rate deltas, suite metric
# deltas, and per-trial tool-call sequence diffs ('n' jumps to regressions).
agent-eval ui --compare baselines/results.json reports/refund_support/results.json

# Or watch a run live: trials tick pending -> running -> pass/fail as they
# execute (composes with --concurrency), then the results browser opens.
agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent echo \
  --concurrency 8 \
  --output reports/refund_support \
  --ui
```

The `ui` command opens a full-screen terminal browser: a task tree with
per-trial pass/fail markers and suite metrics (`pass@k`, `pass^k`) on the
left, and the selected trial's grader verdicts and full transcript (tool
calls, results, errors, timings) on the right. Navigate with the arrow keys,
switch panes with `tab`, quit with `q`.

Artifacts written under `--output`:

```
results.json                 # full SuiteResult
summary.json                 # suite metadata + metrics
transcripts/<task_id>/trial_<n>.json
index.html                   # HTML report (metrics, tables, transcript viewer)
```

### Running against a real agent over HTTP

The `http` adapter POSTs each task to a `/run` endpoint. A runnable mock agent is
included:

```bash
python examples/agents/mock_refund_agent.py            # serves http://127.0.0.1:8080
agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent http --agent-url http://127.0.0.1:8080/run \
  --trials 3 --output reports/refund_support_http
```

The `http` adapter validates the response body against a typed schema, retries
transient failures (connection errors, timeouts, 5xx) with exponential backoff
while failing fast on 4xx / schema errors, sends an `X-Request-ID` header
recorded in trial metadata, and captures an agent-reported version
(`agent_version` field or `X-Agent-Version` header). Tune retries with
`--retry-attempts` (default 3) and `--retry-backoff` (base seconds, default 0.2).

### JSONL / dataset suites

Suites can also be authored as JSONL — one task per line — which makes it easy
to generate evals from logs, datasets, or production traces. An optional first
"header" line carries `suite` / `defaults` metadata; without it, suite metadata
is synthesized from the filename. `validate`, `run`, and `report` accept
`.jsonl` / `.ndjson` files transparently.

```jsonl
{"suite": {"id": "refund_support", "name": "Refund support"}, "defaults": {"trials": 2}}
{"id": "refund_allowed", "input": {"user_message": "..."}, "graders": [{"type": "state_check", "expect": {"refund.status": "processed"}}]}
{"id": "refund_denied",  "input": {"user_message": "..."}, "graders": [{"type": "state_check", "expect": {"refund.status": "denied"}}]}
```

```bash
agent-eval run --suite examples/suites/refund_support.jsonl --agent echo --output reports/refund_jsonl
```

## 5. Example suite

See [`examples/suites/refund_support.yaml`](examples/suites/refund_support.yaml).
The format is:

```yaml
suite: { id, name, description, version }
defaults:
  trials: 3
  timeout_seconds: 60
  scoring: { mode: weighted, pass_threshold: 0.80 }
  # Optional: USD per 1M tokens, used to estimate run cost from the
  # agent's reported token usage. Defaults to 0 (cost reported as $0).
  pricing: { input_per_1m: 3.0, output_per_1m: 15.0 }
tasks:
  - id: refund_allowed_under_30_days
    input: { user_message: "..." }
    reference: { summary: "..." }
    expected_outcome: { refund: { status: processed, order_id: A100 } }
    graders:
      - { type: tool_calls, weight: 0.25, required: [...], forbidden: [...] }
      - { type: state_check, weight: 0.35, expect: { refund.status: processed } }
      - { type: transcript, weight: 0.15, max_turns: 10 }
      - { type: llm_rubric, weight: 0.25, enabled: false, assertions: [...] }
```

**Graders available:** `exact_match`, `regex`, `tool_calls`, `argument_schema`,
`state_check`, `transcript`, `llm_rubric` (disabled by default).

**Scoring:** in `weighted` mode the task score is the weighted average of enabled
graders and passes at `pass_threshold` (and only if no hard-fail grader, e.g. a
forbidden tool, failed). In `binary` mode every enabled grader must pass. Partial
credit is preserved throughout.

## 6. How to add a new grader

1. Create `src/agent_eval/graders/my_grader.py`:

   ```python
   from agent_eval.graders.base import BaseGrader
   from agent_eval.schemas import GraderResult, Task, Trial

   class MyGrader(BaseGrader):
       type = "my_grader"

       async def grade(self, task: Task, trial: Trial) -> GraderResult:
           ok = ...  # inspect trial.transcript / trial.outcome / trial.final_output
           return self.result(score=1.0 if ok else 0.0, passed=ok, reason="...")
   ```

2. Register it in `src/agent_eval/graders/__init__.py` by adding the class to the
   `_GRADERS` map. Configure it in a suite with `- type: my_grader`.

Read `self.options` for grader-specific config; use `self.result(...)` to build
the `GraderResult` with the configured weight. Set `hard_fail=True` for failures
that should sink the task regardless of weighted score.

## 7. How to add a new agent adapter

1. Implement the `AgentAdapter` protocol in `src/agent_eval/adapters/`:

   ```python
   from agent_eval.adapters.base import AgentRunResult
   from agent_eval.environments.base import EvalEnvironment
   from agent_eval.schemas import Task

   class MyAgentAdapter:
       name = "my_agent"

       async def run(self, task: Task, env: EvalEnvironment) -> AgentRunResult:
           # drive your agent, then return a normalized result
           return AgentRunResult(final_output=..., transcript=..., outcome=...)
   ```

2. Register a factory in `src/agent_eval/adapters/__init__.py` with
   `@adapter_registry.register("my_agent")`. Select it via `--agent my_agent`.

### Publishing graders/adapters from an external package

You do not need to fork this repo to add plugins. Any installed distribution can
register graders, adapters, or reporters via entry points; the CLI discovers them
at startup. In your package's `pyproject.toml`:

```toml
[project.entry-points."agent_eval.graders"]
my_grader = "my_pkg.graders:MyGraderFactory"   # (GraderConfig) -> BaseGrader

[project.entry-points."agent_eval.adapters"]
my_agent = "my_pkg.adapters:make_my_agent"     # (**kwargs) -> AgentAdapter
```

The entry-point name becomes the plugin name (used in suites / `--agent`).
Discovery is additive — built-in names always win — idempotent, and resilient:
a plugin that fails to import warns and is skipped rather than aborting the run.

## 8. How to run in CI

A safe example workflow lives at
[`.github/workflows/eval.yml`](.github/workflows/eval.yml). It installs the
package with uv and runs the suite with the deterministic `echo` agent (no
secrets, no network). It also documents the `http`-agent invocation against the
bundled mock server. Adapt it to run against your own agent and fail the build on
a `pass^k` regression.

## 9. Limitations and next steps

- **LLM judge is interface-only.** Only the `mock` provider is wired up; OpenAI /
  Anthropic providers are stubs behind env vars and require implementation.
- **Single-process run backend.** Concurrency is bounded in-process via
  `--concurrency`; there is no distributed run backend yet.
- **Filesystem storage only.** No SQLite/cloud storage backend.
- **No web UI.** Reports are static HTML by design.

Recommended next steps: implement a real LLM judge provider, add JSONL suites,
parallelize trials, add a SQLite storage backend, and add a CI action that
comments `pass@k` / `pass^k` deltas on pull requests.
