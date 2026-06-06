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
  very different stories about reliability. `pass@k` / `pass^k` are computed
  **per task** against that task's own trial count, so they stay correct when
  tasks use different counts; reports then label the metric with the range
  (`pass@2..4`). Set `defaults.enforce_consistent_trials: true` to reject a
  suite whose tasks resolve to differing counts.

Beyond a single `pass@k`, the harness reports the full **reliability curve** —
`pass@1..N` and `pass^1..N` — using the standard unbiased combinatorial
estimators (`pass@k = 1 - C(n-c,k)/C(n,k)`, `pass^k = C(c,k)/C(n,k)`), and
flags **flaky tasks** that pass on some trials and fail on others. Both the
console summary and the HTML report render the curve and the flaky-task list.
See [`agent_eval/reliability.py`](src/agent_eval/reliability.py).

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
make test           # run the pytest suite (measures coverage, fails under 85%)
make coverage       # same gate plus a browsable htmlcov/ report
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

The static HTML report (`index.html`) is self-contained (no external assets) and
interactive: a sticky toolbar filters to **only failing trials**, expands or
collapses every trial at once, a **per-grader aggregation** table shows
passed/total and pass rate per grader, transcripts are individually
collapsible (auto-expanded for failing trials), task IDs have **copy buttons**,
and per-task / per-trial **token totals** are shown when usage is reported.

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
`state_check`, `transcript`, `milestone`, `llm_rubric` (disabled by default).

The `milestone` grader scores *intermediate* progress, not just the final
outcome. Each milestone is satisfied by a tool call, a phrase in the output, or
a final-state expectation, and ordering can be enforced either across the whole
list (`ordered: true`) or pairwise via `after:` — so
"identity verified before refund processed" is expressed as
`{name: refund_processed, tool: process_refund, after: [identity_verified]}`.

```yaml
- type: milestone
  ordered: true
  milestones:
    - { name: identity_verified, tool: verify_identity }
    - { name: refund_processed, tool: process_refund }
```

**Calibrated scoring.** Hand-written weights are guesses; with human labels or
production outcomes you can fit them. [`agent_eval/calibration.py`](src/agent_eval/calibration.py)
learns grader weights via logistic regression (`learn_weights` /
`learn_from_labeled_trials`), reports a calibrated **pass probability**
(`LogisticModel.pass_probability`) instead of an opaque weighted average, and
supports **multi-objective** scoring across quality, safety, latency, and cost
(`Objectives`, `weighted_objective`, `dominates`, `pareto_front`).

**Scoring:** in `weighted` mode the task score is the weighted average of enabled
graders and passes at `pass_threshold` (and only if no hard-fail grader, e.g. a
forbidden tool, failed). In `binary` mode every enabled grader must pass. Partial
credit is preserved throughout.

### Failure taxonomy

Failing trials are classified into a fixed taxonomy so you can see *how* a suite
fails, not just that it did. [`agent_eval/taxonomy.py`](src/agent_eval/taxonomy.py)
inspects each failing trial's transcript and grader results and assigns one
best-fit category — `timeout`, `recovery_failure`, `looping`,
`policy_violation`, `wrong_tool`, `wrong_args`, `state_mismatch`, or `other` —
identifies the first bad step in the transcript, and aggregates the modes across
a run. The console summary prints the aggregated taxonomy table.

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

### Semantic tool-call matching

Exact argument equality is brittle, so the `tool_calls` grader supports semantic
matching via [`agent_eval/canonicalize.py`](src/agent_eval/canonicalize.py).
Per required/forbidden entry you can add `match` (field -> kind) to canonicalize
arguments before comparing — `amount` (`"$1,000.00"` == `1000`), `date`
(`"01/05/2021"` == `"2021-01-05"`), `id` (`"a-100"` == `"A100"`), `text`/`casing`,
`sorted`/`ordering` (order-insensitive lists), or `alias`/`enum` (with a per-field
`aliases` table). Set `fuzzy: true` (optional `fuzzy_threshold`) for whole-argument
fuzzy text equality. Entries may also be written as AST-style call expressions —
`{call: "process_refund(amount=50)"}` — parsed safely with `ast` (literals only).

```yaml
- type: tool_calls
  required:
    - tool: process_refund
      params: { amount: 50, order_id: A100, when: "2021-01-05" }
      match: { amount: amount, order_id: id, when: date }
```

Read `self.options` for grader-specific config; use `self.result(...)` to build
the `GraderResult` with the configured weight. Set `hard_fail=True` for failures
that should sink the task regardless of weighted score.

### User simulators and dual-control environments

For conversational tasks, [`agent_eval/simulators.py`](src/agent_eval/simulators.py)
provides a deterministic, dependency-free toolkit for interactive evals:

- `UserSimulator` / `ScriptedUserSimulator` — a user that responds *dynamically*
  to the agent's latest message (rule-matched), not from a fixed script.
- `DualControlState` — shared environment state both the agent and the user
  write to, with per-key provenance so a grader can tell who changed what.
- `simulate_dialogue` — drives turns between an agent callable and the user over
  the shared state and returns a `Transcript`.
- `score_dialogue` — scores **reasoning**, **communication**, and
  **coordination** as three independent axes (each the fraction of its signals
  satisfied), so a weakness in one is not masked by strength in another.

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

### Calibrating and de-biasing LLM judges

LLM judges are biased, so [`agent_eval/judges.py`](src/agent_eval/judges.py)
provides the standard mitigations (deterministic, testable against mock
providers):

- **Pairwise judging** (`pairwise_judge`) compares two candidates directly.
- **A/B order swap** runs each comparison in both orders and *detects position
  bias*: when the orders disagree the verdict is downgraded to a tie rather than
  trusted.
- **Multi-judge consensus** (`consensus`, `consensus_pairwise`) takes a majority
  vote across judges with an agreement score.
- **Gold-set agreement** (`agreement_rate`, `cohens_kappa`,
  `calibrate_against_gold`) tracks how well a judge matches human labels
  (chance-corrected).

## 8. How to run in CI

A safe example workflow lives at
[`.github/workflows/eval.yml`](.github/workflows/eval.yml). It installs the
package with uv and runs the suite with the deterministic `echo` agent (no
secrets, no network). It also documents the `http`-agent invocation against the
bundled mock server.

The workflow then **gates the run against a committed baseline**
([`baselines/refund_support.json`](baselines/refund_support.json)): the `compare`
step fails the build when pass rate, pass@k, pass^k, or average score regress
beyond the tolerance, and the JSON + HTML reports are uploaded as artifacts on
every run (including failures). Regenerate the baseline after an intentional
change with `make baseline`, and check the gate locally with `make compare`.

For noisy agents, gate on *statistical significance* rather than any raw drop.
`agent-eval compare --significance` runs a paired bootstrap and a paired t-test
over the per-task pass rates shared by both runs and fails the build only when
both agree the regression is significant at `--alpha` (default 0.05). It also
reports the mean delta with a bootstrap 95% confidence interval, the paired
effect size (Cohen's d), and both p-values. The statistics live in
[`agent_eval/stats.py`](src/agent_eval/stats.py) and use only the standard
library (seeded, deterministic).

### Coverage-aware dataset sampling

To build a representative suite from production logs,
[`agent_eval/sampling.py`](src/agent_eval/sampling.py) samples cases to
*maximize coverage* while prioritizing risk, frequency, and failure history.
`sample_cases` greedily picks the records that add the most new coverage across
intent, task type, risk, tools, policies, failure modes, and edge cases
(ties broken by a risk + frequency + prior-failure priority — deterministic, no
RNG). `coverage_matrix` reports the suite coverage matrix (value -> count per
dimension), and `coverage_report` compares a sample against the full universe
and lists the gaps.

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
