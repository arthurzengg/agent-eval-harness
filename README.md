# agent-eval-harness

`agent-eval-harness` is an extensible, CLI-first evaluation harness for
tool-calling and agentic systems. It runs local YAML or JSONL eval suites,
records full transcripts and outcomes, grades each trial, aggregates reliability
metrics, and writes inspectable JSON / HTML / terminal reports.

The project is intentionally small but production-shaped: typed schemas,
deterministic graders by default, optional LLM-judge hooks, HTTP adapters,
statistical regression checks, trace failure taxonomy, user simulators, safety
evals, and CI-friendly artifacts.

## Features

- **Typed eval suites** with tasks, inputs, references, expected outcomes,
  graders, trial counts, timeouts, scoring, and pricing.
- **Multiple input formats**: YAML/JSON suite files and JSONL/NDJSON datasets
  with one task per line.
- **Agent adapters**: deterministic `echo`, hardened HTTP adapter, and
  entry-point plugin registration for external adapters.
- **Trial isolation** through per-trial local environments.
- **Transcript-first records**: user / assistant / tool / environment steps,
  tool calls, tool results, token usage, timing, errors, and final state.
- **Graders**: exact match, regex, tool calls, argument schema, state check,
  transcript, milestone, safety, and LLM rubric.
- **Semantic tool-call matching** for dates, amounts, IDs, casing, aliases,
  unordered lists, fuzzy text, and AST-style call expressions.
- **Reliability metrics**: `pass@k`, `pass^k`, mixed-k labels, full
  `pass@1..N` / `pass^1..N` curves, and flaky-task detection.
- **Statistical comparison**: fixed-tolerance gates plus paired bootstrap,
  paired t-test, confidence intervals, p-values, and effect size.
- **Failure taxonomy** for timeout, recovery failure, looping, policy
  violation, wrong tool, wrong args, state mismatch, and other failures.
- **Calibrated scoring** via learned logistic weights and multi-objective
  quality / safety / latency / cost scoring.
- **User simulators and dual-control state** for interactive conversations where
  both user and agent can affect shared state.
- **Coverage-aware sampling** from production logs by intent, task type, risk,
  tools, policies, failure modes, and edge cases.
- **Reports**: console summary, JSON artifacts, self-contained HTML report, and
  optional Textual TUI.
- **CI support** with linting, strict typing, coverage threshold, example eval,
  baseline regression gate, and uploaded reports.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
make test
make lint
make typecheck
make run-example
```

Run a suite:

```bash
agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent echo \
  --trials 3 \
  --concurrency 8 \
  --output reports/refund_support
```

Validate, report, compare, and browse results:

```bash
agent-eval validate examples/suites/refund_support.yaml
agent-eval report --results reports/refund_support/results.json --output reports/refund_support/index.html
agent-eval compare --baseline baselines/refund_support.json --current reports/refund_support/results.json
agent-eval ui --results reports/refund_support/results.json
```

Use statistical gating for noisy agents:

```bash
agent-eval compare \
  --baseline baselines/refund_support.json \
  --current reports/refund_support/results.json \
  --significance \
  --alpha 0.05
```

## Suite Format

Minimal YAML suite:

```yaml
suite:
  id: refund_support
  name: Refund Support Agent Eval

defaults:
  trials: 3
  timeout_seconds: 60
  scoring: { mode: weighted, pass_threshold: 0.8 }

tasks:
  - id: refund_allowed_under_30_days
    input:
      user_message: "I want a refund for order A100. I bought it 10 days ago."
    reference:
      summary: "Verify identity, fetch policy, process refund, and explain."
    expected_outcome:
      refund: { status: processed, order_id: A100 }
    graders:
      - type: tool_calls
        weight: 0.25
        required:
          - tool: verify_identity
          - tool: fetch_refund_policy
          - tool: process_refund
        forbidden:
          - tool: escalate_to_manager
      - type: state_check
        weight: 0.35
        expect:
          refund.status: processed
          refund.order_id: A100
      - type: transcript
        weight: 0.15
        max_turns: 10
```

JSONL suites are also supported:

```jsonl
{"suite": {"id": "refund_support_jsonl", "name": "Refund support"}, "defaults": {"trials": 2}}
{"id": "refund_allowed", "input": {"user_message": "..."}, "graders": [{"type": "state_check", "expect": {"refund.status": "processed"}}]}
{"id": "refund_denied", "input": {"user_message": "..."}, "graders": [{"type": "state_check", "expect": {"refund.status": "denied"}}]}
```

## Outputs

Each run writes:

```text
results.json
summary.json
transcripts/<task_id>/trial_<n>.json
index.html
```

The HTML report includes suite metrics, reliability curves, flaky tasks,
per-grader aggregation, failing-trial filters, transcript expand/collapse,
tool-call timelines, outcome JSON, token usage, and cost totals when available.

## HTTP Adapter

The `http` adapter sends each task to a remote `/run` endpoint:

```bash
python examples/agents/mock_refund_agent.py
agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent http \
  --agent-url http://127.0.0.1:8080/run \
  --trials 3 \
  --output reports/refund_support_http
```

The adapter validates response schema, retries transient failures with
exponential backoff, sends `X-Request-ID`, records request metadata, captures
agent version metadata, and classifies HTTP failures.

## Advanced Evaluation

- **Milestones** check intermediate progress such as "identity verified before
  refund processed."
- **Safety** checks adversarial prompts, prompt injection, tool abuse,
  unauthorized actions, data exfiltration, refusal behavior, and safe
  completion.
- **Semantic tool matching** canonicalizes brittle arguments before comparing
  required or forbidden tool calls.
- **Failure taxonomy** aggregates failure modes and first bad steps across runs.
- **Judge calibration** supports pairwise judging, A/B order swaps, consensus,
  agreement rate, and Cohen's kappa against gold labels.
- **Coverage sampling** builds representative eval sets from logs by maximizing
  coverage while prioritizing risk, frequency, and prior failures.

See the focused modules under [`src/agent_eval`](src/agent_eval):

- [`reliability.py`](src/agent_eval/reliability.py)
- [`stats.py`](src/agent_eval/stats.py)
- [`taxonomy.py`](src/agent_eval/taxonomy.py)
- [`canonicalize.py`](src/agent_eval/canonicalize.py)
- [`calibration.py`](src/agent_eval/calibration.py)
- [`judges.py`](src/agent_eval/judges.py)
- [`simulators.py`](src/agent_eval/simulators.py)
- [`sampling.py`](src/agent_eval/sampling.py)
- [`graders/milestone.py`](src/agent_eval/graders/milestone.py)
- [`graders/safety.py`](src/agent_eval/graders/safety.py)

## Plugins

External packages can register graders, adapters, or reporters without forking
this repo:

```toml
[project.entry-points."agent_eval.graders"]
my_grader = "my_pkg.graders:MyGraderFactory"

[project.entry-points."agent_eval.adapters"]
my_agent = "my_pkg.adapters:make_my_agent"
```

The entry-point name becomes the suite `type` or `--agent` name. Built-ins win
on name clashes, and failed plugin imports warn instead of aborting a run.

## Development

```bash
make test       # pytest + coverage gate
make coverage   # HTML coverage report
make lint       # ruff check + ruff format --check
make typecheck  # mypy --strict
make baseline   # regenerate committed deterministic baseline
make compare    # compare latest example run to baseline
```

The GitHub Actions workflow runs lint, strict typing, tests with coverage,
example eval, baseline regression comparison, and uploads JSON/HTML artifacts.

## References / Related Work

- [Anthropic: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Anthropic: A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals)
- [Evaluating Large Language Models Trained on Code / HumanEval](https://arxiv.org/abs/2107.03374)
- [tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- [tau2-bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982)
- [ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark](https://arxiv.org/abs/2408.04682)
- [TRAIL: Trace Reasoning and Agentic Issue Localization](https://arxiv.org/abs/2505.08638)
- [Berkeley Function Calling Leaderboard (BFCL)](https://proceedings.mlr.press/v267/patil25a.html)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models](https://arxiv.org/abs/2405.01535)
- [AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents](https://arxiv.org/abs/2410.09024)
- [AlphaEval: Evaluating Agents in Production](https://arxiv.org/abs/2604.12162)
- [Agentic Benchmark Checklist: Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)
- [OpenAI Evals](https://github.com/openai/evals)
- [Inspect AI](https://inspect.aisi.org.uk/)

## Next Steps

- Add real OpenAI / Anthropic / local-model providers for `llm_rubric`.
- Add SQLite or cloud storage for queryable run history.
- Add a distributed run backend for large suites.
- Track dataset lineage, prompt versions, grader versions, and agent versions.
- Add contamination, leakage, and duplicate-task checks.
- Add power analysis for sample-size planning.
- Add adversarial environment fuzzing and property-based task generation.
