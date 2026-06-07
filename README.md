# agent-eval-harness

An extensible, CLI-first evaluation harness for tool-calling and agentic
systems. It runs local YAML or JSONL eval suites, records full transcripts and
outcomes, grades each trial, aggregates reliability metrics, and writes
inspectable JSON / HTML / terminal reports.

Intentionally small but production-shaped: typed schemas, deterministic graders
by default, optional LLM-judge hooks, statistical regression checks, and
CI-friendly artifacts.

## Features

- **Typed suites** in YAML/JSON or JSONL, with per-task graders, trial counts,
  timeouts, weighted scoring, and pricing.
- **Agent adapters**: deterministic `echo`, a hardened HTTP adapter, and
  entry-point plugins for external adapters.
- **Transcript-first records** with per-trial isolation: every step, tool call,
  tool result, token count, timing, error, and final state.
- **Nine graders**: exact match, regex, tool calls, argument schema, state
  check, transcript, milestone, safety, and LLM rubric — with semantic
  tool-call matching for dates, amounts, IDs, aliases, and fuzzy text.
- **Reliability metrics**: `pass@k` / `pass^k`, full `pass@1..N` curves, and
  flaky-task detection.
- **Statistical comparison**: tolerance gates plus paired bootstrap, paired
  t-test, confidence intervals, and effect size.
- **Failure taxonomy**, calibrated multi-objective scoring, judge calibration,
  user simulators with dual-control state, and coverage-aware sampling from
  production logs.
- **Reports**: console summary, JSON artifacts, self-contained HTML report, and
  an optional Textual TUI.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
make install
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

Add `--significance --alpha 0.05` to `compare` for statistical gating of noisy
agents.

## Suite Format

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
    expected_outcome:
      refund: { status: processed, order_id: A100 }
    graders:
      - type: tool_calls
        weight: 0.25
        required:
          - tool: verify_identity
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

JSONL suites use one task per line after a header line:

```jsonl
{"suite": {"id": "refund_support_jsonl", "name": "Refund support"}, "defaults": {"trials": 2}}
{"id": "refund_allowed", "input": {"user_message": "..."}, "graders": [{"type": "state_check", "expect": {"refund.status": "processed"}}]}
```

## Outputs

Each run writes `results.json`, `summary.json`,
`transcripts/<task_id>/trial_<n>.json`, and `index.html`. The HTML report
covers suite metrics, reliability curves, flaky tasks, per-grader aggregation,
transcripts, tool-call timelines, token usage, and cost totals.

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

It validates the response schema, retries transient failures with exponential
backoff, and classifies HTTP failures.

## Advanced Evaluation

- **Milestones** — intermediate progress checks ([`graders/milestone.py`](src/agent_eval/graders/milestone.py))
- **Safety** — prompt injection, tool abuse, refusal behavior ([`graders/safety.py`](src/agent_eval/graders/safety.py))
- **Semantic tool matching** — canonicalize brittle arguments ([`canonicalize.py`](src/agent_eval/canonicalize.py))
- **Failure taxonomy** — failure modes and first bad steps ([`taxonomy.py`](src/agent_eval/taxonomy.py))
- **Judge calibration** — pairwise judging, order swaps, Cohen's kappa ([`judges.py`](src/agent_eval/judges.py), [`calibration.py`](src/agent_eval/calibration.py))
- **Reliability and stats** — ([`reliability.py`](src/agent_eval/reliability.py), [`stats.py`](src/agent_eval/stats.py))
- **User simulation and sampling** — ([`simulators.py`](src/agent_eval/simulators.py), [`sampling.py`](src/agent_eval/sampling.py))

## Plugins

External packages can register graders, adapters, or reporters via entry
points — the entry-point name becomes the suite `type` or `--agent` name:

```toml
[project.entry-points."agent_eval.graders"]
my_grader = "my_pkg.graders:MyGraderFactory"

[project.entry-points."agent_eval.adapters"]
my_agent = "my_pkg.adapters:make_my_agent"
```

Built-ins win on name clashes; failed plugin imports warn instead of aborting.

## Development

```bash
make test       # pytest + coverage gate
make lint       # ruff check + ruff format --check
make typecheck  # mypy --strict
make baseline   # regenerate committed deterministic baseline
make compare    # compare latest example run to baseline
```

CI runs lint, strict typing, tests with coverage, the example eval, a baseline
regression gate, and uploads JSON/HTML artifacts.

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
