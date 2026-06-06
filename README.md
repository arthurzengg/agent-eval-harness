# agent-eval-harness

An extensible, CLI-first evaluation harness for tool-calling / agentic systems,
modeled on the core concepts from Anthropic's "Demystifying evals for AI agents":
**Task, Trial, Grader, Transcript, Outcome, Harness, and Suite**.

> This README is expanded with full usage, architecture, and CI docs in a later
> change. The MVP is built incrementally across several pull requests.

## Status

Work in progress. See `examples/suites/` for the eval-suite format and run:

```bash
make install
make test
make run-example
```
