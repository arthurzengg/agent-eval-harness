# agent-eval-harness

一个可扩展、以命令行为先的智能体（agentic / 工具调用）评测框架。它将 Anthropic
《Demystifying evals for AI agents》一文中的核心概念落地为一个小而完整、带类型、可
测试的框架：运行本地 YAML 评测套件，并生成 JSON 与 HTML 报告。

## 1. 这是什么

`agent-eval-harness` 针对会调用工具的智能体运行自动化评测。它刻意保持精简，但具备
生产形态：清晰的数据模型、默认使用确定性的代码评分器、可选的 LLM 评审插件，以及生成
可检查产物的 CLI。项目不含 Web 应用——对外接口就是 `agent-eval` 命令及其生成的静态
报告。

## 2. 为什么评测需要轨迹、结果、评分器和多次试验

智能体行为是非确定性且多步的，仅对最终答案做字符串匹配往往不够：

- **轨迹（Transcript）** 记录完整过程（assistant 消息、工具调用、工具结果、耗时、
  token 用量、错误），便于评估“过程”而非仅最终输出，并在评分器出错时进行复查。
- **结果（Outcome）** 捕获最终可观测状态（例如“退款已处理”），这通常才是真正关心的。
- **评分器（Grader）** 对轨迹和结果打分。不同任务需要不同检查：必需/禁止的工具、
  参数合法性、最终状态断言、预算限制，或（可选的）LLM 评分。工具调用的“顺序匹配”被
  支持，但绝不是唯一的通过方式。
- **多次试验（Trials）** 暴露不稳定性。框架汇总 `pass@k`（k 次中至少一次通过）与
  `pass^k`（k 次全部通过），二者刻画的可靠性截然不同。

## 3. 与 Anthropic 概念的对应

| 概念 | 所在位置 |
| --- | --- |
| Task 任务 | `schemas.Task` |
| Trial 试验 | `schemas.Trial` |
| Transcript 轨迹 | `schemas.Transcript` 等 |
| Outcome 结果 | `schemas.Outcome` |
| Grader 评分器 | `graders/` |
| Harness 评测引擎 | `runner.Runner` |
| Suite 套件 | `schemas.EvalSuite` + `suite_loader` |
| Environment 环境 | `environments/` |
| Metrics 指标 | `metrics.py` |

## 4. 快速开始

需要 Python 3.11+ 与 [uv](https://docs.astral.sh/uv/)。

```bash
make install        # 准备 Python 3.11 并安装
make test           # 运行测试
make lint           # ruff 检查与格式校验
make typecheck      # mypy 严格类型检查
make run-example    # 用 echo 智能体运行 refund_support 套件
```

CLI 用法：

```bash
agent-eval validate examples/suites/refund_support.yaml

agent-eval run \
  --suite examples/suites/refund_support.yaml \
  --agent echo --trials 3 \
  --concurrency 8 \
  --output reports/refund_support

agent-eval report \
  --results reports/refund_support/results.json \
  --output reports/refund_support/index.html
```

产物（写入 `--output` 目录）：`results.json`、`summary.json`、
`transcripts/<task_id>/trial_<n>.json`、`index.html`。

## 5. 套件示例

参见 [`examples/suites/refund_support.yaml`](examples/suites/refund_support.yaml)。
可用评分器：`exact_match`、`regex`、`tool_calls`、`argument_schema`、`state_check`、
`transcript`、`llm_rubric`（默认关闭）。`weighted` 模式按权重加权求平均并以阈值判定
通过（且不能有 hard-fail 的评分器失败）；`binary` 模式要求所有启用的评分器全部通过。

## 6. 如何新增评分器

在 `src/agent_eval/graders/` 新建实现 `BaseGrader` 的类（设置 `type`、实现 `grade`），
然后在 `graders/__init__.py` 的 `_GRADERS` 映射中注册即可在套件中以 `type` 引用。

## 7. 如何新增智能体适配器

在 `src/agent_eval/adapters/` 实现 `AgentAdapter` 协议（`name` + `async run`），
并在 `adapters/__init__.py` 用 `@adapter_registry.register("名称")` 注册，
通过 `--agent 名称` 选用。

## 8. 如何在 CI 中运行

示例工作流见 [`.github/workflows/eval.yml`](.github/workflows/eval.yml)：用 uv 安装，
以确定性的 `echo` 智能体运行（无需密钥、无需网络），并演示对内置 mock 服务的 `http`
适配器调用方式。

## 9. 限制与后续

- LLM 评审仅有接口：目前仅实现 `mock` provider，OpenAI/Anthropic 为占位需自行实现。
- 仅支持 YAML 套件，尚无 JSONL。
- 任务串行执行，暂无并行/分布式后端。
- 仅文件系统存储，暂无 SQLite/云存储。
- 无 Web UI，报告为静态 HTML（设计如此）。
