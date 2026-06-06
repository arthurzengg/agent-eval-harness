from agent_eval.metrics import compute_metrics, pass_at_k, pass_caret_k
from agent_eval.schemas import (
    Pricing,
    Role,
    TaskResult,
    TokenUsage,
    Transcript,
    TranscriptStep,
    Trial,
    TrialResult,
)


def _task(task_id: str, passes: list[bool]) -> TaskResult:
    trials = [
        TrialResult(trial=Trial(task_id=task_id, index=i), passed=p, score=1.0 if p else 0.0)
        for i, p in enumerate(passes)
    ]
    num = len(trials) or 1
    return TaskResult(
        task_id=task_id,
        trials=trials,
        pass_rate=sum(passes) / num,
        avg_score=sum(1.0 if p else 0.0 for p in passes) / num,
    )


def test_pass_at_k_any_passes() -> None:
    assert pass_at_k(_task("a", [False, True, False])) is True
    assert pass_at_k(_task("a", [False, False])) is False


def test_pass_caret_k_all_pass() -> None:
    assert pass_caret_k(_task("a", [True, True])) is True
    assert pass_caret_k(_task("a", [True, False])) is False


def test_compute_metrics_aggregates() -> None:
    tasks = [_task("a", [True, True, True]), _task("b", [True, False, False])]
    m = compute_metrics(tasks, k=3)
    assert m.total_tasks == 2
    assert m.total_trials == 6
    # task a passes pass^k; task b does not -> 0.5
    assert m.pass_caret_k == 0.5
    # both tasks have at least one pass -> pass@k == 1.0
    assert m.pass_at_k == 1.0
    assert m.per_task["a"] == 1.0


def _trial_with_tokens(input_tokens: int, output_tokens: int) -> TrialResult:
    transcript = Transcript(
        steps=[
            TranscriptStep(
                role=Role.assistant,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )
        ]
    )
    return TrialResult(
        trial=Trial(task_id="a", index=0, transcript=transcript), passed=True, score=1.0
    )


def test_token_and_cost_metrics() -> None:
    task = TaskResult(
        task_id="a",
        trials=[_trial_with_tokens(1000, 500), _trial_with_tokens(3000, 1500)],
        pass_rate=1.0,
        avg_score=1.0,
    )
    pricing = Pricing(input_per_1m=10.0, output_per_1m=30.0)  # USD / 1M tokens
    m = compute_metrics([task], k=2, pricing=pricing)

    assert m.avg_input_tokens == 2000.0
    assert m.avg_output_tokens == 1000.0
    assert m.total_tokens == 6000  # (1000+500) + (3000+1500)
    # trial 1: 1000/1e6*10 + 500/1e6*30 = 0.025; trial 2: 0.075 -> total 0.10
    assert round(m.total_cost_usd, 6) == 0.10
    assert round(m.avg_cost_usd, 6) == 0.05


def test_cost_is_zero_without_pricing() -> None:
    task = TaskResult(
        task_id="a",
        trials=[_trial_with_tokens(1000, 500)],
        pass_rate=1.0,
        avg_score=1.0,
    )
    m = compute_metrics([task], k=1)  # no pricing
    assert m.total_tokens == 1500
    assert m.total_cost_usd == 0.0
    assert m.avg_cost_usd == 0.0
