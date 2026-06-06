from agent_eval.metrics import compute_metrics, pass_at_k, pass_caret_k
from agent_eval.schemas import TaskResult, Trial, TrialResult


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
