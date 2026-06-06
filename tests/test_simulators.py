"""Tests for user simulators and dual-control environments."""

from __future__ import annotations

from agent_eval.schemas import Role, Transcript
from agent_eval.simulators import (
    AxisRubric,
    Dialogue,
    DualControlState,
    Rule,
    ScriptedUserSimulator,
    score_dialogue,
    simulate_dialogue,
)


def test_dual_control_state_tracks_provenance() -> None:
    state = DualControlState()
    state.write("user", "order_id", "A100")
    state.write("agent", "refund.status", "processed")
    assert state.get("order_id") == "A100"
    assert state.written_by("user") == ["order_id"]
    assert state.written_by("agent") == ["refund.status"]
    assert len(state.history) == 2


def test_scripted_user_responds_dynamically_to_agent() -> None:
    user = ScriptedUserSimulator(
        opening="I want a refund",
        rules=[
            Rule(trigger="order number", reply="It is A100"),
            Rule(trigger="processed", reply="Thank you!", done=True),
        ],
    )
    state = DualControlState()
    assert user.opening(state).message == "I want a refund"
    # The reply depends on what the agent said -- a dynamic, not fixed, user.
    assert user.respond("What is your order number?", state).message == "It is A100"
    closing = user.respond("Your refund was processed.", state)
    assert closing.done


def test_scripted_user_default_reply_when_no_rule_matches() -> None:
    user = ScriptedUserSimulator("hi", rules=[Rule(trigger="foo", reply="bar")])
    assert (
        user.respond("totally unrelated", DualControlState()).message
        == "I'm not sure, can you clarify?"
    )


def test_user_can_modify_shared_state() -> None:
    user = ScriptedUserSimulator(
        opening="I want a refund",
        rules=[Rule(trigger="order number", reply="here", writes={"order_id": "A100"})],
        opening_writes={"intent": "refund"},
    )

    def agent(message: str, state: DualControlState) -> str:
        if state.get("order_id"):
            state.write("agent", "refund.status", "processed")
            return "Your refund was processed."
        return "What is your order number?"

    dialogue = simulate_dialogue(agent, user, max_turns=6)
    # Both actors wrote to the shared environment.
    assert dialogue.state.get("intent") == "refund"
    assert "order_id" in dialogue.state.written_by("user")
    assert dialogue.state.get("refund.status") == "processed"


def test_simulate_dialogue_ends_on_user_done() -> None:
    user = ScriptedUserSimulator(
        opening="hello",
        rules=[Rule(trigger="", reply="bye", done=True)],
    )
    dialogue = simulate_dialogue(lambda m, s: "hi there", user, max_turns=10)
    assert dialogue.ended_by_user
    # opening + agent reply + user close
    roles = [step.role for step in dialogue.transcript.steps]
    assert roles[0] == Role.user
    assert roles[1] == Role.assistant


def test_simulate_dialogue_respects_max_turns() -> None:
    # A user that never finishes; the cap must stop the loop.
    user = ScriptedUserSimulator("start", rules=[Rule(trigger="", reply="keep going")])
    dialogue = simulate_dialogue(lambda m, s: "more?", user, max_turns=3)
    assert dialogue.turns <= 3
    assert not dialogue.ended_by_user


def test_score_dialogue_scores_three_axes_separately() -> None:
    transcript = Transcript(steps=[])
    state = DualControlState()
    state.write("user", "order_id", "A100")
    state.write("agent", "refund.status", "processed")
    dialogue = Dialogue(transcript=transcript, state=state, turns=2, ended_by_user=True)

    rubric = AxisRubric(
        reasoning=[lambda t, s: s.get("refund.status") == "processed"],
        communication=[lambda t, s: True, lambda t, s: False],  # 1 of 2
        coordination=[lambda t, s: "order_id" in s.written_by("user")],
    )
    scores = score_dialogue(dialogue, rubric)
    assert scores.reasoning == 1.0
    assert scores.communication == 0.5
    assert scores.coordination == 1.0
    assert abs(scores.overall - (1.0 + 0.5 + 1.0) / 3) < 1e-9


def test_empty_axis_scores_zero() -> None:
    dialogue = Dialogue(Transcript(steps=[]), DualControlState(), 0, False)
    scores = score_dialogue(dialogue, AxisRubric())
    assert scores.reasoning == 0.0
    assert scores.overall == 0.0
