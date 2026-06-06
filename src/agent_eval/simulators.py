"""User simulators and dual-control environments for interactive evals.

Many agent tasks are conversations, not one-shot calls: a user reacts to what
the agent says, and both the agent (via tools) and the user can change shared
environment state. This module provides the pieces to evaluate that:

- ``UserSimulator`` -- a user that responds *dynamically* to the agent's last
  message, optionally mutating shared state (so the "user" is a second actor in
  the environment, not a fixed script).
- ``DualControlState`` -- shared state both agent and user write to, with
  per-key provenance so a grader can tell who changed what.
- ``simulate_dialogue`` -- drive turns between an agent callable and a user
  simulator over the shared state, producing a ``Transcript``.
- ``score_dialogue`` -- score reasoning, communication, and coordination as
  three *separate* axes rather than one blended number.

Everything here is deterministic and dependency-free so interactive evals can be
unit-tested without a live model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_eval.schemas import Role, Transcript, TranscriptStep


@dataclass
class DualControlState:
    """Shared environment state writable by both the agent and the user.

    Tracks who last wrote each key (``"agent"`` or ``"user"``) so coordination
    can be graded -- e.g. that the user supplied a value before the agent acted
    on it.
    """

    values: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)
    history: list[tuple[str, str, Any]] = field(default_factory=list)  # (actor, key, value)

    def write(self, actor: str, key: str, value: Any) -> None:
        self.values[key] = value
        self.provenance[key] = actor
        self.history.append((actor, key, value))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def written_by(self, actor: str) -> list[str]:
        """Keys whose most recent write was by ``actor``."""
        return [k for k, who in self.provenance.items() if who == actor]


@dataclass(frozen=True)
class UserTurn:
    """One user message, plus an optional state mutation and a done signal."""

    message: str
    done: bool = False
    writes: dict[str, Any] = field(default_factory=dict)


class UserSimulator(ABC):
    """A user that reacts to the agent's latest message."""

    @abstractmethod
    def opening(self, state: DualControlState) -> UserTurn:
        """The user's first message, before the agent has said anything."""

    @abstractmethod
    def respond(self, agent_message: str, state: DualControlState) -> UserTurn:
        """React to the agent's latest message."""


@dataclass
class Rule:
    """A keyword-triggered user reply (matched case-insensitively)."""

    trigger: str  # substring to look for in the agent's message ("" = always)
    reply: str
    done: bool = False
    writes: dict[str, Any] = field(default_factory=dict)

    def matches(self, message: str) -> bool:
        return self.trigger.lower() in message.lower()


class ScriptedUserSimulator(UserSimulator):
    """A rule-based dynamic user.

    The opening turn is fixed. Each subsequent reply is chosen by the first
    matching rule (so the user *responds to what the agent said*), falling back
    to ``default_reply``. Rules may also write shared state, making the user a
    co-controller of the environment.
    """

    def __init__(
        self,
        opening: str,
        rules: list[Rule],
        *,
        default_reply: str = "I'm not sure, can you clarify?",
        max_replies: int = 20,
        opening_writes: dict[str, Any] | None = None,
    ) -> None:
        self._opening = opening
        self._opening_writes = opening_writes or {}
        self._rules = rules
        self._default_reply = default_reply
        self._max_replies = max_replies
        self._replies = 0

    def opening(self, state: DualControlState) -> UserTurn:
        return UserTurn(self._opening, writes=dict(self._opening_writes))

    def respond(self, agent_message: str, state: DualControlState) -> UserTurn:
        self._replies += 1
        if self._replies >= self._max_replies:
            return UserTurn("Thanks, goodbye.", done=True)
        for rule in self._rules:
            if rule.matches(agent_message):
                return UserTurn(rule.reply, done=rule.done, writes=dict(rule.writes))
        return UserTurn(self._default_reply)


# An agent callable: given the user's latest message and the shared state, it
# returns its reply. It may mutate ``state`` directly (its tool actions).
AgentCallable = Callable[[str, DualControlState], str]


@dataclass
class Dialogue:
    """The outcome of a simulated agent/user conversation."""

    transcript: Transcript
    state: DualControlState
    turns: int
    ended_by_user: bool


def simulate_dialogue(
    agent: AgentCallable,
    user: UserSimulator,
    *,
    state: DualControlState | None = None,
    max_turns: int = 8,
) -> Dialogue:
    """Run an agent/user conversation over shared state.

    A "turn" is one user message followed by one agent reply. The loop stops
    when the user signals ``done`` or ``max_turns`` is reached. User-declared
    ``writes`` are applied to the shared state (attributed to ``"user"``) before
    the agent responds, so the agent sees them.
    """
    state = state or DualControlState()
    steps: list[TranscriptStep] = []

    turn = user.opening(state)
    ended_by_user = False
    turns = 0
    for _ in range(max_turns):
        turns += 1
        for key, value in turn.writes.items():
            state.write("user", key, value)
        steps.append(TranscriptStep(role=Role.user, content=turn.message))

        reply = agent(turn.message, state)
        steps.append(TranscriptStep(role=Role.assistant, content=reply))

        if turn.done:
            ended_by_user = True
            break
        turn = user.respond(reply, state)
        if turn.done:
            # Record the user's closing message, then stop.
            for key, value in turn.writes.items():
                state.write("user", key, value)
            steps.append(TranscriptStep(role=Role.user, content=turn.message))
            ended_by_user = True
            turns += 1
            break

    return Dialogue(Transcript(steps=steps), state, turns, ended_by_user)


@dataclass(frozen=True)
class DialogueScores:
    """Three independent quality axes for an interactive run."""

    reasoning: float
    communication: float
    coordination: float

    @property
    def overall(self) -> float:
        """Unweighted mean of the three axes (callers may reweight)."""
        return (self.reasoning + self.communication + self.coordination) / 3.0


# A signal is a predicate over (transcript, state) contributing to one axis.
Signal = Callable[[Transcript, DualControlState], bool]


@dataclass
class AxisRubric:
    """Named signals whose satisfied fraction is one axis's score."""

    reasoning: list[Signal] = field(default_factory=list)
    communication: list[Signal] = field(default_factory=list)
    coordination: list[Signal] = field(default_factory=list)


def _axis_score(signals: list[Signal], transcript: Transcript, state: DualControlState) -> float:
    if not signals:
        return 0.0
    hits = sum(1 for s in signals if s(transcript, state))
    return hits / len(signals)


def score_dialogue(dialogue: Dialogue, rubric: AxisRubric) -> DialogueScores:
    """Score a dialogue on reasoning, communication, and coordination separately.

    Each axis score is the fraction of its signals satisfied, so the three are
    independent and a weakness in one is not masked by strength in another.
    """
    t, s = dialogue.transcript, dialogue.state
    return DialogueScores(
        reasoning=_axis_score(rubric.reasoning, t, s),
        communication=_axis_score(rubric.communication, t, s),
        coordination=_axis_score(rubric.coordination, t, s),
    )
