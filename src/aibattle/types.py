"""Core cross-layer data types.

All data that crosses a layer boundary (game -> runner -> agent -> log) is a
plain, serializable dataclass. No layer ever passes a live game-state object
across the agent boundary; agents only ever see an ``Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

PlayerId = str  # e.g. "player_0", "player_1"
Action = str    # game-defined token, e.g. "check", "bet", "call", "fold"

# Sentinel returned by an agent that could not produce any action (e.g. a model
# that never emitted a parseable token). The runner's invalid-action policy
# turns this into a concrete legal action or a forfeit.
INVALID: Action = "__invalid__"


@dataclass(frozen=True)
class Observation:
    """What a single player is allowed to see at a decision point."""

    player: PlayerId
    private: dict          # info only this player sees, e.g. {"card": "K"}
    public: dict           # info all players see, e.g. {"pot": 2}
    history: list          # public action log: [{"player","action"}, ...]
    legal_actions: list    # list[Action] valid right now
    rendered: str          # human/agent-facing text rendering of the above

    def to_dict(self) -> dict:
        return {
            "player": self.player,
            "private": self.private,
            "public": self.public,
            "history": self.history,
            "legal_actions": list(self.legal_actions),
            "rendered": self.rendered,
        }


@dataclass(frozen=True)
class AgentRequest:
    """The standardized request the runner hands to an agent."""

    game: str              # "kuhn_poker"
    game_version: str
    player: PlayerId
    observation: Observation
    instructions: str      # output-format expectations for model agents
    step_index: int
    # Deterministic per-decision seed. Stochastic builtin agents seed their RNG
    # from this so behavior is reproducible regardless of (parallel) execution
    # order. Model/external agents ignore it.
    decision_seed: Optional[int] = None


@dataclass
class AgentResponse:
    """The standardized response an agent returns. Only ``action`` is required."""

    action: Action
    message: Optional[str] = None       # optional natural-language rationale
    raw_output: Optional[str] = None    # optional unparsed model output
    metadata: dict = field(default_factory=dict)  # latency, tokens, retries...

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "message": self.message,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }


@dataclass
class InvalidInfo:
    """Records what happened when an agent returned an illegal/missing action."""

    invalid: bool                       # was the agent's raw action illegal?
    reason: Optional[str] = None        # "illegal_action" | "no_action"
    requested: Optional[Action] = None  # what the agent asked for
    resolution: Optional[str] = None    # "fallback" | "forfeit"

    def to_dict(self) -> dict:
        return {
            "invalid": self.invalid,
            "reason": self.reason,
            "requested": self.requested,
            "resolution": self.resolution,
        }


@dataclass
class StepRecord:
    """One decision point in an episode."""

    step_index: int
    player: PlayerId
    observation: Observation
    response: AgentResponse
    selected_action: Action
    invalid_info: InvalidInfo
