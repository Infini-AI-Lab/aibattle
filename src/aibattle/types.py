"""Core cross-layer data types.

All data that crosses a layer boundary (game -> runner -> agent -> log) is a
plain, serializable dataclass. No layer ever passes a live game-state object
across the agent boundary; agents only ever see an ``Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

PlayerId = str  # e.g. "player_0", "player_1"
Action = str    # an action *type* token, e.g. "check", "bet", "call", "fold"

# Sentinel returned by an agent that could not produce any action (e.g. a model
# that never emitted a parseable token). The runner's invalid-action policy
# turns this into a concrete legal action or a forfeit.
INVALID: Action = "__invalid__"


@dataclass(frozen=True)
class Move:
    """A chosen action: an action *type* plus an optional integer amount.

    Discrete games (Kuhn) use ``amount=None``. Numeric games (Hold'em) carry an
    integer ``amount`` for bet/raise, interpreted as the player's TOTAL
    committed chips for the current betting street.
    """

    type: Action
    amount: Optional[int] = None

    def label(self) -> str:
        return f"{self.type} {self.amount}" if self.amount is not None else self.type


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
class MatchContext:
    """Match-level context spanning episodes (a single hand has none of this).

    Lets agents see which hand they're on and their running chip standing across
    the match so far. ``standing`` maps agent name -> cumulative chip delta
    *before* this hand.
    """

    episode: int            # 0-based index of the current hand
    total_episodes: int     # total hands in the match
    you: str                # this agent's name (to find its own standing)
    standing: dict          # {agent_name: cumulative chip delta so far}

    def describe(self) -> str:
        head = f"Hand {self.episode + 1} of {self.total_episodes}."
        # Standings are omitted under parallel execution (passed empty), since
        # which episodes have completed is nondeterministic and would otherwise
        # leak run-to-run variation into the prompt.
        if not self.standing:
            return head
        mine = self.standing.get(self.you, 0)
        others = ", ".join(f"{n}: {v:+g}" for n, v in self.standing.items()
                           if n != self.you)
        opp = f"  Opponent standing: {others}." if others else ""
        return (f"{head}  Your overall standing so far: {mine:+g} chips.{opp}")

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "you": self.you,
            "standing": self.standing,
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
    match: Optional["MatchContext"] = None   # cross-episode context


@dataclass
class AgentResponse:
    """The standardized response an agent returns. Only ``action`` is required.

    ``action`` is the action TYPE (e.g. "raise"); ``amount`` is the integer
    total street commitment, required for bet/raise in numeric games and left
    ``None`` otherwise.
    """

    action: Action
    amount: Optional[int] = None        # required for bet/raise in numeric games
    message: Optional[str] = None       # optional natural-language rationale
    raw_output: Optional[str] = None    # optional unparsed model output
    metadata: dict = field(default_factory=dict)  # latency, tokens, retries...

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "amount": self.amount,
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
    selected_amount: Optional[int] = None
