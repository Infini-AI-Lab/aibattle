"""Game layer abstraction.

A ``Game`` is a pure, immutable-state module. ``step`` returns a new state and
never mutates its argument. State is opaque to everyone except the game itself;
agents only ever receive an ``Observation``.

Key invariant: the runner is the only caller of ``step``, and it only calls it
with an action that is in ``legal_actions`` for the current player.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from ..types import Action, Move, Observation, PlayerId

State = Any  # game-defined; opaque outside the game module


class Game(ABC):
    name: str
    version: str
    players: list  # fixed roster of PlayerId for v0

    @abstractmethod
    def initial_state(self, rng: random.Random) -> State:
        ...

    @abstractmethod
    def current_player(self, state: State) -> PlayerId:
        ...

    @abstractmethod
    def observation(self, state: State, player: PlayerId) -> Observation:
        """Expose ONLY what ``player`` is allowed to see (enforces hidden info)."""

    @abstractmethod
    def legal_actions(self, state: State, player: PlayerId) -> list:
        """Return the legal action *types* (strings) for the current player."""

    def validate_action(self, state: State, player: PlayerId, move: Move):
        """Validate a chosen Move. Returns (ok: bool, reason: Optional[str]).

        Default: discrete games — the type must be legal and carry no amount.
        Numeric games (Hold'em) override to validate bet/raise amounts.
        """
        if move.type not in self.legal_actions(state, player):
            return False, "illegal_action_type"
        if move.amount is not None:
            return False, "unexpected_amount"
        return True, None

    def fallback_action(self, state: State, player: PlayerId, legal: list) -> Move:
        """Pick a safe legal move when an agent's action is invalid.

        Default (discrete/poker games): prefer check, then fold, then call, then
        the first legal action type. Board games override for a center bias.
        """
        for t in ("check", "fold", "call"):
            if t in legal:
                return Move(type=t)
        return Move(type=legal[0]) if legal else Move(type="__invalid__")

    @abstractmethod
    def step(self, state: State, move: Move) -> State:
        """Advance the game. Precondition: move is valid for the current player."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        ...

    @abstractmethod
    def returns(self, state: State) -> dict:
        """Terminal payoffs in chips, keyed by PlayerId. Zero-sum for Kuhn."""

    def episode_metadata(self, state: State) -> dict:
        """Extra game-specific fields to stamp into the episode summary/log
        (e.g. end reason, big blind). Default: none."""
        return {}

    @abstractmethod
    def render(self, state: State, *, perspective: Optional[PlayerId] = None) -> str:
        ...
