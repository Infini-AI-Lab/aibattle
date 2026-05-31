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

from ..types import Action, Observation, PlayerId

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
        ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        """Advance the game. Precondition: action is legal for current player."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        ...

    @abstractmethod
    def returns(self, state: State) -> dict:
        """Terminal payoffs in chips, keyed by PlayerId. Zero-sum for Kuhn."""

    @abstractmethod
    def render(self, state: State, *, perspective: Optional[PlayerId] = None) -> str:
        ...
