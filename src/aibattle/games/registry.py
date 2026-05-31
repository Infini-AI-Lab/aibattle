"""Game registry: name -> Game class."""

from __future__ import annotations

from .base import Game
from .kuhn import KuhnPoker
from .holdem import HoldemPoker

_GAMES = {
    KuhnPoker.name: KuhnPoker,
    HoldemPoker.name: HoldemPoker,
}


def make_game(name: str, params: dict | None = None) -> Game:
    if name not in _GAMES:
        raise ValueError(
            f"Unknown game {name!r}. Available: {sorted(_GAMES)}"
        )
    params = params or {}
    return _GAMES[name](**params)


def available_games() -> list:
    return sorted(_GAMES)
