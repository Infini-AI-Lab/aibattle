"""Game registry: name -> Game class."""

from __future__ import annotations

from .base import Game
from .kuhn import KuhnPoker
from .holdem import HoldemPoker
from .holdem_match import HoldemMatch
from .holdem_table import HoldemTable
from .connect4 import ConnectFour
from .gomoku import Gomoku
from .othello_lite import OthelloLite6x6
from .blackjack import IndependentBlackjack
from .leduc import LeducPoker

_GAMES = {
    KuhnPoker.name: KuhnPoker,
    HoldemPoker.name: HoldemPoker,
    HoldemMatch.name: HoldemMatch,
    HoldemTable.name: HoldemTable,
    ConnectFour.name: ConnectFour,
    Gomoku.name: Gomoku,
    OthelloLite6x6.name: OthelloLite6x6,
    IndependentBlackjack.name: IndependentBlackjack,
    LeducPoker.name: LeducPoker,
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
