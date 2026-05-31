"""Template registry: game name -> GameTemplate."""

from __future__ import annotations

from .base import GameTemplate
from .kuhn import KuhnTemplate
from .holdem import HoldemTemplate
from .holdem_match import HoldemMatchTemplate
from .holdem_table import HoldemTableTemplate
from .connect4 import Connect4Template
from .gomoku import GomokuTemplate

_TEMPLATES = {
    "kuhn_poker": KuhnTemplate,
    "holdem": HoldemTemplate,
    "holdem_match": HoldemMatchTemplate,
    "holdem_table": HoldemTableTemplate,
    "connect4": Connect4Template,
    "gomoku": GomokuTemplate,
}


def make_template(game_name: str) -> GameTemplate:
    if game_name not in _TEMPLATES:
        raise ValueError(
            f"No model template for game {game_name!r}. "
            f"Available: {sorted(_TEMPLATES)}"
        )
    return _TEMPLATES[game_name]()
