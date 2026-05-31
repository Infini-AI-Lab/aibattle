"""Template registry: game name -> GameTemplate."""

from __future__ import annotations

from .base import GameTemplate
from .kuhn import KuhnTemplate
from .holdem import HoldemTemplate

_TEMPLATES = {
    "kuhn_poker": KuhnTemplate,
    "holdem": HoldemTemplate,
}


def make_template(game_name: str) -> GameTemplate:
    if game_name not in _TEMPLATES:
        raise ValueError(
            f"No model template for game {game_name!r}. "
            f"Available: {sorted(_TEMPLATES)}"
        )
    return _TEMPLATES[game_name]()
