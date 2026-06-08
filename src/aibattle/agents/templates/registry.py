"""Template registry: game name -> GameTemplate."""

from __future__ import annotations

from .base import GameTemplate
from .kuhn import KuhnTemplate
from .holdem import HoldemTemplate
from .holdem_match import HoldemMatchTemplate
from .holdem_table import HoldemTableTemplate
from .connect4 import Connect4Template
from .gomoku import GomokuTemplate
from .othello_lite import OthelloLiteTemplate
from .blackjack import BlackjackTemplate

_TEMPLATES = {
    "kuhn_poker": KuhnTemplate,
    "holdem": HoldemTemplate,
    "holdem_match": HoldemMatchTemplate,
    "holdem_table": HoldemTableTemplate,
    "connect4": Connect4Template,
    "gomoku": GomokuTemplate,
    "othello_lite_6x6": OthelloLiteTemplate,
    "independent_blackjack": BlackjackTemplate,
}

# The canned coaching line per game: one sentence of process scaffolding (which
# factors to consider before deciding), no strategy prescription. Selected by
# coached=True; the same template class just gets the text injected after rules.
_COACHING = {
    "kuhn_poker": ("Before you act, consider your card's strength and what the "
                   "betting so far suggests about your opponent."),
    "holdem": ("Before you act, weigh your hand strength, your position, the "
               "pot odds, and what the betting so far suggests about your opponent."),
    "connect4": ("Before you move, check whether you can win this turn, whether "
                 "you must block the opponent, and which columns build your position."),
    "gomoku": ("Before you move, check whether you can make five, whether you "
               "must block the opponent's line, and how your own stones connect."),
    "othello_lite_6x6": ("Before you move, consider how many pieces each move "
                         "flips, whether it gives you a corner or a stable edge, "
                         "and how it affects your future mobility."),
    "independent_blackjack": ("Before you act, weigh your current total and "
                              "whether it is soft, the dealer's upcard, and the "
                              "chance of busting against the chance of improving."),
}
# Match/Table reuse the Hold'em coaching line.
_COACHING["holdem_match"] = _COACHING["holdem"]
_COACHING["holdem_table"] = _COACHING["holdem"]


def make_template(game_name: str, *, coached: bool = False) -> GameTemplate:
    if game_name not in _TEMPLATES:
        raise ValueError(
            f"No model template for game {game_name!r}. "
            f"Available: {sorted(_TEMPLATES)}"
        )
    coaching = _COACHING[game_name] if coached else ""
    return _TEMPLATES[game_name](coaching=coaching)
