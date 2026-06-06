"""Prompt template for Heads-Up Hold'em Match Mode.

Reuses the single-hand Hold'em parser (identical action grammar) and only swaps
in match-aware rules text. The per-hand match context (hand number, carried
chip standing) is injected by the game's observation, so it appears in the
rendered prompt automatically.
"""

from __future__ import annotations

from ...types import AgentRequest
from .holdem import HoldemTemplate, _RULES

_MATCH_RULES = _RULES + (
    "\nThis is a MATCH, not a single hand: you play up to a fixed number of "
    "hands, your stack carries over between hands, and you win the match by "
    "finishing with more chips than your opponent (or if they bust). Manage risk "
    "across the whole match — protect a lead, take measured risk when behind."
)


class HoldemMatchTemplate(HoldemTemplate):
    # No MatchContext "Hand X of N" line: in Match mode one episode wraps the
    # whole match, so MatchContext.episode is the MATCH index (always "1 of 1"
    # with EPISODES=1) and would contradict the true within-match hand counter
    # the engine already emits in obs.rendered.
    _show_match_ctx = False

    def rules(self, request: AgentRequest) -> str:
        return _MATCH_RULES
