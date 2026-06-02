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
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        legal = ", ".join(obs.legal_actions)
        ctx = f"Match: {request.match.describe()}\n" if request.match else ""
        return (
            f"{_MATCH_RULES}\n\n"
            f"{ctx}"
            f"{obs.rendered}\n\n"
            f"{self._history_block(obs)}"
            f"Choose exactly one legal action: {legal}.\n"
            "Respond with ONLY the action (and an integer amount for bet/raise), "
            "e.g. `call`, `check`, `fold`, `all_in`, `bet 6`, or `raise 12`. "
            "Put the action on the last line if you reason first."
        )
