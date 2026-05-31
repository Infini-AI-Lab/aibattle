"""Heads-Up Texas Hold'em Lite prompt template + tolerant action/amount parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate

_RULES = (
    "You are playing Heads-Up Texas Hold'em Lite (no-limit style, one hand). "
    "Each player starts with 50 chips; small blind 1, big blind 2. You are dealt "
    "two private hole cards; five community cards are revealed across preflop, "
    "flop, turn, and river. Make the best five-card hand at showdown.\n"
    "Actions:\n"
    "  fold  - give up the hand\n"
    "  check - pass (only when there is no bet to call)\n"
    "  call  - match the current bet\n"
    "  bet N - bet when no bet is facing you; N = your TOTAL chips committed this street\n"
    "  raise N - raise a facing bet; N = your TOTAL chips committed this street (raise-TO)\n"
    "  all_in - commit your entire remaining stack\n"
    "N must be an integer within the stated legal range."
)

# action type -> regex; longer/odd spellings handled (all-in/allin/shove).
_ALIASES = {
    "all_in": r"\ball[\s_-]?in\b|\bshove\b|\bjam\b",
    "raise": r"\braise\b",
    "bet": r"\bbet\b",
    "call": r"\bcall\b",
    "check": r"\bcheck\b",
    "fold": r"\bfold\b",
}


class HoldemTemplate(GameTemplate):
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        legal = ", ".join(obs.legal_actions)
        ctx = f"Match: {request.match.describe()}\n" if request.match else ""
        return (
            f"{_RULES}\n\n"
            f"{ctx}"
            f"{obs.rendered}\n\n"
            f"Choose exactly one legal action: {legal}.\n"
            "Respond with ONLY the action (and an integer amount for bet/raise), "
            "e.g. `call`, `check`, `fold`, `all_in`, `bet 6`, or `raise 12`. "
            "Put the action on the last line if you reason first."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = request.observation.legal_actions
        text = raw.lower()
        # Prefer the last non-empty line (the conclusion) if it contains an action.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for candidate in ([lines[-1]] if lines else []) + [text]:
            for atype in ("all_in", "raise", "bet", "call", "check", "fold"):
                if atype not in legal:
                    continue
                if re.search(_ALIASES[atype], candidate):
                    if atype in ("bet", "raise"):
                        m = re.search(r"(\d+)", candidate)
                        if not m:
                            return None  # amount required but absent
                        return Move(type=atype, amount=int(m.group(1)))
                    return Move(type=atype)
        return None

    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"{self.render_prompt(request)}\n\n"
            f"Your previous reply did not contain a valid action. "
            f"Reply with exactly one of: {legal} (with an integer amount for bet/raise)."
        )
