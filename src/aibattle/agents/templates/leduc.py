"""Leduc Poker prompt template + action parser.

Bet sizes are fixed per street, so the model only needs to choose an action word
(fold/check/call/bet/raise). For bet/raise the template fills in the canonical
total street commitment (raise-to ``amount``) from the observation, so the model
is never asked to compute chip totals itself.
"""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate, clean_answer_line


_RULES = (
    "You are playing Leduc Poker (2 players). Deck: J J Q Q K K, ranked J < Q < "
    "K. Each player antes 1 and gets one private card. There are two betting "
    "rounds; one public card is revealed between them. At showdown, a private "
    "card that matches the public card (a pair) beats any non-pair hand; "
    "otherwise the higher private card wins; equal hands split the pot. Betting "
    "is fixed-limit with at most one raise per round. Actions: 'check' (no bet "
    "to call), 'bet' (open the betting), 'call' (match a bet), 'raise' (raise "
    "once), or 'fold' (give up the pot)."
)

_ALIASES = {
    "raise": r"\braise\b",
    "fold": r"\bfold\b",
    "call": r"\bcall\b",
    "check": r"\bcheck\b",
    "bet": r"\bbet\b",
}


class LeducTemplate(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"Choose one legal action from: {legal}.\n"
            "Respond with ONLY the action word (bet sizes are fixed, so you do "
            "not need a number). Put it on the last line if you reason first."
        )

    def _amount_for(self, atype: str, request: AgentRequest) -> Optional[int]:
        """Canonical total street commitment for a bet/raise (raise-to)."""
        pub = request.observation.public
        size = int(pub.get("bet_size", 0))
        your_commit = int(pub.get("your_commit", 0))
        to_call = int(pub.get("to_call", 0))
        cur_max = your_commit + to_call
        if atype == "bet":
            return your_commit + size
        if atype == "raise":
            return cur_max + size
        return None

    def _move_for(self, atype: str, request: AgentRequest) -> Move:
        if atype in ("bet", "raise"):
            return Move(type=atype, amount=self._amount_for(atype, request))
        return Move(type=atype)

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = list(request.observation.legal_actions)
        text = raw.lower()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # Fast path: the last line IS exactly one legal action word (modulo
        # decoration) — unambiguous, no scanning.
        if lines:
            bare = clean_answer_line(lines[-1])
            if bare in legal:
                return self._move_for(bare, request)
        # Fuzzy path: the LATEST-mentioned legal action wins. Models state their
        # conclusion last, so a fixed priority order would turn "I shouldn't
        # raise, just call" into a raise; position order does not.
        for chunk in ([lines[-1]] if lines else []) + [text]:
            best = None  # (position, action)
            for atype in legal:
                for m in re.finditer(_ALIASES[atype], chunk):
                    if best is None or m.start() > best[0]:
                        best = (m.start(), atype)
            if best:
                return self._move_for(best[1], request)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            "Your previous reply had no valid action. Reply with exactly one of: "
            f"{legal}."
        )
