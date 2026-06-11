"""Independent Blackjack prompt template + action parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate


_RULES = (
    "You are playing Blackjack against a fixed-policy dealer. Aim to beat the "
    "dealer without exceeding 21. Number cards count their face value, J/Q/K "
    "count 10, and an ace counts 11 or 1 (whichever is better). You see your own "
    "hand and the dealer's single upcard. Actions: 'hit' (take a card), 'stand' "
    "(end your turn), or 'double' (only on your first two cards: double the "
    "stake, take exactly one more card, then stand). After you stand, the dealer "
    "reveals its hand and hits until reaching 17 (standing on all 17s). A two-"
    "card 21 is a blackjack. Going over 21 is a bust and an immediate loss. "
    "Scoring: normal wins pay +1, normal losses pay -1, pushes pay 0, double "
    "wins/losses pay +/-2, your blackjack pays +1.5, dealer blackjack pays -1, "
    "and both blackjack is a push."
)

_ALIASES = {
    "double": r"\bdouble\b|\bdouble\s*down\b|\bdd\b",
    "stand": r"\bstand\b|\bstay\b|\bhold\b",
    "hit": r"\bhit\b|\bdraw\b|\btake\b",
}


class BlackjackTemplate(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"Choose one legal action from: {legal}.\n"
            "Respond with ONLY the action word (put it on the last line if you "
            "reason first)."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = list(request.observation.legal_actions)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for chunk in ([lines[-1]] if lines else []) + [raw]:
            low = chunk.lower()
            # Check 'double' before 'hit'/'stand' so "double down" is not eaten.
            for atype in ("double", "stand", "hit"):
                if atype in legal and re.search(_ALIASES[atype], low):
                    return Move(type=atype)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            "Your previous reply had no valid action. Reply with exactly one of: "
            f"{legal}."
        )
