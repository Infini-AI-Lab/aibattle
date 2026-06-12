"""Independent Blackjack prompt template + action parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate, clean_answer_line


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

# No 'take'/'draw' aliases for hit: they match incidental prose ("take the
# upcard into account", "the dealer could draw"). Unparseable output goes
# through the repair-retry loop instead of being guessed at.
_ALIASES = {
    "double": r"\bdouble(?:\s*down)?\b|\bdd\b",
    "stand": r"\bstand\b|\bstay\b|\bhold\b",
    "hit": r"\bhit\b",
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
        text = raw.lower()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # Fast path: the last line IS exactly one legal action word (modulo
        # decoration) — unambiguous, no scanning.
        if lines:
            bare = clean_answer_line(lines[-1])
            if bare in legal:
                return Move(type=bare)
        # Fuzzy path: the LATEST-mentioned legal action wins, so "don't double,
        # stand" parses as stand (a fixed priority order would pick double).
        for chunk in ([lines[-1]] if lines else []) + [text]:
            best = None  # (position, action)
            for atype in legal:
                for m in re.finditer(_ALIASES[atype], chunk):
                    if best is None or m.start() > best[0]:
                        best = (m.start(), atype)
            if best:
                return Move(type=best[1])
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            "Your previous reply had no valid action. Reply with exactly one of: "
            f"{legal}."
        )
