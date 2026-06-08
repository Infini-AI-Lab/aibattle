"""Gomoku-Lite prompt template + coordinate parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate

# Letter and digit must be adjacent (optional single dash). Tolerating spaces
# would misread prose like "a 3-in-a-row" or "I 5" as a move (A and I are
# English words); requiring adjacency closes that without rejecting E5 / e-5.
_COORD = re.compile(r"\b([A-I])-?([1-9])\b", re.IGNORECASE)


_RULES = (
    "You are playing Gomoku-Lite (9x9). Place a stone on any empty cell; connect "
    "five in a row (horizontal, vertical, or diagonal) to win. Columns are A-I, "
    "rows 1-9; center is E5."
)


class GomokuTemplate(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        return (
            "Respond with ONLY a coordinate for an empty cell, e.g. E5 "
            "(column letter A-I, row number 1-9). Put it on the last line if you "
            "reason first."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = set(request.observation.legal_actions)
        # The answer is the conclusion: scan lines bottom-up and return the first
        # line that names exactly ONE distinct legal coordinate. A line listing
        # several different cells is ambiguous -> skip it -> None -> repair, then
        # the runner's random fallback. (The rendered board uses space-separated
        # column letters and X/O/. cells, so it yields no "E5"-style matches.)
        for ln in reversed([l for l in raw.splitlines() if l.strip()]):
            coords = {
                f"{m.group(1).upper()}{m.group(2)}"
                for m in _COORD.finditer(ln)
            } & legal
            if len(coords) == 1:
                return Move(type=next(iter(coords)))
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        return (
            "Your previous reply was not a valid empty cell. Reply with one "
            "coordinate like E5 that is currently empty."
        )
