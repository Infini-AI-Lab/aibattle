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
            "(column letter A-I, row number 1-9). Think privately before you answer."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = set(request.observation.legal_actions)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for chunk in ([lines[-1]] if lines else []) + [raw]:
            for m in _COORD.finditer(chunk):
                coord = f"{m.group(1).upper()}{m.group(2)}"
                if coord in legal:
                    return Move(type=coord)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        return (
            "Your previous reply was not a valid empty cell. Reply with one "
            "coordinate like E5 that is currently empty."
        )
