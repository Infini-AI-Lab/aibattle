"""Connect Four prompt template + column parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate


_RULES = (
    "You are playing Connect Four (6 rows x 7 columns). Drop a piece into a "
    "column; it falls to the lowest empty cell. Connect four (horizontal, "
    "vertical, or diagonal) to win."
)


class Connect4Template(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"Choose one legal column from: {legal}.\n"
            "Respond with ONLY the column number (put it on the last line if you "
            "reason first)."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = set(request.observation.legal_actions)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        # Prefer the last line (the conclusion), then the whole text.
        for chunk in ([lines[-1]] if lines else []) + [raw]:
            for tok in re.findall(r"\d+", chunk):
                if tok in legal:
                    return Move(type=tok)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return f"Your previous reply had no valid column. Reply with exactly one of: {legal}."
