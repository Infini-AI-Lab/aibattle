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
        # The answer is the conclusion, so scan lines bottom-up and return the
        # first line that names exactly ONE distinct legal column. This:
        #   * ignores the echoed board's "0 1 2 3 4 5 6" header (many distinct
        #     legal digits -> never a single answer), fenced or not;
        #   * accepts a clean answer or a repetition loop ("4", "4 4 4");
        #   * rejects an ambiguous line ("4, not 2") -> None -> repair, then the
        #     runner's random fallback.
        # (Board rows render as dots/X/O, so they contribute no digits.)
        for ln in reversed([l for l in raw.splitlines() if l.strip()]):
            cols = {t for t in re.findall(r"\d+", ln) if t in legal}
            if len(cols) == 1:
                return Move(type=next(iter(cols)))
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return f"Your previous reply had no valid column. Reply with exactly one of: {legal}."
