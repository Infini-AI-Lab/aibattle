"""Othello-lite 6x6 prompt template + move parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate


_RULES = (
    "You are playing Othello-lite (Reversi) on a 6x6 board. Rows are numbered "
    "1-6 (top to bottom) and columns A-F (left to right); a cell is a column "
    "letter then a row number, e.g. C3. B is Black, W is White, '.' is empty. "
    "On your turn you place one piece on an empty cell such that it flips at "
    "least one of the opponent's pieces: a flip happens when your new piece and "
    "another of your pieces bracket an unbroken line of opponent pieces in any "
    "of the eight directions (horizontal, vertical, or diagonal); every "
    "bracketed opponent piece becomes yours. If you have no flipping move you "
    "must pass. The game ends when both players have no move; whoever has more "
    "pieces wins."
)

_COORD_RE = re.compile(r"\b([A-Fa-f])\s*-?\s*([1-6])\b")


class OthelloLiteTemplate(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        legal = ", ".join(request.observation.legal_actions)
        if request.observation.legal_actions == ["pass"]:
            return (
                "You have no legal move, so you must pass. Respond with ONLY the "
                "word: pass (put it on the last line if you reason first)."
            )
        return (
            f"Choose one legal move from: {legal}.\n"
            "Respond with ONLY the move as a column letter and row number "
            "(e.g. C3), on the last line if you reason first."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = set(request.observation.legal_actions)
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        # Prefer the last line (the conclusion), then the whole text.
        for chunk in ([lines[-1]] if lines else []) + [raw]:
            if "pass" in legal and re.search(r"\bpass\b", chunk, re.IGNORECASE):
                return Move(type="pass")
            for m in _COORD_RE.finditer(chunk):
                coord = f"{m.group(1).upper()}{m.group(2)}"
                if coord in legal:
                    return Move(type=coord)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        if request.observation.legal_actions == ["pass"]:
            return "You have no legal move. Reply with exactly: pass."
        return (
            "Your previous reply had no valid move. Reply with exactly one of: "
            f"{legal}."
        )
