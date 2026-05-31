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


class GomokuTemplate(GameTemplate):
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        ctx = f"Match: {request.match.describe()}\n" if request.match else ""
        return (
            f"{ctx}{obs.rendered}\n\n"
            "Respond with ONLY a coordinate for an empty cell, e.g. E5 "
            "(column letter A-I, row number 1-9). Put it on the last line if you "
            "reason first."
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

    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        return (
            f"{self.render_prompt(request)}\n\n"
            "Your previous reply was not a valid empty cell. Reply with one "
            "coordinate like E5 that is currently empty."
        )
