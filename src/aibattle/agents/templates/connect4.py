"""Connect Four prompt template + column parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate


class Connect4Template(GameTemplate):
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        legal = ", ".join(obs.legal_actions)
        ctx = f"Match: {request.match.describe()}\n" if request.match else ""
        return (
            f"{ctx}{obs.rendered}\n\n"
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

    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"{self.render_prompt(request)}\n\n"
            f"Your previous reply had no valid column. Reply with exactly one of: {legal}."
        )
