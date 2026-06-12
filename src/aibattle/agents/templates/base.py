"""Game template interface for the default model-backed agent.

A template knows how to (a) turn an ``AgentRequest`` into a model prompt,
(b) parse a model's raw text back into a legal action, and (c) produce a
repair nudge when parsing fails. It is the only game-specific part of the
otherwise generic ``ModelAgent``.

The prompt is assembled from four ordered sections so that agent *variants*
(e.g. a coached agent) can override one section without rewriting the whole
prompt::

    rules  ->  coaching  ->  state  ->  instruction

``rules`` and ``coaching`` are agent-side framing; ``state`` is the game's
rendered observation; ``instruction`` is the output-format demand. Empty
sections are skipped, the rest joined by a blank line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...types import AgentRequest, Move


def clean_answer_line(line: str) -> str:
    """A candidate final-answer line with markdown/punctuation decoration
    stripped (e.g. '**C5.**' or '- call:' -> 'C5' / 'call'), so a parser can
    test it for an EXACT action match before falling back to fuzzy scanning."""
    return line.strip().strip("*_`'\"()[]{}.:;,!->#= ").strip()


class GameTemplate(ABC):
    def __init__(self, *, coaching: str = "") -> None:
        # Coaching is just text: an optional line of process scaffolding to
        # inject after the rules. Empty = the plain agent. A 'coached' agent is
        # the same template built with this set, not a separate subclass.
        self._coaching = coaching

    # --- prompt sections: override to customize; render_prompt assembles them ---
    def rules(self, request: AgentRequest) -> str:
        """Natural-language rules of the game. Default: none (a game may instead
        carry its rules inside the rendered observation)."""
        return ""

    def coaching(self, request: AgentRequest) -> str:
        """The coaching line, inserted right after the rules. Default: the text
        passed at construction (empty for the plain agent). Override only if a
        game needs coaching that depends on the live ``request``."""
        return self._coaching

    def state(self, request: AgentRequest) -> str:
        """The decision context the model reasons over: optional match context
        followed by the game's rendered observation."""
        obs = request.observation
        ctx = f"Match: {request.match.describe()}\n" if request.match else ""
        return f"{ctx}{obs.rendered}"

    @abstractmethod
    def instruction(self, request: AgentRequest) -> str:
        """How to answer: the legal actions and the output-format demand."""

    def render_prompt(self, request: AgentRequest) -> str:
        sections = [
            self.rules(request),
            self.coaching(request),
            self.state(request),
            self.instruction(request),
        ]
        return "\n\n".join(s for s in sections if s)

    # --- parsing ---
    @abstractmethod
    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        """Return a Move (action type + optional amount), or None if unparseable."""

    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        """Re-issue the full prompt with a corrective nudge appended."""
        return f"{self.render_prompt(request)}\n\n{self.repair_hint(request, bad_output)}"

    @abstractmethod
    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        """The corrective sentence appended when the previous answer failed to parse."""
