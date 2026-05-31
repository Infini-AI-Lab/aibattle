"""Game template interface for the default model-backed agent.

A template knows how to (a) turn an ``AgentRequest`` into a model prompt,
(b) parse a model's raw text back into a legal action, and (c) produce a
repair nudge when parsing fails. It is the only game-specific part of the
otherwise generic ``ModelAgent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...types import AgentRequest, Move


class GameTemplate(ABC):
    @abstractmethod
    def render_prompt(self, request: AgentRequest) -> str:
        ...

    @abstractmethod
    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        """Return a Move (action type + optional amount), or None if unparseable."""

    @abstractmethod
    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        ...
