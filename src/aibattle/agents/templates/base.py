"""Game template interface for the default model-backed agent.

A template knows how to (a) turn an ``AgentRequest`` into a model prompt,
(b) parse a model's raw text back into a legal action, and (c) produce a
repair nudge when parsing fails. It is the only game-specific part of the
otherwise generic ``ModelAgent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...types import Action, AgentRequest


class GameTemplate(ABC):
    @abstractmethod
    def render_prompt(self, request: AgentRequest) -> str:
        ...

    @abstractmethod
    def parse(self, raw: str, legal_actions: list) -> Optional[Action]:
        """Return a legal action, or None if nothing parseable was found."""

    @abstractmethod
    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        ...
