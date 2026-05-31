"""Agent layer abstraction.

Every participant in the arena is an ``Agent``, whether it is a built-in
baseline, a model-backed wrapper, or an external pipeline. The runner only
ever sees this interface; it never knows how an agent works internally.

``act`` is async so that model/remote agents can do I/O without blocking;
local agents simply return immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AgentRequest, AgentResponse


class Agent(ABC):
    name: str          # instance label, e.g. "gpt-oss-default"
    agent_type: str    # "builtin" | "model" | "external"

    @abstractmethod
    async def act(self, request: AgentRequest) -> AgentResponse:
        """Return an action. May return an illegal one — the runner handles that."""
