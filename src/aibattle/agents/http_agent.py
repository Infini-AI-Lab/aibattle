"""External custom agent over HTTP.

POSTs the serialized ``AgentRequest`` to a user endpoint and reads back an
``AgentResponse``. This is the documented contract for remote agent pipelines:
the JSON shape is exactly the dataclasses in ``aibattle.types``.

Request JSON:
  {"game", "game_version", "player", "step_index", "instructions",
   "observation": {player, private, public, history, legal_actions, rendered}}
Response JSON:
  {"action", "message"?, "raw_output"?, "metadata"?}
"""

from __future__ import annotations

import asyncio

from ..types import INVALID, AgentRequest, AgentResponse
from .base import Agent


class HttpAgent(Agent):
    agent_type = "external"

    def __init__(self, *, name: str, url: str, timeout_s: float = 30.0):
        self.name = name
        self.url = url
        self.timeout_s = timeout_s

    def _payload(self, request: AgentRequest) -> dict:
        return {
            "game": request.game,
            "game_version": request.game_version,
            "player": request.player,
            "step_index": request.step_index,
            "instructions": request.instructions,
            "observation": request.observation.to_dict(),
            "match": request.match.to_dict() if request.match else None,
        }

    async def act(self, request: AgentRequest) -> AgentResponse:
        import requests  # local import; only needed for HTTP agents

        payload = self._payload(request)

        def _post():
            resp = requests.post(self.url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            return resp.json()

        try:
            data = await asyncio.to_thread(_post)
        except Exception as e:  # network/endpoint failure -> invalid, runner decides
            return AgentResponse(action=INVALID, metadata={"error": str(e)})

        return AgentResponse(
            action=data.get("action", INVALID),
            amount=data.get("amount"),
            message=data.get("message"),
            raw_output=data.get("raw_output"),
            metadata=data.get("metadata", {}),
        )
