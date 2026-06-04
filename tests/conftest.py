"""Shared pytest fixtures for the agent / harness test suite.

All default-tier tests run fully offline: no network, no real model. The model
is replaced by ``FakeModelClient`` (scripted ``ModelOutput`` responses), and
requests are built without spinning up a game via ``make_request``. The real
``GameTemplate``s (Kuhn discrete, Hold'em numeric) are used so prompt rendering
and tolerant parsing are exercised for real.
"""

from __future__ import annotations

import pytest

from aibattle.models.base import ModelClient, ModelOutput
from aibattle.types import AgentRequest, MatchContext, Observation


class FakeModelClient(ModelClient):
    """A scripted ModelClient test double.

    ``scripted`` is a list consumed in order on each ``generate`` call. Entries
    may be ``ModelOutput`` instances or bare strings (auto-wrapped into
    ``ModelOutput(content=...)``). Every call records the prompt and kwargs into
    ``self.calls``. Exhausting the script raises AssertionError so an
    over-generating loop is caught loudly rather than hanging.
    """

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._i = 0
        self.calls = []  # list of {"prompt", "temperature", "max_tokens"}

    @staticmethod
    def _as_output(item):
        if isinstance(item, ModelOutput):
            return item
        return ModelOutput(content=str(item))

    async def generate(self, prompt, *, temperature: float = 0.0, max_tokens: int = 256):
        self.calls.append({"prompt": prompt, "temperature": temperature,
                            "max_tokens": max_tokens})
        assert self._i < len(self._scripted), (
            f"FakeModelClient script exhausted after {self._i} calls "
            f"(too many generations?)"
        )
        out = self._as_output(self._scripted[self._i])
        self._i += 1
        return out

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ConstantModelClient(ModelClient):
    """A non-exhausting ModelClient that always returns the same text.

    Useful for end-to-end matches where an agent's ``act`` is called an unknown
    number of times (every step of every episode). Records nothing.
    """

    def __init__(self, text: str):
        self._text = text

    async def generate(self, prompt, *, temperature: float = 0.0, max_tokens: int = 256):
        return ModelOutput(content=self._text)


@pytest.fixture
def fake_client():
    """Factory: fake_client([...]) -> FakeModelClient with that script."""
    return lambda scripted: FakeModelClient(scripted)


def _make_request(*, numeric: bool = False, game: str = "kuhn_poker",
                  legal_actions=None, decision_seed=None, match=None,
                  player: str = "player_0", step_index: int = 0,
                  game_version: str = "1.0.0") -> AgentRequest:
    if numeric:
        legal = legal_actions if legal_actions is not None else ["fold", "call", "raise"]
        private = {"hole": ["As", "Kd"]}
        public = {"pot": 6, "to_call": 2,
                  "amount_range": {"raise": {"min": 4, "max": 200}}}
        rendered = ("Your hole cards: As Kd. Pot 6, to call 2. "
                    "Legal: " + ", ".join(legal) + ". raise range 4-200.")
    else:
        legal = legal_actions if legal_actions is not None else ["check", "bet"]
        private = {"card": "K"}
        public = {"pot": 2}
        rendered = "Your card: K. Pot 2. Legal: " + ", ".join(legal) + "."
    obs = Observation(player=player, private=private, public=public,
                      history=[], legal_actions=legal, rendered=rendered)
    return AgentRequest(
        game=game, game_version=game_version, player=player, observation=obs,
        instructions="Respond with exactly one legal action token.",
        step_index=step_index, decision_seed=decision_seed, match=match,
    )


@pytest.fixture
def make_request():
    """Factory fixture building a minimal AgentRequest (discrete or numeric)."""
    return _make_request


@pytest.fixture
def match_ctx():
    """A simple MatchContext for prompts that render match info."""
    return MatchContext(episode=0, total_episodes=10, you="me", standing={})
